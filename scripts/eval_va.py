"""Closed-loop evaluation for sft_online concat-format models.

Training-inference alignment (CRITICAL):
  - Uses `thor_view_env.ThorViewEnv` which renders at 640×360 and burns
    CURRENT / TARGET labels via the SAME `burn_label` function the SFT data
    was built with.
  - Per-turn user text comes from `thor_view_env.build_concat_user_text`
    (single source of truth shared with build_sft_online_data.py).
  - System prompt = `thor_view_env.SYSTEM_PROMPT` byte-for-byte.
  - Multi-turn chat history accumulates per task — the model sees the same
    structure it was trained on (system + N×[user_obs, assistant_action]).
  - chat_template_kwargs={"enable_thinking": False} matches training.

This eval CANNOT be substituted with scripts/evaluate.py + sft_single.yaml,
because that pipeline uses a per-step prompt with "Your recent actions:"
text block — a DIFFERENT input distribution than what sft_online learned.

GPU layout (4×H100):
  - vLLM serve (external): GPU 2,3 (TP=2)
  - THOR env workers:      GPU 0,1 (--gpu_ids 0 1 --procs_per_gpu 2 = 4 workers)

Usage:
    python scripts/eval_sft_online.py \
        --task_file data/tasks/eval.json \
        --model_name sft_online_v1 \
        --api_base http://localhost:8000/v1 \
        --output_dir outputs/eval_sft_online_v1 \
        --gpu_ids 0 1 --procs_per_gpu 2 \
        --max_steps 30 \
        --resume
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import defaultdict
from multiprocessing import Lock, Process, Queue
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tvrbench.online_rl.thor_view_env import (
    IMAGE_PH,
    SYSTEM_PROMPT,
    ThorViewEnv,
)
from tvrbench.evaluation.metrics import (
    horizon_error,
    is_exact_match,
    position_error,
    rotation_error,
)


# ---------------------------------------------------------------------------
# OpenAI content helpers
# ---------------------------------------------------------------------------

def _img_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _obs_to_oai_content(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert env obs (with <image> placeholders + PIL multi_modal_input) to
    OpenAI chat content blocks. Order of <image> in obs_str must match the
    order of images in multi_modal_input[IMAGE_PH] — env guarantees this.
    """
    text = obs["obs_str"]
    images = obs["multi_modal_input"][IMAGE_PH]
    parts = text.split(IMAGE_PH)
    assert len(parts) - 1 == len(images), (
        f"obs has {len(parts)-1} <image> placeholders but {len(images)} images"
    )
    blocks: List[Dict[str, Any]] = []
    for i, part in enumerate(parts):
        if part:
            blocks.append({"type": "text", "text": part})
        if i < len(images):
            blocks.append({
                "type": "image_url",
                "image_url": {"url": _img_to_data_url(images[i])},
            })
    return blocks


# ---------------------------------------------------------------------------
# Action parsing — tolerant of </think> artifact from enable_thinking=False
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"Action\s*:\s*([A-Za-z]+)", re.IGNORECASE)

_VALID_ACTIONS = {
    "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
    "RotateLeft", "RotateRight", "LookUp", "LookDown", "Stop",
}


def parse_action(text: str) -> Optional[str]:
    """Return the canonical action name, or None if unparseable."""
    if not text:
        return None
    text = text.replace("</think>", "").strip()
    m = _ACTION_RE.search(text)
    if not m:
        return None
    cand = m.group(1)
    # Case-normalize against known actions
    for v in _VALID_ACTIONS:
        if cand.lower() == v.lower():
            return v
    return None


# ---------------------------------------------------------------------------
# Per-task eval
# ---------------------------------------------------------------------------

def eval_one_task(env: ThorViewEnv, client, model_name: str, task: Dict,
                  max_steps_ithor: int, max_steps_procthor: int,
                  request_kwargs: Dict, log_fp=None) -> Dict:
    """Run one closed-loop eval. Returns trajectory record dict.

    max_steps is dataset-aware (iTHOR vs ProcTHOR) since ProcTHOR BFS is 10-20
    vs iTHOR 2-8.
    """
    max_steps = (max_steps_procthor if task.get("dataset") == "procthor"
                 else max_steps_ithor)
    # Use sync env API directly (skip asyncio overhead).
    obs, info = env._sync_reset(task)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _obs_to_oai_content(obs)},
    ]

    actions: List[str] = []
    raw_first: Optional[str] = None
    raw_last: Optional[str] = None
    n_invalid = 0
    stopped = False

    for step in range(max_steps):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **request_kwargs,
            )
            text = resp.choices[0].message.content or ""
        except Exception as e:
            text = f"__API_ERROR__ {type(e).__name__}: {e}"
            if log_fp:
                log_fp.write(f"  step {step} API error: {e}\n"); log_fp.flush()

        if raw_first is None:
            raw_first = text
        raw_last = text

        messages.append({"role": "assistant", "content": text})

        action = parse_action(text)
        if action is None:
            n_invalid += 1
            # No env progress, but a turn is "spent". We still call env.step
            # with the raw text so format_invalid_penalty applies; env stays
            # at same pose.
            action_for_env = "InvalidAction"
        else:
            action_for_env = action
        actions.append(action_for_env)

        new_obs, reward, done, step_info = env._sync_step(f"Action: {action_for_env}")

        if action_for_env == "Stop":
            stopped = True
            done = True

        if done:
            break

        messages.append({"role": "user", "content": _obs_to_oai_content(new_obs)})

    # Compute final metrics.
    state = env._env.get_state()
    target = task["target"]
    pos_err = position_error(state, target)
    rot_err = rotation_error(state, target)
    hor_err = horizon_error(state, target)
    success = stopped and is_exact_match(state, target)

    return {
        "task_id": task["task_id"],
        "dataset": task["dataset"],
        "scene": task["scene"],
        "difficulty": task["difficulty"],
        "result": {
            "success": success,
            "stopped": stopped,
            "num_steps": len(actions),
            "n_invalid_actions": n_invalid,
            "final_pos_error": pos_err,
            "final_rot_error": rot_err,
            "final_hor_error": hor_err,
        },
        "actions": actions,
        "start": task["start"],
        "target": task["target"],
        "raw_first_response": (raw_first or "")[:300],
        "raw_last_response":  (raw_last or "")[:300],
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker_fn(worker_id: int, gpu_id: int, task_queue: Queue, init_lock: Lock,
              args, results_dir: str):
    """One worker process: dedicated ThorViewEnv, sequentially eval tasks."""
    from openai import OpenAI

    log_path = os.path.join(results_dir, f"worker_{worker_id}.log")
    log_fp = open(log_path, "a", buffering=1)

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] [w{worker_id}|gpu{gpu_id}] {msg}"
        log_fp.write(line + "\n"); log_fp.flush()
        print(line, flush=True)

    log("starting")

    # Serialize Unity/Vulkan startup.
    with init_lock:
        log("acquiring ThorViewEnv...")
        # Set env max_steps to the larger of the two caps so env doesn't
        # auto-terminate before our task-specific outer cap kicks in.
        env_max_steps = max(args.max_steps_ithor, args.max_steps_procthor)
        env = ThorViewEnv(env_config={
            "gpu_device": gpu_id,
            "width": 640,
            "height": 360,
            "procthor_dir": "data/procthor-10k",
            "max_steps": env_max_steps,
        })
        # Warm up by force-creating the controller (else first task pays the cost).
        env._ensure_env()
        log("ThorViewEnv ready")

    client = OpenAI(base_url=args.api_base, api_key="EMPTY")

    # enable_thinking=False  → server jinja injects empty closed <think></think>;
    #                          model emits "Action: X" directly (no-CoT path).
    # enable_thinking=True   → with our custom jinja
    #                          (configs/qwen3_5_preserve_history_think.jinja),
    #                          no <think> injection; model emits <think>...</think>
    #                          itself, byte-identical to LF-training CoT data.
    request_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "extra_body": {
            "top_k": args.top_k,
            "chat_template_kwargs": {"enable_thinking": bool(args.cot)},
        },
    }

    n_done = 0
    n_success = 0
    while True:
        try:
            item = task_queue.get(timeout=2.0)
        except Exception:
            continue
        if item is None:
            break
        task = item
        task_id = task["task_id"]

        out_dir = os.path.join(results_dir, task_id)
        out_file = os.path.join(out_dir, "trajectory.json")
        if args.resume and os.path.exists(out_file):
            n_done += 1
            if n_done % 20 == 0:
                log(f"resume-skipped {n_done}")
            continue

        os.makedirs(out_dir, exist_ok=True)
        log(f"  task {task_id} starting")
        try:
            t0 = time.time()
            record = eval_one_task(
                env, client, args.model_name, task,
                args.max_steps_ithor, args.max_steps_procthor,
                request_kwargs, log_fp=log_fp,
            )
            with open(out_file, "w") as f:
                json.dump(record, f, indent=2)
            dt = time.time() - t0
            res = record["result"]
            tag = "SUCCESS" if res["success"] else ("STOPPED" if res["stopped"] else "TIMEOUT")
            log(f"  task {task_id} {tag} "
                f"steps={res['num_steps']} inv={res['n_invalid_actions']} "
                f"pos_err={res['final_pos_error']:.3f} dt={dt:.1f}s")
            if res["success"]:
                n_success += 1
            n_done += 1
            if n_done % 10 == 0:
                log(f"progress: {n_done} done, {n_success} success")
        except Exception as e:
            log(f"  task {task_id} ERROR: {type(e).__name__}: {e}")
            log_fp.write(traceback.format_exc() + "\n"); log_fp.flush()

    log(f"worker done. {n_done} tasks, {n_success} success")

    try:
        env._env.close()
    except Exception:
        pass
    log_fp.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Aggregate (extracted into a function so --aggregate_only can call it without
# spinning up workers / vLLM).
# ---------------------------------------------------------------------------

# Maps (dataset, difficulty) → paper-table column abbreviation.
_CAT_LABELS = {
    ("ithor", "easy"):    "SR-easy",
    ("ithor", "hard"):    "SR-hard",
    ("procthor", "easy"): "LR-easy",
    ("procthor", "hard"): "LR-hard",
}
_CAT_ORDER = [("ithor", "easy"), ("ithor", "hard"),
              ("procthor", "easy"), ("procthor", "hard")]


def _fmt_pct(num, den, width=4):
    """'XX.X' or '—' if den==0."""
    if not den:
        return "—".rjust(width)
    return f"{100*num/den:{width}.1f}"


def _spl(success: bool, num_steps: int, opt_steps: Optional[int]) -> float:
    """Standard SPL: success × opt / max(opt, actual). 0 if unsuccessful
    or no optimal-steps reference; clamped to ≤1 if model is faster than opt."""
    if not success or opt_steps is None or num_steps <= 0:
        return 0.0
    return min(opt_steps, num_steps) / max(opt_steps, num_steps)


def aggregate(results_dir: str, model_name: str,
              optimal_steps_path: Optional[str] = None) -> None:
    """Read per-task records, compute SR / SPL / Stop / F-stop, write
    `results.json` + `summary.txt`, print a copy-pasteable text dump."""
    print(f"\n[aggregate] reading per-task results from {results_dir}")
    records = []
    for entry in sorted(os.listdir(results_dir)):
        traj_path = os.path.join(results_dir, entry, "trajectory.json")
        if os.path.isfile(traj_path):
            with open(traj_path) as f:
                records.append(json.load(f))

    if not records:
        print("[aggregate] no records! something went wrong.")
        return

    # Load optimal-step lookup for SPL.
    opt_lookup: Dict[str, int] = {}
    if optimal_steps_path and os.path.exists(optimal_steps_path):
        with open(optimal_steps_path) as f:
            raw = json.load(f)
        for tid, v in raw.items():
            if v.get("feasible") and v.get("optimal_steps") is not None:
                opt_lookup[tid] = v["optimal_steps"]
        print(f"[aggregate] loaded optimal_steps for {len(opt_lookup)} tasks "
              f"from {optimal_steps_path}")
    else:
        print(f"[aggregate] no optimal_steps file at {optimal_steps_path}; "
              f"SPL will be reported as —")

    n = len(records)
    n_success = sum(1 for r in records if r["result"]["success"])
    n_stopped = sum(1 for r in records if r["result"]["stopped"])
    sum_steps = sum(r["result"]["num_steps"] for r in records)
    spl_terms = [_spl(r["result"]["success"], r["result"]["num_steps"],
                      opt_lookup.get(r["task_id"]))
                 for r in records]
    spl_overall = sum(spl_terms) / n if opt_lookup else None

    by_cat: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in records:
        by_cat[(r["dataset"], r["difficulty"])].append(r)

    # Per-category numbers.
    per_cat = {}
    for key in _CAT_ORDER:
        rs = by_cat.get(key, [])
        if not rs:
            per_cat[key] = None
            continue
        ns = len(rs)
        nsu = sum(1 for r in rs if r["result"]["success"])
        nst = sum(1 for r in rs if r["result"]["stopped"])
        msteps = sum(r["result"]["num_steps"] for r in rs) / ns
        if opt_lookup:
            spl_c = sum(_spl(r["result"]["success"], r["result"]["num_steps"],
                              opt_lookup.get(r["task_id"]))
                        for r in rs) / ns
        else:
            spl_c = None
        mean_pos = sum(r["result"]["final_pos_error"] for r in rs) / ns
        mean_rot = sum(r["result"]["final_rot_error"] for r in rs) / ns
        mean_hor = sum(r["result"]["final_hor_error"] for r in rs) / ns
        per_cat[key] = dict(n=ns, n_success=nsu, n_stopped=nst,
                            mean_steps=msteps, spl=spl_c,
                            mean_pos_err=mean_pos, mean_rot_err=mean_rot,
                            mean_hor_err=mean_hor)

    # ---- Save JSON summary (richer than the text dump). ----
    summary = {
        "model_name": model_name,
        "n_tasks": n,
        "n_success": n_success,
        "n_stopped": n_stopped,
        "overall_success_rate": n_success / n,
        "overall_stop_rate": n_stopped / n,
        "overall_false_stop_given_stop":
            (n_stopped - n_success) / n_stopped if n_stopped else None,
        "overall_false_stop_over_all":
            (n_stopped - n_success) / n,
        "overall_spl": spl_overall,
        "mean_num_steps": sum_steps / n,
        "mean_invalid_actions":
            sum(r["result"]["n_invalid_actions"] for r in records) / n,
        "mean_pos_error":
            sum(r["result"]["final_pos_error"] for r in records) / n,
        "mean_rot_error":
            sum(r["result"]["final_rot_error"] for r in records) / n,
        "mean_hor_error":
            sum(r["result"]["final_hor_error"] for r in records) / n,
        "by_category": {
            f"{ds}_{diff}": (
                None if per_cat[(ds, diff)] is None else {
                    "label": _CAT_LABELS[(ds, diff)],
                    "n": per_cat[(ds, diff)]["n"],
                    "n_success": per_cat[(ds, diff)]["n_success"],
                    "n_stopped": per_cat[(ds, diff)]["n_stopped"],
                    "success_rate":
                        per_cat[(ds, diff)]["n_success"] / per_cat[(ds, diff)]["n"],
                    "stop_rate":
                        per_cat[(ds, diff)]["n_stopped"] / per_cat[(ds, diff)]["n"],
                    "mean_num_steps": per_cat[(ds, diff)]["mean_steps"],
                    "spl": per_cat[(ds, diff)]["spl"],
                    "mean_pos_error": per_cat[(ds, diff)]["mean_pos_err"],
                    "mean_rot_error": per_cat[(ds, diff)]["mean_rot_err"],
                    "mean_hor_error": per_cat[(ds, diff)]["mean_hor_err"],
                })
            for (ds, diff) in _CAT_ORDER
        },
    }
    out_json = os.path.join(results_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Build text dump (师兄 copy-pastes this block back). ----
    lines = []
    lines.append(f"=== TVRBench EVAL: {model_name} ===")
    lines.append(f"n_tasks: {n}")
    lines.append("")
    lines.append("Overall:")
    lines.append(f"  SR:           {_fmt_pct(n_success, n)}% ({n_success}/{n})")
    if spl_overall is not None:
        lines.append(f"  SPL:          {spl_overall:.3f}")
    else:
        lines.append(f"  SPL:          —  (no optimal_steps file)")
    lines.append(f"  Mean steps:   {sum_steps/n:.1f}")
    lines.append(f"  Stop rate:    {_fmt_pct(n_stopped, n)}% ({n_stopped}/{n})")
    fstop_given = _fmt_pct(n_stopped - n_success, n_stopped)
    fstop_all = _fmt_pct(n_stopped - n_success, n)
    lines.append(f"  F-stop|stop:  {fstop_given}% ({n_stopped - n_success}/{n_stopped})"
                 f"    # of Stops, fraction at wrong pose")
    lines.append(f"  F-stop|all:   {fstop_all}% ({n_stopped - n_success}/{n})"
                 f"     # of all episodes, fraction stopped-wrong")
    lines.append(f"  Final |Δp|:   {summary['mean_pos_error']:.3f} m"
                 f"            # mean position error")
    lines.append(f"  Final |Δθ|:   {summary['mean_rot_error']:.1f}°"
                 f"               # mean body yaw error")
    lines.append(f"  Final |Δφ|:   {summary['mean_hor_error']:.1f}°"
                 f"               # mean head pitch error")
    lines.append("")
    lines.append("Per-category:")
    lines.append("  Category    n     SR(%)   SPL    Steps   Stop(%)  "
                 "F-stop|stop(%)  |Δp|(m)  |Δθ|°  |Δφ|°")
    for key in _CAT_ORDER:
        c = per_cat[key]
        label = _CAT_LABELS[key]
        if c is None:
            lines.append(f"  {label:<10}  —     —       —      —       —        "
                         "—              —       —     —")
            continue
        sr = _fmt_pct(c["n_success"], c["n"])
        spl_s = f"{c['spl']:.3f}" if c["spl"] is not None else "  —  "
        stop_pct = _fmt_pct(c["n_stopped"], c["n"])
        fs = _fmt_pct(c["n_stopped"] - c["n_success"], c["n_stopped"])
        lines.append(f"  {label:<10}  {c['n']:<4d}  {sr}    {spl_s}  "
                     f"{c['mean_steps']:5.1f}   {stop_pct}     {fs}        "
                     f"{c['mean_pos_err']:5.3f}  "
                     f"{c['mean_rot_err']:5.1f}  {c['mean_hor_err']:5.1f}")
    lines.append("=== end ===")
    text_dump = "\n".join(lines)

    out_txt = os.path.join(results_dir, "summary.txt")
    with open(out_txt, "w") as f:
        f.write(text_dump + "\n")

    print()
    print(text_dump)
    print()
    print(f"[aggregate] saved {out_json}")
    print(f"[aggregate] saved {out_txt}  (copy the block above into the paper)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task_file", required=True)
    ap.add_argument("--model_name", required=True,
                    help="must match --served-model-name in vLLM serve")
    ap.add_argument("--api_base", default="http://localhost:8000/v1")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--gpu_ids", type=int, nargs="+", default=[0])
    ap.add_argument("--procs_per_gpu", type=int, default=2)
    ap.add_argument("--max_steps_ithor", type=int, default=30,
                    help="hard cap for iTHOR tasks (BFS 2-8, short range)")
    ap.add_argument("--max_steps_procthor", type=int, default=40,
                    help="hard cap for ProcTHOR tasks (BFS 10-20, cross-room)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="greedy by default (reproducible)")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--max_tokens", type=int, default=64,
                    help="assistant only outputs 'Action: X' — 64 is plenty. "
                         "For --cot models bump to 256+ to fit <think>...</think>.")
    ap.add_argument("--cot", action="store_true",
                    help="CoT model. Passes chat_template_kwargs.enable_thinking=True "
                         "so server-side template does NOT inject empty <think></think> "
                         "(it lets the model emit its own <think> as the first token, "
                         "matching LF-training response format). Requires vLLM to be "
                         "served with --chat-template that preserves historical <think> "
                         "(configs/qwen3_5_preserve_history_think.jinja).")
    ap.add_argument("--resume", action="store_true",
                    help="skip tasks with existing trajectory.json")
    ap.add_argument("--optimal_steps_file", type=str,
                    default="data/tasks/eval_optimal_steps.json",
                    help="precomputed optimal steps for SPL "
                         "(produced by precompute_eval_optimal_steps.py).")
    ap.add_argument("--aggregate_only", action="store_true",
                    help="Skip eval; just re-aggregate existing trajectory.json "
                         "records in output_dir. Useful when adding new metrics "
                         "to old runs.")
    ap.add_argument("--limit_per_category", type=int, default=None,
                    help="Smoke-test mode: keep only the first N tasks per "
                         "(dataset, difficulty) bucket so all 4 categories "
                         "appear in the text dump.")
    args = ap.parse_args()

    # --aggregate_only fast path: no task_file load, no vLLM, no workers.
    if args.aggregate_only:
        results_dir = os.path.join(args.output_dir, args.model_name,
                                   "concat", "eval")
        if not os.path.isdir(results_dir):
            print(f"[aggregate_only] dir not found: {results_dir}")
            sys.exit(2)
        aggregate(results_dir, args.model_name,
                  optimal_steps_path=args.optimal_steps_file)
        return

    with open(args.task_file) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks from {args.task_file}")

    if args.limit_per_category:
        per_cat_count: Dict[Tuple[str, str], int] = defaultdict(int)
        kept = []
        for t in tasks:
            key = (t["dataset"], t["difficulty"])
            if per_cat_count[key] < args.limit_per_category:
                kept.append(t)
                per_cat_count[key] += 1
        tasks = kept
        print(f"[limit] kept {len(tasks)} tasks "
              f"(<={args.limit_per_category} per category)")

    cat_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for t in tasks:
        cat_counts[(t["dataset"], t["difficulty"])] += 1
    print("Categories: " + ", ".join(f"{d}/{diff}={n}"
                                     for (d, diff), n in sorted(cat_counts.items())))

    # Output layout: {output_dir}/{model_name}/concat/eval/
    results_dir = os.path.join(args.output_dir, args.model_name, "concat", "eval")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Output dir: {results_dir}")

    # Health-check vLLM
    try:
        from openai import OpenAI
        client = OpenAI(base_url=args.api_base, api_key="EMPTY")
        models = client.models.list()
        names = [m.id for m in models.data]
        print(f"[health] vLLM models available: {names}")
        if args.model_name not in names:
            print(f"  WARN: model_name {args.model_name!r} not in served list")
    except Exception as e:
        print(f"[health] vLLM unreachable: {e}")
        print(f"  Make sure vllm serve is running at {args.api_base}")
        return

    # Build task queue.
    task_queue: Queue = Queue()
    init_lock = Lock()
    for t in tasks:
        task_queue.put(t)

    n_workers = len(args.gpu_ids) * args.procs_per_gpu
    for _ in range(n_workers):
        task_queue.put(None)

    workers: List[Process] = []
    wid = 0
    for gpu in args.gpu_ids:
        for _ in range(args.procs_per_gpu):
            p = Process(
                target=worker_fn,
                args=(wid, gpu, task_queue, init_lock, args, results_dir),
            )
            p.start()
            workers.append(p)
            wid += 1
    print(f"Launched {n_workers} workers on GPUs {args.gpu_ids}")

    # Background progress reporter — polls results_dir every N seconds, prints
    # overall completion + ETA. Threading (not multiprocessing) so it can see
    # main process's state.
    stop_event = threading.Event()

    def progress_loop():
        t0 = time.time()
        n0 = 0
        try:
            n0 = sum(1 for e in os.listdir(results_dir)
                     if os.path.isfile(os.path.join(results_dir, e, "trajectory.json")))
        except Exception:
            pass
        total = len(tasks)
        print(f"[progress] starting from {n0}/{total} "
              f"({'resume' if n0 > 0 else 'fresh'})")
        last_print = 0
        while not stop_event.wait(timeout=15):
            try:
                entries = os.listdir(results_dir)
            except Exception:
                continue
            done = 0
            n_succ = 0
            n_stop = 0
            for e in entries:
                p = os.path.join(results_dir, e, "trajectory.json")
                if not os.path.isfile(p):
                    continue
                done += 1
                try:
                    with open(p) as f:
                        r = json.load(f)
                    res = r.get("result", {})
                    if res.get("success"):
                        n_succ += 1
                    if res.get("stopped"):
                        n_stop += 1
                except Exception:
                    pass
            if done == last_print:
                continue
            last_print = done
            dt = time.time() - t0
            new_done = done - n0
            if new_done > 0 and dt > 0:
                rate = new_done / dt
                remain = total - done
                eta_s = remain / rate if rate > 0 else 0
                eta_m = int(eta_s / 60)
                eta_h = eta_m // 60
                eta_mm = eta_m % 60
                eta_str = f"{eta_h}h{eta_mm:02d}m" if eta_h > 0 else f"{eta_mm}m"
                print(f"[progress] {done}/{total} ({100*done/total:.1f}%) | "
                      f"success {n_succ}/{done} ({100*n_succ/max(done,1):.1f}%) | "
                      f"stopped {n_stop}/{done} | rate={rate:.2f}/s ETA={eta_str}")
            else:
                print(f"[progress] {done}/{total} (warming up...)")

    progress_thread = threading.Thread(target=progress_loop, daemon=True)
    progress_thread.start()

    for p in workers:
        p.join()

    # Stop progress reporter
    stop_event.set()
    progress_thread.join(timeout=5)

    # Aggregate (writes results.json + summary.txt + prints text dump).
    aggregate(results_dir, args.model_name,
              optimal_steps_path=args.optimal_steps_file)


if __name__ == "__main__":
    main()
