"""Single-instance async AI2-THOR env for online RL viewpoint matching.

Wraps `tvrbench.env.thor_env.ThorEnv` with the GymImageEnv async protocol
(VAGEN pattern, but no runtime dep on `vagen`). Obs schema and human-message
template match SFT data byte-for-byte; see `reference_sft_data_format.md`.

Interface:
    env = ThorViewEnv(env_config={"gpu_device": 0, ...})
    obs = await env.system_prompt()               # constant 942-char system
    obs, info = await env.reset(task=<task_dict>) # task from data/tasks/rl.json
    obs, reward, done, info = await env.step(action_str)
    await env.close()

obs = {"obs_str": "...<image>...", "multi_modal_input": {"<image>": [PIL.Image, ...]}}
Where the two images are always [current, target] in that order.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import threading
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tvrbench.env.thor_env import ACTION_NAMES, HORIZON_MAX, HORIZON_MIN, ThorEnv
from tvrbench.evaluation.metrics import (
    horizon_error,
    is_exact_match,
    position_error,
    rotation_error,
)


# Image placeholder used in obs_str (matches SFT data).
IMAGE_PH = "<image>"


# ---------------------------------------------------------------------------
# System prompt (constant 942 chars, identical for sft_single & sft_cot_think_v2)
# ---------------------------------------------------------------------------

# Hardcoded so env works without runtime data-file dependency. Verified byte-
# identical to data[0]["system"] in both sft_train.json and sft_train_cot_think_v2.json.
SYSTEM_PROMPT = (
    "You are a navigation agent in an indoor environment. "
    "Your task is to navigate and adjust your viewpoint to PRECISELY match a target image. "
    "You must match the exact position, orientation, and camera angle "
    "— the goal is for your observation to look identical to the target.\n\n"
    "Available actions:\n"
    "- MoveAhead: Move forward 0.25m\n"
    "- MoveBack: Move backward 0.25m\n"
    "- MoveLeft: Move left 0.25m\n"
    "- MoveRight: Move right 0.25m\n"
    "- RotateRight: Rotate clockwise 45°\n"
    "- RotateLeft: Rotate counter-clockwise 45°\n"
    "- LookUp: Tilt camera up 30°\n"
    "- LookDown: Tilt camera down 30°\n"
    "- Stop: Declare that you have reached the target viewpoint\n\n"
    "You will receive:\n"
    "1. Your current observation\n"
    "2. Your recent action history (if available)\n"
    "3. The target viewpoint you need to match\n\n"
    "Use your action history to avoid repeating ineffective actions "
    "(e.g. if MoveAhead caused a collision, try a different direction).\n\n"
    "You MUST respond in exactly this format:\n"
    "Action: <action name>"
)


def burn_label(image_array: np.ndarray, label: str,
               bg_color: Tuple[int, int, int] = (60, 60, 60)) -> Image.Image:
    """Burn a "CURRENT" or "TARGET" badge into the top-right corner of an
    image. Matches the visual style of `base_agent._burn_label` (eval time)
    and `build_sft_data.burn_label` (no_concat SFT).

    Font size auto-scales with image width so the badge stays at ~7% of
    image height across rendering resolutions:
        width=1280 → font 32  (eval / old SFT)
        width=640  → font 16  (online RL + new SFT online)
    Padding/inner margin scale proportionally.

    Args:
        image_array: HWC uint8 ndarray straight from AI2-THOR.
        label: "CURRENT" or "TARGET".

    Returns:
        PIL Image with the badge burned in. Callers can `.save()` or pass it
        through to multi_modal_input directly.
    """
    img = Image.fromarray(image_array.astype(np.uint8)).copy()
    draw = ImageDraw.Draw(img)

    # Scale: 32px at width=1280 baseline.
    scale = img.width / 1280
    font_size = max(12, round(32 * scale / 2) * 2)  # nearest even, floor 12
    pad = max(4, int(8 * scale))
    inner = max(5, int(10 * scale))

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    W = img.width
    rx1 = W - tw - inner * 2 - pad
    ry1 = pad
    rx2 = W - pad
    ry2 = pad + th + inner * 2
    draw.rectangle([rx1, ry1, rx2, ry2], fill=bg_color)
    cx = (rx1 + rx2) // 2
    cy = (ry1 + ry2) // 2
    draw.text((cx, cy), label, fill=(255, 255, 255), font=font, anchor="mm")
    return img


def build_concat_user_text(valid_actions: List[str]) -> str:
    """Per-turn user text in concat multi-turn mode (history is the chat —
    do NOT inject a "Your recent actions" text block; that's the no_concat /
    SFT-aligned path).

    Used by both `ThorViewEnv._build_obs_str` at RL rollout time and by
    `scripts/build_sft_online_data.py` at SFT data build time. Single source
    of truth — they cannot drift.
    """
    return "\n".join([
        "Your CURRENT observation:",
        IMAGE_PH,
        "TARGET viewpoint you must match:",
        IMAGE_PH,
        f"Valid actions at this step: {', '.join(valid_actions)}",
    ])


def verify_system_prompt_matches_sft(sft_train_json_path: str) -> bool:
    """Optional sanity check at startup: SYSTEM_PROMPT == data[0]['system']."""
    with open(sft_train_json_path) as f:
        data = json.load(f)
    return data[0]["system"] == SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ThorViewEnvConfig:
    gpu_device: int = 0
    # 1280/2 × 720/2 — half-resolution to keep visual tokens ~220 each so long
    # concat trajectories don't OOM. See project_online_rl_plan.md.
    width: int = 640
    height: int = 360
    fov: int = 60
    quality: str = "Ultra"
    # PoliFormer-style reward (see Zeng et al. 2024, App. B.1):
    #   per-step:
    #     - step_penalty (efficiency pressure, breaks ties between trajectories)
    #     - format_valid_reward  if model emitted parseable Action: X
    #     - format_invalid_penalty if it didn't (drift protection only — smoke
    #       showed base Qwen3.5-9B already emits strict 100% of the time)
    #     - progress_scale × Δpose_distance  IFF new minimum distance reached
    #       (asymmetric, no penalty for moving away — encourages exploration;
    #       only-new-min prevents oscillation reward farming)
    #   terminal:
    #     - +success_reward  on Stop at target pose
    #     - -stop_fail_penalty  on Stop NOT at target (anti-cheap-stop)
    step_penalty: float = 0.01
    format_valid_reward: float = 0.005
    format_invalid_penalty: float = 0.01
    progress_scale: float = 1.0
    # 2026-05-19: scaled down from (success=10, stop_fail=1) to align with
    # Yu et al 2025 ("Thinking in 360°") reward magnitude (~1.0). Old scale
    # had pg_loss ~10x KL constraint → β=0.01 effectively no-op → mode collapse
    # to 8-turn Stop-gamble policy within 1 PPO step. New scale: trajectory
    # reward in [-1, +2] range matches KL magnitude → β actually constrains.
    success_reward: float = 1.0
    stop_fail_penalty: float = 0.5
    # pose_distance weights — position dominates (PoliFormer uses pure L2);
    # rotation & horizon folded in via "how many 45°/30° steps away".
    pose_rot_weight: float = 0.25
    pose_hor_weight: float = 0.25
    # Termination
    max_steps: int = 30
    # ProcTHOR support: scenes with id `procthor_XXXXX` need a house dict
    # loaded from this dir's `train.jsonl.gz` (line index = XXXXX).
    procthor_dir: str = "data/procthor-10k"


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


# Tier-1 STRICT: whole response is just "Action: X" with at most a leading
# `</think>\n\n` artifact from Qwen3.5 chat-template enable_thinking=False.
# Anything else (CoT-style wrap, extra commentary, etc.) demotes to medium.
_STRICT_RE = re.compile(
    r"^\s*(?:</think>\s*)?Action:\s*(?P<action>\w+)\.?\s*$",
    re.DOTALL,
)
# Tier-2 MEDIUM: "Action: <name>" keyword appears somewhere in the response.
_KEYWORD_RE = re.compile(r"Action:\s*(?P<action>\w+)")
# Tier-3 WEAK: any of the 9 action names appears as a whole word, no keyword.
_ANY_ACTION_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in ACTION_NAMES) + r")\b")


def score_response(response: str, weak: float, medium: float, strict: float
                   ) -> Tuple[Optional[str], float, bool]:
    """Return (canonical_action_or_None, format_score, is_strict).

    Tier ladder, mutually exclusive (NOT additive within a step):
      strict (^Action: X$ tolerant of </think> prefix)  → strict_reward
      medium ("Action: X" keyword anywhere)             → medium_reward
      weak   (any action name as whole word)            → weak_reward
      none                                              → 0.0
    """
    if not response:
        return None, 0.0, False
    m = _STRICT_RE.match(response)
    if m:
        token = m.group("action")
        if token in ACTION_NAMES:
            return token, strict, True
    m = _KEYWORD_RE.search(response)
    if m:
        token = m.group("action")
        if token in ACTION_NAMES:
            return token, medium, False
    m = _ANY_ACTION_RE.search(response)
    if m:
        return m.group(1), weak, False
    return None, 0.0, False


class ThorViewEnv:
    """AI2-THOR viewpoint-matching env (async GymImageEnv protocol)."""

    # Class-level cache for ProcTHOR house dicts. All instances in a Ray worker
    # share this — loading the same scene twice is wasteful. Lazy-populate on
    # first request via _resolve_scene().
    _PROCTHOR_CACHE: Dict[str, Dict] = {}
    _CACHE_LOCK = threading.Lock()

    def __init__(self, env_config: Optional[Dict[str, Any]] = None):
        env_config = env_config or {}
        cfg_keys = {f.name for f in fields(ThorViewEnvConfig)}
        self.cfg = ThorViewEnvConfig(
            **{k: v for k, v in env_config.items() if k in cfg_keys}
        )

        self._env: Optional[ThorEnv] = None
        self._task: Optional[Dict] = None
        self._scene: Optional[str] = None
        self._target_img: Optional[Image.Image] = None
        self._action_history: List[str] = []
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        # min pose-distance seen so far this episode — supports asymmetric
        # progress reward (only when reaching a new minimum, no penalty for
        # moving away).
        self._min_pose_distance: float = float("inf")

    # --- scene resolution --------------------------------------------------

    @classmethod
    def _resolve_procthor_dir(cls, procthor_dir: str) -> str:
        """Make `procthor_dir` robust to Ray-worker CWD changes.

        Order tried: as-given → ${PWD}/<dir> → <repo_root>/<dir>.
        Repo root is inferred from this file's path (3 levels up).
        """
        if os.path.isabs(procthor_dir) and os.path.exists(procthor_dir):
            return procthor_dir
        candidates = [procthor_dir]
        if not os.path.isabs(procthor_dir):
            candidates.append(os.path.join(os.getcwd(), procthor_dir))
            repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            candidates.append(os.path.join(repo_root, procthor_dir))
        for c in candidates:
            if os.path.exists(c):
                return c
        # All failed — return the original so the downstream error message
        # cites what the user actually passed.
        return procthor_dir

    @classmethod
    def _load_procthor_house(cls, scene_id: str, procthor_dir: str) -> Dict:
        """Load (or return cached) ProcTHOR house dict for `scene_id`.

        Scene ID convention: `procthor_XXXXX` where XXXXX is a zero-padded line
        index into `procthor-10k/train.jsonl.gz`. Matches loader logic in
        `scripts/evaluate.py:load_procthor_houses` and `collect_sft_rule.py`.
        """
        with cls._CACHE_LOCK:
            if scene_id in cls._PROCTHOR_CACHE:
                return cls._PROCTHOR_CACHE[scene_id]
        # Outside lock: do the actual file read. Re-check inside lock at end
        # to avoid storing duplicates if two threads race.
        try:
            idx = int(scene_id.split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Cannot parse procthor scene id {scene_id!r}; expected 'procthor_NNNNN'"
            ) from exc

        resolved_dir = cls._resolve_procthor_dir(procthor_dir)
        gz_path = os.path.join(resolved_dir, "train.jsonl.gz")
        plain_path = os.path.join(resolved_dir, "train.jsonl")
        if os.path.exists(gz_path):
            opener = lambda: gzip.open(gz_path, "rt")
            path_used = gz_path
        elif os.path.exists(plain_path):
            opener = lambda: open(plain_path, "rt")
            path_used = plain_path
        else:
            raise FileNotFoundError(
                f"No train.jsonl(.gz) under {procthor_dir!r} "
                f"(resolved to {resolved_dir!r}) for scene {scene_id}"
            )

        house = None
        with opener() as f:
            for line_idx, line in enumerate(f):
                if line_idx == idx:
                    house = json.loads(line)
                    break
        if house is None:
            raise ValueError(
                f"ProcTHOR scene {scene_id} (line {idx}) not found in {path_used}"
            )

        with cls._CACHE_LOCK:
            cls._PROCTHOR_CACHE.setdefault(scene_id, house)
            return cls._PROCTHOR_CACHE[scene_id]

    def _resolve_scene(self, scene_id: str) -> Union[str, Dict]:
        """Map iTHOR scene name → str (pass-through), procthor_XXXXX → house dict."""
        if scene_id.startswith("procthor_"):
            return self._load_procthor_house(scene_id, self.cfg.procthor_dir)
        return scene_id

    # --- sync helpers (always called from a thread via to_thread) -----------

    def _ensure_env(self):
        if self._env is None:
            self._env = ThorEnv(
                scene="FloorPlan1",  # placeholder; replaced in reset()
                width=self.cfg.width,
                height=self.cfg.height,
                fov=self.cfg.fov,
                quality=self.cfg.quality,
                gpu_device=self.cfg.gpu_device,
            )
            self._scene = "FloorPlan1"

    def _render_target(self, target_pose: Dict) -> Image.Image:
        """Teleport to target pose, capture frame, burn TARGET badge.
        Caller restores start pose after."""
        frame = self._env.reset(
            position=target_pose["position"],
            rotation_y=target_pose["rotation_y"],
            horizon=target_pose["horizon"],
        )
        return burn_label(frame, "TARGET")

    def _current_obs_img(self) -> Image.Image:
        return burn_label(self._env.get_observation(), "CURRENT")

    def _build_obs_str(self, valid_actions: List[str]) -> str:
        return build_concat_user_text(valid_actions)

    def _build_obs(self) -> Dict[str, Any]:
        valid = self._env.get_valid_actions(include_stop=True)
        current = self._current_obs_img()
        return {
            "obs_str": self._build_obs_str(valid),
            "multi_modal_input": {IMAGE_PH: [current, self._target_img]},
        }

    def _success_at_current_pose(self) -> bool:
        target = self._task["target"]
        state = self._env.get_state()
        return is_exact_match(state, target)

    def _pose_distance(self, state: Dict, target: Dict) -> float:
        """Weighted 7D pose distance: position L2 (m) + rot/hor folded as
        "discrete-steps remaining". Position dominates (PoliFormer is pure L2);
        rotation step = 45°, horizon step = 30°.
        """
        sp = state["position"]
        tp = target["position"]
        dx = sp["x"] - tp["x"]
        dz = sp["z"] - tp["z"]
        dpos = (dx * dx + dz * dz) ** 0.5
        dr = abs((state["rotation_y"] - target["rotation_y"]) % 360)
        dr = min(dr, 360 - dr) / 45.0
        dh = abs(state["horizon"] - target["horizon"]) / 30.0
        return dpos + self.cfg.pose_rot_weight * dr + self.cfg.pose_hor_weight * dh

    # --- sync entrypoints ---------------------------------------------------

    def _sync_reset(self, task: Dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._ensure_env()
        self._task = task

        scene = task["scene"]
        if scene != self._scene:
            scene_data = self._resolve_scene(scene)  # iTHOR str | ProcTHOR dict
            self._env.load_scene(scene_data)
            self._scene = scene

        # Render target view (teleports to target pose internally).
        self._target_img = self._render_target(task["target"])

        # Teleport to start pose for the actual rollout.
        self._env.reset(
            position=task["start"]["position"],
            rotation_y=task["start"]["rotation_y"],
            horizon=task["start"]["horizon"],
        )

        self._action_history = []
        self._step_count = 0
        self._cumulative_reward = 0.0
        # Initialize min distance with the start-state distance — any future
        # state that gets closer earns progress reward.
        self._min_pose_distance = self._pose_distance(self._env.get_state(), task["target"])

        info = {
            "task_id": task.get("task_id", "?"),
            "scene": scene,
            "step": 0,
            "init_pose_distance": self._min_pose_distance,
        }
        return self._build_obs(), info

    def _sync_step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        # Parse action — we no longer reward by tier, but we still need to
        # distinguish "executable" (strict/medium = clear `Action: X`) from
        # "weak match" (action name buried in random prose, no keyword).
        # Weak gets weight 0 so executable=False; medium/strict get 1 → True.
        action, _score, is_strict = score_response(
            action_str, weak=0.0, medium=1.0, strict=1.0,
        )
        executable = action is not None and _score > 0  # strict or medium only

        # ── per-step components ───────────────────────────────────────────
        reward = -self.cfg.step_penalty
        format_reward = self.cfg.format_valid_reward if executable else -self.cfg.format_invalid_penalty
        reward += format_reward

        done = False
        info: Dict[str, Any] = {
            "format_ok": is_strict,
            "format_strict": is_strict,
            "format_executable": executable,
            "action": action,
            "step": self._step_count,
            "action_executed": False,
            "step_penalty": -self.cfg.step_penalty,
            "format_reward": format_reward,
            "progress_reward": 0.0,
            "terminal_reward": 0.0,
        }

        target = self._task["target"]

        # ── execute env transition ────────────────────────────────────────
        if not executable:
            info["error"] = "no_action"
            # No env step; pose unchanged → progress = 0
        elif action == "Stop":
            done = True
            success = self._success_at_current_pose()
            terminal = self.cfg.success_reward if success else -self.cfg.stop_fail_penalty
            reward += terminal
            info["success"] = success
            info["stopped"] = True
            info["action_executed"] = True
            info["terminal_reward"] = terminal
        else:
            _, action_success = self._env.step(action)
            info["action_success"] = action_success
            info["action_executed"] = True
            self._action_history.append(action)

        # ── distance progress reward (asymmetric, only new-min) ───────────
        # Recompute distance AFTER step. Skip when episode ended via Stop
        # (terminal reward dominates; no need for progress on Stop step).
        if not done:
            d_now = self._pose_distance(self._env.get_state(), target)
            if d_now < self._min_pose_distance:
                progress = self.cfg.progress_scale * (self._min_pose_distance - d_now)
                self._min_pose_distance = d_now
                reward += progress
                info["progress_reward"] = progress
                info["pose_distance"] = d_now
            else:
                info["pose_distance"] = d_now
        else:
            info["pose_distance"] = self._pose_distance(self._env.get_state(), target)

        # ── termination by step cap ───────────────────────────────────────
        self._step_count += 1
        if not done and self._step_count >= self.cfg.max_steps:
            done = True
            info["truncated"] = True

        self._cumulative_reward += reward
        info["cumulative_reward"] = self._cumulative_reward
        return self._build_obs(), reward, done, info

    def _sync_close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    # --- async GymImageEnv protocol -----------------------------------------

    async def system_prompt(self) -> Dict[str, Any]:
        # System prompt has no images.
        return {"obs_str": SYSTEM_PROMPT}

    async def reset(self, *, task: Dict) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_reset, task)

    async def step(self, action_str: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        return await asyncio.to_thread(self._sync_step, action_str)

    async def close(self) -> None:
        await asyncio.to_thread(self._sync_close)


# ---------------------------------------------------------------------------
# __main__: standalone correctness test (5 assertions)
# ---------------------------------------------------------------------------


def _load_one_task(task_file: str) -> Dict:
    with open(task_file) as f:
        tasks = json.load(f)
    return tasks[0] if isinstance(tasks, list) else tasks["tasks"][0]


async def _main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--task_file", default="data/tasks/rl.json",
                    help="JSON with task dicts (start/target/scene)")
    ap.add_argument("--gpu_device", type=int, default=0)
    ap.add_argument("--sft_data_for_verify", default=None,
                    help="Optional: path to sft_train.json to verify SYSTEM_PROMPT matches")
    args = ap.parse_args()

    if args.sft_data_for_verify:
        assert verify_system_prompt_matches_sft(args.sft_data_for_verify), (
            "SYSTEM_PROMPT in thor_view_env.py does not match SFT data — "
            "model will see a different prompt at RL time than at SFT time."
        )
        print("[OK] SYSTEM_PROMPT matches SFT data byte-for-byte")

    task = _load_one_task(args.task_file)
    print(f"[task] {task.get('task_id', '?')}  scene={task['scene']}")

    env = ThorViewEnv({"gpu_device": args.gpu_device})  # default 640×360

    # --- assertion (a): system_prompt has the expected structure
    sys_obs = await env.system_prompt()
    assert "obs_str" in sys_obs and len(sys_obs["obs_str"]) == 942, (
        f"system prompt should be 942 chars, got {len(sys_obs['obs_str'])}"
    )
    assert "multi_modal_input" not in sys_obs or not sys_obs.get("multi_modal_input"), \
        "system prompt should have no images"
    print(f"[OK a] system_prompt OK ({len(sys_obs['obs_str'])} chars)")

    # --- assertion (b): reset returns valid obs schema
    obs, info = await env.reset(task=task)
    assert "obs_str" in obs
    assert IMAGE_PH in obs["obs_str"], f"obs_str must contain {IMAGE_PH} placeholder"
    assert obs["obs_str"].count(IMAGE_PH) == 2, (
        f"obs_str must have exactly 2 image placeholders (current + target), got "
        f"{obs['obs_str'].count(IMAGE_PH)}"
    )
    assert "multi_modal_input" in obs
    imgs = obs["multi_modal_input"][IMAGE_PH]
    assert len(imgs) == 2, f"expected 2 images (current+target), got {len(imgs)}"
    assert all(isinstance(im, Image.Image) for im in imgs)
    assert "Your recent actions" not in obs["obs_str"], "step-0 obs should have no history block"
    print(f"[OK b] reset obs has correct schema (obs_str {len(obs['obs_str'])} chars, 2 images)")

    # --- (c): valid action → step_penalty + format_valid_reward; may also get
    #          progress reward IF the action moves us closer to target.
    obs, info = await env.reset(task=task)
    init_d = info["init_pose_distance"]
    print(f"[info] init pose distance to target: {init_d:.3f}")
    obs, reward, done, info = await env.step("Action: MoveAhead")
    base = -env.cfg.step_penalty + env.cfg.format_valid_reward
    assert info["action"] == "MoveAhead"
    assert info["format_strict"] is True
    assert info["action_executed"] is True
    # reward = base + (progress if got closer) — progress can be 0 or positive
    assert reward >= base - 1e-9, f"reward {reward} should be at least base {base}"
    print(f"[OK c] strict MoveAhead: reward={reward:.4f} = base {base:.4f} + progress {info['progress_reward']:.4f}")

    # --- (c2): concat mode — obs_str must NOT contain history block
    obs, _, _, _ = await env.step("Action: RotateLeft")
    assert "Your recent actions" not in obs["obs_str"], (
        "concat mode: obs_str must NOT contain 'Your recent actions' text block"
    )
    print(f"[OK c2] no history text block in obs_str (concat-correct)")

    # --- (d): Stop at target → step_penalty + format_valid + success_reward
    obs, info = await env.reset(task=task)
    env._env.reset(
        position=task["target"]["position"],
        rotation_y=task["target"]["rotation_y"],
        horizon=task["target"]["horizon"],
    )
    # Reset the min-distance tracker so progress from the manual teleport
    # doesn't fire (we want to isolate the Stop terminal reward).
    env._min_pose_distance = env._pose_distance(env._env.get_state(), task["target"])
    obs, reward, done, info = await env.step("Action: Stop")
    assert done and info.get("success") is True
    expected = -env.cfg.step_penalty + env.cfg.format_valid_reward + env.cfg.success_reward
    assert abs(reward - expected) < 1e-6, f"expected {expected}, got {reward}"
    print(f"[OK d] Stop at target: reward={reward:.4f} (step {-env.cfg.step_penalty} + fmt {env.cfg.format_valid_reward} + success {env.cfg.success_reward})")

    # --- (e): Stop far from target → step_penalty + format_valid - stop_fail_penalty
    obs, info = await env.reset(task=task)
    obs, reward, done, info = await env.step("Action: Stop")
    assert done and info.get("success") is False
    expected = -env.cfg.step_penalty + env.cfg.format_valid_reward - env.cfg.stop_fail_penalty
    assert abs(reward - expected) < 1e-6, f"expected {expected}, got {reward}"
    print(f"[OK e] Stop at start: reward={reward:.4f} (step + fmt - stop_fail {env.cfg.stop_fail_penalty})")

    # --- (f): no action signal → step_penalty + format_invalid_penalty
    obs, info = await env.reset(task=task)
    obs, reward, done, info = await env.step("blah blah completely random output")
    assert info["action"] is None
    assert info["action_executed"] is False
    expected = -env.cfg.step_penalty - env.cfg.format_invalid_penalty
    assert abs(reward - expected) < 1e-6, f"expected {expected}, got {reward}"
    print(f"[OK f] no signal: reward={reward:.4f} (step + format_invalid_penalty)")

    # --- (g): action name in prose (no "Action:" keyword) → still parsed but
    #          NOT executed; treated as format_invalid (we want strict/medium only)
    obs, info = await env.reset(task=task)
    obs, reward, done, info = await env.step("I think MoveAhead might be a good choice here")
    # weak parser returns action=MoveAhead BUT score_response returns it with
    # weak score; we now collapse: only strict/medium count as executable.
    # The action is parsed (info["action"]=MoveAhead) but executable=False.
    assert info["action"] == "MoveAhead"
    assert info["action_executed"] is False
    expected = -env.cfg.step_penalty - env.cfg.format_invalid_penalty
    assert abs(reward - expected) < 1e-6, f"expected {expected}, got {reward}"
    print(f"[OK g] weak (no 'Action:' keyword): not executed, reward={reward:.4f}")

    # --- (h): MEDIUM ("Action:" keyword in CoT wrap) — executes, format_valid
    obs, info = await env.reset(task=task)
    obs, reward, done, info = await env.step(
        "<think>I should move forward.</think>\n\nAction: MoveAhead"
    )
    assert info["action"] == "MoveAhead"
    assert info["format_strict"] is False
    assert info["action_executed"] is True
    base = -env.cfg.step_penalty + env.cfg.format_valid_reward
    assert reward >= base - 1e-9, f"reward {reward} should be ≥ base {base}"
    print(f"[OK h] medium (CoT-style): executed, reward={reward:.4f}")

    # --- (i): STRICT tolerates leading </think>\n\n
    obs, info = await env.reset(task=task)
    obs, reward, done, info = await env.step("</think>\n\nAction: MoveAhead")
    assert info["format_strict"] is True
    assert info["action_executed"] is True
    print(f"[OK i] strict tolerates </think> prefix, reward={reward:.4f}")

    # --- (j): progress only on NEW min — go forward then back; second move should
    #          get zero progress reward (still farther than the new-min set on first)
    obs, info = await env.reset(task=task)
    obs, r1, _, info1 = await env.step("Action: MoveAhead")  # may or may not progress
    d_after_forward = info1.get("pose_distance", env._min_pose_distance)
    obs, r2, _, info2 = await env.step("Action: MoveBack")   # likely moves away
    # MoveBack should NOT trigger progress reward (not new min)
    assert info2["progress_reward"] == 0.0, (
        f"MoveBack after MoveAhead should not trigger progress, got {info2['progress_reward']}"
    )
    print(f"[OK j] asymmetric progress: MoveBack progress_reward={info2['progress_reward']}")

    await env.close()
    print("\n[PASS] all assertions passed.")


if __name__ == "__main__":
    asyncio.run(_main())
