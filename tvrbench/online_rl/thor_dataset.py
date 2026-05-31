"""Minimal dataset for online RL — yields one row per task from data/tasks/rl.json.

verl 0.8.0 wires this in via `data.custom_cls.{path,name}`. Each row is a dict
in `non_tensor_batch` that gets passed as kwargs to `ThorAgentLoop.run()`.

Why we don't reuse VAGEN's `AgenticDataset`: VAGEN's spec is built around an
EnvSpec YAML with `n_envs` + seed-directive expansion (Sokoban-style). For
ActiveSpatial each row IS a task dict (scene + start + target poses), so we
just read rl.json directly. Simpler.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# Required by verl's agent_loop router — every row must carry this.
AGENT_NAME = "active_spatial_thor"
DATA_SOURCE = "active_spatial_thor"


def _load_tasks(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]
    if not isinstance(data, list):
        raise TypeError(f"Expected list of task dicts in {path}, got {type(data)}")
    return data


class ThorTaskDataset(Dataset):
    """One row per task. verl will run rollout_n samples per row at training time.

    Constructor signature matches verl's `custom_cls` contract:
    `ctor(data_files, tokenizer, processor, config)` — but verl is permissive
    here, accepting any subset. We use only `data_files` (path to rl.json).
    """

    def __init__(
        self,
        data_files: str,
        tokenizer: Any = None,
        processor: Any = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # data_files may be a string or a list (when passed via OmegaConf).
        if isinstance(data_files, (list, tuple)):
            if len(data_files) != 1:
                raise ValueError(
                    f"ThorTaskDataset expects a single file path, got {len(data_files)} entries"
                )
            data_files = data_files[0]
        if not os.path.exists(data_files):
            raise FileNotFoundError(f"task file not found: {data_files}")

        self.tasks = _load_tasks(data_files)

        # Deterministic shuffle. rl.json is grouped by scene/dataset (procthor
        # tasks first, then ithor), so without shuffling batches end up
        # type-homogeneous → model overfits to whichever dataset comes first.
        # verl's DataLoader shuffle behavior with custom_cls isn't reliable
        # enough to depend on; do it ourselves with a fixed seed.
        import random
        rng = random.Random(0xACE)  # fixed seed for reproducibility
        rng.shuffle(self.tasks)

        logger.warning(f"[ThorTaskDataset] loaded {len(self.tasks)} tasks from {data_files} (shuffled)")
        print(f"[ThorTaskDataset] first 5 task_ids after shuffle: "
              f"{[t.get('task_id') for t in self.tasks[:5]]}", flush=True)

        # Optional filtering by difficulty / dataset, controlled via config.
        if config is not None:
            allowed_diff = config.get("difficulties")  # e.g. ["easy"]
            allowed_dataset = config.get("datasets")    # e.g. ["ithor"]
            if allowed_diff:
                allowed_diff = set(allowed_diff)
                self.tasks = [t for t in self.tasks if t.get("difficulty") in allowed_diff]
            if allowed_dataset:
                allowed_dataset = set(allowed_dataset)
                self.tasks = [t for t in self.tasks if t.get("dataset") in allowed_dataset]
            if allowed_diff or allowed_dataset:
                logger.warning(
                    f"[ThorTaskDataset] after filter (diff={allowed_diff}, dataset={allowed_dataset}): "
                    f"{len(self.tasks)} tasks"
                )

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task = self.tasks[idx]
        return {
            # routing: matches @register("active_spatial_thor") in thor_agent_loop
            "agent_name": AGENT_NAME,
            "data_source": DATA_SOURCE,
            # env input — full task dict consumed by env.reset(task=...)
            "task": task,
            # convenience copies (handy in logs / extra_info)
            "task_id": task.get("task_id", f"idx_{idx}"),
            "scene": task.get("scene", "?"),
            "difficulty": task.get("difficulty", "?"),
            # verl's _agent_loop_postprocess does `output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]`
            # so this key must exist even if we don't use it.
            "raw_prompt": [],
            # extra_info is consumed by verl's reward path & traces.
            "extra_info": {"task_id": task.get("task_id", f"idx_{idx}")},
            # IMPORTANT: do NOT include 'input_ids' / 'attention_mask' / 'position_ids' here.
            # verl 0.8.0's `_get_gen_batch` no longer pops them before rollout (unlike
            # VAGEN's vendored 0.6.1), and `batch.union(gen_batch_output)` asserts that
            # any colliding key has equal tensor values. The agent_loop produces real
            # padded prompt+response tokens under those names; our dataset dummies
            # would NEVER match → AssertionError at protocol.py:118.
            #
            # We still need ONE tensor field so the dataloader's collate gives the
            # TensorDict a batch_size. Use a non-colliding sentinel.
            "_row_idx": torch.tensor([idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# __main__: standalone sanity test
# ---------------------------------------------------------------------------


def _main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_file", default="data/tasks/rl.json")
    ap.add_argument("--difficulties", nargs="+", default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()

    cfg = {}
    if args.difficulties:
        cfg["difficulties"] = args.difficulties
    if args.datasets:
        cfg["datasets"] = args.datasets
    ds = ThorTaskDataset(data_files=args.task_file, config=cfg or None)

    print(f"[OK a] dataset size: {len(ds)}")

    row = ds[0]
    assert row["agent_name"] == AGENT_NAME, f"agent_name must be {AGENT_NAME!r}"
    assert "task" in row and isinstance(row["task"], dict)
    for key in ("scene", "start", "target"):
        assert key in row["task"], f"task missing key {key!r}"
    print(f"[OK b] row[0]: task_id={row['task_id']} scene={row['scene']} difficulty={row['difficulty']}")
    print(f"           agent_name={row['agent_name']!r}")
    print(f"           task keys: {sorted(row['task'].keys())}")

    # Single sentinel tensor for collate's batch_size; no input_ids dummy.
    assert "_row_idx" in row and isinstance(row["_row_idx"], torch.Tensor)
    assert "input_ids" not in row, "must NOT have input_ids dummy — see comment in __getitem__"
    print(f"[OK c] sentinel tensor present; no colliding input_ids dummy")

    # Spot-check a few more rows
    for i in [len(ds) // 2, len(ds) - 1]:
        r = ds[i]
        assert r["agent_name"] == AGENT_NAME
        assert "task" in r
    print(f"[OK d] random-access {len(ds) // 2} and {len(ds) - 1} OK")

    print("\n[PASS] dataset sanity OK")


if __name__ == "__main__":
    _main()
