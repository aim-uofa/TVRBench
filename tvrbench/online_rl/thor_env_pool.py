"""Per-GPU env pool for online RL.

A `ThorEnvPool` lives at the agent_loop class level (per Ray worker / per GPU),
shared across all concurrent `agent_loop.run()` calls in that worker. It bounds
the number of concurrent `ThorViewEnv` instances on the GPU, lazily creating
them up to `max_size` and reusing them across rollouts.

Design (see project_online_rl_plan.md):
- No cache, no eviction, no scene affinity — envs are persistent for the pool's
  lifetime; switching scenes is handled inside `env.reset(task)` via
  `controller.reset(scene=...)`.
- Pool size = concurrent rollouts on this GPU. Default 8 — profile-tune on
  smoke test (#12) if vLLM + envs OOM.

Usage:
    pool = ThorEnvPool(gpu_device=0, max_size=8)
    async with pool.acquire() as env:
        obs, info = await env.reset(task=task)
        # ... rollout
    await pool.close_all()
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from tvrbench.online_rl.thor_view_env import ThorViewEnv


class ThorEnvPool:
    """Bounded async pool of ThorViewEnv instances, one pool per Ray worker.

    The pool spans MULTIPLE physical GPUs: each lazy-created env is assigned a
    GPU in round-robin order from `gpu_devices`. This is the "Option C"
    distribution strategy — each AgentLoopWorker's pool covers all 4 GPUs,
    so 4 workers × 8 envs / 4 GPUs = 8 envs per GPU on the cluster, perfectly
    balanced regardless of where the AgentLoopWorker actors land.

    (We can't easily colocate AgentLoopWorker actors with their vLLM replicas
    in verl 0.8 — they're CPU-only orchestrators by design. So we balance at
    the env level instead.)
    """

    def __init__(
        self,
        gpu_devices: List[int],
        max_size: int = 8,
        env_config_extra: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            gpu_devices: physical GPU indices to spread envs across, round-robin.
                Example: [0, 1, 2, 3] for 4-GPU node.
            max_size: max number of concurrently-acquired envs (also caps total alive envs).
            env_config_extra: extra kwargs passed to each ThorViewEnv (e.g. width, height).
                Note: `gpu_device` is injected per-env at creation time from
                `gpu_devices[round_robin]`, overriding any gpu_device in env_config_extra.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if not gpu_devices:
            raise ValueError("gpu_devices must be a non-empty list")

        self.gpu_devices = list(gpu_devices)
        self.max_size = max_size
        self._env_config_extra = dict(env_config_extra or {})
        # `gpu_device` is per-env, not pool-wide; strip if present so callers
        # don't accidentally pin all envs to one GPU via env_config.
        self._env_config_extra.pop("gpu_device", None)

        # Round-robin pointer for env→GPU assignment at lazy-create time.
        self._next_gpu_idx: int = 0
        # Track GPU assignment per env for introspection / debugging.
        self._env_gpu: dict[int, int] = {}  # id(env) → gpu_device

        # Semaphore caps concurrent acquires; queue holds idle envs.
        self._sem = asyncio.Semaphore(max_size)
        self._idle: asyncio.Queue[ThorViewEnv] = asyncio.Queue()
        # Tracks all envs ever created (for close_all). Append-only.
        self._all_envs: list[ThorViewEnv] = []
        self._create_lock = asyncio.Lock()
        self._closed = False

    # Legacy accessor: many callers used `pool.gpu_device`. Preserve as a
    # readable summary (returns the list joined as string) for log messages.
    @property
    def gpu_device(self) -> str:  # type: ignore[override]
        return ",".join(str(g) for g in self.gpu_devices)

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[ThorViewEnv]:
        """Acquire an env from the pool, lazy-creating up to max_size.

        Blocks (await) while pool is full; resumes as soon as another caller releases.
        """
        if self._closed:
            raise RuntimeError("ThorEnvPool is closed; cannot acquire")

        await self._sem.acquire()
        env: Optional[ThorViewEnv] = None
        try:
            # Re-check after sem acquire — close_all() could have run while we waited
            if self._closed:
                raise RuntimeError("ThorEnvPool closed while waiting to acquire")

            # Try to grab an existing idle env; otherwise lazy-create.
            try:
                env = self._idle.get_nowait()
            except asyncio.QueueEmpty:
                async with self._create_lock:
                    # Re-check inside the lock — another coroutine might have just released
                    try:
                        env = self._idle.get_nowait()
                    except asyncio.QueueEmpty:
                        # Lazy-create with round-robin GPU assignment.
                        gpu = self.gpu_devices[self._next_gpu_idx % len(self.gpu_devices)]
                        self._next_gpu_idx += 1
                        env_cfg = dict(self._env_config_extra)
                        env_cfg["gpu_device"] = gpu
                        env = ThorViewEnv(env_cfg)
                        self._env_gpu[id(env)] = gpu
                        self._all_envs.append(env)

            yield env
        finally:
            if env is not None:
                # Return to idle queue unless the pool was closed
                if not self._closed:
                    self._idle.put_nowait(env)
            self._sem.release()

    async def close_all(self) -> None:
        """Close every env created by this pool. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Close all envs we ever created (idle queue is a subset of this).
        # No need to drain _idle separately — _all_envs covers it.
        for env in self._all_envs:
            try:
                await env.close()
            except Exception as e:
                # Best-effort cleanup; surface but don't re-raise (we want to close as many as we can)
                import logging
                logging.getLogger(__name__).warning(
                    f"Error closing pool env on GPU{self._env_gpu.get(id(env), '?')}: "
                    f"{type(e).__name__}: {e}"
                )

    # Per-GPU env count (for introspection / nvidia-smi sanity).
    def envs_per_gpu(self) -> Dict[int, int]:
        counts: Dict[int, int] = {g: 0 for g in self.gpu_devices}
        for env in self._all_envs:
            g = self._env_gpu.get(id(env))
            if g is not None:
                counts[g] = counts.get(g, 0) + 1
        return counts

    # --- introspection (mostly for tests / debugging) -----------------------

    @property
    def num_created(self) -> int:
        return len(self._all_envs)

    @property
    def num_idle(self) -> int:
        return self._idle.qsize()

    @property
    def closed(self) -> bool:
        return self._closed


# ---------------------------------------------------------------------------
# __main__: standalone correctness test
# ---------------------------------------------------------------------------


async def _worker(pool: ThorEnvPool, task: Dict, worker_id: int,
                  active_counter: Dict[str, int], max_active: Dict[str, int]) -> Dict:
    """Simulate one rollout: acquire env, run one reset+step, release."""
    async with pool.acquire() as env:
        # Track concurrent active count
        active_counter["n"] += 1
        max_active["n"] = max(max_active["n"], active_counter["n"])
        try:
            obs, info = await env.reset(task=task)
            obs, reward, done, info = await env.step("Action: RotateRight")
            return {
                "worker_id": worker_id,
                "reward": reward,
                "done": done,
                "obs_str_len": len(obs["obs_str"]),
            }
        finally:
            active_counter["n"] -= 1


async def _main():
    import argparse
    import json
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--task_file", default="data/tasks/rl.json")
    ap.add_argument("--gpu_devices", type=int, nargs="+", default=[0],
                    help="Physical GPU indices to round-robin across (e.g. 0 1 2 3)")
    ap.add_argument("--max_size", type=int, default=4,
                    help="Pool max_size for the test")
    ap.add_argument("--n_concurrent", type=int, default=16,
                    help="Number of concurrent workers (should be > max_size to test queueing)")
    args = ap.parse_args()

    with open(args.task_file) as f:
        tasks = json.load(f)
        if isinstance(tasks, dict) and "tasks" in tasks:
            tasks = tasks["tasks"]
    print(f"[task_file] loaded {len(tasks)} tasks; using first {args.n_concurrent}")

    pool = ThorEnvPool(
        gpu_devices=args.gpu_devices,
        max_size=args.max_size,
        env_config_extra={"width": 256, "height": 256},
    )

    active_counter = {"n": 0}
    max_active = {"n": 0}

    print(f"[run] {args.n_concurrent} concurrent workers, pool max_size={args.max_size}")
    t0 = time.time()
    results = await asyncio.gather(*[
        _worker(pool, tasks[i % len(tasks)], i, active_counter, max_active)
        for i in range(args.n_concurrent)
    ])
    elapsed = time.time() - t0
    print(f"[run] elapsed: {elapsed:.1f}s")

    # --- assertion (a): all workers completed
    assert len(results) == args.n_concurrent
    assert all(r["done"] is False or r["done"] is True for r in results)  # tautology, just ensure dict ok
    assert all("reward" in r for r in results)
    print(f"[OK a] all {args.n_concurrent} workers completed")

    # --- assertion (b): max concurrent acquires never exceeded max_size
    assert max_active["n"] <= args.max_size, (
        f"max concurrent active = {max_active['n']}, exceeds pool max_size = {args.max_size}"
    )
    print(f"[OK b] max concurrent active = {max_active['n']} ≤ max_size = {args.max_size}")

    # --- assertion (c): pool created at most max_size envs (not n_concurrent)
    assert pool.num_created <= args.max_size, (
        f"pool created {pool.num_created} envs, exceeds max_size {args.max_size}"
    )
    print(f"[OK c] pool created {pool.num_created} envs (≤ max_size={args.max_size})")

    # --- assertion (d): post-run, idle == created (all returned)
    assert pool.num_idle == pool.num_created, (
        f"idle={pool.num_idle} != created={pool.num_created} — some envs leaked"
    )
    print(f"[OK d] all {pool.num_idle} envs returned to idle queue")

    # --- assertion (d2): GPU assignment is round-robin balanced
    per_gpu = pool.envs_per_gpu()
    print(f"[info] envs per GPU: {per_gpu}")
    if len(args.gpu_devices) > 1:
        # With round-robin, max-min should be ≤ 1
        max_count = max(per_gpu.values())
        min_count = min(per_gpu.values())
        assert max_count - min_count <= 1, (
            f"round-robin failed: per_gpu={per_gpu} (max-min > 1)"
        )
        print(f"[OK d2] round-robin balanced: max={max_count}, min={min_count} (Δ≤1)")

    # --- assertion (e): close_all closes envs, subsequent acquire raises
    await pool.close_all()
    assert pool.closed
    try:
        async with pool.acquire() as _env:
            assert False, "acquire on closed pool should have raised"
    except RuntimeError as e:
        print(f"[OK e] post-close acquire raises: {e}")

    # --- assertion (f): batch timing — ceil(n_concurrent/max_size) waves
    # rough sanity: elapsed should be at least ceil(N/max_size) * single_task_time,
    # but we don't know single_task_time precisely. Just print for visual check.
    waves = -(-args.n_concurrent // args.max_size)
    print(f"[info] elapsed {elapsed:.1f}s = {elapsed / waves:.1f}s/wave × {waves} waves "
          f"(work-stealing means stragglers compress this)")

    print("\n[PASS] all pool assertions passed.")


if __name__ == "__main__":
    asyncio.run(_main())
