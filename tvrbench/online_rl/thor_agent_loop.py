"""verl 0.8.0 agent_loop for ActiveSpatial viewpoint matching (concat multi-turn).

Ported from `vagen/agent_loop/gym_agent_loop.py` (VAGEN's 0.6.1 verl) to verl
0.8.0 native API. Key differences from VAGEN's version:

1. `AgentLoopBase` uses standard `__init__(*args, **kwargs)` + `super().__init__`,
   NOT VAGEN's `init_class` classmethod hook.
2. `apply_chat_template(messages, images=..., remove_system_prompt=True)` is
   provided by AgentLoopBase (handles processor + system prompt slicing). VAGEN
   manually cached system_prompt_prefix_ids — 0.8 has `self.system_prompt`.
3. `AgentLoopOutput.multi_modal_data` keys are `"images"` and `"videos"`
   (plural). VAGEN used `{"image": [...]}` (singular) which silently breaks 0.8
   postprocess.
4. server_manager.generate API matches (image_data kwarg same).

Env pool lives at **class level** (one per Ray worker / per GPU), shared across
all concurrent `run()` invocations in that worker. Per-instance state is
trajectory-only (messages, prompt_ids, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from tvrbench.online_rl.thor_env_pool import ThorEnvPool
from tvrbench.online_rl.thor_view_env import IMAGE_PH, ThorViewEnv

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_images(imgs: Optional[List[Image.Image]]) -> List[Image.Image]:
    """Ensure PIL RGB, drop Nones."""
    out: List[Image.Image] = []
    for im in imgs or []:
        if im is None:
            continue
        out.append(im.convert("RGB") if isinstance(im, Image.Image) else im)
    return out


_IMAGE_PLACEHOLDER_PATTERN = re.compile(re.escape(IMAGE_PH))


def _obs_to_content(obs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert env obs (`{"obs_str": "...<image>...", "multi_modal_input": {"<image>": [PIL]}}`)
    into a structured content list for chat messages.

    Splits obs_str on `<image>` placeholders, replacing each with a structured
    `{"type": "image"}` block. The image count in obs_str MUST match the image
    count in multi_modal_input — asserted here as a load-bearing invariant
    (mismatched counts are the #1 cause of training crashes).
    """
    text = obs.get("obs_str", "")
    images = obs.get("multi_modal_input", {}).get(IMAGE_PH, []) or []
    num_placeholders = text.count(IMAGE_PH)
    assert num_placeholders == len(images), (
        f"obs invariant violated: {num_placeholders} '{IMAGE_PH}' placeholders in "
        f"obs_str but {len(images)} images in multi_modal_input. "
        f"obs_str preview: {text[:200]!r}"
    )

    if num_placeholders == 0:
        return [{"type": "text", "text": text}] if text else []

    content: List[Dict[str, Any]] = []
    segments = _IMAGE_PLACEHOLDER_PATTERN.split(text)
    # segments: [text_before_img1, text_between_1_2, ..., text_after_last_img]
    # interleave with image blocks
    for i, seg in enumerate(segments):
        if seg:
            content.append({"type": "text", "text": seg})
        if i < num_placeholders:
            content.append({"type": "image"})
    return content


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class AgentState(Enum):
    PENDING = "pending"            # initial messages built, prompt not yet tokenized
    GENERATING = "generating"      # prompt ready, call LLM
    INTERACTING = "interacting"    # LLM response in hand, step env
    TERMINATED = "terminated"


class AgentData:
    """Per-trajectory mutable state."""

    def __init__(self, messages, image_data, metrics, request_id, per_turn_response_limit):
        self.messages: List[Dict[str, Any]] = messages
        self.image_data: List[Image.Image] = image_data
        self.metrics: Dict[str, Any] = metrics
        self.request_id: str = request_id
        self.per_turn_response_limit: int = per_turn_response_limit

        # Token buffers (accumulated across turns)
        self.prompt_ids: List[int] = []
        self.response_ids: List[int] = []          # last turn's generation only
        self.response_mask: List[int] = []         # 1 for LLM, 0 for env-injected
        self.response_logprobs: List[float] = []

        # Per-trajectory stats
        self.env_rewards: List[float] = []
        self.traj_success: bool = False
        self.env_turns: int = 0
        self.last_assistant_text: Optional[str] = None

        # Reward-ladder counters (logged via extra_fields → wandb)
        self.format_strict_count: int = 0
        self.format_medium_count: int = 0
        self.format_weak_count: int = 0
        self.format_none_count: int = 0


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


@register("active_spatial_thor")
class ThorAgentLoop(AgentLoopBase):
    """Concat multi-turn agent loop for ActiveSpatial viewpoint matching.

    Lifecycle:
        - Per Ray worker / per GPU: `ThorEnvPool` is created once at class level
          (lazy, guarded by `_class_init_lock`).
        - Per rollout: one instance is created via `hydra.utils.instantiate`, runs
          `run(sampling_params, **kwargs)`, returns `AgentLoopOutput`.
        - kwargs is a per-row dict from the dataset, expected to carry `task`
          (full task dict with `scene`, `start`, `target`).
    """

    # Class-level pool state
    _class_init_lock = threading.Lock()
    _env_pool: Optional[ThorEnvPool] = None
    _pool_init_failed: bool = False

    def __init__(
        self,
        *args,
        max_turns: int = 8,
        response_length_per_turn: int = 32,
        env_pool_size: int = 8,
        env_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Construct from `hydra.utils.instantiate(agent_loop_config, **standard_kwargs)`.

        Verl-supplied standard kwargs (passed via `*args` / `**kwargs` from
        AgentLoopWorker._run_agent_loop): `trainer_config`, `server_manager`,
        `tokenizer`, `processor`, `dataset_cls`, `data_config`.

        Agent-specific kwargs (set in `configs/online_rl/agent.yaml`):
            max_turns: hard cap on env.step calls per trajectory
            response_length_per_turn: per-turn max_new_tokens for vLLM generation
            env_pool_size: concurrent envs per Ray worker (per GPU)
            env_config: dict passed to ThorViewEnv constructor (width, height, …)
        """
        super().__init__(*args, **kwargs)

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_turns = int(max_turns)
        self.per_turn_response_limit = int(response_length_per_turn)

        # Sanity log so agent.yaml sync failures don't silently produce wrong cap.
        # If you see max_turns=8 here but agent.yaml says 30, the yaml didn't sync.
        # Use print (not logger) because verl/Ray's stdout capture is reliable
        # but the file-named logger gets filtered.
        if not getattr(type(self), "_logged_init", False):
            print(
                f"[ThorAgentLoop] init: max_turns={self.max_turns}, "
                f"response_length_per_turn={self.per_turn_response_limit}, "
                f"prompt_length={self.prompt_length}, response_length={self.response_length}",
                flush=True,
            )
            type(self)._logged_init = True

        cls = type(self)
        with cls._class_init_lock:
            if cls._env_pool is None and not cls._pool_init_failed:
                try:
                    cls._env_pool = cls._build_pool(int(env_pool_size), dict(env_config or {}))
                    logger.warning(
                        f"[ThorAgentLoop] env pool ready on GPU{cls._env_pool.gpu_device} "
                        f"max_size={cls._env_pool.max_size}"
                    )
                except Exception:
                    cls._pool_init_failed = True
                    logger.error(
                        "[ThorAgentLoop] env pool init failed:\n%s",
                        traceback.format_exc(),
                    )
                    raise

    @classmethod
    def _build_pool(cls, max_size: int, env_config: Dict[str, Any]) -> ThorEnvPool:
        # Each AgentLoopWorker's pool spans all physical GPUs and round-robins
        # envs across them. Net effect across N workers: envs distributed evenly.
        #
        # Rationale: THOR uses Vulkan, which sees all physical GPUs regardless
        # of CUDA_VISIBLE_DEVICES. AgentLoopWorker actors are CPU-only by design
        # in verl 0.8 (they orchestrate vLLM RPC + env.step), so we can't rely
        # on Ray's GPU placement to colocate them. Pool-internal round-robin
        # gives perfect balance without cross-actor coordination.
        #
        # Two ways to specify which physical GPUs to use:
        #   1. `gpu_devices: [0, 2, 7]`     — explicit list. Use when some GPUs
        #      on the box have broken Vulkan paths (H200D: 5/8 GPUs hang in
        #      Unity Vulkan submit; only 0/2/7 work).
        #   2. `num_physical_gpus: 8`       — implicit `range(N)`. Use when all
        #      GPUs work uniformly (H100 box, H200B).
        # `gpu_devices` takes precedence when both are present.
        explicit = env_config.pop("gpu_devices", None)
        if explicit is not None:
            gpu_devices = list(explicit)
        else:
            num_phys_gpus = int(env_config.pop("num_physical_gpus", 4))
            gpu_devices = list(range(num_phys_gpus))
        # Drop num_physical_gpus too if it was set alongside gpu_devices (harmless,
        # but prevents accidental kwarg pollution downstream).
        env_config.pop("num_physical_gpus", None)
        logger.warning(
            f"[ThorAgentLoop] _build_pool: gpu_devices={gpu_devices}, max_size={max_size}"
        )
        return ThorEnvPool(
            gpu_devices=gpu_devices,
            max_size=max_size,
            env_config_extra=env_config,
        )

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        if "task" not in kwargs:
            raise KeyError(
                "ThorAgentLoop expects 'task' in kwargs (per-row dataset field). "
                f"Got keys: {list(kwargs.keys())}"
            )
        task: Dict[str, Any] = kwargs["task"]
        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex

        cls = type(self)
        if cls._env_pool is None:
            raise RuntimeError(
                "ThorEnvPool not initialized — class init must run before any rollout. "
                "Check __init__ logs for pool-build errors."
            )

        async with cls._env_pool.acquire() as env:
            try:
                output = await self._run_inner(env, task, sampling_params, metrics, request_id)
            except Exception:
                logger.error(
                    "[ThorAgentLoop] rollout failed for task %s:\n%s",
                    task.get("task_id", "?"), traceback.format_exc(),
                )
                # Return a minimal valid output so the batch doesn't lose the slot.
                # verl's _agent_loop_postprocess calls .dim() on padded tensors —
                # if response_ids is [], tokenizer.pad returns a plain list and
                # the postprocess crashes (AttributeError 'list' has no .dim()).
                # So inject a single eos token to keep tensor types alive.
                pad_id = self.tokenizer.pad_token_id
                if pad_id is None:
                    pad_id = self.tokenizer.eos_token_id or 0
                output = AgentLoopOutput(
                    prompt_ids=[pad_id],
                    response_ids=[pad_id],
                    response_mask=[0],     # mark as non-LLM token — no PPO gradient
                    multi_modal_data={},
                    reward_score=0.0,
                    num_turns=0,
                    metrics=AgentLoopMetrics(),
                    # Schema MUST match the success path's extra_fields (line ~370)
                    # — verl's DataProto.concat asserts all rollouts in a batch
                    # have identical non_tensor_batch keys.
                    extra_fields={
                        "traj_success": 0.0,
                        "format_strict_count": 0,
                        "format_medium_count": 0,
                        "format_weak_count": 0,
                        "format_none_count": 0,
                        "rollout_error": 1.0,
                    },
                )
        return output

    async def _run_inner(
        self,
        env: ThorViewEnv,
        task: Dict[str, Any],
        sampling_params: Dict[str, Any],
        metrics: Dict[str, Any],
        request_id: str,
    ) -> AgentLoopOutput:
        # Initial obs: system prompt + first user turn (current + target images)
        sys_obs = await env.system_prompt()
        init_obs, _info = await env.reset(task=task)

        messages: List[Dict[str, Any]] = []
        image_data: List[Image.Image] = []

        # System (text-only).
        messages.append({"role": "system", "content": sys_obs["obs_str"]})

        # First user turn — structured content list, with 2 images (current+target).
        messages.append({"role": "user", "content": _obs_to_content(init_obs)})
        first_imgs = init_obs.get("multi_modal_input", {}).get(IMAGE_PH, []) or []
        image_data.extend(_normalize_images(first_imgs))

        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            per_turn_response_limit=self.per_turn_response_limit,
        )

        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending(agent_data)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_env(agent_data, env)
            else:  # pragma: no cover
                logger.error(f"unreachable state: {state}")
                break

        # Finalize: prompt_ids = initial encoded; response_ids = everything after.
        resp_len = len(agent_data.response_mask)
        if resp_len:
            response_ids = agent_data.prompt_ids[-resp_len:]
            prompt_ids = agent_data.prompt_ids[:-resp_len]
        else:
            response_ids = []
            prompt_ids = agent_data.prompt_ids

        multi_modal_data = {"images": agent_data.image_data} if agent_data.image_data else {}

        if len(prompt_ids) > self.prompt_length:
            logger.warning(
                f"prompt_ids length {len(prompt_ids)} exceeds prompt_length {self.prompt_length}; truncating left"
            )
        if len(response_ids) > self.response_length:
            logger.warning(
                f"response_ids length {len(response_ids)} exceeds response_length {self.response_length}; truncating right"
            )

        return AgentLoopOutput(
            prompt_ids=prompt_ids[-self.prompt_length:],
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length]
                if agent_data.response_logprobs else None
            ),
            reward_score=float(sum(agent_data.env_rewards)),
            num_turns=agent_data.env_turns,
            metrics=AgentLoopMetrics(**metrics) if metrics else AgentLoopMetrics(),
            extra_fields={
                "traj_success": float(agent_data.traj_success),
                "format_strict_count": agent_data.format_strict_count,
                "format_medium_count": agent_data.format_medium_count,
                "format_weak_count": agent_data.format_weak_count,
                "format_none_count": agent_data.format_none_count,
                "rollout_error": 0.0,   # keep schema identical to error path (verl concat)
            },
        )

    # -----------------------------------------------------------------------
    # State handlers
    # -----------------------------------------------------------------------

    async def _handle_pending(self, agent_data: AgentData) -> AgentState:
        """Encode initial (system + first user) messages into prompt_ids."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            images=agent_data.image_data or None,
        )
        agent_data.prompt_ids = list(prompt_ids)
        if len(agent_data.prompt_ids) > self.prompt_length:
            logger.warning(
                f"initial prompt length {len(agent_data.prompt_ids)} exceeds {self.prompt_length}"
            )
        return AgentState.GENERATING

    async def _handle_generating(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """Call vLLM to generate the next assistant turn; cap per-turn tokens."""
        sp = dict(sampling_params)
        per_turn_cap = min(
            sp.get("max_new_tokens", self.response_length) or self.response_length,
            agent_data.per_turn_response_limit,
        )
        # Also don't exceed the remaining response_length budget for this trajectory.
        remaining = self.response_length - len(agent_data.response_mask)
        per_turn_cap = min(per_turn_cap, max(remaining, 1))
        sp["max_new_tokens"] = per_turn_cap

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sp,
                image_data=agent_data.image_data,
            )

        agent_data.response_ids = list(output.token_ids)
        agent_data.prompt_ids.extend(agent_data.response_ids)
        agent_data.response_mask.extend([1] * len(agent_data.response_ids))
        if getattr(output, "log_probs", None):
            agent_data.response_logprobs.extend(output.log_probs)

        # Decode for env-step input
        assistant_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True),
        )
        agent_data.last_assistant_text = assistant_text
        agent_data.messages.append({"role": "assistant", "content": assistant_text})

        return AgentState.INTERACTING

    async def _handle_env(self, agent_data: AgentData, env: ThorViewEnv) -> AgentState:
        """Step env with the last assistant text; encode new obs into prompt."""
        action_str = agent_data.last_assistant_text or ""
        try:
            obs, reward, done, info = await env.step(action_str)
        except Exception as exc:
            logger.error(
                "env.step failed with action=%r: %s\n%s",
                action_str[:200], exc, traceback.format_exc(),
            )
            obs = {"obs_str": "Environment error.", "multi_modal_input": {IMAGE_PH: []}}
            reward, done, info = 0.0, True, {"traj_success": False, "error": "env_step_exc"}

        agent_data.env_rewards.append(float(reward))
        agent_data.traj_success = bool(info.get("success", False) or info.get("traj_success", False))
        agent_data.env_turns += 1

        # Format-ladder counters from env info (logged via wandb).
        fs = float(info.get("format_score", 0.0))
        if info.get("format_strict"):
            agent_data.format_strict_count += 1
        elif fs >= 0.10:
            agent_data.format_medium_count += 1
        elif fs > 0:
            agent_data.format_weak_count += 1
        else:
            agent_data.format_none_count += 1

        # Termination: done from env (success OR max_steps) OR turn cap OR token cap.
        if done:
            return AgentState.TERMINATED
        if agent_data.env_turns >= self.max_turns:
            return AgentState.TERMINATED
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED

        # Encode the new user turn — pass only NEW images so apply_chat_template
        # embeds the right token count for them. The system prompt is stripped
        # via remove_system_prompt=True so we get just the user-suffix ids.
        new_images_raw = obs.get("multi_modal_input", {}).get(IMAGE_PH, []) or []
        new_images = _normalize_images(new_images_raw)
        user_msg = {"role": "user", "content": _obs_to_content(obs)}
        agent_data.messages.append(user_msg)

        # tokenize JUST this user message, stripped of system prefix
        suffix_ids = await self.apply_chat_template(
            [user_msg],
            images=new_images or None,
            remove_system_prompt=True,
        )
        agent_data.prompt_ids.extend(suffix_ids)
        agent_data.response_mask.extend([0] * len(suffix_ids))
        if agent_data.response_logprobs:
            agent_data.response_logprobs.extend([0.0] * len(suffix_ids))

        if new_images:
            agent_data.image_data.extend(new_images)

        return AgentState.GENERATING
