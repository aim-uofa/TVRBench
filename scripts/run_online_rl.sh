#!/bin/bash
# ActiveSpatial online RL — concat multi-turn GRPO from base Qwen3.5-9B.
#
# Differences from run_grpo_active_spatial_single.sh (offline):
#   - MODEL_PATH = base Qwen3.5-9B (NOT sft_single ckpt)
#   - rollout.mode=async + agent_loop_config_path → triggers ThorAgentLoop
#   - data.custom_cls = ThorTaskDataset (loads data/tasks/rl.json directly)
#   - max_prompt_length=2048, max_response_length=16384 (concat accumulates env tokens)
#   - Reward comes from env.step(), NOT a custom_reward_function — see thor_view_env.py
#     for the 3-tier ladder + Stop success bonus.
#   - apply_chat_template_kwargs.enable_thinking=False (base model)
#
# Pre-req:
#   conda activate verl-vllm
#   data/tasks/rl.json present locally
#   tvrbench/, configs/online_rl/, scripts/ all synced to remote
#
# Usage (either works):
#   bash scripts/run_online_rl.sh smoke
#   bash scripts/run_online_rl.sh full
#   MODE=full   KL_LOSS_COEF=0.01 EXP_SUFFIX=_base  bash scripts/run_online_rl.sh
#
# Precedence: $1 (CLI arg) > $MODE (env var) > smoke (default).

set -x

MODE=${1:-${MODE:-smoke}}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
EXP_SUFFIX=${EXP_SUFFIX:-}

# Working dir (shared HDD)
BASEDIR=$(cd "$(dirname "$0")/.." && pwd)
# Init ckpt: base Qwen3.5-9B (untrained).
MODEL_PATH=${MODEL_PATH:-/path/to/model}

# Data (raw JSON, NOT parquet — custom dataset loads directly)
TRAIN_TASKS=${TRAIN_TASKS:-$BASEDIR/data/tasks/rl.json}
VAL_TASKS=${VAL_TASKS:-$BASEDIR/data/tasks/eval.json}

AGENT_LOOP_YAML=$BASEDIR/configs/online_rl/agent.yaml

CKPT_ROOT=${CKPT_ROOT:-$BASEDIR/outputs/online_rl}
PROJECT_NAME=active_spatial_online

if [ ! -f "$TRAIN_TASKS" ]; then
    echo "ERROR: train task file not found at $TRAIN_TASKS" >&2; exit 1
fi
if [ ! -f "$AGENT_LOOP_YAML" ]; then
    echo "ERROR: agent loop yaml not found at $AGENT_LOOP_YAML" >&2; exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: model path not found at $MODEL_PATH" >&2; exit 1
fi

case "$MODE" in
    # NOTE: max_turns and env_pool_size live in configs/online_rl/agent.yaml
    # (passed to ThorAgentLoop by hydra). The launch script does not control them.
    # To tune for smoke vs full, edit agent.yaml directly.
    smoke)
        # 8-GPU smoke (H200D). FSDP FULL_SHARD across 8 cards.
        EXPERIMENT_NAME=qwen3_5_9b_sftv1_online_smoke
        TOTAL_STEPS=3
        # SAVE_FREQ / TEST_FREQ default to mode-suitable values but can be
        # overridden via env vars. For debug iterations, TEST_FREQ=999999
        # effectively disables periodic validation (end-of-training val still runs).
        SAVE_FREQ=${SAVE_FREQ:-10}; TEST_FREQ=${TEST_FREQ:-10}; VAL_BEFORE_TRAIN=False
        BATCH=8; MINI=8; MICRO=1; LOGPROB_MICRO=2
        ROLLOUT_N=2; TP=1
        GPU_MEM_UTIL=0.4
        PARAM_OFFLOAD=False; OPT_OFFLOAD=False
        NUM_GPUS=8
        MAX_RESPONSE_LENGTH=24576    # 8192 × 3; covers ~50 turns. OOM → drop back
        ;;
    full)
        # 8-GPU full (H200D, sft_online_v1 init). Verified 4-GPU baseline 2026-05-17:
        # - PARAM/OPT_OFFLOAD=True: ~25s slower but ~30 GB less peak memory,
        #   safer against vLLM wake_up OOM. Marginal speedup of OFFLOAD=False
        #   not worth the memory risk.
        # - MICRO=2: optimal for 9B+long-seq on H200. MICRO=4 was slower
        #   (memory-bound forward + grad_checkpoint recompute overhead).
        # 8 GPU: doubled BATCH=64 MINI=16 vs 4-GPU 32/8 baseline.
        EXPERIMENT_NAME=qwen3_5_9b_sftv1_online_concat${EXP_SUFFIX}
        TOTAL_STEPS=${TOTAL_STEPS:-2000}
        SAVE_FREQ=${SAVE_FREQ:-100}; TEST_FREQ=${TEST_FREQ:-100}; VAL_BEFORE_TRAIN=False
        BATCH=64; MINI=16; MICRO=2; LOGPROB_MICRO=2
        ROLLOUT_N=8; TP=2
        GPU_MEM_UTIL=0.4
        PARAM_OFFLOAD=True; OPT_OFFLOAD=True
        NUM_GPUS=8
        MAX_RESPONSE_LENGTH=24576    # 8192 × 3; covers ~50 turns. OOM → drop back
        ;;
    *)
        echo "Unknown MODE='$MODE'. Use smoke / full." >&2; exit 2 ;;
esac

SAVE_DIR=$CKPT_ROOT/$EXPERIMENT_NAME
mkdir -p "$SAVE_DIR"
cd "$BASEDIR"

# CUDA / runtime env (same as offline scripts)
export CUDA_HOME=${CUDA_HOME:-$CONDA_PREFIX}
export LD_LIBRARY_PATH=$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}

HOST_TAG=$(hostname -s)
export FLASHINFER_WORKSPACE_BASE=$BASEDIR/.cache/flashinfer-$HOST_TAG
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=$BASEDIR/.cache/torchinductor-$HOST_TAG
mkdir -p $FLASHINFER_WORKSPACE_BASE $TORCHINDUCTOR_CACHE_DIR

export WANDB_MODE=offline
export WANDB_DIR=$BASEDIR/outputs/wandb
mkdir -p "$WANDB_DIR"

export VLLM_USE_V1=1

# NOTE: Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here.
# vLLM's CuMemAllocator (which powers sleep/wake) explicitly asserts it is
# absent — they are mutually exclusive. See vllm/device_allocator/cumem.py:132.

# AI2-THOR Vulkan ICD fix (CloudRendering on H200B)
export VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="$TRAIN_TASKS" \
    data.val_files="$VAL_TASKS" \
    data.train_batch_size=$BATCH \
    data.max_prompt_length=2048 \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=True \
    +data.custom_cls.path=$BASEDIR/tvrbench/online_rl/thor_dataset.py \
    +data.custom_cls.name=ThorTaskDataset \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-7} \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPT_OFFLOAD \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='[Qwen3_5DecoderLayer,Qwen3_5VisionBlock]' \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEM_UTIL \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$AGENT_LOOP_YAML \
    actor_rollout_ref.rollout.agent.num_workers=$NUM_GPUS \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOGPROB_MICRO \
    actor_rollout_ref.ref.fsdp_config.param_offload=$PARAM_OFFLOAD \
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='[Qwen3_5DecoderLayer,Qwen3_5VisionBlock]' \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_STEPS \
    +trainer.max_ckpt_to_keep=3 \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir="$SAVE_DIR" \
    2>&1 | tee "$SAVE_DIR/train.log"
