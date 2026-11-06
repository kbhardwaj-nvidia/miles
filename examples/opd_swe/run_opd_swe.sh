#!/bin/bash
# Pure on-policy distillation (OPD) on agentic SWE episodes: Qwen3.5-122B-A10B teacher -> Qwen3.5-35B-A3B
# student (see README.md). Run inside the Miles container on the Ray head node once every student node has
# joined the Ray cluster; the teacher SGLang server and the NeMo Gym agent servers run elsewhere.
# Defaults are the settings of our baseline (2 student nodes x 4 GPUs + 1 teacher node).
set -euo pipefail
MILES_ROOT=${MILES_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
EXAMPLE=$MILES_ROOT/examples/opd_swe
: "${STUDENT_HF:?HF checkpoint dir of the student}" "${STUDENT_TORCH_DIST:?Megatron torch_dist checkpoint of the student}"
: "${TEACHER_URL:?teacher SGLang server, e.g. http://teacher-host:30000}" "${TRAIN_DATA:?training JSONL}"
: "${SAVE_DIR:?checkpoint dir}" "${GYM_AGENT_URLS:?JSON {agent_ref.name: Gym agent server URL}}"
EVAL_CONFIG=${EVAL_CONFIG:-$EXAMPLE/eval_config.yaml}
NUM_TRAIN_NODES=${NUM_TRAIN_NODES:-2}; GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TP=${TP:-2}; CP=${CP:-2}; EP=${EP:-8}; MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-65536}
MAX_TOKENS=${MAX_TOKENS:-131072}   # per-episode budget = --max-seq-len; the session server answers an empty stop at it
TURN_TOKENS=${TURN_TOKENS:-32768}  # per-turn generation cap
MAX_TURNS=${MAX_TURNS:-250}        # per-episode model-call cap
ROLLOUT_BATCH=${ROLLOUT_BATCH:-64}; N_SAMPLES=${N_SAMPLES:-1}; NUM_ROLLOUT=${NUM_ROLLOUT:-200}
LR=${LR:-1e-6}; LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-10}; LR_WARMUP_INIT=${LR_WARMUP_INIT:-1e-7}
OPD_KL_COEF=${OPD_KL_COEF:-1.0}
SAVE_INTERVAL=${SAVE_INTERVAL:-5}; EVAL_INTERVAL=${EVAL_INTERVAL:-20}
export GYM_AGENT_URLS GYM_RUN_TIMEOUT=${GYM_RUN_TIMEOUT:-9000} OPD_TASK_REWARD=${OPD_TASK_REWARD:-0}
export MILES_SESSION_MAX_TOKENS=$MAX_TOKENS MILES_SESSION_MAX_TURN_TOKENS=$TURN_TOKENS MILES_SESSION_MAX_TURNS=$MAX_TURNS
CTX=$((MAX_TOKENS + 8192))

MODEL_ARGS=$(python3 "$MILES_ROOT/miles/utils/external_utils/model_args_utils.py" qwen3.5-35B-A3B)
set -f; MODEL=($MODEL_ARGS); set +f   # --moe-layer-freq [1,1,...] must not glob

ARGS=(
  --actor-num-nodes "$NUM_TRAIN_NODES" --actor-num-gpus-per-node "$GPUS_PER_NODE" --num-gpus-per-node "$GPUS_PER_NODE"
  --colocate
  "${MODEL[@]}"
  --hf-checkpoint "$STUDENT_HF" --ref-load "$STUDENT_TORCH_DIST"
  --load "$SAVE_DIR" --save "$SAVE_DIR" --save-interval "$SAVE_INTERVAL"
  # data: one Gym task per prompt, the full Gym task row in metadata
  --prompt-data "$TRAIN_DATA" --input-key prompt --metadata-key metadata
  --rollout-shuffle --rollout-seed "${ROLLOUT_SEED:-20261102}" --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH" --n-samples-per-prompt "$N_SAMPLES"
  --rollout-temperature 1.0 --rollout-max-response-len "$TURN_TOKENS"
  --global-batch-size $((ROLLOUT_BATCH * N_SAMPLES)) --num-steps-per-rollout 1 --balance-data
  --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
  # agentic rollout: opencode (in NeMo Gym) calls the TITO session server; parsers are not derived from --tito-model
  --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
  --custom-agent-function-path miles_ext.gym_swe_agent.run
  --use-session-server --session-server-port "${SESSION_SERVER_PORT:-30000}" --session-server-workers 8
  --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3
  --tito-model qwen35 --session-message-matcher miles_ext.matchers.loose_tool_call_reasoning_optional
  --max-seq-len "$MAX_TOKENS"
  # OPD: sampled-token reverse KL against the SGLang teacher; the Gym grade is logged, not trained on
  --use-opd --opd-type sglang --opd-kl-coef "$OPD_KL_COEF" --opd-log-prob-top-k 0
  --rm-url "${TEACHER_URL%/}/generate"
  --custom-rm-path miles_ext.opd_reward.reward_func
  --custom-reward-post-process-path miles_ext.opd_reward.post_process_rewards
  --advantage-estimator grpo --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 --use-tis
  --optimizer adam --lr "$LR" --lr-decay-style constant --weight-decay 0.0
  --lr-warmup-iters "$LR_WARMUP_ITERS" --lr-warmup-init "$LR_WARMUP_INIT"
  --adam-beta1 0.9 --adam-beta2 0.98 --clip-grad 1.0
  --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer
  --calculate-per-token-loss --moe-router-freeze-gate
  # student parallelism (alltoall: the flex/DeepEP dispatcher failed cross-node in our containers)
  --tensor-model-parallel-size "$TP" --sequence-parallel --pipeline-model-parallel-size 1
  --context-parallel-size "$CP" --expert-model-parallel-size "$EP" --expert-tensor-parallel-size 1
  --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
  --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" --log-probs-chunk-size 4096
  # rollout engines: one TP1 SGLang engine per GPU, weights pushed from the trainer (dummy load)
  --rollout-num-gpus-per-engine 1 --sglang-load-format dummy --sglang-mem-fraction-static 0.7
  --sglang-context-length "$CTX"
  --sglang-moe-runner-backend flashinfer_cutlass --sglang-attention-backend trtllm_mha
  --sglang-mamba-radix-cache-strategy extra_buffer --sglang-watchdog-timeout 1800
  --attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32 --attention-backend flash
  --eval-interval "$EVAL_INTERVAL" --eval-config "$EVAL_CONFIG"
  --override-opt_param-scheduler   # lets a resume change --num-rollout
)
[[ -n ${WANDB_PROJECT:-} ]] && ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
[[ -n ${DUMP_DIR:-} ]] && ARGS+=(--dump-details "$DUMP_DIR")
[[ -f $SAVE_DIR/latest_checkpointed_iteration.txt ]] && ARGS+=(--skip-eval-before-train)

# Behavioral env knobs ride in the job's runtime env, not in the raylets' environment.
RUNTIME_ENV=$(JOB_PYTHONPATH="/root/Megatron-LM:$MILES_ROOT:$EXAMPLE" python3 -c '
import json, os
keys = ["GYM_AGENT_URLS", "GYM_RUN_TIMEOUT", "OPD_TASK_REWARD",
        "MILES_SESSION_MAX_TOKENS", "MILES_SESSION_MAX_TURN_TOKENS", "MILES_SESSION_MAX_TURNS"]
env = {k: os.environ[k] for k in keys}
env["PYTHONPATH"] = os.environ["JOB_PYTHONPATH"]
env["PYTHONUNBUFFERED"] = "1"  # live driver and worker logs (Ray buffers their stdout otherwise)
print(json.dumps({"env_vars": env}))')

ray job submit --address="${RAY_ADDRESS:-http://127.0.0.1:8265}" --runtime-env-json="$RUNTIME_ENV" \
  -- python3 "$MILES_ROOT/train.py" "${ARGS[@]}"
