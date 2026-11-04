# On-policy distillation on agentic SWE episodes (Qwen3.5-122B-A10B -> Qwen3.5-35B-A3B)

A baseline for on-policy distillation (OPD) on long, multi-turn coding-agent episodes. The student
writes code in a sandboxed repo through the opencode agent; the teacher scores every token the
student generated; the student is trained on the per-token reverse KL. The task grade is logged, not
trained on.

This branch is Miles v0.1.3 plus:
- three session-server changes (below), needed for agent harnesses that do not stop on their own;
- this directory: the Miles plug-ins (`miles_ext/`), a launcher, an eval config and tests.

## Session-server changes (v1 session server)

- **Episode budget** (`MILES_SESSION_MAX_TOKENS`, `MILES_SESSION_MAX_TURNS`): once a session reaches its
  budget, the next chat call gets an empty `stop` reply instead of reaching SGLang. opencode cannot
  stop at a token budget, and its own context compaction would break the append-only TITO session.
- **Per-turn cap** (`MILES_SESSION_MAX_TURN_TOKENS`): each turn's `max_tokens` is clamped to the cap and
  to the remaining episode budget (opencode asks for its full context window every turn).
- **`POST /drain`**: every open session gets the empty `stop` at its next call; sessions created later
  are unaffected. `miles_ext.gym_swe_agent.abort` calls it after a rollout abort, because opencode
  retries aborted requests and leftover episodes would otherwise run to completion.

## How it fits together

```
Miles trainer (Megatron, colocated with SGLang rollout engines; synchronous, strictly on-policy)
  per sample: miles_ext.gym_swe_agent.run  --POST /run {task row, policy_base_url = session URL}-->  Gym agent server
                                                   opencode in a sandbox --chat calls--> Miles session server --> SGLang
                                                   (exact token ids + logprobs recorded; TITO, no re-tokenization)
                                              <--  {"reward": grade}  (logged as raw_reward)
  miles_ext.opd_reward.reward_func: teacher SGLang server scores the finished sequence (prefill only)
  advantage per token = -opd_kl_coef * (log p_student - log p_teacher); loss averaged over tokens
```

The agent function only needs an HTTP `/run` that takes the task row, sends every model call to
`policy_base_url` (an OpenAI-compatible chat endpoint served by the Miles session server) and returns
`{"reward": float}`. We use NeMo Gym's opencode agent server and SWE resource servers; any harness with
that contract works.

## Requirements

- Miles v0.1.3 container with this branch over `/root/miles`.
- Student `Qwen/Qwen3.5-35B-A3B` (HF checkpoint + Megatron `torch_dist` conversion); teacher
  `Qwen/Qwen3.5-122B-A10B` served by SGLang. Teacher and student must share the tokenizer (Qwen3.5,
  vocab 248320) for token-level scoring; Qwen3-30B-A3B does not.
- Teacher server we use (one 4-GPU node): `python3 -m sglang.launch_server --model-path <teacher> --tp-size 4
  --context-length <MAX_TOKENS + 8192> --chunked-prefill-size 16384 --disable-radix-cache`.
- An agent environment as above. With NeMo Gym's opencode agent we needed two small Gym changes:
  honor a per-request `policy_base_url`, and the permission ordering fix (Pitfalls, 2).
- Our hardware: 3 nodes x 4 GPUs. 2 student nodes (Megatron TP2/CP2/EP8 plus 8 TP1 SGLang engines), 1
  teacher node.

## Data

JSONL, one task per line; the full agent-environment task row rides in `metadata`:

```json
{"prompt": <row.responses_create_params.input>, "metadata": <the full task row>}
```

`metadata.agent_ref.name` selects the agent server URL from `GYM_AGENT_URLS`. Eval rows use the same
format; `eval_config.yaml` tags them `miles_opd_eval` so they are graded without teacher scoring.

## Run

```bash
STUDENT_HF=... STUDENT_TORCH_DIST=... TEACHER_URL=http://<teacher>:30000 \
TRAIN_DATA=train.jsonl EVAL_CONFIG=my_eval.yaml SAVE_DIR=ckpt \
GYM_AGENT_URLS='{"<agent_ref.name>": "http://<gym-host>:<port>"}' \
  bash examples/opd_swe/run_opd_swe.sh
```

Defaults (all env-overridable in `run_opd_swe.sh`): 64 tasks x 1 episode per step, one optimizer step per
rollout; 131072-token episodes, 32768 tokens per turn, 250 turns; Adam lr 1e-6 constant after 10 warmup
steps from 1e-7, grad clip 1.0; TIS for the rollout/train mismatch; frozen MoE router; no reference
model or KL loss; held-out eval every 20 steps with 4 samples per task.

Tests: `python3 -I examples/opd_swe/tests/test_matchers.py`, and
`STUDENT_HF=... PYTHONPATH=.:examples/opd_swe python -m pytest -q examples/opd_swe/tests/test_session_drain.py`.

Healthy run, for comparison: train/rollout log-prob KL ~1e-3, TIS clip fraction ~1e-4, grad norm below
the clip, prefix-cache hit rate ~98%, every rollout at the current weight version. The TITO mismatch
rate was 3-14% per rollout before the matcher in Pitfalls 3 (mostly opencode rewriting turns). A step
takes ~40 min on 3 nodes, mostly the rollout's longest episode.

## Pitfalls we hit

1. In v0.1.3 the SGLang tool-call and reasoning parsers are not derived from `--tito-model`; without
   `--sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3` every episode ends after one turn.
2. opencode >= 1.17 resolves permission rules last-match-wins, and turns a `tools` map into rules placed
   before the `permission` block. A catch-all `"*": "allow"` then comes last and silently re-allows
   everything: disabled tools (subagents, web fetch) and every bash deny pattern (e.g. `git log --all`).
   Send one ordered `permission` map, catch-all first, and no `tools` map.
3. opencode repairs tool calls that fail validation when it replays history (lowercased name, or a call
   to `invalid`). With the strict session matcher those turns are rolled back and lost from training;
   `miles_ext.matchers.loose_tool_call_reasoning_optional` accepts the repairs.
4. Use the token-level loss (`--calculate-per-token-loss`): with per-sample averaging, a token of a
   1.7K-token episode weighs ~40x a token of a 68K-token one.
5. The flex/DeepEP MoE dispatcher failed cross-node in our containers (no CUDA IPC across nodes); use alltoall.
6. Step time is set by the slowest episode (one long episode can run alone for 15-40 min).
7. Some clusters hand out ephemeral ports from 9000 up, which covers the fixed ports here and Miles'
   dynamic ports (20000+, `miles/utils/misc.py`); an engine or server can then fail with EADDRINUSE at startup.
