"""OPD reward hooks for agentic SWE samples: teacher scoring plus the Gym task reward.

    --custom-rm-path miles_ext.opd_reward.reward_func
    --custom-reward-post-process-path miles_ext.opd_reward.post_process_rewards

reward_func scores each training sample's full token sequence with the SGLang teacher (Miles'
sampled-token OPD request to --rm-url), stores the teacher log-probs of the response span on the
sample, and returns the Gym grade as the scalar reward. Eval samples (tagged miles_opd_eval by the
eval config's metadata_overrides, see eval_config.yaml) skip teacher scoring and return the grade only.

post_process_rewards logs the Gym grade as raw_reward and trains on reward 0 (pure OPD: the
advantage is -opd_kl_coef * (log p_student - log p_teacher) per token), unless OPD_TASK_REWARD=1
adds the grade through the selected estimator (grounded OPD).
"""

import asyncio
import logging
import os
from argparse import Namespace
from typing import Any

from miles.rollout import on_policy_distillation as opd
from miles.utils.types import Sample

logger = logging.getLogger(__name__)
_TEACHER_ATTEMPTS = 4


async def _score_with_teacher(args: Namespace, sample: Sample) -> dict[str, Any]:
    for attempt in range(_TEACHER_ATTEMPTS):
        try:
            return await opd.reward_func(args, sample)
        except Exception as e:  # transient teacher-server errors must not kill the rollout
            if attempt == _TEACHER_ATTEMPTS - 1:
                raise
            logger.warning("teacher scoring failed (attempt %d, %d tokens): %r", attempt + 1, len(sample.tokens), e)
            await asyncio.sleep(15 * (attempt + 1))
    raise AssertionError("unreachable")


async def reward_func(args: Namespace, sample: Sample, **kwargs: Any) -> float:
    metadata = sample.metadata or {}
    task_reward = float(metadata.get("reward") or 0.0)
    if metadata.get("miles_opd_eval") or sample.status == Sample.Status.ABORTED or sample.response_length == 0:
        return task_reward
    teacher = await _score_with_teacher(args, sample)
    sample.teacher_log_probs = opd._teacher_sampled_log_probs(teacher, sample.response_length).tolist()
    return task_reward


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs: Any) -> tuple[list[float], list[float]]:
    task_rewards = []
    for sample in samples:
        task_rewards.append(float(sample.reward or 0.0))
        if sample.teacher_log_probs is None:
            # Aborted samples are dropped by check_no_aborted before this point; an unscored non-empty
            # response would turn the reverse KL into -log p_student, so refuse it.
            if sample.response_length > 0:
                raise ValueError(f"sample {sample.index} has a {sample.response_length}-token response but no teacher score")
            sample.teacher_log_probs = []
    train_rewards = list(task_rewards) if os.environ.get("OPD_TASK_REWARD") == "1" else [0.0] * len(samples)
    return task_rewards, train_rewards
