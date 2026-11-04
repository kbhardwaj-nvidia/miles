"""Miles agent function: one NeMo Gym SWE episode (opencode coding agent in a sandbox) per sample.

Wired with
    --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
    --custom-agent-function-path miles_ext.gym_swe_agent.run
Each prompt row carries its full Gym task row in ``metadata`` (see README.md, "Data").
The row is posted to the Gym agent server named by ``agent_ref.name`` with ``policy_base_url`` set to
this sample's Miles session URL, so every opencode model call goes through the session server, which
records token ids and logprobs (TITO). The Gym episode grade is returned as ``reward``.

Env:
  GYM_AGENT_URLS     JSON {agent_name: base_url} of the Gym agent servers.
  GYM_RUN_TIMEOUT    seconds before the trainer gives up on one /run (default 6000; the Gym agent's
                     sandbox_timeout + evaluation_timeout must stay below it).
"""

import asyncio
import json
import logging
import os
import random
import time
from typing import Any

import aiohttp

from miles.rollout.agentic.session import openai_session_url

logger = logging.getLogger(__name__)

# Keys Miles adds to sample metadata before calling the agent; not part of the Gym task row.
_MILES_METADATA_KEYS = ("max_seq_len", "session_server_instance_id", "session_server_id", "miles_opd_eval")
_POST_ATTEMPTS = 3


def _agent_urls() -> dict[str, str]:
    urls = json.loads(os.environ["GYM_AGENT_URLS"])
    return {name: url.rstrip("/") for name, url in urls.items()}


def build_run_body(base_url: str, request_kwargs: dict[str, Any], metadata: dict[str, Any]) -> tuple[str, dict]:
    row = {k: v for k, v in metadata.items() if k not in _MILES_METADATA_KEYS}
    agent_name = (row.get("agent_ref") or {}).get("name")
    urls = _agent_urls()
    if agent_name not in urls:
        raise KeyError(f"no Gym agent server for agent_ref.name={agent_name!r}; known: {sorted(urls)}")
    params = dict(row.get("responses_create_params") or {})
    for key in ("temperature", "top_p"):
        if request_kwargs.get(key) is not None:
            params[key] = request_kwargs[key]
    row["responses_create_params"] = params
    row["policy_base_url"] = openai_session_url(base_url)
    return urls[agent_name], row


async def _post_run(url: str, body: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60)
    for attempt in range(_POST_ATTEMPTS):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{url}/run", json=body) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        raise RuntimeError(f"Gym /run HTTP {resp.status}: {text[:2000]}")
                    return json.loads(text)
        except aiohttp.ClientConnectorError:
            # Only a connection that was never established is safe to retry: once Gym has accepted /run,
            # a retry would start a second episode against the same Miles session.
            if attempt == _POST_ATTEMPTS - 1:
                raise
            await asyncio.sleep(random.uniform(2.0, 10.0))
    raise AssertionError("unreachable")


def _scalar_fields(response: dict) -> dict[str, Any]:
    """Small scalar verifier fields worth keeping in the sample metadata (the full transcript is not)."""
    keep = {}
    for key, value in response.items():
        if key in ("response", "responses_create_params", "metadata"):
            continue
        if isinstance(value, (bool, int, float)) or (isinstance(value, str) and len(value) <= 200):
            keep[key] = value
    return keep


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any] | None:
    metadata = metadata or {}
    request_kwargs = request_kwargs or {}
    url, body = build_run_body(base_url, request_kwargs, metadata)
    timeout_s = float(os.environ.get("GYM_RUN_TIMEOUT", "6000"))
    t0 = time.monotonic()
    try:
        response = await asyncio.wait_for(_post_run(url, body), timeout=timeout_s)
    except asyncio.TimeoutError:
        logger.error("Gym /run timed out after %.0fs (%s)", timeout_s, metadata.get("instance_id"))
        return {"reward": 0.0, "gym_error": "timeout", "gym_seconds": time.monotonic() - t0}
    except Exception as e:  # the recorded session still becomes a sample; it scores 0
        logger.error("Gym /run failed for %s: %r", metadata.get("instance_id"), e)
        return {"reward": 0.0, "gym_error": repr(e)[:2000], "gym_seconds": time.monotonic() - t0}
    reward = float(response.get("reward") or 0.0)
    return {"reward": reward, "gym_error": None, "gym_seconds": time.monotonic() - t0,
            "gym_result": _scalar_fields(response)}


async def abort(args) -> None:
    """Rollout-abort hook (Miles calls it after aborting SGLang, then waits for every pending episode).

    An aborted SGLang request does not end an opencode episode: opencode retries, so leftover episodes
    (e.g. a whole extra batch submitted after one dropped group) would run to completion before
    training. Drain every session server instead: each session created so far gets the empty "stop"
    reply at its next model call, so those episodes finish (and verify) right away. Sessions created
    later, by the next rollout, are unaffected. Requires the v1 session server's POST /drain (this branch).
    """
    instances = getattr(args, "session_server_instances", None) or []
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def drain(url: str) -> str:
            try:
                async with session.post(f"{url}/drain") as resp:
                    return f"{url}: HTTP {resp.status} {(await resp.text())[:120]}"
            except Exception as e:  # best effort: an undrained episode only costs time
                return f"{url}: {e!r}"

        results = await asyncio.gather(*(drain(instance.url) for instance in instances))
    logger.info("Rollout abort: drained %d session servers: %s", len(instances), "; ".join(results))
