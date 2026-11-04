"""Session history matcher for opencode, which does not replay assistant turns verbatim.

    --session-message-matcher miles_ext.matchers.loose_tool_call_reasoning_optional

opencode (AI SDK openai-compatible provider, v1.17) changes a stored assistant turn in two ways when
it sends the history back:

1. It drops reasoning_content.
2. It repairs tool calls that failed validation (session/llm.ts experimental_repairToolCall): a call
   whose name only differs in case from a real tool is replayed under the lowercase name, and any
   other failing call (unknown tool, schema-invalid arguments, garbled name) is replayed as a call to
   `invalid` with arguments {"tool": <original name>, "error": <message>}, keeping the call id.

Under the strict matcher either change counts as divergence: the session rolls the generated turn back
and records opencode's rewritten copy as prompt tokens, so the turn the policy actually produced is
lost from training. Accept both forms, with role, content, call ids and call order still required to
match; on a match Miles keeps its stored tokens, so the policy sees and trains on what it generated,
followed by opencode's tool result for the same call id.
"""

import json
from typing import Any

from miles.utils.chat_template_utils.message_matcher_hub.funcs import (
    _arguments_match,
    _tool_call_matches,
    loose_tool_call_message_matches,
)
from miles.utils.chat_template_utils.message_matcher_hub.utils import _normalize_value


def _repaired_call_matches(stored: Any, replayed: Any) -> bool:
    """Whether `replayed` is opencode's repair of the stored call `stored`."""
    if not isinstance(stored, dict) or not isinstance(replayed, dict):
        return False
    if stored.get("id") != replayed.get("id") or stored.get("type") != replayed.get("type"):
        return False
    stored_fn, replayed_fn = stored.get("function") or {}, replayed.get("function") or {}
    stored_name, replayed_name = stored_fn.get("name"), replayed_fn.get("name")
    if not isinstance(stored_name, str) or not isinstance(replayed_name, str):
        return False
    if replayed_name != stored_name and replayed_name == stored_name.lower():
        return _arguments_match(stored_fn.get("arguments"), replayed_fn.get("arguments"))
    if replayed_name == "invalid" and stored_name != "invalid":
        try:
            args = json.loads(replayed_fn.get("arguments") or "")
        except (TypeError, ValueError):
            return False
        return isinstance(args, dict) and args.get("tool") == stored_name and isinstance(args.get("error"), str)
    return False


def _tool_calls_match_allowing_repairs(stored: dict[str, Any], replayed: dict[str, Any]) -> bool:
    stored_calls = _normalize_value(stored.get("tool_calls"))
    replayed_calls = _normalize_value(replayed.get("tool_calls"))
    if not isinstance(stored_calls, list) or not isinstance(replayed_calls, list):
        return False
    if len(stored_calls) != len(replayed_calls):
        return False
    return all(
        _tool_call_matches(s, r) or _repaired_call_matches(s, r) for s, r in zip(stored_calls, replayed_calls, strict=True)
    )


def loose_tool_call_reasoning_optional(stored: dict[str, Any], replayed: dict[str, Any]) -> bool:
    if loose_tool_call_message_matches(stored, replayed):
        return True
    if stored.get("role") != "assistant" or replayed.get("role") != "assistant":
        return False
    if _normalize_value(replayed.get("reasoning_content")) is None:
        replayed = {**replayed, "reasoning_content": stored.get("reasoning_content")}
        if loose_tool_call_message_matches(stored, replayed):
            return True
    for key in ("content", "reasoning_content"):
        if _normalize_value(stored.get(key)) != _normalize_value(replayed.get(key)):
            return False
    return _tool_calls_match_allowing_repairs(stored, replayed)
