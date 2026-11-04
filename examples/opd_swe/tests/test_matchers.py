"""Tests for miles_ext.matchers (pure python: loads Miles' matcher modules by path, no torch needed).

    python3 -I examples/opd_swe/tests/test_matchers.py
"""

import importlib.util
import json
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[3]  # the Miles repo root
HUB = ROOT / "miles" / "utils" / "chat_template_utils" / "message_matcher_hub"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for pkg in ("miles", "miles.utils", "miles.utils.chat_template_utils", "miles.utils.chat_template_utils.message_matcher_hub"):
    sys.modules.setdefault(pkg, types.ModuleType(pkg))
_load("miles.utils.chat_template_utils.message_matcher_hub.utils", HUB / "utils.py")
_load("miles.utils.chat_template_utils.message_matcher_hub.funcs", HUB / "funcs.py")
matchers = _load("miles_ext.matchers", ROOT / "examples" / "opd_swe" / "miles_ext" / "matchers.py")
match = matchers.loose_tool_call_reasoning_optional


def call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def asst(calls, content="I'll check.", reasoning="think"):
    m = {"role": "assistant", "content": content, "tool_calls": calls}
    if reasoning is not None:
        m["reasoning_content"] = reasoning
    return m


def invalid(cid, tool):
    return call(cid, "invalid", {"tool": tool, "error": f"Model tried to call unavailable tool '{tool}'."})


CASES = [
    ("identical", asst([call("c1", "bash", {"command": "ls"})]), asst([call("c1", "bash", {"command": "ls"})]), True),
    ("reasoning dropped", asst([call("c1", "bash", {"command": "ls"})]), asst([call("c1", "bash", {"command": "ls"})], reasoning=None), True),
    ("invalid repair", asst([call("c1", "todo", {"x": 1})]), asst([invalid("c1", "todo")], reasoning=None), True),
    ("invalid repair of garbled name", asst([call("c1", "gre<parameter=path", {})]), asst([invalid("c1", "gre<parameter=path")]), True),
    ("lowercase repair", asst([call("c1", "Bash", {"command": "ls"})]), asst([call("c1", "bash", {"command": "ls"})]), True),
    ("lowercase repair, args changed", asst([call("c1", "Bash", {"command": "ls"})]), asst([call("c1", "bash", {"command": "pwd"})]), False),
    ("mixed: one repaired, one same", asst([call("c1", "todo", {}), call("c2", "read", {"filePath": "/a"})]),
     asst([invalid("c1", "todo"), call("c2", "read", {"filePath": "/a"})]), True),
    ("swapped call order", asst([call("c1", "todo", {}), call("c2", "read", {"filePath": "/a"})]),
     asst([call("c2", "read", {"filePath": "/a"}), invalid("c1", "todo")]), False),
    ("invalid names another tool", asst([call("c1", "todo", {})]), asst([invalid("c1", "lfs")]), False),
    ("invalid with different call id", asst([call("c1", "todo", {})]), asst([invalid("c9", "todo")]), False),
    ("invalid without error field", asst([call("c1", "todo", {})]), asst([call("c1", "invalid", {"tool": "todo"})]), False),
    ("content differs", asst([call("c1", "todo", {})]), asst([invalid("c1", "todo")], content="Something else."), False),
    ("reasoning differs (present)", asst([call("c1", "bash", {"command": "ls"})]), asst([call("c1", "bash", {"command": "ls"})], reasoning="other"), False),
    ("extra call", asst([call("c1", "todo", {})]), asst([invalid("c1", "todo"), call("c2", "bash", {"command": "ls"})]), False),
    ("valid call renamed to invalid", asst([call("c1", "bash", {"command": "ls"})]), asst([invalid("c1", "read")]), False),
    ("user message differs", {"role": "user", "content": "a"}, {"role": "user", "content": "b"}, False),
    ("tool message same", {"role": "tool", "tool_call_id": "c1", "content": "ok"}, {"role": "tool", "tool_call_id": "c1", "content": "ok"}, True),
]

failures = 0
for name, stored, replayed, expected in CASES:
    got = match(stored, replayed)
    ok = got is expected
    failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got}, expected {expected}")
print(f"{len(CASES) - failures}/{len(CASES)} passed")
sys.exit(1 if failures else 0)
