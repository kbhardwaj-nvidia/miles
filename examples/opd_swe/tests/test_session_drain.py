"""Session-server drain (POST /drain, this branch) end to end through the FastAPI routes, real Qwen3.5 tokenizer.

From the Miles repo root (CPU is enough; STUDENT_HF = a local Qwen3.5-35B-A3B checkpoint dir):
    STUDENT_HF=... PYTHONPATH=.:examples/opd_swe python -m pytest -q examples/opd_swe/tests/test_session_drain.py
"""

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.sessions import setup_session_routes

STUDENT_HF = os.environ.get("STUDENT_HF")
pytestmark = pytest.mark.skipif(not STUDENT_HF, reason="set STUDENT_HF to a Qwen3.5 checkpoint dir (tokenizer)")
BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}


class _CountingBackend:
    def __init__(self):
        self.calls = 0

    async def do_proxy(self, *args, **kwargs):
        self.calls += 1
        return {"status_code": 503, "headers": {}, "response_body": b'{"error": "backend reached"}'}


@pytest.fixture()
def client_and_backend():
    app = FastAPI()
    backend = _CountingBackend()
    cfg = make_session_server_config(
        hf_checkpoint=STUDENT_HF,
        tito_model="qwen35",
        session_message_matcher="miles_ext.matchers.loose_tool_call_reasoning_optional",
    )
    setup_session_routes(app, backend, cfg)
    return TestClient(app), backend


def _new_session(client):
    r = client.post("/sessions", json={})
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def test_drained_session_gets_empty_stop_without_backend(client_and_backend):
    client, backend = client_and_backend
    old = _new_session(client)
    r = client.post("/drain")
    assert r.status_code == 200 and r.json()["open_sessions_drained"] == 1
    r = client.post(f"/sessions/{old}/v1/chat/completions", json=BODY)
    assert r.status_code == 200, r.text
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "stop" and choice["message"]["content"] == ""
    assert backend.calls == 0
    # Streaming clients (opencode) get a single SSE chunk plus [DONE].
    r = client.post(f"/sessions/{old}/v1/chat/completions", json={**BODY, "stream": True})
    assert r.status_code == 200 and "data: [DONE]" in r.text and '"finish_reason":"stop"' in r.text
    assert backend.calls == 0
    # Nothing was recorded for the drained session.
    info = client.get(f"/sessions/{old}").json()
    assert not info.get("records"), info


def test_session_created_after_drain_is_served(client_and_backend):
    client, backend = client_and_backend
    _new_session(client)
    client.post("/drain")
    new = _new_session(client)
    r = client.post(f"/sessions/{new}/v1/chat/completions", json=BODY)
    assert backend.calls == 1, "a session created after the drain must reach the backend"
    assert r.status_code == 503
