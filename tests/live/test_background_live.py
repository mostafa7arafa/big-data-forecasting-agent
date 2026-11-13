"""Live: a slow forecast runs as a background task under ``langgraph dev``.

Needs model and LangSmith keys plus the dataset (``.env``). Run with
``uv run pytest -m live tests/live -v``; ``make test`` excludes it.

On one thread it:
1. asks for a slow forecast;
2. sees a background task id in state within 60 s;
3. asks a side question and gets an answer while the task runs;
4. within 300 s sees an ``async_subagent`` notification followed by an answer
   whose verification label is ``verified``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from analyst.background import is_background_notification
from analyst.config import get_settings

pytestmark = pytest.mark.live

REPO = Path(__file__).resolve().parents[2]
SLOW_FORECAST = "Forecast daily trips for the next 30 days by borough."
SIDE_QUESTION = "What was the total in the first month?"
TASK_WAIT_S = 60
DONE_WAIT_S = 300


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_ok(url: str, timeout_s: float, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"langgraph dev exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{url}/ok", timeout=2) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError(f"langgraph dev not ready at {url}")


@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    settings = get_settings()
    if settings.openai_api_key is None and settings.openrouter_api_key is None:
        pytest.skip("no model API key configured")
    if not settings.dataset:
        pytest.skip("ANALYST_DATASET not set")
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    port = _free_port()
    log = tmp_path_factory.mktemp("langgraph") / "dev.log"
    with log.open("w") as out:
        proc = subprocess.Popen(
            [
                "uv",
                "run",
                "langgraph",
                "dev",
                "--port",
                str(port),
                "--no-browser",
                "--no-reload",
                "--n-jobs-per-worker",
                "10",
            ],
            cwd=REPO,
            stdout=out,
            stderr=subprocess.STDOUT,
            env={**os.environ},
        )
    url = f"http://127.0.0.1:{port}"
    try:
        _wait_ok(url, 120, proc)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def _messages(state: dict) -> list[dict]:
    return list((state.get("values") or {}).get("messages") or [])


def _is_ai_answer(message: dict) -> bool:
    return message.get("type") == "ai" and not message.get("tool_calls")


_ACTIVE_RUNS = frozenset({"pending", "running"})


async def _task_status(client, task: dict) -> str:
    """The server-side status of the task thread's newest run (as background.py reads it)."""
    runs = await client.runs.list(thread_id=task["thread_id"], limit=1)
    return runs[0].get("status", "?") if runs else "no run"


async def _scenario(url: str) -> None:
    from langgraph_sdk import get_client

    client = get_client(url=url)
    thread = await client.threads.create()
    tid = thread["thread_id"]
    context = {"background": True}

    # 1-2. The slow forecast starts a background task.
    await client.runs.wait(
        tid,
        "analyst",
        input={"messages": [{"role": "user", "content": SLOW_FORECAST}]},
        context=context,
    )
    deadline = time.monotonic() + TASK_WAIT_S
    tasks: dict = {}
    while time.monotonic() < deadline:
        state = await client.threads.get_state(tid)
        tasks = (state.get("values") or {}).get("async_tasks") or {}
        if tasks:
            break
        await asyncio.sleep(2)
    assert tasks, "no background task started within 60 s"
    task = next(iter(tasks.values()))

    # 3. A side question is answered while the task runs: the task must be
    # running before the question is sent and still running once it is answered
    # (an answer queued behind the forecast would fail here).
    status = await _task_status(client, task)
    assert status in _ACTIVE_RUNS, (
        f"background task not running before the side question (status {status!r})"
    )
    before = len(_messages(await client.threads.get_state(tid)))
    await client.runs.wait(
        tid,
        "analyst",
        input={"messages": [{"role": "user", "content": SIDE_QUESTION}]},
        context=context,
        multitask_strategy="enqueue",
    )
    status = await _task_status(client, task)
    after = _messages(await client.threads.get_state(tid))
    assert any(_is_ai_answer(m) for m in after[before:]), "side question unanswered"
    if status not in _ACTIVE_RUNS:
        pytest.fail(
            f"background task already finished (status {status!r}) when the side answer "
            "arrived; the answer did not prove chatting while the task runs"
        )

    # 4. The notification arrives, then a verified answer.
    deadline = time.monotonic() + DONE_WAIT_S
    while time.monotonic() < deadline:
        state = await client.threads.get_state(tid)
        messages = _messages(state)
        notes = [i for i, m in enumerate(messages) if is_background_notification(m)]
        verification = (state.get("values") or {}).get("verification") or {}
        if (
            notes
            and any(_is_ai_answer(m) for m in messages[notes[-1] + 1 :])
            and not state.get("next")
        ):
            assert verification.get("label") == "verified", verification
            return
        await asyncio.sleep(5)
    pytest.fail("no async_subagent notification and verified answer within 300 s")


def test_background_forecast_while_chatting(server_url):
    asyncio.run(_scenario(server_url))
