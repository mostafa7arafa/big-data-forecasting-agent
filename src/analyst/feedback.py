"""Thumbs up/down feedback on agent runs, appended to a local JSON Lines file."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from analyst.config import get_settings

LOCAL_USER = "local"

_lock = threading.Lock()


class Feedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thread_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    rating: Literal["up", "down"]
    correction: str | None = Field(default=None, max_length=2000)


def feedback_path() -> Path:
    return get_settings().artifacts_dir / "feedback" / "feedback.jsonl"


def append_feedback(item: Feedback) -> None:
    record = {**item.model_dump(), "user_id": LOCAL_USER, "at": datetime.now(UTC).isoformat()}
    line = json.dumps(record, ensure_ascii=False) + "\n"
    path = feedback_path()
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
