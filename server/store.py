"""Jobs on disk, so a restart does not forget them.

One SQLite table, written through the standard library.  The rows hold what
the browser needs to draw a card and what the manager needs to pick a job
back up; the heavy things -- the transcript, the proposals, the video --
stay in the job's work directory and are re-read from there.

Every state change already funnels through ``JobManager._publish``, which is
where the row is written, so nothing here needs to be remembered by callers.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    owner         TEXT NOT NULL,
    filename      TEXT NOT NULL,
    preset        TEXT NOT NULL,
    caption_style TEXT NOT NULL,
    status        TEXT NOT NULL,
    stage         TEXT NOT NULL,
    message       TEXT NOT NULL,
    error         TEXT,
    keep          TEXT,
    summary       TEXT,
    output        TEXT,
    publish       TEXT,
    created       TEXT NOT NULL,
    bytes         INTEGER NOT NULL DEFAULT 0,
    -- Reserved for sharing a finished video by link; nothing sets it yet.
    share_token   TEXT
);
CREATE INDEX IF NOT EXISTS jobs_owner ON jobs (owner, created);
"""


@dataclass
class Row:
    """A job as the table holds it.  Plain data; ``Job`` is built from it."""

    id: str
    owner: str
    filename: str
    preset: str
    caption_style: str
    status: str
    stage: str
    message: str
    error: str | None
    keep: list[int] | None
    summary: dict | None
    output: str | None
    publish: dict | None
    created: datetime
    bytes: int


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def save(self, row: Row) -> None:
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO jobs (id, owner, filename, preset, caption_style, status, stage,
                                  message, error, keep, summary, output, publish, created, bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    caption_style = excluded.caption_style, status = excluded.status,
                    stage = excluded.stage, message = excluded.message, error = excluded.error,
                    keep = excluded.keep, summary = excluded.summary, output = excluded.output,
                    publish = excluded.publish, bytes = excluded.bytes
                """,
                (
                    row.id, row.owner, row.filename, row.preset, row.caption_style,
                    row.status, row.stage, row.message, row.error,
                    json.dumps(row.keep) if row.keep is not None else None,
                    json.dumps(row.summary) if row.summary is not None else None,
                    row.output,
                    json.dumps(row.publish) if row.publish is not None else None,
                    row.created.isoformat(), row.bytes,
                ),
            )

    def delete(self, job_id: str) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def all(self) -> list[Row]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created").fetchall()
        return [self._row(r) for r in rows]

    def usage(self, owner: str) -> tuple[int, int]:
        """(jobs, bytes) an owner currently holds."""
        with self._connect() as db:
            count, total = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM jobs WHERE owner = ?", (owner,)
            ).fetchone()
        return int(count), int(total)

    @staticmethod
    def _row(r: sqlite3.Row) -> Row:
        return Row(
            id=r["id"], owner=r["owner"], filename=r["filename"], preset=r["preset"],
            caption_style=r["caption_style"], status=r["status"], stage=r["stage"],
            message=r["message"], error=r["error"],
            keep=json.loads(r["keep"]) if r["keep"] else None,
            summary=json.loads(r["summary"]) if r["summary"] else None,
            output=r["output"],
            publish=json.loads(r["publish"]) if r["publish"] else None,
            created=datetime.fromisoformat(r["created"]), bytes=r["bytes"],
        )
