"""Several people on one instance: identity, isolation, persistence, budgets.

The identity model is a header set by a proxy that already did the login.
What matters is that the app never guesses: no header configured means the
single-user tool it always was; header configured and missing means refuse.
And a job that is not yours does not exist -- 404, so that an id leaks
nothing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autocut.models import KeepSegment, Timeline, Word
from server import app as server_app
from server import auth, retention
from server.jobs import BadUpload, Job, JobManager, Publication, QuotaExceeded
from server.store import Store

HEADER = "Cf-Access-Authenticated-User-Email"


@pytest.fixture
def multiuser(monkeypatch):
    monkeypatch.setenv(auth.HEADER_ENV, HEADER)
    monkeypatch.setenv(auth.OWNERS_ENV, "me@example.com")


@pytest.fixture
def manager(tmp_path, monkeypatch) -> JobManager:
    m = JobManager(tmp_path / "work")
    monkeypatch.setattr(server_app, "manager", m)
    return m


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_app.app)


def as_(email: str) -> dict:
    return {HEADER: email}


def make_timeline(work_dir: Path) -> Timeline:
    words = [Word(text="hi", start=0.0, end=0.5), Word(text="there", start=0.5, end=1.0)]
    return Timeline(
        source=work_dir / "input.mp4", duration=2.0, fps=30, width=1080, height=1920,
        words=words, keep_segments=[KeepSegment(start=0.0, end=2.0)],
    )


def plant(manager: JobManager, job_id: str, owner: str, status: str = "review", **extra) -> Job:
    work_dir = manager.root / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "input.mp4").write_bytes(b"video")
    job = Job(id=job_id, filename=f"{job_id}.mp4", source=work_dir / "input.mp4", work_dir=work_dir,
              owner=owner, status=status, timeline=make_timeline(work_dir), **extra)
    (work_dir / "timeline.json").write_text(job.timeline.model_dump_json())
    manager.jobs[job_id] = job
    manager.store.save(job.row())
    return job


class TestIdentity:
    def test_without_a_header_configured_everyone_is_local(self, client, manager) -> None:
        assert client.get("/me").json() == {"email": "local", "can_publish": True, "local": True}

    def test_with_a_header_configured_it_is_required(self, client, manager, multiuser) -> None:
        assert client.get("/me").status_code == 401
        assert client.get("/jobs").status_code == 401

    def test_the_header_names_the_user(self, client, manager, multiuser) -> None:
        assert client.get("/me", headers=as_("Ann@Example.com")).json() == {
            "email": "ann@example.com", "can_publish": False, "local": False,
        }

    def test_owners_may_publish(self, client, manager, multiuser) -> None:
        assert client.get("/me", headers=as_("me@example.com")).json()["can_publish"] is True


class TestIsolation:
    def test_each_user_sees_only_their_own_jobs(self, client, manager, multiuser) -> None:
        plant(manager, "aaa", "ann@example.com")
        plant(manager, "bbb", "bob@example.com")
        assert [j["id"] for j in client.get("/jobs", headers=as_("ann@example.com")).json()] == ["aaa"]
        assert [j["id"] for j in client.get("/jobs", headers=as_("bob@example.com")).json()] == ["bbb"]

    @pytest.mark.parametrize("path,method,body", [
        ("/jobs/{id}", "GET", None),
        ("/jobs/{id}/parts", "GET", None),
        ("/jobs/{id}/preview", "GET", None),
        ("/jobs/{id}/result", "GET", None),
        ("/jobs/{id}/publish", "GET", None),
        ("/jobs/{id}/render", "POST", {"keep": [0]}),
        ("/jobs/{id}/captions", "POST", {"edits": []}),
        ("/jobs/{id}/publish", "POST", {"caption": ""}),
        ("/jobs/{id}", "DELETE", None),
    ])
    def test_another_users_job_does_not_exist(self, client, manager, multiuser, path, method, body) -> None:
        plant(manager, "aaa", "ann@example.com")
        response = client.request(method, path.format(id="aaa"), headers=as_("bob@example.com"), json=body)
        assert response.status_code == 404
        assert manager.get("aaa") is not None

    def test_the_owner_can_reach_theirs(self, client, manager, multiuser) -> None:
        plant(manager, "aaa", "ann@example.com")
        assert client.get("/jobs/aaa", headers=as_("ann@example.com")).status_code == 200
        assert client.get("/jobs/aaa/parts", headers=as_("ann@example.com")).status_code == 200

    def test_publishing_is_the_operators_alone(self, client, manager, multiuser) -> None:
        plant(manager, "aaa", "ann@example.com", status="done")
        status = client.get("/instagram", headers=as_("ann@example.com")).json()
        assert status == {"connected": False, "error": "publishing is only available to the operator", "operator_only": True}
        assert client.post("/jobs/aaa/publish", headers=as_("ann@example.com"), json={"caption": ""}).status_code == 403


class TestDeletion:
    def test_delete_removes_files_row_and_job(self, client, manager, multiuser) -> None:
        job = plant(manager, "aaa", "ann@example.com")
        assert client.delete("/jobs/aaa", headers=as_("ann@example.com")).json() == {"deleted": "aaa"}
        assert not job.work_dir.exists()
        assert manager.get("aaa") is None
        assert manager.store.all() == []

    def test_a_running_job_cannot_be_deleted(self, client, manager, multiuser) -> None:
        plant(manager, "aaa", "ann@example.com", status="rendering")
        assert client.delete("/jobs/aaa", headers=as_("ann@example.com")).status_code == 409


class TestPersistence:
    def test_jobs_come_back_after_a_restart(self, tmp_path) -> None:
        first = JobManager(tmp_path / "work")
        job = plant(first, "aaa", "ann@example.com", status="done", caption_style="neon", keep=[0])
        job.output = job.work_dir / "final-1.mp4"
        job.output.write_bytes(b"cut")
        job.summary = {"output": "final-1.mp4", "words": 2}
        job.publish = Publication(status="published", permalink="https://instagram.com/reel/x/")
        job.bytes = 5
        first._publish(job)

        second = JobManager(tmp_path / "work")
        second.load()
        back = second.get("aaa")
        assert back is not None
        assert (back.owner, back.status, back.caption_style, back.keep, back.bytes) == ("ann@example.com", "done", "neon", [0], 5)
        assert back.output == job.output and back.summary == job.summary
        assert back.publish.permalink == "https://instagram.com/reel/x/"
        assert back.timeline is not None and len(back.timeline.words) == 2

    def test_a_render_cut_short_goes_back_to_review(self, tmp_path) -> None:
        first = JobManager(tmp_path / "work")
        job = plant(first, "aaa", "ann@example.com", status="rendering", keep=[0])
        second = JobManager(tmp_path / "work"); second.load()
        back = second.get("aaa")
        assert back.status == "review"
        assert "render again" in back.message
        assert back.keep == [0]

    def test_an_analysis_cut_short_asks_for_the_upload_again(self, tmp_path) -> None:
        first = JobManager(tmp_path / "work")
        job = plant(first, "aaa", "ann@example.com", status="analyzing")
        (job.work_dir / "timeline.json").unlink()
        first._publish(job)
        second = JobManager(tmp_path / "work"); second.load()
        assert second.get("aaa").status == "error"

    def test_a_finished_video_that_vanished_is_an_error(self, tmp_path) -> None:
        first = JobManager(tmp_path / "work")
        job = plant(first, "aaa", "ann@example.com", status="done")
        job.output = job.work_dir / "final-gone.mp4"
        job.summary = {"output": "final-gone.mp4"}
        first._publish(job)
        second = JobManager(tmp_path / "work"); second.load()
        back = second.get("aaa")
        assert back.status == "error" and back.summary is None

    def test_the_store_round_trips_a_row(self, tmp_path) -> None:
        store = Store(tmp_path / "db.sqlite")
        job = Job(id="x", filename="x.mp4", source=tmp_path / "in.mp4", work_dir=tmp_path,
                  owner="ann@example.com", keep=[1, 2], summary={"a": 1}, bytes=9)
        store.save(job.row())
        [row] = store.all()
        assert (row.id, row.owner, row.keep, row.summary, row.bytes) == ("x", "ann@example.com", [1, 2], {"a": 1}, 9)
        assert store.usage("ann@example.com") == (1, 9)


class TestBudgets:
    def test_too_many_jobs_is_refused_before_upload(self, manager, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCUT_MAX_JOBS_PER_USER", "2")
        plant(manager, "a1", "ann@example.com", status="done")
        plant(manager, "a2", "ann@example.com", status="done")
        with pytest.raises(QuotaExceeded, match="delete one"):
            manager.check_room("ann@example.com")
        manager.check_room("bob@example.com")

    def test_too_many_bytes_is_refused(self, manager, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCUT_MAX_BYTES_PER_USER", "100")
        plant(manager, "a1", "ann@example.com", status="done", bytes=100)
        with pytest.raises(QuotaExceeded, match="100 B"):
            manager.check_room("ann@example.com")

    def test_too_many_in_progress_is_refused(self, manager, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCUT_MAX_ACTIVE_PER_USER", "1")
        plant(manager, "a1", "ann@example.com", status="analyzing")
        with pytest.raises(QuotaExceeded, match="in progress"):
            manager.check_room("ann@example.com")

    def test_an_oversized_upload_leaves_nothing_behind(self, manager, monkeypatch) -> None:
        import io
        monkeypatch.setenv("AUTOCUT_MAX_UPLOAD_BYTES", "10")
        with pytest.raises(BadUpload, match="upload limit"):
            asyncio.run(manager.submit("big.mp4", io.BytesIO(b"x" * 64), owner="ann@example.com"))
        assert list(manager.root.iterdir()) == [manager.root / "autocut.db"]

    def test_a_non_video_is_refused_and_removed(self, manager) -> None:
        import io
        with pytest.raises(BadUpload, match="does not look like a video"):
            asyncio.run(manager.submit("notes.mp4", io.BytesIO(b"hello"), owner="ann@example.com"))
        assert list(manager.root.iterdir()) == [manager.root / "autocut.db"]

    def test_the_refusals_reach_the_browser_with_the_right_status(self, client, manager, multiuser, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCUT_MAX_JOBS_PER_USER", "0")
        response = client.post("/jobs", headers=as_("ann@example.com"), files={"file": ("a.mp4", b"x")})
        assert response.status_code == 429
        assert "delete one" in response.json()["detail"]


class TestRetention:
    def test_intermediates_go_after_a_render_and_the_upload_stays(self, tmp_path) -> None:
        work = tmp_path / "j"; work.mkdir()
        for name in ("input.mov", "audio.wav", "cut.mkv", "cut-part001.mkv", "compose.filter", "captions.txt",
                     "preview.mp4", "transcript.json", "timeline.json", "final-1.mp4"):
            (work / name).write_bytes(b"x" * 10)
        (work / "captions").mkdir(); (work / "captions" / "0.png").write_bytes(b"x" * 10)
        (work / "holes").mkdir(); (work / "holes" / "00.wav").write_bytes(b"x" * 10)
        freed = retention.drop_intermediates(work)
        assert freed == 70
        assert sorted(p.name for p in work.iterdir()) == ["final-1.mp4", "input.mov", "preview.mp4", "timeline.json", "transcript.json"]

    def test_expired_jobs_are_swept(self, manager, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCUT_RETENTION_DAYS", "7")
        old = plant(manager, "old", "ann@example.com", status="done",
                    created=datetime.now(timezone.utc) - timedelta(days=8))
        fresh = plant(manager, "new", "ann@example.com", status="done")
        busy = plant(manager, "busy", "ann@example.com", status="rendering",
                     created=datetime.now(timezone.utc) - timedelta(days=30))
        assert asyncio.run(manager.sweep()) == 1
        assert not old.work_dir.exists()
        assert fresh.work_dir.exists() and busy.work_dir.exists()
        assert set(manager.jobs) == {"new", "busy"}

    def test_the_expiry_is_told_to_the_browser(self, manager, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCUT_RETENTION_DAYS", "3")
        job = plant(manager, "aaa", "ann@example.com")
        expires = datetime.fromisoformat(job.snapshot()["expires"])
        assert expires - job.created == timedelta(days=3)


class TestQueuePosition:
    def test_waiting_jobs_know_how_many_are_ahead(self, manager) -> None:
        async def go():
            a = plant(manager, "a", "ann@example.com", status="queued")
            b = plant(manager, "b", "bob@example.com", status="queued")
            await manager._enqueue(a, "propose")
            await manager._enqueue(b, "propose")
            assert (a.position, b.position) == (0, 1)
            assert b.snapshot()["position"] == 1
            await manager.delete(a)
            assert b.position == 0
        asyncio.run(go())
