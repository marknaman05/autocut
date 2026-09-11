"""Publishing to Instagram, with the network faked out.

What is worth checking is the *sequence* -- container, upload, wait, publish
-- and that each way Instagram says no comes back as a message a person can
act on, rather than a traceback.  The API itself is not under test.
"""

from __future__ import annotations

import asyncio
import io
import json
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autocut.publish import instagram
from server.jobs import Job, JobManager


@dataclass
class FakeMeta:
    """Answers requests the way Meta's hosts do, and remembers what was asked."""

    statuses: list[str] = field(default_factory=lambda: ["IN_PROGRESS", "FINISHED"])
    calls: list[tuple[str, str, dict]] = field(default_factory=list)
    fail: dict[str, tuple[int, str]] = field(default_factory=dict)

    def __call__(self, request, timeout=None):
        url = request.full_url
        method = request.get_method()
        headers = dict(request.header_items())
        self.calls.append((method, url, headers))

        for needle, (code, message) in self.fail.items():
            if needle in url:
                body = json.dumps({"error": {"message": message}}).encode()
                raise urllib.error.HTTPError(url, code, "nope", {}, io.BytesIO(body))

        if "/me?" in url:
            reply = {"id": "app-scoped", "user_id": "17841400000", "username": "someone"}
        elif url.endswith("/media") and method == "POST":
            self.container_params = dict(
                pair.split("=", 1) for pair in request.data.decode().split("&")
            )
            reply = {"id": "container-1"}
        elif "rupload.facebook.com" in url:
            self.uploaded = request.data
            reply = {"success": True}
        elif "container-1?" in url:
            reply = {"status_code": self.statuses.pop(0)}
        elif url.endswith("/media_publish"):
            reply = {"id": "media-9"}
        elif "media-9?" in url:
            reply = {"permalink": "https://www.instagram.com/reel/abc/"}
        elif "refresh_access_token" in url:
            reply = {"access_token": "fresh", "expires_in": 5184000}
        else:  # pragma: no cover
            raise AssertionError(f"unexpected request {method} {url}")
        return io.BytesIO(json.dumps(reply).encode())



class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


@pytest.fixture
def meta(monkeypatch):
    fake = FakeMeta()

    def urlopen(request, timeout=None):
        return _Response(fake(request, timeout).read())

    monkeypatch.setattr(instagram.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv(instagram.TOKEN_ENV, "tok")
    monkeypatch.delenv(instagram.USER_ID_ENV, raising=False)
    # The server picks its backend from the environment; these tests are
    # about the direct Meta client, whatever keys the developer has in .env.
    monkeypatch.setenv("AUTOCUT_PUBLISH_BACKEND", "instagram")
    return fake


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "finished.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"x" * 100)
    return path


class TestPublishReel:
    def test_the_whole_sequence_runs_in_order(self, meta, video) -> None:
        said = []
        posted = instagram.publish_reel(video, "hello", report=said.append, sleep=lambda _: None)

        assert posted.media_id == "media-9"
        assert posted.permalink == "https://www.instagram.com/reel/abc/"
        steps = [url.split(instagram.API_VERSION + "/", 1)[1].split("?")[0] for _, url, _ in meta.calls]
        assert steps == [
            "me",
            "17841400000/media",
            "container-1",           # the upload, on the other host
            "container-1",           # IN_PROGRESS
            "container-1",           # FINISHED
            "17841400000/media_publish",
            "media-9",
        ]
        assert said[0].startswith("checking")
        assert "publishing" in said

    def test_the_container_is_a_resumable_reel_with_the_caption(self, meta, video) -> None:
        instagram.publish_reel(video, "a caption", sleep=lambda _: None)
        assert meta.container_params["media_type"] == "REELS"
        assert meta.container_params["upload_type"] == "resumable"
        assert meta.container_params["caption"] == "a+caption"

    def test_the_bytes_go_to_the_upload_host_with_the_right_headers(self, meta, video) -> None:
        instagram.publish_reel(video, sleep=lambda _: None)
        method, url, headers = next(c for c in meta.calls if "rupload" in c[1])
        assert method == "POST"
        assert url.endswith(f"/ig-api-upload/{instagram.API_VERSION}/container-1")
        assert headers["Authorization"] == "OAuth tok"
        assert headers["Offset"] == "0"
        assert headers["File_size"] == str(video.stat().st_size)
        assert meta.uploaded == video.read_bytes()

    def test_a_rejected_video_is_reported_not_published(self, meta, video) -> None:
        meta.statuses = ["ERROR"]
        with pytest.raises(instagram.InstagramError, match="rejected"):
            instagram.publish_reel(video, sleep=lambda _: None)
        assert not any(url.endswith("/media_publish") for _, url, _ in meta.calls)

    def test_metas_error_message_is_surfaced(self, meta, video) -> None:
        meta.fail["/media"] = (400, "The video file you selected is in a format that we don't support.")
        with pytest.raises(instagram.InstagramError, match="400: The video file"):
            instagram.publish_reel(video, sleep=lambda _: None)

    def test_a_bad_token_fails_before_anything_is_uploaded(self, meta, video) -> None:
        meta.fail["/me"] = (190, "Invalid OAuth access token.")
        with pytest.raises(instagram.InstagramError, match="Invalid OAuth"):
            instagram.publish_reel(video, sleep=lambda _: None)
        assert len(meta.calls) == 1

    def test_no_token_is_a_plain_message(self, monkeypatch, video) -> None:
        monkeypatch.delenv(instagram.TOKEN_ENV, raising=False)
        assert not instagram.configured()
        with pytest.raises(instagram.InstagramError, match=instagram.TOKEN_ENV):
            instagram.publish_reel(video)

    def test_an_overlong_caption_is_refused_locally(self, meta, video) -> None:
        with pytest.raises(instagram.InstagramError, match="character limit"):
            instagram.publish_reel(video, "x" * (instagram.CAPTION_LIMIT + 1))
        assert meta.calls == []

    def test_a_missing_permalink_does_not_fail_the_publish(self, meta, video) -> None:
        meta.fail["media-9?"] = (500, "flaky")
        posted = instagram.publish_reel(video, sleep=lambda _: None)
        assert posted.media_id == "media-9"
        assert posted.permalink is None

    def test_an_explicit_account_id_wins(self, meta, video, monkeypatch) -> None:
        monkeypatch.setenv(instagram.USER_ID_ENV, "123")
        instagram.publish_reel(video, sleep=lambda _: None)
        assert any("/123/media" in url for _, url, _ in meta.calls)


class TestAccount:
    def test_reports_who_the_token_belongs_to(self, meta) -> None:
        who = instagram.account()
        assert who == instagram.Account(user_id="17841400000", username="someone")

    def test_refresh_replaces_the_token_in_this_process(self, meta, monkeypatch) -> None:
        assert instagram.refresh_token() == 5184000
        assert instagram._token() == "fresh"


class TestServer:
    """The job manager's side: one publish at a time, only of a finished video."""

    def _finished_job(self, tmp_path, video) -> tuple[JobManager, Job]:
        from autocut.pipeline import Result
        from autocut.models import Timeline

        manager = JobManager(tmp_path / "work")
        job = Job(id="j", filename="a.mp4", source=video, work_dir=tmp_path)
        job.status = "done"
        job.result = Result(
            output=video, timeline=Timeline(source=video, duration=1.0, fps=30, width=1080, height=1920), tracked=False, elapsed=1.0
        )
        return manager, job

    def test_publishes_and_records_the_link(self, meta, video, tmp_path) -> None:
        manager, job = self._finished_job(tmp_path, video)

        async def go():
            await manager.publish(job, "hi")
            assert job.snapshot()["publish"]["status"] == "publishing"
            await asyncio.gather(*manager._publishing)

        asyncio.run(go())
        assert job.publish.status == "published"
        assert job.publish.permalink == "https://www.instagram.com/reel/abc/"
        assert job.snapshot()["publish"]["caption"] == "hi"

    def test_a_failure_is_recorded_not_raised(self, meta, video, tmp_path) -> None:
        meta.fail["/me"] = (190, "Invalid OAuth access token.")
        manager, job = self._finished_job(tmp_path, video)

        async def go():
            await manager.publish(job, "")
            await asyncio.gather(*manager._publishing)

        asyncio.run(go())
        assert job.publish.status == "error"
        assert "Invalid OAuth" in job.publish.error

    def test_an_unfinished_job_cannot_be_published(self, meta, video, tmp_path) -> None:
        manager, job = self._finished_job(tmp_path, video)
        job.status = "rendering"
        with pytest.raises(ValueError, match="not finished"):
            asyncio.run(manager.publish(job, ""))

    def test_a_second_click_while_publishing_is_refused(self, meta, video, tmp_path) -> None:
        manager, job = self._finished_job(tmp_path, video)

        async def go():
            await manager.publish(job, "")
            with pytest.raises(ValueError, match="already"):
                await manager.publish(job, "")
            await asyncio.gather(*manager._publishing)

        asyncio.run(go())
