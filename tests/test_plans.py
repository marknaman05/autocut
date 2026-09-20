"""Free and Pro: everyone edits and watches, only Pro downloads.

The plan is decided from the environment (``AUTOCUT_PRO``; owners and the
local user are Pro without being listed) and enforced in one place, the
download route; the page reads it from ``/me`` and draws the locked button
itself, so the route is the backstop, not the experience.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import app as server_app
from server import auth
from server.jobs import JobManager
from tests.test_review import make_job

HEADER = "Cf-Access-Authenticated-User-Email"


@pytest.fixture
def multiuser(monkeypatch):
    monkeypatch.setenv(auth.HEADER_ENV, HEADER)
    monkeypatch.setenv(auth.OWNERS_ENV, "me@example.com")
    monkeypatch.setenv(auth.PRO_ENV, "Paid@Example.com, other@example.com")


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


def watermark_exists(job) -> bool:
    from server import watermark
    return watermark.preview_path(job.output).exists()


def finished(manager, owner: str):
    job = make_job(manager.root)
    job.owner = owner
    job.status = "done"
    job.output = job.work_dir / "final-20260913-120000.mp4"
    job.output.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    manager.jobs[job.id] = job
    return job


class TestWhoIsPro:
    def test_the_local_user_is_pro(self, client, manager) -> None:
        assert client.get("/me").json()["plan"] == "pro"

    def test_listed_addresses_are_pro_regardless_of_case(self, client, manager, multiuser) -> None:
        assert client.get("/me", headers=as_("paid@example.com")).json()["plan"] == "pro"
        assert client.get("/me", headers=as_("OTHER@example.com")).json()["plan"] == "pro"

    def test_owners_are_pro_without_being_listed(self, client, manager, multiuser) -> None:
        assert client.get("/me", headers=as_("me@example.com")).json()["plan"] == "pro"

    def test_everyone_else_is_free(self, client, manager, multiuser) -> None:
        assert client.get("/me", headers=as_("ann@example.com")).json()["plan"] == "free"


class TestDownload:
    def test_free_watches_the_watermarked_copy_not_the_render(self, client, manager, multiuser, monkeypatch) -> None:
        from server import watermark

        job = finished(manager, "ann@example.com")
        # The real preview needs ffmpeg on a real video; here the point is
        # which file is served, so the builder is a stand-in.
        def fake_preview(output):
            path = watermark.preview_path(output)
            path.write_bytes(b"WATERMARKED")
            return path
        monkeypatch.setattr(watermark, "ensure_preview", fake_preview)

        watch = client.get(f"/jobs/{job.id}/watch", headers=as_("ann@example.com"))
        assert watch.status_code == 200
        assert watch.content == b"WATERMARKED"
        assert "attachment" not in watch.headers.get("content-disposition", "")
        result = client.get(f"/jobs/{job.id}/result", headers=as_("ann@example.com"))
        assert result.status_code == 402
        assert "/pricing" in result.json()["detail"]

    def test_pro_watches_the_render_itself(self, client, manager, multiuser) -> None:
        job = finished(manager, "paid@example.com")
        watch = client.get(f"/jobs/{job.id}/watch", headers=as_("paid@example.com"))
        assert watch.status_code == 200
        assert watch.content == job.output.read_bytes()
        assert not watermark_exists(job)

    def test_pro_downloads(self, client, manager, multiuser) -> None:
        job = finished(manager, "paid@example.com")
        result = client.get(f"/jobs/{job.id}/result", headers=as_("paid@example.com"))
        assert result.status_code == 200
        assert "attachment" in result.headers["content-disposition"]

    def test_the_local_tool_is_not_gated(self, client, manager) -> None:
        job = finished(manager, "local")
        assert client.get(f"/jobs/{job.id}/result").status_code == 200

    def test_pro_still_cannot_take_someone_elses(self, client, manager, multiuser) -> None:
        job = finished(manager, "ann@example.com")
        assert client.get(f"/jobs/{job.id}/result", headers=as_("paid@example.com")).status_code == 404
        assert client.get(f"/jobs/{job.id}/watch", headers=as_("paid@example.com")).status_code == 404


class TestPricingPage:
    def test_is_served_and_names_both_plans(self, client, manager) -> None:
        response = client.get("/pricing")
        assert response.status_code == 200
        assert "Free" in response.text and "Pro" in response.text
        assert 'id="upgrade"' in response.text
