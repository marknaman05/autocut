"""The trial: the whole product for a few videos, then upgrade to upload more.

Credits are spent on upload and kept in a ledger, so deleting a video does
not hand one back; a refused upload costs nothing.  Only uploading is
metered -- a trial that has spent its videos still renders them again --
and both ways in (one request, or in parts) meet the same gate.
"""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi.testclient import TestClient

from server import app as server_app
from server import auth, retention
from server.jobs import BadUpload, JobManager
from tests.test_multiuser import plant

HEADER = "Cf-Access-Authenticated-User-Email"


@pytest.fixture
def multiuser(monkeypatch):
    monkeypatch.setenv(auth.HEADER_ENV, HEADER)
    monkeypatch.setenv(auth.OWNERS_ENV, "me@example.com")
    monkeypatch.setenv(auth.PRO_ENV, "paid@example.com")
    monkeypatch.setenv(auth.TRIAL_ENV, "Trial@Example.com")
    monkeypatch.setenv("AUTOCUT_TRIAL_CREDITS", "2")


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


def spend(manager, owner: str, n: int = 1) -> None:
    for _ in range(n):
        manager.store.count_upload(owner)


class TestPlan:
    def test_listed_addresses_are_on_trial(self, client, manager, multiuser) -> None:
        me = client.get("/me", headers=as_("trial@example.com")).json()
        assert me["plan"] == "trial"
        assert me["credits"] == {"used": 0, "total": 2, "left": 2}

    def test_pro_wins_over_trial(self, client, manager, multiuser, monkeypatch) -> None:
        monkeypatch.setenv(auth.TRIAL_ENV, "paid@example.com")
        assert client.get("/me", headers=as_("paid@example.com")).json()["plan"] == "pro"

    def test_others_have_no_credits_to_speak_of(self, client, manager, multiuser) -> None:
        assert client.get("/me", headers=as_("paid@example.com")).json()["credits"] is None
        assert client.get("/me", headers=as_("ann@example.com")).json()["credits"] is None

    def test_a_trial_may_download(self, client, manager, multiuser) -> None:
        from tests.test_plans import finished
        job = finished(manager, "trial@example.com")
        assert client.get(f"/jobs/{job.id}/result", headers=as_("trial@example.com")).status_code == 200


class TestLedger:
    def test_an_accepted_upload_is_counted(self, manager) -> None:
        assert manager.uploads_by("trial@example.com") == 0
        manager.store.count_upload("trial@example.com")
        manager.store.count_upload("trial@example.com")
        assert manager.uploads_by("trial@example.com") == 2
        assert manager.uploads_by("other@example.com") == 0

    def test_a_refused_upload_costs_nothing(self, manager) -> None:
        with pytest.raises(BadUpload):
            asyncio.run(manager.submit("notes.mp4", io.BytesIO(b"hello"), owner="trial@example.com"))
        assert manager.uploads_by("trial@example.com") == 0

    def test_deleting_a_video_does_not_refund(self, manager) -> None:
        job = plant(manager, "t1", "trial@example.com", status="done")
        spend(manager, "trial@example.com")
        asyncio.run(manager.delete(job))
        assert manager.uploads_by("trial@example.com") == 1


class TestGate:
    def test_the_last_credit_still_uploads(self, client, manager, multiuser) -> None:
        spend(manager, "trial@example.com", 1)
        assert client.get("/me", headers=as_("trial@example.com")).json()["credits"]["left"] == 1
        # Twenty bytes are not a video, but the refusal is the probe's (400),
        # not the paywall's (402): the gate let it through.
        response = client.post("/jobs", headers=as_("trial@example.com"), files={"file": ("a.mp4", b"x" * 20)})
        assert response.status_code == 400

    def test_spent_the_single_request_upload_is_402(self, client, manager, multiuser) -> None:
        spend(manager, "trial@example.com", 2)
        response = client.post("/jobs", headers=as_("trial@example.com"), files={"file": ("a.mp4", b"x" * 20)})
        assert response.status_code == 402
        assert "upgrade" in response.json()["detail"]

    def test_spent_the_part_upload_is_402_before_a_byte_moves(self, client, manager, multiuser) -> None:
        spend(manager, "trial@example.com", 2)
        response = client.post("/uploads", headers=as_("trial@example.com"), json={"filename": "a.mov", "size": 100})
        assert response.status_code == 402
        assert server_app.parts.pending == {}

    def test_spent_re_rendering_is_still_free(self, client, manager, multiuser) -> None:
        spend(manager, "trial@example.com", 2)
        job = plant(manager, "t1", "trial@example.com", status="review")
        response = client.post(f"/jobs/{job.id}/render", headers=as_("trial@example.com"), json={"keep": [0]})
        assert response.status_code == 200, response.text

    def test_pro_and_free_are_never_metered(self, client, manager, multiuser) -> None:
        for who in ("paid@example.com", "ann@example.com"):
            spend(manager, who, 5)
            response = client.post("/jobs", headers=as_(who), files={"file": ("a.mp4", b"x" * 20)})
            assert response.status_code == 400  # the probe's refusal, not the gate's
