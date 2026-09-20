"""Large uploads in parts: the browser PUTs 50 MB slices to the app, which
lands each at its offset in one file and queues the whole thing.

What is under test: the app refuses early what it can, binds an upload to
the person who started it, checks every part against the size declared at
the start, hands the job manager the same bytes that were put -- in any
order, with retries -- and leaves nothing behind whether or not the job was
accepted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import app as server_app
from server import auth, uploads
from server.jobs import JobManager

HEADER = "Cf-Access-Authenticated-User-Email"


@pytest.fixture
def small_parts(monkeypatch) -> None:
    monkeypatch.setattr(uploads, "PART_SIZE", 8)


@pytest.fixture
def manager(tmp_path, monkeypatch, small_parts) -> JobManager:
    m = JobManager(tmp_path / "work")
    monkeypatch.setattr(server_app, "manager", m)
    monkeypatch.setattr(server_app, "parts", uploads.Uploads(tmp_path / "work"))
    return m


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv(auth.HEADER_ENV, HEADER)
    return TestClient(server_app.app)


def as_(email: str) -> dict:
    return {HEADER: email}


def start(client, email="ann@example.com", size=20, filename="take.mov"):
    return client.post("/uploads", headers=as_(email), json={"filename": filename, "size": size})


def put_all(client, upload_id: str, data: bytes, order=None, email="ann@example.com") -> None:
    count = (len(data) + 7) // 8
    for n in order or range(1, count + 1):
        response = client.put(f"/uploads/{upload_id}/{n}", headers=as_(email), content=data[(n - 1) * 8:n * 8])
        assert response.status_code == 200, response.text


def catch_submit(manager, monkeypatch) -> dict:
    received = {}

    async def fake_submit(filename, stream, preset="default", owner=None):
        received.update(filename=filename, data=stream.read(1024), preset=preset, owner=owner)
        from tests.test_review import make_job
        return make_job(manager.root)

    monkeypatch.setattr(manager, "submit", fake_submit)
    return received


class TestStart:
    def test_the_page_is_told_where_the_line_is(self, client, manager) -> None:
        assert client.get("/me", headers=as_("ann@example.com")).json()["part_size"] == 8

    def test_the_part_count_comes_from_the_size(self, client, manager) -> None:
        response = start(client, size=20)
        assert response.status_code == 200
        body = response.json()
        assert (body["part_size"], body["parts"]) == (8, 3)  # 8 + 8 + 4

    def test_an_oversized_file_is_refused_before_a_byte_moves(self, client, manager, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCUT_MAX_UPLOAD_BYTES", "10")
        response = start(client, size=11)
        assert response.status_code == 413
        assert "upload limit" in response.json()["detail"]
        assert server_app.parts.pending == {}

    def test_a_full_quota_is_refused_before_a_byte_moves(self, client, manager, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCUT_MAX_JOBS_PER_USER", "0")
        assert start(client).status_code == 429

    def test_an_unknown_preset_is_refused(self, client, manager) -> None:
        response = client.post("/uploads", headers=as_("ann@example.com"),
                               json={"filename": "a.mov", "size": 5, "preset": "nope"})
        assert response.status_code == 400


class TestParts:
    def test_the_assembled_bytes_reach_the_job_manager_and_nothing_is_left(self, client, manager, monkeypatch) -> None:
        received = catch_submit(manager, monkeypatch)
        data = bytes(range(20))
        upload_id = start(client, size=20).json()["id"]
        put_all(client, upload_id, data)

        response = client.post(f"/uploads/{upload_id}/finish", headers=as_("ann@example.com"))
        assert response.status_code == 200, response.text
        assert received == {"filename": "take.mov", "data": data, "preset": "default", "owner": "ann@example.com"}
        assert server_app.parts.pending == {}
        assert list(server_app.parts.dir.iterdir()) == []

    def test_parts_may_arrive_in_any_order_and_be_retried(self, client, manager, monkeypatch) -> None:
        received = catch_submit(manager, monkeypatch)
        data = bytes(range(20))
        upload_id = start(client, size=20).json()["id"]
        put_all(client, upload_id, data, order=[3, 1, 2, 1])
        assert client.post(f"/uploads/{upload_id}/finish", headers=as_("ann@example.com")).status_code == 200
        assert received["data"] == data

    def test_a_part_of_the_wrong_size_is_refused(self, client, manager) -> None:
        upload_id = start(client, size=20).json()["id"]
        short = client.put(f"/uploads/{upload_id}/1", headers=as_("ann@example.com"), content=b"x" * 7)
        assert short.status_code == 400 and "7 B" in short.text
        long = client.put(f"/uploads/{upload_id}/3", headers=as_("ann@example.com"), content=b"x" * 5)
        assert long.status_code == 400 and "larger" in long.text
        assert client.put(f"/uploads/{upload_id}/4", headers=as_("ann@example.com"), content=b"").status_code == 400

    def test_someone_elses_upload_does_not_exist(self, client, manager) -> None:
        upload_id = start(client, email="ann@example.com").json()["id"]
        bob = as_("bob@example.com")
        assert client.put(f"/uploads/{upload_id}/1", headers=bob, content=b"x" * 8).status_code == 404
        assert client.post(f"/uploads/{upload_id}/finish", headers=bob).status_code == 404
        assert client.delete(f"/uploads/{upload_id}", headers=bob).status_code == 404
        assert upload_id in server_app.parts.pending

    def test_a_refused_job_still_removes_the_file(self, client, manager) -> None:
        # Twenty bytes are not a video; the manager's probe refuses them.
        upload_id = start(client, size=20).json()["id"]
        put_all(client, upload_id, bytes(20))
        response = client.post(f"/uploads/{upload_id}/finish", headers=as_("ann@example.com"))
        assert response.status_code == 400
        assert "does not look like a video" in response.json()["detail"]
        assert list(server_app.parts.dir.iterdir()) == []
        assert sorted(p.name for p in manager.root.iterdir()) == ["autocut.db", "uploads"]

    def test_abandoning_removes_the_file(self, client, manager) -> None:
        upload_id = start(client).json()["id"]
        assert client.delete(f"/uploads/{upload_id}", headers=as_("ann@example.com")).status_code == 200
        assert server_app.parts.pending == {}
        assert list(server_app.parts.dir.iterdir()) == []

    def test_a_restart_sweeps_what_was_left(self, tmp_path, small_parts) -> None:
        first = uploads.Uploads(tmp_path)
        first.start("ann@example.com", "a.mov", 20, "default")
        assert len(list(first.dir.iterdir())) == 1
        second = uploads.Uploads(tmp_path)
        second.sweep()
        assert list(second.dir.iterdir()) == []
