"""The web app's caption-style endpoints, over HTTP."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from autocut.config import CAPTION_STYLE_LABELS, CAPTION_STYLES
from server import app as server_app


@pytest.fixture
def client() -> TestClient:
    # Not used as a context manager on purpose: that would run the lifespan
    # and start the render worker, which nothing here needs.
    return TestClient(server_app.app)


class TestCaptionStyles:
    def test_lists_every_style_in_order_with_its_label(self, client) -> None:
        response = client.get("/caption-styles")
        assert response.status_code == 200
        listed = response.json()
        assert [s["name"] for s in listed] == list(CAPTION_STYLES)
        assert all(s["label"] == CAPTION_STYLE_LABELS[s["name"]] for s in listed)

    @pytest.mark.parametrize("name", list(CAPTION_STYLES))
    def test_every_style_has_a_png_sample(self, client, name) -> None:
        response = client.get(f"/caption-styles/{name}.png")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG")

    def test_an_unknown_style_is_404(self, client) -> None:
        assert client.get("/caption-styles/sideways.png").status_code == 404

    def test_the_sample_is_the_same_bytes_each_time(self, client) -> None:
        a = client.get("/caption-styles/pill.png").content
        b = client.get("/caption-styles/pill.png").content
        assert a == b


class TestRenderStyle:
    def test_an_unknown_style_is_rejected_before_the_render(self, client, tmp_path) -> None:
        from tests.test_review import make_job

        job = make_job(tmp_path)
        server_app.manager.jobs[job.id] = job
        try:
            response = client.post(f"/jobs/{job.id}/render", json={"keep": [1], "style": "nope"})
        finally:
            del server_app.manager.jobs[job.id]
        assert response.status_code == 400
        assert "nope" in response.text
        assert job.keep is None
