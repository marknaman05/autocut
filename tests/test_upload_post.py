"""The Upload-Post relay, with the network faked out.

Checks the request it builds -- one multipart POST with the fields Upload-Post
wants -- and that both of its answers, the immediate result and the
``request_id`` to poll, end up as the same ``Published``.
"""

from __future__ import annotations

import io
import json
import re
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from autocut import publish
from autocut.publish import instagram, upload_post


@dataclass
class FakeUploadPost:
    profiles: list[dict] = field(
        default_factory=lambda: [
            {"username": "default", "social_accounts": {"instagram": {"handle": "someone"}, "tiktok": "x"}}
        ]
    )
    #: The reply to POST /upload.
    upload_reply: dict = field(
        default_factory=lambda: {
            "success": True,
            "results": {"instagram": {"success": True, "url": "https://www.instagram.com/reel/abc/", "post_id": "p1"}},
        }
    )
    statuses: list[dict] = field(default_factory=list)
    calls: list[tuple[str, str]] = field(default_factory=list)
    body: bytes = b""
    fail: dict[str, tuple[int, dict]] = field(default_factory=dict)

    def __call__(self, request, timeout=None):
        url, method = request.full_url, request.get_method()
        self.calls.append((method, url))
        assert request.get_header("Authorization") == "Apikey key"
        for needle, (code, payload) in self.fail.items():
            if needle in url:
                raise urllib.error.HTTPError(url, code, "nope", {}, io.BytesIO(json.dumps(payload).encode()))
        if url.endswith("/uploadposts/users"):
            reply = {"success": True, "profiles": self.profiles}
        elif url.endswith("/upload"):
            self.body = request.data
            self.content_type = request.get_header("Content-type")
            reply = self.upload_reply
        elif "/uploadposts/status" in url:
            reply = self.statuses.pop(0)
        else:  # pragma: no cover
            raise AssertionError(url)
        return _Response(json.dumps(reply).encode())

    def fields(self) -> dict[str, str]:
        """The plain form fields out of the multipart body."""
        text = self.body.decode("latin-1")
        return dict(re.findall(r'name="([^"]+)"\r\n\r\n([^\r]*)\r\n', text))


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
        return False


@pytest.fixture
def relay(monkeypatch):
    fake = FakeUploadPost()
    monkeypatch.setattr(upload_post.urllib.request, "urlopen", fake)
    monkeypatch.setenv(upload_post.API_KEY_ENV, "key")
    monkeypatch.delenv(upload_post.USER_ENV, raising=False)
    monkeypatch.delenv("AUTOCUT_PUBLISH_BACKEND", raising=False)
    return fake


@pytest.fixture
def video(tmp_path) -> Path:
    path = tmp_path / "finished.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"v" * 64)
    return path


class TestPublishReel:
    def test_one_multipart_post_with_the_right_fields(self, relay, video) -> None:
        posted = upload_post.publish_reel(video, "hi there", sleep=lambda _: None)

        assert posted.permalink == "https://www.instagram.com/reel/abc/"
        assert posted.media_id == "p1"
        assert [m for m, _ in relay.calls] == ["GET", "POST"]
        assert relay.content_type.startswith("multipart/form-data; boundary=")
        fields = relay.fields()
        assert fields["user"] == "default"
        assert fields["platform[]"] == "instagram"
        assert fields["title"] == "hi there"
        assert fields["media_type"] == "REELS"
        assert fields["async_upload"] == "false"
        assert b'filename="finished.mp4"' in relay.body
        assert video.read_bytes() in relay.body

    def test_a_slow_upload_is_polled_to_completion(self, relay, video) -> None:
        relay.upload_reply = {"success": True, "request_id": "r1", "total_platforms": 1}
        relay.statuses = [
            {"status": "in_progress", "results": []},
            {"status": "completed", "results": [{"platform": "instagram", "success": True, "message": "ok"}]},
        ]
        posted = upload_post.publish_reel(video, sleep=lambda _: None)
        assert posted.media_id == "r1"
        assert posted.permalink is None
        assert sum("/uploadposts/status" in u for _, u in relay.calls) == 2
        assert "request_id=r1" in relay.calls[-1][1]

    def test_a_platform_failure_is_reported(self, relay, video) -> None:
        relay.upload_reply = {
            "success": False,
            "results": {"instagram": {"success": False, "error": "Video is too long for Reels"}},
        }
        with pytest.raises(instagram.InstagramError, match="too long"):
            upload_post.publish_reel(video, sleep=lambda _: None)

    def test_a_polled_failure_is_reported(self, relay, video) -> None:
        relay.upload_reply = {"success": True, "request_id": "r1"}
        relay.statuses = [
            {"status": "completed", "results": [{"platform": "instagram", "success": False, "message": "nope"}]}
        ]
        with pytest.raises(instagram.InstagramError, match="nope"):
            upload_post.publish_reel(video, sleep=lambda _: None)

    def test_the_services_error_message_is_surfaced(self, relay, video) -> None:
        relay.fail["/upload"] = (403, {"error": "Monthly upload limit reached"})
        with pytest.raises(instagram.InstagramError, match="403: Monthly upload limit"):
            upload_post.publish_reel(video, sleep=lambda _: None)

    def test_a_bad_key_stops_before_the_upload(self, relay, video) -> None:
        relay.fail["/users"] = (401, {"success": False, "message": "Invalid API key"})
        with pytest.raises(instagram.InstagramError, match="Invalid API key"):
            upload_post.publish_reel(video, sleep=lambda _: None)
        assert len(relay.calls) == 1

    def test_no_key_is_a_plain_message(self, monkeypatch, video) -> None:
        monkeypatch.delenv(upload_post.API_KEY_ENV, raising=False)
        assert not upload_post.configured()
        with pytest.raises(instagram.InstagramError, match=upload_post.API_KEY_ENV):
            upload_post.publish_reel(video)


class TestAccount:
    def test_the_only_profile_with_instagram_is_chosen(self, relay) -> None:
        assert upload_post.account() == instagram.Account(user_id="default", username="someone")

    def test_a_profile_without_instagram_does_not_count(self, relay) -> None:
        relay.profiles.insert(0, {"username": "other", "social_accounts": {"tiktok": "t"}})
        assert upload_post.account().user_id == "default"

    def test_two_candidates_need_a_decision(self, relay) -> None:
        relay.profiles.append({"username": "second", "social_accounts": {"instagram": "b"}})
        with pytest.raises(instagram.InstagramError, match=upload_post.USER_ENV):
            upload_post.account()

    def test_an_explicit_profile_is_used(self, relay, monkeypatch) -> None:
        relay.profiles.append({"username": "second", "social_accounts": {"instagram": "b"}})
        monkeypatch.setenv(upload_post.USER_ENV, "second")
        assert upload_post.account() == instagram.Account(user_id="second", username="b")

    def test_an_unknown_profile_names_the_real_ones(self, relay, monkeypatch) -> None:
        monkeypatch.setenv(upload_post.USER_ENV, "typo")
        with pytest.raises(instagram.InstagramError, match="it has: default"):
            upload_post.account()

    def test_nothing_connected_says_where_to_go(self, relay) -> None:
        relay.profiles = [{"username": "default", "social_accounts": {}}]
        with pytest.raises(instagram.InstagramError, match="Manage users"):
            upload_post.account()


class TestBackendChoice:
    def test_upload_post_wins_when_its_key_is_set(self, relay, monkeypatch) -> None:
        monkeypatch.setenv(instagram.TOKEN_ENV, "tok")
        assert publish.backend() is upload_post

    def test_meta_when_only_its_token_is_set(self, monkeypatch) -> None:
        monkeypatch.delenv(upload_post.API_KEY_ENV, raising=False)
        monkeypatch.delenv("AUTOCUT_PUBLISH_BACKEND", raising=False)
        monkeypatch.setenv(instagram.TOKEN_ENV, "tok")
        assert publish.backend() is instagram
        assert publish.configured()

    def test_an_explicit_choice_overrides(self, relay, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCUT_PUBLISH_BACKEND", "instagram")
        assert publish.backend() is instagram

    def test_an_unknown_choice_is_an_error(self, monkeypatch) -> None:
        monkeypatch.setenv("AUTOCUT_PUBLISH_BACKEND", "carrier-pigeon")
        with pytest.raises(instagram.InstagramError, match="carrier-pigeon"):
            publish.backend()

    def test_nothing_set_is_not_configured(self, monkeypatch) -> None:
        monkeypatch.delenv(upload_post.API_KEY_ENV, raising=False)
        monkeypatch.delenv(instagram.TOKEN_ENV, raising=False)
        monkeypatch.delenv("AUTOCUT_PUBLISH_BACKEND", raising=False)
        assert not publish.configured()
