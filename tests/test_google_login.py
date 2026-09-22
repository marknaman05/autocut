"""Sign in with Google, done by the app.

Google itself is mocked at the two HTTP calls (token exchange, tokeninfo);
what is tested is ours: the state check, the verified-email rule, the signed
session cookie, what an anonymous visitor sees, and that the other two
identity modes are untouched.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import app as server_app
from server import auth, login
from server.jobs import JobManager

CLIENT_ID = "123-abc.apps.googleusercontent.com"


@pytest.fixture
def google(monkeypatch, tmp_path):
    monkeypatch.delenv(auth.HEADER_ENV, raising=False)
    monkeypatch.setenv(auth.GOOGLE_CLIENT_ID_ENV, CLIENT_ID)
    monkeypatch.setenv(auth.GOOGLE_CLIENT_SECRET_ENV, "shh")
    monkeypatch.setenv(auth.SESSION_SECRET_ENV, "x" * 40)
    monkeypatch.setenv("AUTOCUT_PUBLIC_URL", "https://autocut.example.com")
    monkeypatch.setattr(server_app, "manager", JobManager(tmp_path / "work"))


@pytest.fixture
def client() -> TestClient:
    # https, because the cookies are Secure when the public URL is https.
    return TestClient(server_app.app, base_url="https://testserver", follow_redirects=False)


class FakeGoogle:
    """Stands in for httpx.AsyncClient: answers the token and tokeninfo calls."""

    def __init__(self, claims):
        self.claims = claims
        self.posted = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, data):
        self.posted = data
        return FakeResponse({"id_token": "tok"})

    async def get(self, url, params):
        return FakeResponse(self.claims)


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def with_google(monkeypatch, claims):
    fake = FakeGoogle(claims)
    monkeypatch.setattr(login.httpx, "AsyncClient", lambda timeout=0: fake)
    return fake


class TestSessions:
    def test_round_trip(self, google):
        token = auth.make_session("Ann@Example.com")
        assert auth.read_session(token) == "Ann@Example.com"

    def test_tampered_or_expired_is_nothing(self, google):
        token = auth.make_session("ann@example.com", days=1, now=1_000_000)
        assert auth.read_session(token, now=1_000_000 + 2 * 86400) is None
        assert auth.read_session(token[:-4] + "AAAA") is None
        assert auth.read_session("garbage") is None and auth.read_session(None) is None

    def test_short_secret_is_refused(self, google, monkeypatch):
        monkeypatch.setenv(auth.SESSION_SECRET_ENV, "short")
        with pytest.raises(RuntimeError):
            auth.make_session("a@b.c")


class TestFlow:
    def test_anonymous_visitor_is_sent_to_login(self, google, client):
        r = client.get("/")
        assert r.status_code == 302 and r.headers["location"] == "/login"
        assert client.get("/me").status_code == 401
        page = client.get("/login")
        assert page.status_code == 200 and "Continue with Google" in page.text

    def test_start_redirects_to_google_with_state(self, google, client):
        r = client.get("/auth/google?next=/pricing")
        assert r.status_code == 302
        assert r.headers["location"].startswith(login.AUTH_URL)
        assert "client_id=" + CLIENT_ID.replace(".", "%2E") in r.headers["location"] or CLIENT_ID in r.headers["location"]
        assert "redirect_uri=https%3A%2F%2Fautocut.example.com%2Fauth%2Fcallback" in r.headers["location"]
        assert login.STATE_COOKIE in r.cookies

    def test_callback_signs_in_and_sets_cookie(self, google, client, monkeypatch):
        fake = with_google(monkeypatch, {"aud": CLIENT_ID, "email": "Ann@Example.com", "email_verified": "true"})
        client.get("/auth/google?next=/pricing")
        state = client.cookies[login.STATE_COOKIE].split(".")[0]
        r = client.get(f"/auth/callback?code=abc&state={state}")
        assert r.status_code == 302 and r.headers["location"] == "/pricing"
        assert fake.posted["code"] == "abc" and fake.posted["redirect_uri"] == "https://autocut.example.com/auth/callback"
        assert auth.SESSION_COOKIE in client.cookies
        me = client.get("/me")
        assert me.status_code == 200 and me.json()["email"] == "ann@example.com"
        assert me.json()["plan"] == "free" and me.json()["can_sign_out"] is True
        assert client.get("/").status_code == 200

    def test_wrong_state_is_rejected(self, google, client, monkeypatch):
        with_google(monkeypatch, {"aud": CLIENT_ID, "email": "a@b.c", "email_verified": "true"})
        client.get("/auth/google")
        r = client.get("/auth/callback?code=abc&state=not-the-one")
        assert r.headers["location"] == "/login?error=state"
        assert auth.SESSION_COOKIE not in client.cookies

    def test_unverified_email_is_rejected(self, google, client, monkeypatch):
        with_google(monkeypatch, {"aud": CLIENT_ID, "email": "a@b.c", "email_verified": "false"})
        client.get("/auth/google")
        state = client.cookies[login.STATE_COOKIE].split(".")[0]
        r = client.get(f"/auth/callback?code=abc&state={state}")
        assert r.headers["location"] == "/login?error=unverified"

    def test_token_for_another_client_is_rejected(self, google, client, monkeypatch):
        with_google(monkeypatch, {"aud": "someone-else", "email": "a@b.c", "email_verified": "true"})
        client.get("/auth/google")
        state = client.cookies[login.STATE_COOKIE].split(".")[0]
        r = client.get(f"/auth/callback?code=abc&state={state}")
        assert r.headers["location"] == "/login?error=failed"

    def test_logout_clears_the_session(self, google, client, monkeypatch):
        with_google(monkeypatch, {"aud": CLIENT_ID, "email": "a@b.c", "email_verified": "true"})
        client.get("/auth/google")
        state = client.cookies[login.STATE_COOKIE].split(".")[0]
        client.get(f"/auth/callback?code=abc&state={state}")
        assert client.get("/me").status_code == 200
        r = client.post("/logout")
        assert r.status_code == 303
        assert client.get("/me").status_code == 401

    def test_pro_list_still_applies(self, google, client, monkeypatch):
        monkeypatch.setenv(auth.PRO_ENV, "paid@example.com")
        with_google(monkeypatch, {"aud": CLIENT_ID, "email": "Paid@Example.com", "email_verified": "true"})
        client.get("/auth/google")
        state = client.cookies[login.STATE_COOKIE].split(".")[0]
        client.get(f"/auth/callback?code=abc&state={state}")
        assert client.get("/me").json()["plan"] == "pro"

    def test_open_redirects_are_not_followed(self, google, client, monkeypatch):
        with_google(monkeypatch, {"aud": CLIENT_ID, "email": "a@b.c", "email_verified": "true"})
        client.get("/auth/google?next=https://evil.example/steal")
        state = client.cookies[login.STATE_COOKIE].split(".")[0]
        r = client.get(f"/auth/callback?code=abc&state={state}")
        assert r.headers["location"] == "/"


class TestOtherModesUnchanged:
    def test_header_mode_wins_when_both_are_set(self, google, client, monkeypatch):
        monkeypatch.setenv(auth.HEADER_ENV, "X-User")
        assert auth.mode() == "header"
        assert client.get("/me", headers={"X-User": "h@example.com"}).json()["email"] == "h@example.com"
        assert client.get("/login").status_code == 404

    def test_local_mode_when_nothing_is_set(self, client, monkeypatch, tmp_path):
        for name in (auth.HEADER_ENV, auth.GOOGLE_CLIENT_ID_ENV, auth.GOOGLE_CLIENT_SECRET_ENV):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(server_app, "manager", JobManager(tmp_path / "work"))
        assert client.get("/").status_code == 200
        assert client.get("/me").json()["email"] == "local"
