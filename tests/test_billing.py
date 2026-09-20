"""Pro through Dodo Payments.

The gateway is never called here: checkout and portal are patched, and the
webhook is signed the way Dodo signs it (Standard Webhooks) so the real
verifier runs.  What is tested is ours -- that a verified ``subscription.*``
event is what makes someone Pro, that a bad signature or a replay changes
nothing, and that the pricing page's data in ``/me`` says the right thing.
"""

from __future__ import annotations

import base64
import json
import time
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from standardwebhooks import Webhook

from server import app as server_app
from server import auth, billing
from server.jobs import JobManager
from tests.test_plans import as_, finished

SECRET = "whsec_" + base64.b64encode(b"a secret of thirty-two bytes...!").decode()


@pytest.fixture
def wired(monkeypatch, tmp_path):
    monkeypatch.setenv(auth.HEADER_ENV, "Cf-Access-Authenticated-User-Email")
    monkeypatch.setenv(billing.API_KEY_ENV, "test-key")
    monkeypatch.setenv(billing.PRODUCT_ENV, "pdt_pro")
    monkeypatch.setenv(billing.WEBHOOK_KEY_ENV, SECRET)
    monkeypatch.setenv(billing.PUBLIC_URL_ENV, "https://autocut.example.com")
    m = JobManager(tmp_path / "work")
    monkeypatch.setattr(server_app, "manager", m)
    return m


@pytest.fixture
def client() -> TestClient:
    return TestClient(server_app.app)


def signed(event: dict, event_id: str = "msg_1") -> tuple[bytes, dict]:
    body = json.dumps(event)
    ts = int(time.time())
    sig = Webhook(SECRET).sign(event_id, datetime.fromtimestamp(ts, tz=timezone.utc), body)
    return body.encode(), {
        "webhook-id": event_id,
        "webhook-timestamp": str(ts),
        "webhook-signature": sig,
        "content-type": "application/json",
    }


def subscription_event(kind: str, status: str, owner: str = "buyer@example.com") -> dict:
    return {
        "business_id": "biz_1",
        "type": kind,
        "timestamp": "2026-09-18T10:00:00Z",
        "data": {
            "payload_type": "Subscription",
            "subscription_id": "sub_1",
            "status": status,
            "customer": {"customer_id": "cus_1", "email": owner, "name": "Buyer"},
            "metadata": {"owner": owner},
        },
    }


class TestWebhook:
    def test_active_subscription_makes_pro(self, wired, client):
        assert client.get("/me", headers=as_("buyer@example.com")).json()["plan"] == "free"
        body, headers = signed(subscription_event("subscription.active", "active"))
        r = client.post("/billing/webhook", content=body, headers=headers)
        assert r.status_code == 200 and r.json() == {"received": True}

        me = client.get("/me", headers=as_("buyer@example.com")).json()
        assert me["plan"] == "pro"
        assert me["billing"]["subscription"]["status"] == "active"

        job = finished(wired, "buyer@example.com")
        assert client.get(f"/jobs/{job.id}/result", headers=as_("buyer@example.com")).status_code == 200

    def test_cancelled_subscription_drops_back_to_free(self, wired, client):
        body, headers = signed(subscription_event("subscription.active", "active"), "msg_1")
        client.post("/billing/webhook", content=body, headers=headers)
        body, headers = signed(subscription_event("subscription.cancelled", "cancelled"), "msg_2")
        client.post("/billing/webhook", content=body, headers=headers)
        assert client.get("/me", headers=as_("buyer@example.com")).json()["plan"] == "free"

    def test_past_due_keeps_access(self, wired, client):
        body, headers = signed(subscription_event("subscription.past_due", "past_due"))
        client.post("/billing/webhook", content=body, headers=headers)
        assert client.get("/me", headers=as_("buyer@example.com")).json()["plan"] == "pro"

    def test_bad_signature_is_refused(self, wired, client):
        body, headers = signed(subscription_event("subscription.active", "active"))
        headers["webhook-signature"] = "v1,bm90IHRoZSBzaWduYXR1cmU="
        assert client.post("/billing/webhook", content=body, headers=headers).status_code == 401
        assert client.get("/me", headers=as_("buyer@example.com")).json()["plan"] == "free"

    def test_no_secret_refuses_everything(self, wired, client, monkeypatch):
        monkeypatch.delenv(billing.WEBHOOK_KEY_ENV)
        body, headers = signed(subscription_event("subscription.active", "active"))
        assert client.post("/billing/webhook", content=body, headers=headers).status_code == 401

    def test_replay_is_ignored(self, wired, client):
        body, headers = signed(subscription_event("subscription.active", "active"), "msg_same")
        client.post("/billing/webhook", content=body, headers=headers)
        body, headers = signed(subscription_event("subscription.cancelled", "cancelled"), "msg_same")
        r = client.post("/billing/webhook", content=body, headers=headers)
        assert r.json()["duplicate"] is True
        assert client.get("/me", headers=as_("buyer@example.com")).json()["plan"] == "pro"

    def test_other_events_are_acknowledged(self, wired, client):
        event = {"business_id": "b", "type": "payment.succeeded", "timestamp": "2026-09-18T10:00:00Z",
                 "data": {"payload_type": "Payment", "payment_id": "pay_1"}}
        body, headers = signed(event)
        assert client.post("/billing/webhook", content=body, headers=headers).status_code == 200

    def test_webhook_needs_no_login(self, wired, client):
        # No identity header on purpose: Dodo does not sign in.
        body, headers = signed(subscription_event("subscription.active", "active"))
        assert client.post("/billing/webhook", content=body, headers=headers).status_code == 200


class TestCheckout:
    def test_checkout_returns_hosted_url(self, wired, client, monkeypatch):
        seen = {}

        def fake(email):
            seen["email"] = email
            return "https://test.checkout.dodopayments.com/session/cks_1"

        monkeypatch.setattr(billing, "checkout_url", fake)
        r = client.post("/billing/checkout", headers=as_("Buyer@Example.com"))
        assert r.status_code == 200
        assert r.json()["url"].startswith("https://test.checkout.dodopayments.com/")
        assert seen["email"] == "buyer@example.com"

    def test_already_subscribed_is_told_so(self, wired, client, monkeypatch):
        body, headers = signed(subscription_event("subscription.active", "active"))
        client.post("/billing/webhook", content=body, headers=headers)
        assert client.post("/billing/checkout", headers=as_("buyer@example.com")).status_code == 409

    def test_off_without_keys(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv(auth.HEADER_ENV, "Cf-Access-Authenticated-User-Email")
        monkeypatch.delenv(billing.API_KEY_ENV, raising=False)
        monkeypatch.setattr(server_app, "manager", JobManager(tmp_path / "work"))
        assert client.post("/billing/checkout", headers=as_("x@example.com")).status_code == 503
        assert client.get("/me", headers=as_("x@example.com")).json()["billing"]["enabled"] is False

    def test_portal_for_subscriber_only(self, wired, client, monkeypatch):
        monkeypatch.setattr(billing, "portal_url", lambda cid: f"https://portal/{cid}")
        assert client.post("/billing/portal", headers=as_("buyer@example.com")).status_code == 404
        body, headers = signed(subscription_event("subscription.active", "active"))
        client.post("/billing/webhook", content=body, headers=headers)
        r = client.post("/billing/portal", headers=as_("buyer@example.com"))
        assert r.status_code == 200 and r.json()["url"] == "https://portal/cus_1"


class TestEventParsing:
    def test_metadata_owner_beats_checkout_email(self):
        event = subscription_event("subscription.active", "active")
        event["data"]["customer"]["email"] = "typo@example.com"
        assert billing.subscription_from(event)["owner"] == "buyer@example.com"

    def test_customer_email_when_no_metadata(self):
        event = subscription_event("subscription.active", "active")
        event["data"]["metadata"] = {}
        assert billing.subscription_from(event)["owner"] == "buyer@example.com"

    def test_non_subscription_events_are_none(self):
        assert billing.subscription_from({"type": "payment.succeeded", "data": {}}) is None
