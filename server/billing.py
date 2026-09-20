"""Pro, bought through Dodo Payments.

The app never sees a card.  "Upgrade" creates a hosted checkout session for
the signed-in address and sends the browser there; Dodo tells us what
happened through a webhook, and the subscription's latest status is what
makes someone Pro (see ``auth.plan_for``).  Test mode is the default -- set
``DODO_PAYMENTS_ENVIRONMENT=live_mode`` only when the product is real.

Configuration (all optional; without ``DODO_PAYMENTS_API_KEY`` and
``DODO_PRODUCT_ID`` the pricing page says payments are not wired up):

- ``DODO_PAYMENTS_API_KEY``      API key from the dashboard (Developer > API keys).
- ``DODO_PRODUCT_ID``            The Pro subscription product, ``pdt_...``.
- ``DODO_PAYMENTS_WEBHOOK_KEY``  Signing secret of the webhook endpoint
                                 (``whsec_...``); without it webhooks are refused.
- ``DODO_PAYMENTS_ENVIRONMENT``  ``test_mode`` (default) or ``live_mode``.
- ``AUTOCUT_PUBLIC_URL``         Where Dodo sends the customer back, e.g.
                                 ``https://autocut.example.com``.

The webhook route must be reachable by Dodo without a login, so a proxy that
signs users in (Cloudflare Access) needs a bypass for ``/billing/webhook``.
That is safe: every delivery is verified against the signing secret first.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

API_KEY_ENV = "DODO_PAYMENTS_API_KEY"
PRODUCT_ENV = "DODO_PRODUCT_ID"
WEBHOOK_KEY_ENV = "DODO_PAYMENTS_WEBHOOK_KEY"
ENVIRONMENT_ENV = "DODO_PAYMENTS_ENVIRONMENT"
PUBLIC_URL_ENV = "AUTOCUT_PUBLIC_URL"

#: Subscription states that keep the download button working.  ``past_due``
#: is a failed renewal inside its grace period: Dodo says the customer keeps
#: access until it ends, so we do too.
PRO_STATUSES = frozenset({"active", "past_due"})


def environment() -> str:
    return os.environ.get(ENVIRONMENT_ENV) or "test_mode"


def enabled() -> bool:
    return bool(os.environ.get(API_KEY_ENV) and os.environ.get(PRODUCT_ENV))


def public_url() -> str:
    return (os.environ.get(PUBLIC_URL_ENV) or "http://localhost:8000").rstrip("/")


def client():
    from dodopayments import DodoPayments

    return DodoPayments(
        bearer_token=os.environ[API_KEY_ENV],
        environment=environment(),
        webhook_key=os.environ.get(WEBHOOK_KEY_ENV) or None,
    )


def checkout_url(email: str) -> str:
    """A hosted checkout for the Pro subscription, in this person's name."""
    session = client().checkout_sessions.create(
        product_cart=[{"product_id": os.environ[PRODUCT_ENV], "quantity": 1}],
        customer={"email": email},
        # Comes back on the subscription and every webhook about it, so the
        # event can be tied to an account even if the email is edited at checkout.
        metadata={"owner": email},
        return_url=f"{public_url()}/pricing?checkout=done",
    )
    return session.checkout_url


def portal_url(customer_id: str) -> str:
    """Dodo's own page for changing the card or cancelling."""
    session = client().customers.customer_portal.create(
        customer_id, return_url=f"{public_url()}/pricing",
    )
    return session.link


def verify(body: bytes, headers: dict[str, str]) -> dict[str, Any]:
    """The event, if the signature checks out; raises otherwise."""
    if not os.environ.get(WEBHOOK_KEY_ENV):
        raise ValueError(f"{WEBHOOK_KEY_ENV} is not set; refusing unverified webhooks")
    event = client().webhooks.unwrap(
        body.decode("utf-8"),
        headers={
            "webhook-id": headers.get("webhook-id", ""),
            "webhook-signature": headers.get("webhook-signature", ""),
            "webhook-timestamp": headers.get("webhook-timestamp", ""),
        },
    )
    return event.model_dump(mode="json")


def subscription_from(event: dict[str, Any]) -> dict[str, str] | None:
    """What the subscriptions table needs from a ``subscription.*`` event,
    or ``None`` for events about anything else."""
    if not str(event.get("type", "")).startswith("subscription."):
        return None
    data = event.get("data") or {}
    customer = data.get("customer") or {}
    metadata = data.get("metadata") or {}
    owner = (metadata.get("owner") or customer.get("email") or "").strip().lower()
    if not owner or not data.get("subscription_id"):
        return None
    return {
        "owner": owner,
        "subscription_id": data["subscription_id"],
        "customer_id": customer.get("customer_id") or "",
        "status": data.get("status") or "",
        "updated": (event.get("timestamp") or datetime.now(timezone.utc).isoformat()),
    }
