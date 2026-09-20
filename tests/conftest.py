import pytest


@pytest.fixture(autouse=True)
def _no_payment_gateway(monkeypatch):
    """Tests describe an instance without payments unless they say otherwise;
    a developer's .env with real Dodo keys must not leak in."""
    for name in ("DODO_PAYMENTS_API_KEY", "DODO_PRODUCT_ID", "DODO_PAYMENTS_WEBHOOK_KEY", "DODO_PAYMENTS_ENVIRONMENT"):
        monkeypatch.delenv(name, raising=False)
