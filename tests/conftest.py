"""Shared test setup.

The login requirement is off for the existing endpoint tests, which exercise the models and not the accounts.
tests/test_auth.py turns it on and uses its own temporary database.
"""

import pytest


@pytest.fixture(autouse=True)
def _login_not_required(monkeypatch):
    monkeypatch.setenv("APEX3D_REQUIRE_LOGIN", "0")
