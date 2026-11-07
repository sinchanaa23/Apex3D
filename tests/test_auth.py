"""Accounts: sign up, log in, log out, and which routes need a login. Uses a temporary database."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import auth, database
from backend.main import app

GOOD = {"name": "Asha Rao", "email": "asha@example.com", "password": "correct horse 42"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setenv("APEX3D_REQUIRE_LOGIN", "1")
    monkeypatch.setattr(auth, "_ready_for", None)
    auth.reset_limits()
    with TestClient(app) as test_client:
        yield test_client
    auth.reset_limits()


def signup(client, **overrides):
    return client.post("/api/auth/signup", json={**GOOD, **overrides})


def test_signup_logs_the_user_in(client):
    response = signup(client)
    assert response.status_code == 201
    assert response.json()["user"]["email"] == "asha@example.com"
    assert "password" not in response.text
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    me = client.get("/api/auth/me").json()
    assert me["user"]["name"] == "Asha Rao" and me["login_required"] is True


def test_password_is_hashed_and_token_is_not_stored(client):
    response = signup(client)
    token = client.cookies.get(auth.COOKIE_NAME)
    conn = sqlite3.connect(str(database.DB_PATH))
    stored_password = conn.execute("SELECT password_hash FROM users").fetchone()[0]
    stored_tokens = [row[0] for row in conn.execute("SELECT token_hash FROM sessions")]
    conn.close()
    assert response.status_code == 201
    assert GOOD["password"] not in stored_password and stored_password.startswith("scrypt$")
    assert token and token not in stored_tokens and len(stored_tokens) == 1


@pytest.mark.parametrize(
    "overrides",
    [{"email": "not-an-email"}, {"password": "short"}, {"password": "password"}, {"password": "aaaaaaaaaa"}, {"name": "   "}, {"password": "x" * 200}],
)
def test_signup_rejects_bad_input(client, overrides):
    response = signup(client, **overrides)
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_duplicate_email_is_refused_whatever_the_case(client):
    assert signup(client).status_code == 201
    other = TestClient(app)
    assert other.post("/api/auth/signup", json={**GOOD, "email": " ASHA@Example.com "}).status_code == 409


def test_login_and_logout(client):
    signup(client)
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").json()["user"] is None

    wrong = client.post("/api/auth/login", json={"email": GOOD["email"], "password": "wrong password"})
    assert wrong.status_code == 401 and wrong.json()["detail"] == "Wrong email or password."
    unknown = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "whatever 123"})
    assert unknown.status_code == 401 and unknown.json()["detail"] == wrong.json()["detail"]

    ok = client.post("/api/auth/login", json={"email": "Asha@example.com", "password": GOOD["password"]})
    assert ok.status_code == 200 and client.get("/api/auth/me").json()["user"]["email"] == GOOD["email"]
    client.post("/api/auth/logout")
    assert client.get("/api/auth/me").json()["user"] is None


def test_repeated_wrong_passwords_are_limited(client):
    signup(client)
    client.post("/api/auth/logout")
    for _ in range(auth.MAX_LOGIN_FAILURES):
        assert client.post("/api/auth/login", json={"email": GOOD["email"], "password": "nope nope 1"}).status_code == 401
    blocked = client.post("/api/auth/login", json={"email": GOOD["email"], "password": GOOD["password"]})
    assert blocked.status_code == 429 and "Retry-After" in blocked.headers


def test_protected_routes_need_a_login(client):
    for method, path in [("post", "/api/recognize"), ("post", "/api/classical"), ("post", "/api/reconstruct"), ("get", "/api/references"), ("post", "/api/maintenance/cleanup")]:
        assert getattr(client, method)(path).status_code == 401, path
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/metrics").status_code != 401
    signup(client)
    # Logged in, the same routes get past the login check (and then complain about the missing upload).
    assert client.post("/api/recognize").status_code == 422
    client.post("/api/auth/logout")
    assert client.post("/api/recognize").status_code == 401


def test_a_session_cookie_for_a_deleted_session_is_refused(client):
    signup(client)
    conn = sqlite3.connect(str(database.DB_PATH))
    conn.execute("DELETE FROM sessions")
    conn.commit()
    conn.close()
    assert client.post("/api/recognize").status_code == 401


def test_login_can_be_switched_off_for_development(client, monkeypatch):
    monkeypatch.setenv("APEX3D_REQUIRE_LOGIN", "0")
    assert client.post("/api/recognize").status_code == 422
    assert client.get("/api/auth/me").json() == {"user": None, "login_required": False}


def test_another_website_cannot_read_responses_with_cookies(client):
    signup(client)
    response = client.get("/api/auth/me", headers={"Origin": "https://evil.example"})
    assert response.headers.get("access-control-allow-origin") != "https://evil.example"
    assert response.headers.get("access-control-allow-origin") != "*"
