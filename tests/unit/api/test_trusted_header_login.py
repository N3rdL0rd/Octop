"""Reverse-proxy header login: secret gating, provisioning, role sync, break-glass."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from tests.support.app import ensure_control_plane_bound
from tests.support.auth import TEST_PASSWORD, bearer, bootstrap_admin

from octop.api.app import build_app
from octop.infra.auth.trusted_header import TrustedHeaderConfig
from octop.infra.server import OctopServer

SECRET = "s3cret-value"


def _headers(uid: str = "1001", name: str = "alice", organizer: bool = False, **extra: str):
    out = {"X-Proxy-Secret": SECRET, "X-User-Id": uid, "X-User-Name": name}
    if organizer:
        out["X-User-Is-Organizer"] = "true"
    out.update(extra)
    return out


@pytest.fixture
async def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OCTOP_TRUSTED_HEADER_SECRET", SECRET)
    srv = OctopServer(home=tmp_path)
    await srv.start()
    await ensure_control_plane_bound(srv)
    app = build_app(srv)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as c:
        await bootstrap_admin(c, tmp_path)
        yield c
    await srv.stop()


def test_config_off_without_secret() -> None:
    assert TrustedHeaderConfig.from_env({}) is None
    assert TrustedHeaderConfig.from_env({"OCTOP_TRUSTED_HEADER_SECRET": "  "}) is None


async def test_requires_exact_secret_and_subject(client) -> None:
    for bad in (
        {"X-User-Id": "1001"},
        _headers(**{"X-Proxy-Secret": "wrong"}),
        _headers(uid=""),
        {**_headers(), "X-Proxy-Secret": "sécret".encode("latin-1")},  # non-ASCII must 401, not 500
    ):
        r = await client.post("/api/auth/proxy/login", headers=bad)
        assert r.status_code == 401, bad


async def test_disabled_without_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCTOP_TRUSTED_HEADER_SECRET", raising=False)
    srv = OctopServer(home=tmp_path)
    await srv.start()
    await ensure_control_plane_bound(srv)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=build_app(srv)), base_url="http://testserver"
        ) as c:
            await bootstrap_admin(c, tmp_path)
            assert (await c.get("/api/auth/proxy/status")).json() == {"enabled": False}
            assert (await c.post("/api/auth/proxy/login", headers=_headers())).status_code == 401
    finally:
        await srv.stop()


async def test_provisions_once_and_token_works(client) -> None:
    first = await client.post("/api/auth/proxy/login", headers=_headers())
    assert first.status_code == 200
    me = await client.get("/api/auth/me", headers=bearer(first.json()["access_token"]))
    assert me.status_code == 200
    assert me.json()["role"] == "user"
    # Same Discord id after a rename is the same account.
    again = await client.post("/api/auth/proxy/login", headers=_headers(name="alice2"))
    assert again.json()["user"]["id"] == first.json()["user"]["id"]
    other = await client.post("/api/auth/proxy/login", headers=_headers(uid="1002", name="bob"))
    assert other.json()["user"]["id"] != first.json()["user"]["id"]


async def test_organizer_role_syncs_both_ways(client) -> None:
    up = await client.post("/api/auth/proxy/login", headers=_headers(organizer=True))
    assert up.json()["user"]["role"] == "admin"
    down = await client.post("/api/auth/proxy/login", headers=_headers(organizer=False))
    assert down.json()["user"]["role"] == "user"


async def test_break_glass_restricts_password_login(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OCTOP_TRUSTED_HEADER_LOCAL_LOGIN_USERS", "admin")
    ok = await client.post("/api/auth/login", json={"username": "admin", "password": TEST_PASSWORD})
    assert ok.status_code == 200
    # Another local account (here: any other name) is refused before any lookup.
    no = await client.post("/api/auth/login", json={"username": "alice", "password": TEST_PASSWORD})
    assert no.status_code == 401
