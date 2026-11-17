"""Per-user LLM provider provisioning on proxy login, and provider-list redaction."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from tests.support.app import ensure_control_plane_bound
from tests.support.auth import bearer, bootstrap_admin, login

from octop.api.app import build_app
from octop.infra.auth import llm_provisioning
from octop.infra.auth.llm_provisioning import LlmGatewayConfig, provider_name_for
from octop.infra.server import OctopServer

SECRET = "s3cret-value"


def _h(uid: str, name: str) -> dict[str, str]:
    return {"X-Proxy-Secret": SECRET, "X-User-Id": uid, "X-User-Name": name}


@pytest.fixture
async def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OCTOP_TRUSTED_HEADER_SECRET", SECRET)
    monkeypatch.setenv("OCTOP_LLM_GATEWAY_URL", "http://gw/v1")
    monkeypatch.setenv("OCTOP_LLM_MINT_URL", "http://gw/service/agents/key")
    monkeypatch.setenv("OCTOP_LLM_MINT_SECRET", "mint")
    calls: list[tuple[str, str]] = []

    async def fake_mint(config, *, discord_id, username, client=None):
        calls.append((discord_id, username))
        if fake_mint.fail:
            raise httpx.ConnectError("down")
        return f"sk-{discord_id}-{len(calls)}"

    fake_mint.fail = False  # type: ignore[attr-defined]
    monkeypatch.setattr(llm_provisioning, "mint_key", fake_mint)
    srv = OctopServer(home=tmp_path)
    await srv.start()
    await ensure_control_plane_bound(srv)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=build_app(srv)), base_url="http://testserver"
    ) as c:
        await bootstrap_admin(c, tmp_path)
        yield c, srv, calls, fake_mint
    await srv.stop()


def test_config_requires_all_three(monkeypatch: pytest.MonkeyPatch) -> None:
    full = {
        "OCTOP_LLM_GATEWAY_URL": "http://gw/v1/",
        "OCTOP_LLM_MINT_URL": "http://gw/mint",
        "OCTOP_LLM_MINT_SECRET": "x",
    }
    cfg = LlmGatewayConfig.from_env(full)
    assert cfg is not None and cfg.gateway_url == "http://gw/v1" and cfg.model == "auto"
    for missing in full:
        assert LlmGatewayConfig.from_env({k: v for k, v in full.items() if k != missing}) is None


async def test_first_login_provisions_private_provider(env) -> None:
    c, srv, calls, _ = env
    a = (await c.post("/api/auth/proxy/login", headers=_h("1001", "alice"))).json()
    uid = a["user"]["id"]
    name = provider_name_for(uid)
    row = srv.services.provider_repo.get_by_name(name)
    assert row is not None and row.api_key == "sk-1001-1" and row.base_url == "http://gw/v1"
    # A second login must not mint again.
    await c.post("/api/auth/proxy/login", headers=_h("1001", "alice"))
    assert calls == [("1001", "alice")]

    b = (await c.post("/api/auth/proxy/login", headers=_h("1002", "bob"))).json()
    alice_tok, bob_tok = a["access_token"], b["access_token"]

    mine = (await c.get("/api/providers", headers=bearer(alice_tok))).json()
    assert [p["name"] for p in mine] == [name]
    assert mine[0]["api_key"] is None
    theirs = (await c.get("/api/providers", headers=bearer(bob_tok))).json()
    assert name not in [p["name"] for p in theirs]
    assert [p["name"] for p in theirs] == [provider_name_for(b["user"]["id"])]

    resolved_a = (await c.get("/api/providers/resolved", headers=bearer(alice_tok))).json()
    assert {(m["provider_name"], m["model"]) for m in resolved_a} == {(name, "auto")}

    admin = await login(c)
    full = (await c.get("/api/providers", headers=bearer(admin))).json()
    assert {p["name"]: p["api_key"] for p in full}[name] == "sk-1001-1"


async def test_mint_failure_keeps_login_and_retries_next_time(env) -> None:
    c, srv, calls, fake = env
    fake.fail = True
    r = await c.post("/api/auth/proxy/login", headers=_h("1003", "carol"))
    assert r.status_code == 200
    name = provider_name_for(r.json()["user"]["id"])
    assert srv.services.provider_repo.get_by_name(name) is None
    fake.fail = False
    await c.post("/api/auth/proxy/login", headers=_h("1003", "carol"))
    assert srv.services.provider_repo.get_by_name(name) is not None
    assert len(calls) == 2
