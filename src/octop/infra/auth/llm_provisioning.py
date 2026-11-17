"""Per-user LLM provider provisioning for proxy-SSO accounts (valgrind fork).

On a member's first login Octop mints an API key for them on the team LLM
gateway and creates a private provider row (``valgrind-llm-<user id>``) that only
that user (and admins) can see. Usage then shows up under their own Discord id
on the gateway, and their preferred model starts out as the gateway's ``auto``.

Environment:

* ``OCTOP_LLM_GATEWAY_URL`` - OpenAI-compatible base URL the agents call.
* ``OCTOP_LLM_MINT_URL`` - endpoint that mints a key for ``{discord_id, username}``.
* ``OCTOP_LLM_MINT_SECRET`` / ``OCTOP_LLM_MINT_SECRET_FILE`` - shared secret sent
  as ``X-Agents-Secret``.
* ``OCTOP_LLM_MODEL`` - model id to expose, default ``auto``.

All three of gateway URL, mint URL and secret must be set; otherwise this is off.

This is attribution, not secrecy: agents run shell commands as the same OS user
as Octop, so anything in the Octop database is readable from an agent.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from octop.infra.users.groups import RESOURCE_MODEL, model_resource_key

logger = logging.getLogger(__name__)

PROVIDER_PREFIX = "valgrind-llm-"
_MINT_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class LlmGatewayConfig:
    gateway_url: str
    mint_url: str
    secret: str
    model: str = "auto"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LlmGatewayConfig | None:
        source = os.environ if env is None else env
        gateway = (source.get("OCTOP_LLM_GATEWAY_URL") or "").strip().rstrip("/")
        mint = (source.get("OCTOP_LLM_MINT_URL") or "").strip()
        secret = (source.get("OCTOP_LLM_MINT_SECRET") or "").strip()
        secret_file = (source.get("OCTOP_LLM_MINT_SECRET_FILE") or "").strip()
        if not secret and secret_file:
            try:
                secret = Path(secret_file).read_text(encoding="utf-8").strip()
            except OSError:
                logger.exception("cannot read llm mint secret file %s", secret_file)
                return None
        if not (gateway and mint and secret):
            return None
        model = (source.get("OCTOP_LLM_MODEL") or "").strip() or "auto"
        return cls(gateway_url=gateway, mint_url=mint, secret=secret, model=model)


def provider_name_for(user_id: int) -> str:
    return f"{PROVIDER_PREFIX}{user_id}"


async def mint_key(
    config: LlmGatewayConfig,
    *,
    discord_id: str,
    username: str,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Ask the gateway for a fresh key. Raises ``httpx.HTTPError`` / ``ValueError``."""
    owns = client is None
    http = client or httpx.AsyncClient(timeout=_MINT_TIMEOUT_SECONDS)
    try:
        resp = await http.post(
            config.mint_url,
            headers={"X-Agents-Secret": config.secret},
            json={"discord_id": discord_id, "username": username},
        )
        resp.raise_for_status()
        key = resp.json().get("key")
    finally:
        if owns:
            await http.aclose()
    if not isinstance(key, str) or not key:
        raise ValueError("llm gateway returned no key")
    return key


async def ensure_user_provider(
    server: Any,
    user: Any,
    *,
    discord_id: str,
    config: LlmGatewayConfig,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    """Create the user's private provider once; returns its name, or ``None`` on failure.

    Failure (gateway down, bad secret) is logged and leaves the account usable;
    the next login retries because no provider row exists yet.
    """
    services = server.services
    name = provider_name_for(user.id)
    if services.provider_repo.get_by_name(name) is not None:
        return name
    try:
        key = await mint_key(
            config,
            discord_id=discord_id,
            username=user.display_name or user.username,
            client=client,
        )
    except (httpx.HTTPError, ValueError):
        logger.exception("llm key mint failed for user %s", user.username)
        return None

    models = [{"id": config.model, "name": config.model}]
    services.provider_repo.create(
        name=name,
        kind="openai",
        base_url=config.gateway_url,
        api_key=key,
        extra_json=None,
        models_json=json.dumps(models),
        note=f"Team LLM gateway key for {user.username} (auto-provisioned)",
    )
    # Private to this user: a one-member group granted this provider's models.
    repos = services.repos
    group = repos.user_group_repo.get_by_name(name) or repos.user_group_repo.create(
        name=name, description=f"Private LLM access for {user.username}"
    )
    repos.user_group_member_repo.replace(group.user_group_id, [user.id])
    repos.resource_group_grant_repo.replace(
        RESOURCE_MODEL, model_resource_key(name, config.model), [group.user_group_id]
    )
    if server.app_runtime:
        await server.app_runtime.agent_registry.on_provider_changed(provider_name=name)
    await server.user_manager.set_model_preferences(
        user.username, preferred_model=model_resource_key(name, config.model)
    )
    services.audit_repo.write(actor=user.username, action="llm.provision", target=name)
    return name
