"""Reverse-proxy header login HTTP routes (see ``infra/auth/trusted_header``)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from octop.api.deps import get_server
from octop.api.routers.auth import _user_json
from octop.infra.auth.llm_provisioning import LlmGatewayConfig, ensure_user_provider
from octop.infra.auth.mfa import MfaService
from octop.infra.auth.trusted_header import (
    TrustedHeaderAuthService,
    TrustedHeaderConfig,
    read_identity,
)
from octop.infra.errors import ErrorCode, OctopError

router = APIRouter()


@router.get("/proxy/status", summary="Whether reverse-proxy header login is enabled")
async def proxy_status() -> dict[str, bool]:
    return {"enabled": TrustedHeaderConfig.from_env() is not None}


@router.post("/proxy/login", summary="Sign in from trusted reverse-proxy headers")
async def proxy_login(request: Request, server: Any = Depends(get_server)) -> dict[str, Any]:
    """Exchange proxy-asserted identity headers for an Octop access token.

    Needs the shared secret header; without it (or with the feature off) this is
    a plain 401 so the dashboard falls back to the normal login form.
    """
    config = TrustedHeaderConfig.from_env()
    if config is None:
        raise OctopError(ErrorCode.AUTH_FAILED, "proxy authentication is not enabled")
    identity = read_identity(config, request.headers)
    service = TrustedHeaderAuthService(server.services, server.user_manager, config)
    user = await service.resolve_user(identity)
    server.services.audit_repo.write(
        actor=user.username, action="auth.proxy_login", target=user.username
    )
    llm = LlmGatewayConfig.from_env()
    if llm is not None:
        await ensure_user_provider(server, user, discord_id=identity.subject, config=llm)
    token, ttl = MfaService(server.services).mint_access_token(user)
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": ttl,
        "user": _user_json(user, locale=user.locale),
    }
