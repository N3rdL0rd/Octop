"""Trusted reverse-proxy header login ("proxy SSO").

A front proxy that has already authenticated the user (e.g. valgrind's Discord
OAuth proxy) forwards the identity in request headers. Octop accepts those
headers as a login only when the request also carries a shared secret, so a
caller that reaches Octop around the proxy cannot mint an identity by setting
``X-User-Id`` itself.

Configuration is environment-only (no stored secret, nothing for the admin UI
to leak):

* ``OCTOP_TRUSTED_HEADER_SECRET`` (or ``..._SECRET_FILE`` naming a file, e.g. a
  docker secret) - required; the feature is off when neither yields a value.
* ``OCTOP_TRUSTED_HEADER_SECRET_HEADER`` - default ``X-Proxy-Secret``.
* ``OCTOP_TRUSTED_HEADER_ID_HEADER`` - stable subject, default ``X-User-Id``.
* ``OCTOP_TRUSTED_HEADER_NAME_HEADER`` - display name, default ``X-User-Name``.
* ``OCTOP_TRUSTED_HEADER_ADMIN_HEADER`` - default ``X-User-Is-Organizer``;
  a value of ``true`` makes the user an admin.
* ``OCTOP_TRUSTED_HEADER_SYNC_ROLE`` - default ``true``: admin/user is
  re-derived from the header on every login, so demotion works. ``false``
  decides the role once at account creation.
* ``OCTOP_TRUSTED_HEADER_LOCAL_LOGIN_USERS`` - comma-separated usernames that
  may still use password login (break-glass). When set, everyone else is
  refused at ``/api/auth/login``. Empty leaves password login unrestricted.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from octop.infra.errors import ErrorCode, OctopError
from octop.infra.users.identity import Role, User

logger = logging.getLogger(__name__)

PROXY_KIND = "proxy"
_MAX_SUBJECT_LENGTH = 128
_MAX_NAME_LENGTH = 128


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TrustedHeaderConfig:
    secret: str
    secret_header: str = "X-Proxy-Secret"
    id_header: str = "X-User-Id"
    name_header: str = "X-User-Name"
    admin_header: str = "X-User-Is-Organizer"
    sync_role: bool = True
    local_login_users: frozenset[str] = frozenset()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TrustedHeaderConfig | None:
        """The configuration, or ``None`` when no shared secret is set."""
        source = os.environ if env is None else env
        secret = (source.get("OCTOP_TRUSTED_HEADER_SECRET") or "").strip()
        secret_file = (source.get("OCTOP_TRUSTED_HEADER_SECRET_FILE") or "").strip()
        if not secret and secret_file:
            try:
                secret = Path(secret_file).read_text(encoding="utf-8").strip()
            except OSError:
                # Fail closed (header login off) rather than raising: this runs on
                # the password login path too, and a bad mount must not lock out
                # the break-glass admin.
                logger.exception("cannot read trusted header secret file %s", secret_file)
                return None
        if not secret:
            return None

        def header(name: str, default: str) -> str:
            return (source.get(f"OCTOP_TRUSTED_HEADER_{name}") or "").strip() or default

        users = (source.get("OCTOP_TRUSTED_HEADER_LOCAL_LOGIN_USERS") or "").split(",")
        return cls(
            secret=secret,
            secret_header=header("SECRET_HEADER", "X-Proxy-Secret"),
            id_header=header("ID_HEADER", "X-User-Id"),
            name_header=header("NAME_HEADER", "X-User-Name"),
            admin_header=header("ADMIN_HEADER", "X-User-Is-Organizer"),
            sync_role=_truthy(source.get("OCTOP_TRUSTED_HEADER_SYNC_ROLE"), default=True),
            local_login_users=frozenset(u.strip() for u in users if u.strip()),
        )


@dataclass(frozen=True)
class ProxyIdentity:
    subject: str
    display_name: str | None
    is_admin: bool


def read_identity(config: TrustedHeaderConfig, headers: Mapping[str, str]) -> ProxyIdentity:
    """Validate the shared secret and extract the identity, or raise ``AUTH_FAILED``.

    ``headers`` must be case-insensitive (Starlette's ``request.headers`` is).
    """
    presented = headers.get(config.secret_header) or ""
    # compare_digest on bytes: str comparison raises on non-ASCII input.
    if not hmac.compare_digest(presented.encode(), config.secret.encode()):
        raise OctopError(ErrorCode.AUTH_FAILED, "proxy authentication required")
    subject = (headers.get(config.id_header) or "").strip()
    if not subject or len(subject) > _MAX_SUBJECT_LENGTH:
        raise OctopError(ErrorCode.AUTH_FAILED, "proxy authentication required")
    name = (headers.get(config.name_header) or "").strip()[:_MAX_NAME_LENGTH] or None
    is_admin = (headers.get(config.admin_header) or "").strip().lower() == "true"
    return ProxyIdentity(subject=subject, display_name=name, is_admin=is_admin)


class TrustedHeaderAuthService:
    def __init__(self, services: object, user_manager: object, config: TrustedHeaderConfig) -> None:
        self._services = services
        self._user_manager = user_manager
        self.config = config

    def _provider_id(self) -> int:
        repo = self._services.sso_repo  # type: ignore[attr-defined]
        row = repo.get_by_kind(PROXY_KIND)
        if row is None:
            # Identity anchor only: disabled so it never shows up as a login button.
            row = repo.upsert_by_kind(
                PROXY_KIND,
                enabled=False,
                display_name="Reverse proxy",
                issuer="",
                client_id="",
                client_secret_enc=None,
                scopes="",
                dashboard_origin=None,
            )
        return int(row.id)

    async def resolve_user(self, identity: ProxyIdentity) -> User:
        """The Octop user for a proxy identity, provisioned on first sight."""
        manager = self._user_manager
        claims: dict[str, str] = {}
        if identity.display_name:
            claims["preferred_username"] = identity.display_name
            claims["name"] = identity.display_name
        provider_id = self._provider_id()
        existing = self._services.user_repo.get_by_sso(provider_id, identity.subject)  # type: ignore[attr-defined]
        user: User = await manager.resolve_or_create_sso_user(  # type: ignore[attr-defined]
            provider_id=provider_id, subject=identity.subject, claims=claims
        )
        if existing is not None and identity.display_name:
            await manager.set_display_name(user.username, identity.display_name)  # type: ignore[attr-defined]
        if existing is None or self.config.sync_role:
            wanted = Role.ADMIN if identity.is_admin else None
            await self._apply_role(user, wanted, created=existing is None)
        self._services.user_repo.touch_last_login(user.id)  # type: ignore[attr-defined]
        refreshed = manager.get_by_id(user.id)  # type: ignore[attr-defined]
        return refreshed or user

    async def _apply_role(self, user: User, wanted: Role | None, *, created: bool) -> None:
        manager = self._user_manager
        if wanted is Role.ADMIN:
            if user.role != Role.ADMIN:
                await manager.set_role(user.username, Role.ADMIN)  # type: ignore[attr-defined]
            return
        # Not an organizer. A new account already got the default role template;
        # an existing one is demoted only when it is currently the built-in admin
        # (custom role templates are left to Octop admins).
        if not created and user.role == Role.ADMIN:
            await manager.set_role(user.username, Role.USER)  # type: ignore[attr-defined]
