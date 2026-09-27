"""OpenAI Codex subscription OAuth integration for DeepClaw.

This module deliberately delegates OAuth storage, refresh, and the Codex
Responses API behavior to langchain-openai.  DeepClaw owns only its separate
credential-store location and non-secret status presentation.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from deepclaw.config import DeepClawConfig

OPENAI_CODEX_PROVIDER = "openai_codex"
CHATGPT_AUTH_STORE_PATH = Path("~/.deepclaw/auth/chatgpt-auth.json").expanduser()
_LOGIN_COMMAND = "deepclaw auth login openai_codex"


class OpenAICodexError(RuntimeError):
    """Base class for safe OpenAI Codex integration errors."""


class OpenAICodexDependencyError(OpenAICodexError):
    """The installed langchain-openai package cannot provide Codex OAuth."""


class OpenAICodexAuthError(OpenAICodexError):
    """A safe, actionable OAuth credential-store error."""


def load_openai_codex_symbols():
    """Lazily load the langchain-openai OAuth and Codex implementation.

    These are intentionally imported here so non-Codex DeepClaw installs do
    not need to import the private upstream OAuth implementation at startup.
    """
    try:
        oauth_module = import_module("langchain_openai.chatgpt_oauth")
        codex_module = import_module("langchain_openai.chat_models.codex")
        return (
            oauth_module._FileChatGPTOAuthTokenProvider,
            oauth_module.login_chatgpt,
            codex_module._ChatOpenAICodex,
        )
    except (ImportError, AttributeError) as exc:
        msg = (
            "OpenAI Codex OAuth requires langchain-openai==1.3.2 or a compatible "
            "version exposing chatgpt_oauth and _ChatOpenAICodex. Run `uv sync`."
        )
        raise OpenAICodexDependencyError(msg) from exc


def _auth_error() -> OpenAICodexAuthError:
    """Return a non-secret error for every runtime credential failure."""
    return OpenAICodexAuthError(
        "OpenAI Codex OAuth credentials are unavailable, corrupt, expired, or could not be "
        f"refreshed. Run `{_LOGIN_COMMAND}` to sign in again."
    )


def _logout_error() -> OpenAICodexAuthError:
    """Return safe recovery guidance when the local credential store cannot be removed."""
    return OpenAICodexAuthError(
        "OpenAI Codex logout failed because DeepClaw could not remove its local OAuth "
        "credentials. Check file permissions and try again."
    )


class _DeepClawTokenProvider:
    """Translate upstream store errors while delegating all token mechanics.

    The upstream provider continues to own its locks, refresh flow, and atomic
    writes. This proxy only prevents its implementation-specific recovery
    guidance from escaping through an eventual model invocation or stream.
    """

    def __init__(self, provider: Any):
        self._provider = provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def get_token(self) -> Any:
        try:
            return self._provider.get_token()
        except Exception:
            raise _auth_error() from None

    async def aget_token(self) -> Any:
        try:
            return await self._provider.aget_token()
        except Exception:
            raise _auth_error() from None

    def get_access_token(self) -> str:
        try:
            return self._provider.get_access_token()
        except Exception:
            raise _auth_error() from None

    async def aget_access_token(self) -> str:
        try:
            return await self._provider.aget_access_token()
        except Exception:
            raise _auth_error() from None


def _new_token_provider():
    provider_class, _, _ = load_openai_codex_symbols()
    return provider_class(path=CHATGPT_AUTH_STORE_PATH)


def _safe_token_details(token: Any) -> dict[str, str | None]:
    """Return only explicitly non-secret fields from an upstream token."""
    expires_at = getattr(token, "expires_at", None)
    return {
        "account_id": getattr(token, "account_id", None),
        "plan_type": getattr(token, "plan_type", None),
        "user_id": getattr(token, "user_id", None),
        "expires_at": str(expires_at) if expires_at is not None else None,
    }


def get_openai_codex_auth_status() -> dict[str, str | bool | None]:
    """Read/refresh DeepClaw's store and return non-secret account status only."""
    if not CHATGPT_AUTH_STORE_PATH.is_file():
        return {"logged_in": False, "state": "missing"}

    try:
        token = _new_token_provider().get_token()
    except FileNotFoundError:
        return {"logged_in": False, "state": "missing"}
    except OpenAICodexDependencyError:
        # Preserve the pinned-version recovery guidance from the private-symbol
        # import instead of treating an incompatible installation as bad OAuth.
        raise
    except Exception:
        # Never surface upstream exception text: it may contain transport data
        # or implementation details. The state still distinguishes recovery.
        raise _auth_error() from None

    return {"logged_in": True, "state": "valid", **_safe_token_details(token)}


def login_openai_codex() -> None:
    """Run langchain-openai's browser OAuth flow using DeepClaw's own store."""
    _, login_chatgpt, _ = load_openai_codex_symbols()
    CHATGPT_AUTH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    login_chatgpt(store_path=CHATGPT_AUTH_STORE_PATH)


def logout_openai_codex() -> bool:
    """Remove only DeepClaw's OAuth store, never the Codex CLI store."""
    try:
        CHATGPT_AUTH_STORE_PATH.unlink()
    except FileNotFoundError:
        return False
    except OSError:
        raise _logout_error() from None
    return True


def resolve_openai_codex_model(config: DeepClawConfig):
    """Resolve ``openai_codex:<model>`` to the upstream Codex chat model."""
    model_spec = (config.model or "").strip()
    provider, separator, model_name = model_spec.partition(":")
    if separator == "" or provider != OPENAI_CODEX_PROVIDER:
        return model_spec
    if not model_name.strip():
        msg = "OpenAI Codex model name cannot be empty"
        raise ValueError(msg)

    generation = config.generation
    if generation.repetition_penalty is not None:
        msg = (
            "generation.repetition_penalty is unsupported for openai_codex:* models: "
            "the Codex Responses API does not support repetition_penalty. Remove "
            "generation.repetition_penalty or use a provider that supports it."
        )
        raise ValueError(msg)
    if generation.max_tokens is not None:
        msg = (
            "generation.max_tokens is unsupported for openai_codex:* models: "
            "the Codex Responses API rejects max_output_tokens. Remove "
            "generation.max_tokens or use a provider that supports it."
        )
        raise ValueError(msg)

    provider_class, _, chat_model_class = load_openai_codex_symbols()
    token_provider = _DeepClawTokenProvider(provider_class(path=CHATGPT_AUTH_STORE_PATH))
    kwargs: dict[str, Any] = {
        "model": model_name,
        "token_provider": token_provider,
        "originator": "deepclaw",
    }
    if generation.temperature is not None:
        kwargs["temperature"] = generation.temperature
    if generation.top_p is not None:
        kwargs["top_p"] = generation.top_p

    # _ChatOpenAICodex itself forces Responses API, store=False, and streaming.
    return chat_model_class(**kwargs)
