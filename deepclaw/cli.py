"""CLI entry point and subcommand routing for DeepClaw."""

import logging
import sys

from deepclaw.config import load_config

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def _handle_doctor_command() -> None:
    """Handle 'deepclaw doctor' CLI command."""
    import asyncio

    from deepclaw.doctor import format_report, run_checks

    try:
        config = load_config()
    except Exception as exc:
        print(f"Configuration error: {exc}")  # noqa: T201
        raise SystemExit(1) from exc
    checks = asyncio.run(run_checks(config))
    print(format_report(checks))  # noqa: T201


def _handle_service_command(args: list[str]) -> None:
    """Handle 'deepclaw service <subcommand>' CLI commands."""
    from deepclaw.service import (
        detect_platform,
        install_service,
        service_status,
        uninstall_service,
    )

    plat = detect_platform()

    if not args:
        print("Usage: deepclaw service {install|uninstall|status}")  # noqa: T201
        raise SystemExit(1)

    subcommand = args[0]
    if subcommand == "install":
        print(install_service(plat))  # noqa: T201
    elif subcommand == "uninstall":
        print(uninstall_service(plat))  # noqa: T201
    elif subcommand == "status":
        print(service_status(plat))  # noqa: T201
    else:
        print(f"Unknown service subcommand: {subcommand}")  # noqa: T201
        print("Usage: deepclaw service {install|uninstall|status}")  # noqa: T201
        raise SystemExit(1)


def _handle_auth_command(args: list[str]) -> None:
    """Handle ``deepclaw auth {login|status|logout} openai_codex``."""
    usage = "Usage: deepclaw auth {login|status|logout} openai_codex"
    if len(args) != 2 or args[1] != "openai_codex":
        print(usage)  # noqa: T201
        raise SystemExit(1)

    action = args[0]
    if action not in {"login", "status", "logout"}:
        print(f"Unknown auth subcommand: {action}")  # noqa: T201
        print(usage)  # noqa: T201
        raise SystemExit(1)

    from deepclaw.integrations.openai_codex import (
        OpenAICodexAuthError,
        OpenAICodexDependencyError,
        get_openai_codex_auth_status,
        login_openai_codex,
        logout_openai_codex,
    )

    if action == "login":
        try:
            login_openai_codex()
        except OpenAICodexDependencyError as exc:
            print(str(exc))  # noqa: T201
            raise SystemExit(1) from None
        except Exception:
            print("OpenAI Codex login failed. Run `deepclaw auth login openai_codex` to try again.")  # noqa: T201
            raise SystemExit(1) from None
        print("OpenAI Codex login completed.")  # noqa: T201
        return
    if action == "status":
        try:
            status = get_openai_codex_auth_status()
        except (OpenAICodexAuthError, OpenAICodexDependencyError) as exc:
            print(str(exc))  # noqa: T201
            raise SystemExit(1) from None
        if not status["logged_in"]:
            print("OpenAI Codex is not logged in. Run `deepclaw auth login openai_codex`.")  # noqa: T201
            return
        details = [
            f"{key}={status[key]}"
            for key in ("account_id", "plan_type", "user_id", "expires_at")
            if status.get(key) is not None
        ]
        suffix = f" ({', '.join(details)})" if details else ""
        print(f"OpenAI Codex is logged in{suffix}.")  # noqa: T201
        return
    if action == "logout":
        try:
            removed = logout_openai_codex()
        except OpenAICodexAuthError as exc:
            print(str(exc))  # noqa: T201
            raise SystemExit(1) from None
        if removed:
            print("OpenAI Codex local logout complete.")  # noqa: T201
            print("Restart the DeepClaw service to discard its cached credentials.")  # noqa: T201
        else:
            print("OpenAI Codex was not logged in.")  # noqa: T201
        return


def main() -> None:
    """Entry point: start the Telegram bot with long-polling."""
    args = sys.argv[1:]
    if args and args[0] == "service":
        _handle_service_command(args[1:])
        return
    if args and args[0] == "auth":
        _handle_auth_command(args[1:])
        return
    if args and args[0] == "doctor":
        _handle_doctor_command()
        return

    config = load_config()

    from deepclaw.channels.telegram import run_telegram

    run_telegram(config)


if __name__ == "__main__":
    main()
