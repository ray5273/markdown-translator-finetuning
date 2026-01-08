"""Dependency checks and helpful install guidance."""

from __future__ import annotations

from typing import Iterable, Sequence


def require_dependencies(modules: Sequence[tuple[str, str]]) -> None:
    """Ensure modules can be imported, otherwise raise helpful error.

    Args:
        modules: Sequence of (import_name, pip_name) pairs.
    """
    missing: list[str] = []
    for import_name, pip_name in modules:
        try:
            __import__(import_name)
        except ModuleNotFoundError:
            missing.append(pip_name)

    if missing:
        packages = " ".join(sorted(set(missing)))
        raise ModuleNotFoundError(
            "Missing required dependencies. Install with:\n"
            f"  python -m pip install {packages}\n"
            "If you're installing from requirements.txt, run:\n"
            "  python -m pip install -r requirements.txt\n"
        )
