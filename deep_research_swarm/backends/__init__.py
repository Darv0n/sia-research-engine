"""Search backend registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import SearchBackend

_REGISTRY: dict[str, type] = {}


def register_backend(name: str, cls: type) -> None:
    _REGISTRY[name] = cls


def get_backend(name: str, **kwargs) -> "SearchBackend":
    if name not in _REGISTRY:
        available = ", ".join(_REGISTRY) or "(none)"
        raise KeyError(f"Unknown backend {name!r}. Available: {available}")
    return _REGISTRY[name](**kwargs)


def available_backends() -> list[str]:
    return list(_REGISTRY)
