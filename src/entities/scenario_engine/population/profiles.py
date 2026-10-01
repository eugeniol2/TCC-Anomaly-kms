from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    id_prefix: str
    operators: int
    scopes_each: int
    addresses_range: tuple[int, int]
    regime: str


PROFILES = (
    Profile(
        name="end_user",
        id_prefix="user",
        operators=30,
        scopes_each=1,
        addresses_range=(2, 4),
        regime="routine",
    ),
    Profile(
        name="automated_service",
        id_prefix="service",
        operators=6,
        scopes_each=2,
        addresses_range=(1, 2),
        regime="periodic_batch",
    ),
    Profile(
        name="administrator",
        id_prefix="admin",
        operators=8,
        scopes_each=4,
        addresses_range=(2, 4),
        regime="occasional_custody",
    ),
)


def total_operators() -> int:
    """Soma dos operadores de todos os perfis legitimos."""
    total = 0

    for profile in PROFILES:
        total += profile.operators

    return total


def total_scope_assignments() -> int:
    """Quantas atribuicoes de escopo a populacao inteira distribui.

    E o teto de quantos escopos podem existir: acima dele sobraria escopo sem detentor.
    """
    total = 0

    for profile in PROFILES:
        total += profile.operators * profile.scopes_each

    return total
