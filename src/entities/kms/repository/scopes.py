from __future__ import annotations


def scope_pool(quantity: int) -> list[str]:
    """Nomes dos escopos existentes no repositorio."""
    resultado = []

    for number in range(1, quantity + 1):
        resultado.append(f"scope_{number:02d}")

    return resultado
