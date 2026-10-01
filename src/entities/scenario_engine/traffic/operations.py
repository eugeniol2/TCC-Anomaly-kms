from __future__ import annotations

import numpy as np
from numpy.random import Generator

from src.entities.scenario_engine.traffic.parameters import (
    ADMIN_OPERATION_MIX,
    SERVICE_OPERATION_MIX,
    USER_OPERATION_MIX,
)

OPERATIONS = ("Decrypt", "Encrypt", "DescribeKey", "ExportKeyMaterial")


def in_operation_order(mix: dict[str, float]) -> tuple[float, ...]:
    """As fracoes na ordem de `OPERATIONS`, que e a ordem que o sorteio espera.

    Operacao ausente de uma mistura falha aqui.
    """
    shares = []

    for operation in OPERATIONS:
        shares.append(mix[operation])

    return tuple(shares)


OPERATION_MIX: dict[str, tuple[float, ...]] = {
    "end_user": in_operation_order(USER_OPERATION_MIX),
    "automated_service": in_operation_order(SERVICE_OPERATION_MIX),
    "administrator": in_operation_order(ADMIN_OPERATION_MIX),
}


def draw_operations(rng: Generator, profile: str, quantity: int) -> list[str]:
    """As operacoes de uma sessao, sorteadas pela mistura do perfil."""
    probabilities = OPERATION_MIX[profile]
    chosen = rng.choice(len(OPERATIONS), size=quantity, p=np.asarray(probabilities))
    operations = []

    for index in chosen:
        operations.append(OPERATIONS[index])

    return operations
