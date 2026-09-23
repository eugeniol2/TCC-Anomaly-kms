"""A mistura de operacoes de cada perfil (D-055).

Quatro operacoes, e **todos os perfis exercem todas**. Nenhuma celula da tabela
e zero: operacao privativa de um perfil funcionaria como marcador do perfil por
construcao, e os operadores incapazes de exerce-la formariam uma populacao
negativa de antemao.

As cinco operacoes que alterariam o estado do repositorio — `CreateKey`,
`Rotate`, `EnableKey`, `DisableKey`, `DeleteKey` — ficaram de fora: `keys.csv`
tem um unico modulo que o escreve, e `DeleteKey` executada pelo atacante seria
marcador quase perfeito, ja que ele exfiltra e nao administra.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from src.traffic.parameters import (
    ADMIN_OPERATION_MIX,
    SERVICE_OPERATION_MIX,
    USER_OPERATION_MIX,
)

OPERATIONS = ("Decrypt", "Encrypt", "DescribeKey", "ExportKeyMaterial")


def in_operation_order(mix: dict[str, float]) -> tuple[float, ...]:
    """As fracoes na ordem de `OPERATIONS`, que e a ordem que o sorteio espera.

    Os valores sao declarados por nome de operacao em `parameters.py`, e nao
    por posicao. Reordenar `OPERATIONS` deixa de reatribuir porcentagens, e
    operacao ausente de uma mistura falha aqui em vez de passar batido.
    """
    return tuple(mix[operation] for operation in OPERATIONS)


OPERATION_MIX: dict[str, tuple[float, ...]] = {
    "end_user": in_operation_order(USER_OPERATION_MIX),
    "automated_service": in_operation_order(SERVICE_OPERATION_MIX),
    "administrator": in_operation_order(ADMIN_OPERATION_MIX),
}
"""A mistura de cada perfil, ja na ordem do sorteio.

O administrador concentra mais em `DescribeKey` e `ExportKeyMaterial` porque
custodia envolve inspecionar e recuperar material. E o unico perfil que o
atacante personifica, entao e contra esta linha que o comportamento dele sera
comparado.

**A mistura do atacante sai desta mesma tabela**, na linha do administrador que
ele personifica. Mistura de operacoes nao e uma das cinco dimensoes de sigma:
se ele tivesse perfil de exfiltracao proprio, carregaria sinal que sigma nao
controla e continuaria detectavel em 1,0, quebrando a afirmacao de que cada
dimensao se aproxima do operador personificado.
"""


def draw_operations(rng: Generator, profile: str, quantity: int) -> list[str]:
    """As operacoes de uma sessao, sorteadas pela mistura do perfil."""
    probabilities = OPERATION_MIX[profile]
    chosen = rng.choice(len(OPERATIONS), size=quantity, p=np.asarray(probabilities))

    return [OPERATIONS[index] for index in chosen]
