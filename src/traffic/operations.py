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

OPERATIONS = ("Decrypt", "Encrypt", "DescribeKey", "ExportKeyMaterial")

OPERATION_MIX: dict[str, tuple[float, ...]] = {
    "legitimate_user": (0.55, 0.25, 0.12, 0.08),
    "automated_service": (0.45, 0.35, 0.10, 0.10),
    "administrator": (0.35, 0.15, 0.30, 0.20),
}
"""Fracao das requisicoes de um operador daquele perfil, na ordem de `OPERATIONS`.

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
