"""As entidades da arquitetura, na ordem em que o pipeline passa por elas.

Cada passo do pipeline pertence a uma entidade. O `--ate` do comando unico corta
o pipeline numa delas: roda tudo o que vem antes, inclusive ela, e para.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

STAGES = (
    "scenario_engine",
    "kms",
    "audit_logger",
    "dataset_generator",
    "policy_engine",
    "models",
    "evaluation",
    "figures",
)
"""A ordem da arquitetura da proposta (Figura 1), mais a avaliacao e as figuras."""

LAST_STAGE = STAGES[-1]


class Step(NamedTuple):
    """Um passo do pipeline: a entidade a que pertence, e o que ele faz."""

    stage: str
    run: Callable[[object], None]


def reaches(until: str, stage: str) -> bool:
    """Se uma execucao que para em `until` passa pela entidade `stage`."""
    return STAGES.index(stage) <= STAGES.index(until)


def run_steps(steps: tuple[Step, ...], state: object, until: str) -> object:
    """Roda os passos em ordem ate a entidade `until`, e devolve o estado preenchido.

    Os passos vem em ordem de entidade, entao o primeiro que passa do limite
    encerra a lista.
    """
    for step in steps:
        if not reaches(until, step.stage):
            break

        step.run(state)

    return state
