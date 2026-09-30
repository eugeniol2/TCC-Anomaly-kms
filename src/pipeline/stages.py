"""As entidades da arquitetura, na ordem em que o pipeline passa por elas.

Cada passo do pipeline e um par: a entidade a que pertence, e a funcao que ele roda,
como `("kms", warmup_outcomes)`. A funcao vai sem parenteses porque e guardada, e
nao executada: quem a executa e o `run_steps`, na hora certa.

O `--ate` do comando unico corta o pipeline numa entidade: roda tudo o que vem antes,
inclusive ela, e para.
"""

from __future__ import annotations

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


def reaches(until: str, stage: str) -> bool:
    """Se uma execucao que para em `until` passa pela entidade `stage`."""
    return STAGES.index(stage) <= STAGES.index(until)


def run_steps(steps: tuple, state: object, until: str) -> object:
    """Roda os passos em ordem ate a entidade `until`, e devolve o estado preenchido.

    Cada funcao escreve no `state` que recebe, e a seguinte le dali. Os passos vem
    em ordem de entidade, entao o primeiro que passa do limite encerra a lista.
    """
    for stage, run in steps:
        if not reaches(until, stage):
            break

        run(state)

    return state
