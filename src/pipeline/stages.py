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
