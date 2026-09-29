"""O cronometro do tempo de inferencia, o mesmo para regras e modelos (D-106).

Mora aqui, e nao no `parameters.py` de um modulo, porque o baseline (M10) e os
modelos (M11) precisam medir **do mesmo jeito**, e importar do vizinho quebraria
a fronteira entre modulos. E a mesma razao que pos o calendario em
`globals/phases.py` (D-079).

**E a unica medicao nao deterministica do pipeline.** O arquivo que ela produz
(`timing_rules.csv`, e depois o dos modelos) muda a cada execucao, e por isso
fica fora da conferencia byte a byte. As predicoes, nao: elas continuam
reproduziveis.
"""

from __future__ import annotations

from statistics import median
from time import perf_counter_ns
from typing import Callable

import pandas as pd

DISCARDED_ROUNDS = 1
"""Rodadas de aquecimento, cronometradas e jogadas fora (D-106).

A primeira chamada paga o que as seguintes nao pagam: cache frio, alocacao,
importacao preguicosa dentro da biblioteca. Contar essa rodada mediria a
primeira chamada, e nao a decisao.
"""

TIMED_ROUNDS = 10
"""Rodadas que entram na mediana (D-106).

Mediana, e nao media: uma interrupcao do sistema operacional numa rodada so
puxaria a media inteira, e a mediana a ignora.
"""

NANOSECONDS_PER_MICROSECOND = 1_000
NANOSECONDS_PER_SECOND = 1_000_000_000

COLUMNS = (
    "mechanism",
    "sessions",
    "timed_rounds",
    "median_nanoseconds",
    "microseconds_per_session",
    "sessions_per_second",
)
"""As colunas do arquivo de tempo, uma linha por mecanismo.

Microssegundos por sessao e a unidade da tabela do trabalho, e a vazao vai ao
lado porque e o numero que um operador de sistema entende (D-106).
"""


def median_nanoseconds(decide: Callable[[], object]) -> int:
    """A mediana das rodadas cronometradas, depois de descartar o aquecimento.

    Cronometra so a chamada. O que ela recebe ja esta pronto antes do relogio
    comecar, e o que ela devolve nao e escrito em lugar nenhum durante a
    medicao.
    """
    for _ in range(DISCARDED_ROUNDS):
        decide()

    rounds = []

    for _ in range(TIMED_ROUNDS):
        started = perf_counter_ns()
        decide()
        rounds.append(perf_counter_ns() - started)

    return int(median(rounds))


def timing_row(mechanism: str, sessions: int, nanoseconds: int) -> dict[str, object]:
    """Uma linha do arquivo de tempo, com as duas unidades derivadas."""
    per_session = nanoseconds / sessions / NANOSECONDS_PER_MICROSECOND
    throughput = sessions * NANOSECONDS_PER_SECOND / nanoseconds

    return {
        "mechanism": mechanism,
        "sessions": sessions,
        "timed_rounds": TIMED_ROUNDS,
        "median_nanoseconds": nanoseconds,
        "microseconds_per_session": round(per_session, 4),
        "sessions_per_second": round(throughput, 1),
    }


def time_decision(
    mechanism: str, sessions: int, decide: Callable[[], object]
) -> pd.DataFrame:
    """O tempo de um mecanismo decidindo sobre o holdout inteiro."""
    nanoseconds = median_nanoseconds(decide)
    row = timing_row(mechanism, sessions, nanoseconds)

    return pd.DataFrame([row])[list(COLUMNS)]
