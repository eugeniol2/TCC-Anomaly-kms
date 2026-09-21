"""As duas fases de execucao e as semanas que cada uma cobre (D-060, D-069).

Mora aqui, e nao no `parameters.py` de um modulo, porque o calendario nao
pertence a nenhum deles: o M2 o usa para gerar, e o M4, o M5, o M6, o M7, o
M8 e o M9 o usam para saber de que fatia de tempo estao tratando. Pos num
modulo, os outros teriam de importar de um vizinho, e a fronteira entre
modulos e o arquivo.

    semanas 1 e 2    constroem o perfil historico
    semana 3         calibra os limiares do baseline
    semanas 4 a 7    periodo avaliado, onde o atacante age

    --fase warmup      semanas 1 a 3, ramo da semente
    --fase evaluated   semanas 4 a 7, ramo de sigma

A fase e argumento obrigatorio e sem padrao: esquecer tem de falhar, nao
rodar o ramo errado em silencio gravando saida valida no lugar errado.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

FIRST_DAY = date(2026, 1, 5)
"""Primeiro dia simulado (D-067). Segunda-feira, fixa, nunca `date.today()`.

Ancorar numa segunda faz cada semana simulada coincidir com uma semana civil.
Dois dos tres regimes so abrem sessao em dia util, e periodo desalinhado
daria a uma semana quatro dias uteis e a outra seis.
"""

WEEK_COUNT = 7
"""Semanas simuladas (D-069): 2 de perfil + 1 de calibracao + 4 avaliadas."""

DAYS_PER_WEEK = 7
"""Fato de calendario, nao decisao: sem entrada no registro."""

PROFILE_WEEKS = (1, 2)
"""Constroem o `historical_profile`, que fica congelado dali em diante (D-044)."""

CALIBRATION_WEEK = 3
"""Calibra os limiares do baseline, lida contra o perfil das semanas 1 e 2."""

WARMUP = "warmup"
"""Semanas 1 a 3. Anterior ao ataque, mora no ramo da semente."""

EVALUATED = "evaluated"
"""Semanas 4 a 7. Onde a campanha transcorre, no ramo de sigma."""

PHASES = (WARMUP, EVALUATED)

WEEKS_OF = {
    WARMUP: (1, CALIBRATION_WEEK),
    EVALUATED: (CALIBRATION_WEEK + 1, WEEK_COUNT),
}
"""Primeira e ultima semana de cada fase, ambas inclusivas."""


def day_count(week_count: int = WEEK_COUNT) -> int:
    """Dias corridos de um numero de semanas."""
    return week_count * DAYS_PER_WEEK


def first_day_of(week: int) -> date:
    """O primeiro dia de uma semana, contada a partir de 1."""
    return FIRST_DAY + timedelta(days=(week - 1) * DAYS_PER_WEEK)


def day_after_last_of(week: int) -> date:
    """O dia seguinte ao ultimo de uma semana, para comparacao aberta."""
    return first_day_of(week + 1)


def belongs_to(phase: str, moments: pd.Series) -> pd.Series:
    """Quais instantes caem dentro da fase indicada.

    Recebe a coluna de `timestamp` como texto ISO e devolve mascara booleana.
    O limite superior e aberto para que a meia-noite do dia seguinte fique de
    fora sem depender de arredondamento.
    """
    is_unknown = phase not in WEEKS_OF

    if is_unknown:
        raise ValueError(f"fase desconhecida: {phase!r}. Use uma de {PHASES}")

    first_week, last_week = WEEKS_OF[phase]
    opens = pd.Timestamp(first_day_of(first_week))
    closes = pd.Timestamp(day_after_last_of(last_week))

    instants = pd.to_datetime(moments)

    return (instants >= opens) & (instants < closes)
