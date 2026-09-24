"""As duas fases de execucao e as semanas que cada uma cobre (D-060, D-096).

Mora aqui, e nao no `parameters.py` de um modulo, porque o calendario nao
pertence a nenhum deles: o M2 o usa para gerar, e o M4, o M5, o M6, o M7, o
M8 e o M9 o usam para saber de que fatia de tempo estao tratando. Pos num
modulo, os outros teriam de importar de um vizinho, e a fronteira entre
modulos e o arquivo.

    semanas 1 a 4    constroem a regua: o perfil historico E os limiares
    semanas 5 a 8    periodo avaliado, onde o atacante age

    --fase warmup      semanas 1 a 4, ramo da semente
    --fase evaluated   semanas 5 a 8, ramo de sigma

**O aquecimento tem um trabalho so** (D-096). Ate 23/09 ele era dividido
(semanas 1 e 2 faziam o perfil, a semana 3 calibrava os limiares) e a divisao
custava alarme falso sem comprar nada: com a regua vendo quatro semanas em vez
de duas, `atypical_hour` cai de 8,83 % para 4,54 % das sessoes limpas e
`new_source_ip` de 2,05 % para 0,80 %.

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

RULER_WEEKS = 4
"""Semanas que constroem a regua: o perfil historico e os limiares (D-096).

Quatro, e nao tres, porque a curva ainda nao saturou ali. Medido sobre as
mesmas semanas avaliadas, com a regua crescendo:

    regua de 2 semanas   atypical_hour 8,83 %   new_source_ip 2,05 %
    regua de 3 semanas                 5,87 %                 1,14 %
    regua de 4 semanas                 4,54 %                 0,80 %

O que cai e **alarme falso sobre gente inocente**: a janela e a lista de
origens sao estimadas de amostra finita, e amostra pequena demais nao
representa o operador. Baseline mais forte torna a comparacao mais
defensavel, pelo mesmo argumento que deu os oito atributos a ele (D-080).
"""

EVALUATED_WEEKS = 4
"""Semanas do periodo avaliado, onde a campanha transcorre (D-069).

Quatro porque cada sessao comprometida e **uma** positiva (D-061) e so um
administrador e comprometido (D-036): com duas, o holdout ficava com ~12
positivas, contagem na faixa da armadilha de Athapaththu et al. Com quatro,
sao 24.

**E o que impede a simetria de ser gratuita.** Encurtar para tres deixaria o
admin alvo com 30 sessoes legitimas em vez de 40, a campanha com 45 e o
holdout com 18 positivas, abaixo do piso de 20 que a D-081 fixou. A simetria
4 + 4 sai de alongar a regua, nunca de encurtar a avaliacao.
"""

WEEK_COUNT = RULER_WEEKS + EVALUATED_WEEKS
"""Semanas simuladas: 4 de regua + 4 avaliadas (D-096)."""

DAYS_PER_WEEK = 7
"""Fato de calendario, nao decisao: sem entrada no registro."""

WARMUP = "warmup"
"""Semanas 1 a 4. Anterior ao ataque, mora no ramo da semente."""

EVALUATED = "evaluated"
"""Semanas 5 a 8. Onde a campanha transcorre, no ramo de sigma."""

PHASES = (WARMUP, EVALUATED)

WEEKS_OF = {
    WARMUP: (1, RULER_WEEKS),
    EVALUATED: (RULER_WEEKS + 1, WEEK_COUNT),
}
"""Primeira e ultima semana de cada fase, ambas inclusivas.

O aquecimento e a regua sao a mesma coisa desde a D-096: nao ha mais um
recorte dentro dele. `PROFILE_WEEKS` e `CALIBRATION_WEEK` existiam para marcar
essa divisao e foram removidas com ela.
"""


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
