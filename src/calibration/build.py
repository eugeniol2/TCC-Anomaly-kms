"""M8: os limiares do baseline, calibrados nas quatro semanas de aquecimento.

**Um conjunto de limiares por semente**, compartilhado pelas 11 condicoes de
sigma daquela varredura (D-043). O aquecimento e anterior ao ataque, logo
independente de sigma, e por isso mora no ramo da semente e roda 30 vezes e nao
330.

Tres propriedades vem de decisao e nao de conveniencia:

**Calibra sobre tráfego limpo, nunca sobre o holdout** (D-033, D-046). O
baseline nao recebe treino em momento nenhum e chega ao periodo avaliado com os
limiares congelados desde o fim do aquecimento. A assimetria com os modelos e
deliberada:
calibrar limiar exige so comportamento normal, que um administrador teria antes
de qualquer incidente; ajustar hiperparametro exige rotulo de ataque, que ele
nao teria.

**Nao usa rotulo, e nao teria como usar.** O `sessions.csv` do aquecimento nao
tem coluna de rotulo (D-063). A cegueira e estrutural, nao disciplina.

**O limiar e global, nao por operador.** As duas regras que comparam contra o
historico de cada operador — `atypical_hour` e `new_source_ip` — ja vem prontas
do M7, lidas contra o perfil. As seis daqui sao de grandeza absoluta e saem do
percentil sobre todas as sessoes do aquecimento juntas.

**Sai das quatro semanas de aquecimento** (D-096), e nao de uma so. O perfil e
os limiares sao os dois lados da mesma regua, construida de uma vez, no mesmo
periodo: e o que permite dizer que o baseline nasce pronto ao fim do
aquecimento e so entao entra em producao.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration.parameters import PERCENTILE, THRESHOLD_ATTRIBUTES
from src.globals.phases import WARMUP, belongs_to

COLUMNS = ("attribute", "threshold", "percentile", "sessions")
"""As colunas de `thresholds.csv` (D-087).

`percentile` e `sessions` sao redundantes na mesma execucao — valem 99 e o
mesmo numero em todas as linhas — e existem para a **dispersao entre as 30
sementes** ser lida do disco sem reexecutar nada. A D-043 deixou essa medicao
como pendencia explicita: limiares muito instaveis entre sementes seriam achado
sobre a fragilidade do baseline, e material da Discussao.
"""


def ruler_period(sessions: pd.DataFrame) -> pd.DataFrame:
    """As sessoes do aquecimento inteiro, que desde a D-096 **sao** a regua.

    Ate 23/09 so a semana 3 entrava, e as semanas 1 e 2 ficavam de fora porque
    tinham construido o perfil. O argumento nao se sustentava: **as seis
    regras de grandeza nao usam o perfil** — `events`, `duration_minutes` e as
    outras quatro se calculam do log cru —, entao nao havia acoplamento a
    evitar. O que a divisao produzia era um limiar estimado de um terco dos
    dados disponiveis.

    Como guarda, a funcao recusa o que nao for do aquecimento: calibrar sobre
    o periodo avaliado poria o atacante dentro do percentil (D-033).
    """
    return sessions[belongs_to(WARMUP, sessions["opened_at"])]


def threshold_of(values: pd.Series) -> float:
    """O valor acima do qual a regra dispara.

    A comparacao no M10 e `> limiar`, estrita. Com `>=`, uma regra cujo
    percentil encoste no maximo observado marcaria toda sessao que atingisse o
    teto — foi assim que a razao `chaves_por_evento` marcava 9,4 % das sessoes
    ao trocar o operador, medindo tamanho de sessao e nao varredura (D-080).
    """
    return float(np.percentile(values, PERCENTILE))


def build_thresholds(sessions: pd.DataFrame) -> pd.DataFrame:
    """Do conjunto de sessoes do aquecimento aos seis limiares da semente."""
    calibration = ruler_period(sessions)

    is_empty = len(calibration) == 0

    if is_empty:
        raise ValueError(
            "nenhuma sessao do aquecimento; "
            "o `sessions.csv` recebido e da fase avaliada?"
        )

    rows = [
        {
            "attribute": attribute,
            "threshold": round(threshold_of(calibration[attribute]), 4),
            "percentile": PERCENTILE,
            "sessions": len(calibration),
        }
        for attribute in THRESHOLD_ATTRIBUTES
    ]

    return pd.DataFrame(rows)[list(COLUMNS)]
