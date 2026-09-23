"""M8: os limiares do baseline, calibrados na semana 3.

**Um conjunto de limiares por semente**, compartilhado pelas 11 condicoes de
sigma daquela varredura (D-043). A semana 3 e anterior ao ataque, logo
independente de sigma, e por isso mora no ramo da semente e roda 30 vezes e nao
330.

Tres propriedades vem de decisao e nao de conveniencia:

**Calibra sobre tráfego limpo, nunca sobre o holdout** (D-033, D-046). O
baseline nao recebe treino em momento nenhum e chega ao periodo avaliado com os
limiares congelados desde a semana 3. A assimetria com os modelos e deliberada:
calibrar limiar exige so comportamento normal, que um administrador teria antes
de qualquer incidente; ajustar hiperparametro exige rotulo de ataque, que ele
nao teria.

**Nao usa rotulo, e nao teria como usar.** O `sessions.csv` do aquecimento nao
tem coluna de rotulo (D-063). A cegueira e estrutural, nao disciplina.

**O limiar e global, nao por operador.** As duas regras que comparam contra o
historico de cada operador — `atypical_hour` e `novel_source` — ja vem prontas
do M7, lidas contra o perfil das semanas 1 e 2. As seis daqui sao de grandeza
absoluta e saem do percentil sobre todas as sessoes da semana juntas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibration.parameters import PERCENTILE, THRESHOLD_ATTRIBUTES
from src.globals.phases import CALIBRATION_WEEK, day_after_last_of, first_day_of

COLUMNS = ("attribute", "threshold", "percentile", "sessions")
"""As colunas de `thresholds.csv` (D-087).

`percentile` e `sessions` sao redundantes na mesma execucao — valem 99 e o
mesmo numero em todas as linhas — e existem para a **dispersao entre as 30
sementes** ser lida do disco sem reexecutar nada. A D-043 deixou essa medicao
como pendencia explicita: limiares muito instaveis entre sementes seriam achado
sobre a fragilidade do baseline, e material da Discussao.
"""


def calibration_period(sessions: pd.DataFrame) -> pd.DataFrame:
    """Apenas a semana 3, de todo o aquecimento.

    As semanas 1 e 2 construiram o perfil e **nao** entram: calcular o limiar
    sobre os mesmos dados que definiram a janela horaria acoplaria as duas
    coisas, que a D-054 mantem em periodos diferentes de proposito.
    """
    opens = pd.Timestamp(first_day_of(CALIBRATION_WEEK))
    closes = pd.Timestamp(day_after_last_of(CALIBRATION_WEEK))

    instants = pd.to_datetime(sessions["opened_at"])

    return sessions[(instants >= opens) & (instants < closes)]


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
    calibration = calibration_period(sessions)

    is_empty = len(calibration) == 0

    if is_empty:
        raise ValueError(
            f"a semana {CALIBRATION_WEEK} nao tem sessao nenhuma; "
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
