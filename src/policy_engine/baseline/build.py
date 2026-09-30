"""M10: o baseline de regras decidindo sobre o holdout.

Tres propriedades vem de decisao e nao de conveniencia:

**Nao recebe treino** (D-033, D-046). Le o `holdout.csv` e os limiares do
aquecimento, e nunca o `train.csv`. Chega ao periodo avaliado com os limiares
congelados desde o fim da regua, como um defensor real chegaria.

**Oito regras, uma por atributo** (D-080). Seis de grandeza, que disparam
quando o atributo passa **estritamente** do limiar; duas de historico, que
disparam quando valem 1. A sessao vira alerta com **duas ou mais** disparadas
(D-075).

**Carrega o rotulo, mas nao o consulta** (D-113). O `compromised` vem do
holdout e vai para a saida ao lado da decisao, para o M12 e para quem confere o
arquivo a olho. A decisao e calculada so dos oito atributos, e ha teste que
zera o rotulo e confere que nenhuma decisao muda.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.policy_engine.baseline.parameters import HISTORY_RULE_FIRES, MINIMUM_RULES_FIRED
from src.policy_engine.calibration.parameters import THRESHOLD_ATTRIBUTES
from src.dataset_generator.dataset.build import ATTRIBUTES, IDENTIFIERS, LABEL, require_label
from src.shared.timing import time_decision

MECHANISM = "rules"
"""Como o baseline aparece no arquivo de tempo, ao lado dos dois modelos."""

HISTORY_ATTRIBUTES = tuple(
    attribute for attribute in ATTRIBUTES if attribute not in THRESHOLD_ATTRIBUTES
)
"""As duas regras de historico: os atributos que nao tem limiar no M8."""

RULE_ATTRIBUTES = THRESHOLD_ATTRIBUTES + HISTORY_ATTRIBUTES
"""A ordem das oito regras: as seis de grandeza, depois as duas de historico."""

RULE_COLUMNS = tuple(f"rule_{attribute}" for attribute in RULE_ATTRIBUTES)
"""Uma coluna por regra, 0 ou 1. O prefixo separa a regra do atributo que ela le."""

COLUMNS = IDENTIFIERS + RULE_COLUMNS + ("rules_fired", "predicted", LABEL)
"""As colunas do `predictions_rules.csv` (D-113).

Os identificadores para auditar um alerta, as oito regras para saber **por que**
ele saiu, a contagem, a decisao e, por ultimo, a verdade. Decisao e verdade
ficam lado a lado de proposito: e assim que o arquivo se confere sem ferramenta.
"""


class Baseline(NamedTuple):
    """O que o M10 produz: as decisoes e o tempo que elas levaram."""

    predictions: pd.DataFrame
    timing: pd.DataFrame


def magnitude_cutoffs(thresholds: pd.DataFrame) -> np.ndarray:
    """Os seis limiares, na ordem das regras de grandeza.

    Recusa um `thresholds.csv` que nao tenha exatamente as seis regras: uma
    linha faltando desligaria uma regra em silencio, e o baseline passaria a ser
    outro mecanismo sem que nada acusasse.
    """
    by_attribute = thresholds.set_index("attribute")["threshold"]
    is_other_set = set(by_attribute.index) != set(THRESHOLD_ATTRIBUTES)

    if is_other_set:
        raise ValueError(
            f"o `thresholds.csv` tem as regras {sorted(by_attribute.index)}, "
            f"e o baseline espera {sorted(THRESHOLD_ATTRIBUTES)}"
        )

    return by_attribute.loc[list(THRESHOLD_ATTRIBUTES)].to_numpy()


def fired_rules(attributes: pd.DataFrame, cutoffs: np.ndarray) -> np.ndarray:
    """Quais das oito regras disparam em cada sessao.

    `>` estrito nas de grandeza: com `>=`, um limiar que encostasse no maximo
    observado marcaria toda sessao que o atingisse (D-080).
    """
    magnitude = attributes[list(THRESHOLD_ATTRIBUTES)].to_numpy() > cutoffs
    history = attributes[list(HISTORY_ATTRIBUTES)].to_numpy() == HISTORY_RULE_FIRES

    return np.hstack([magnitude, history])


def decide(attributes: pd.DataFrame, cutoffs: np.ndarray) -> np.ndarray:
    """A decisao do baseline: alerta onde duas ou mais regras disparam.

    E esta a funcao cronometrada (D-106). Recebe o quadro dos oito atributos,
    o mesmo formato que o `predict` dos modelos vai receber, e devolve um vetor.
    """
    fired = fired_rules(attributes, cutoffs)

    return fired.sum(axis=1) >= MINIMUM_RULES_FIRED


def predictions_of(sessions: pd.DataFrame, cutoffs: np.ndarray) -> pd.DataFrame:
    """O arquivo de predicoes: por que cada sessao alertou, ou nao."""
    attributes = sessions[list(ATTRIBUTES)]
    fired = fired_rules(attributes, cutoffs).astype(int)

    rules = pd.DataFrame(fired, columns=list(RULE_COLUMNS), index=sessions.index)
    rules_fired = rules.sum(axis=1)
    predicted = decide(attributes, cutoffs).astype(int)

    return sessions[list(IDENTIFIERS)].join(rules).assign(
        rules_fired=rules_fired,
        predicted=predicted,
        **{LABEL: sessions[LABEL]},
    )[list(COLUMNS)].reset_index(drop=True)


def build_baseline(holdout: pd.DataFrame, thresholds: pd.DataFrame) -> Baseline:
    """Do holdout e dos limiares do aquecimento as decisoes, e ao tempo delas.

    O tempo e medido sobre o holdout ja carregado e com os limiares ja lidos:
    so a decisao entra no relogio (D-106).
    """
    require_label(holdout)
    cutoffs = magnitude_cutoffs(thresholds)
    attributes = holdout[list(ATTRIBUTES)]

    timing = time_decision(
        MECHANISM, len(holdout), lambda: decide(attributes, cutoffs)
    )

    return Baseline(predictions_of(holdout, cutoffs), timing)
