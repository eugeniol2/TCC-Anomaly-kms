from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from src.entities.policy_engine.baseline.parameters import (
    HISTORY_RULE_FIRES_AT,
    MINIMUM_RULES_FIRED,
)
from src.entities.policy_engine.calibration.parameters import THRESHOLD_ATTRIBUTES
from src.entities.dataset_generator.dataset.build import ATTRIBUTES, IDENTIFIERS, LABEL, require_label
from src.shared.timing import time_decision

MECHANISM = "rules"

def history_attributes() -> tuple[str, ...]:
    """As duas regras de historico: os atributos que nao tem limiar calibrado."""
    attributes = []

    for attribute in ATTRIBUTES:
        has_threshold = attribute in THRESHOLD_ATTRIBUTES

        if not has_threshold:
            attributes.append(attribute)

    return tuple(attributes)


HISTORY_ATTRIBUTES = history_attributes()

RULE_ATTRIBUTES = THRESHOLD_ATTRIBUTES + HISTORY_ATTRIBUTES


def rule_columns() -> tuple[str, ...]:
    """Uma coluna por regra, 0 ou 1. O prefixo separa a regra do atributo que ela le."""
    columns = []

    for attribute in RULE_ATTRIBUTES:
        columns.append(f"rule_{attribute}")

    return tuple(columns)


RULE_COLUMNS = rule_columns()

COLUMNS = IDENTIFIERS + RULE_COLUMNS + ("rules_fired", "predicted", LABEL)


class Baseline(NamedTuple):
    predictions: pd.DataFrame
    timing: pd.DataFrame


def magnitude_cutoffs(thresholds: pd.DataFrame) -> np.ndarray:
    """Os seis limiares, na ordem das regras de grandeza.

    Recusa um `thresholds.csv` que nao tenha exatamente as seis regras.
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
    """Quais das oito regras disparam em cada sessao. As de grandeza disparam acima do
    limiar, com `>` estrito.
    """
    magnitude = attributes[list(THRESHOLD_ATTRIBUTES)].to_numpy() > cutoffs
    history = attributes[list(HISTORY_ATTRIBUTES)].to_numpy() == HISTORY_RULE_FIRES_AT

    return np.hstack([magnitude, history])


def decide(attributes: pd.DataFrame, cutoffs: np.ndarray) -> np.ndarray:
    """A decisao do baseline: alerta onde duas ou mais regras disparam.

    E a funcao cronometrada: recebe o quadro dos oito atributos e devolve um vetor.
    """
    fired = fired_rules(attributes, cutoffs)

    return fired.sum(axis=1) >= MINIMUM_RULES_FIRED


def predictions_of(sessions: pd.DataFrame, cutoffs: np.ndarray) -> pd.DataFrame:
    """O arquivo de predicoes: por que cada sessao alertou, ou nao."""
    attributes = sessions[list(ATTRIBUTES)]
    fired = fired_rules(attributes, cutoffs).astype(int)

    rules = pd.DataFrame(fired, columns=list(RULE_COLUMNS), index=sessions.index)

    predictions = sessions[list(IDENTIFIERS)].join(rules)
    predictions["rules_fired"] = rules.sum(axis=1)
    predictions["predicted"] = decide(attributes, cutoffs).astype(int)
    predictions[LABEL] = sessions[LABEL]

    return predictions[list(COLUMNS)].reset_index(drop=True)


def build_baseline(holdout: pd.DataFrame, thresholds: pd.DataFrame) -> Baseline:
    """Do holdout e dos limiares do aquecimento as decisoes, e ao tempo delas.

    So a decisao entra no relogio: o holdout chega carregado e os limiares ja lidos.
    """
    require_label(holdout)
    cutoffs = magnitude_cutoffs(thresholds)
    attributes = holdout[list(ATTRIBUTES)]

    timing = time_decision(
        MECHANISM, len(holdout), lambda: decide(attributes, cutoffs)
    )

    return Baseline(predictions_of(holdout, cutoffs), timing)
