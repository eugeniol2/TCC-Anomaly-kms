"""Testes do M10: o baseline de regras, e o cronometro que ele divide com o M11.

Nenhum teste aqui mede desempenho de deteccao. Eles conferem **o mecanismo**,
nunca o quanto ele acerta: olhar acerto no holdout das 30 replicas e um momento
que o registro marca, e nao pode acontecer dentro de um teste.

Quatro coisas sao guardadas:

1. **As regras fazem o que a D-080 diz.** Grandeza dispara so **acima** do
   limiar; historico dispara em 1. Conferido em sessoes sinteticas, uma regra
   de cada vez.
2. **O alerta e a D-075.** Uma regra sozinha nunca alerta, e qualquer par das
   oito alerta: as 28 combinacoes.
3. **O rotulo vai junto, mas nao decide** (D-113). Zerar ou inverter o rotulo nao
   muda decisao nenhuma.
4. **O mesmo resultado pelos dois caminhos.** Lendo do CSV, como o comando
   isolado, ou da memoria, como o orquestrador.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

from src.entities.scenario_engine.attack.build import build_attack
from src.entities.scenario_engine.attack.parameters import AttackSpecification
from src.entities.audit_logger.build import build_log
from src.entities.policy_engine.baseline.build import (
    COLUMNS,
    HISTORY_ATTRIBUTES,
    RULE_ATTRIBUTES,
    RULE_COLUMNS,
    build_baseline,
)
from src.entities.policy_engine.baseline.parameters import HISTORY_RULE_FIRES, MINIMUM_RULES_FIRED
from src.entities.policy_engine.calibration.build import build_thresholds
from src.entities.policy_engine.calibration.parameters import THRESHOLD_ATTRIBUTES
from src.entities.dataset_generator.dataset.build import ATTRIBUTES, LABEL, build_dataset
from src.shared.phases import EVALUATED, WARMUP
from src.shared.tables import write_csv
from src.shared.timing import (
    COLUMNS as TIMING_COLUMNS,
    DISCARDED_ROUNDS,
    TIMED_ROUNDS,
    median_nanoseconds,
    timing_row,
)
from src.entities.dataset_generator.historical_profiles.build import build_profiles
from src.entities.kms.build import build_outcomes
from src.entities.dataset_generator.partition.build import build_partition
from src.pipeline.population import Population, build_population
from src.entities.kms.repository.parameters import KeyRepositorySpecification
from src.entities.scenario_engine.traffic.build import build_traffic
from src.entities.scenario_engine.traffic.parameters import TrafficSpecification

REPOSITORY_SPECIFICATION = KeyRepositorySpecification()
TRAFFIC_SPECIFICATION = TrafficSpecification()
ATTACK_SPECIFICATION = AttackSpecification()

SAMPLE_SEEDS = (1, 15, 30)
SIGMA = 0.5

SYNTHETIC_THRESHOLD = 10.0
"""O limiar de todas as regras de grandeza nas sessoes sinteticas."""

ABOVE = SYNTHETIC_THRESHOLD + 0.0001
"""O menor passo acima do limiar que o CSV representa, com quatro casas."""


# Sessoes sinteticas, para conferir uma regra de cada vez.


def synthetic_thresholds() -> pd.DataFrame:
    return pd.DataFrame({
        "attribute": list(THRESHOLD_ATTRIBUTES),
        "threshold": SYNTHETIC_THRESHOLD,
        "percentile": 99,
        "sessions": 1000,
    })


def quiet_session(number: int) -> dict[str, object]:
    """Uma sessao em que nenhuma regra dispara."""
    return {
        "session_id": f"session_{number:05d}",
        "operator_id": "admin_01",
        "opened_at": "2026-02-02T10:00:00",
        **{attribute: 0.0 for attribute in THRESHOLD_ATTRIBUTES},
        **{attribute: 0 for attribute in HISTORY_ATTRIBUTES},
        LABEL: 0,
    }


def firing(attribute: str) -> float:
    """O valor que faz a regra daquele atributo disparar."""
    is_history = attribute in HISTORY_ATTRIBUTES

    return HISTORY_RULE_FIRES if is_history else ABOVE


def sessions_firing(rule_sets: list[tuple[str, ...]]) -> pd.DataFrame:
    """Uma sessao por conjunto de regras, com exatamente aquelas disparando."""
    rows = []

    for number, attributes in enumerate(rule_sets, start=1):
        row = quiet_session(number)
        row.update({attribute: firing(attribute) for attribute in attributes})
        rows.append(row)

    return pd.DataFrame(rows)


def decisions_for(rule_sets: list[tuple[str, ...]]) -> pd.DataFrame:
    return build_baseline(sessions_firing(rule_sets), synthetic_thresholds()).predictions


@pytest.mark.parametrize("attribute", THRESHOLD_ATTRIBUTES)
def test_a_magnitude_rule_fires_only_strictly_above_its_threshold(attribute: str) -> None:
    """D-080: `>` estrito. No limiar exato a regra fica quieta."""
    sessions = pd.DataFrame([quiet_session(1), quiet_session(2)])
    sessions.loc[0, attribute] = SYNTHETIC_THRESHOLD
    sessions.loc[1, attribute] = ABOVE

    predictions = build_baseline(sessions, synthetic_thresholds()).predictions

    assert list(predictions[f"rule_{attribute}"]) == [0, 1]


@pytest.mark.parametrize("attribute", HISTORY_ATTRIBUTES)
def test_a_history_rule_fires_when_the_attribute_is_one(attribute: str) -> None:
    predictions = decisions_for([(), (attribute,)])

    assert list(predictions[f"rule_{attribute}"]) == [0, 1]


def test_a_quiet_session_fires_nothing() -> None:
    predictions = decisions_for([()])

    assert predictions[list(RULE_COLUMNS)].sum(axis=1).iloc[0] == 0
    assert predictions["predicted"].iloc[0] == 0


@pytest.mark.parametrize("attribute", RULE_ATTRIBUTES)
def test_one_rule_alone_never_alerts(attribute: str) -> None:
    """D-075: nenhuma regra e forte o bastante sozinha, qualquer que seja."""
    predictions = decisions_for([(attribute,)])

    assert predictions["rules_fired"].iloc[0] == 1
    assert predictions["predicted"].iloc[0] == 0


@pytest.mark.parametrize("pair", list(combinations(RULE_ATTRIBUTES, MINIMUM_RULES_FIRED)))
def test_any_two_rules_alert(pair: tuple[str, str]) -> None:
    """D-075: qualquer combinacao de duas serve, sem regra obrigatoria."""
    predictions = decisions_for([pair])

    assert predictions["rules_fired"].iloc[0] == MINIMUM_RULES_FIRED
    assert predictions["predicted"].iloc[0] == 1


def test_every_rule_firing_counts_all_eight() -> None:
    predictions = decisions_for([RULE_ATTRIBUTES])

    assert predictions["rules_fired"].iloc[0] == len(RULE_ATTRIBUTES)
    assert predictions["predicted"].iloc[0] == 1


def test_the_eight_rules_are_the_eight_attributes() -> None:
    """Uma regra por atributo (D-080), nem mais nem menos."""
    assert sorted(RULE_ATTRIBUTES) == sorted(ATTRIBUTES)
    assert len(RULE_COLUMNS) == len(ATTRIBUTES)


# Recusas.


def test_a_thresholds_file_missing_a_rule_is_refused() -> None:
    """Uma linha faltando desligaria uma regra em silencio."""
    incomplete = synthetic_thresholds().iloc[1:]

    with pytest.raises(ValueError, match="thresholds"):
        build_baseline(sessions_firing([()]), incomplete)


def test_a_set_without_label_is_refused() -> None:
    unlabeled = sessions_firing([()]).drop(columns=LABEL)

    with pytest.raises(ValueError, match="rotulo"):
        build_baseline(unlabeled, synthetic_thresholds())


# Sobre o dado de verdade: forma, e o rotulo que nao decide.


@lru_cache(maxsize=None)
def population(seed: int) -> Population:
    return build_population(seed, REPOSITORY_SPECIFICATION)


@lru_cache(maxsize=None)
def traffic(seed: int) -> pd.DataFrame:
    tables = population(seed)

    return build_traffic(seed, tables.operators, tables.keys, TRAFFIC_SPECIFICATION)


@lru_cache(maxsize=None)
def warmup(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """O perfil e os limiares da semente."""
    tables = population(seed)
    outcomes = build_outcomes(traffic(seed), tables.keys, tables.operators, WARMUP)
    log = build_log(traffic(seed), outcomes, WARMUP)
    profiles = build_profiles(log)

    return profiles, build_thresholds(build_dataset(log, profiles, WARMUP))


@lru_cache(maxsize=None)
def holdout(seed: int) -> pd.DataFrame:
    tables = population(seed)
    profiles, _ = warmup(seed)
    campaign = build_attack(
        seed, SIGMA, tables.operators, tables.keys, traffic(seed),
        TRAFFIC_SPECIFICATION, ATTACK_SPECIFICATION,
    )
    outcomes = build_outcomes(
        campaign.requests, tables.keys, tables.operators, EVALUATED
    )
    log = build_log(campaign.requests, outcomes, EVALUATED)
    sessions = build_dataset(log, profiles, EVALUATED, campaign.compromised)

    return build_partition(seed, sessions).holdout


def thresholds(seed: int) -> pd.DataFrame:
    return warmup(seed)[1]


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_one_row_per_holdout_session_in_the_same_order(seed: int) -> None:
    predictions = build_baseline(holdout(seed), thresholds(seed)).predictions

    assert tuple(predictions.columns) == COLUMNS
    assert list(predictions["session_id"]) == list(holdout(seed)["session_id"])


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_counts_and_the_decision_agree_with_the_rule_columns(seed: int) -> None:
    predictions = build_baseline(holdout(seed), thresholds(seed)).predictions
    counted = predictions[list(RULE_COLUMNS)].sum(axis=1)

    assert (predictions["rules_fired"] == counted).all()
    assert (predictions["predicted"] == (counted >= MINIMUM_RULES_FIRED)).all()
    assert set(predictions[list(RULE_COLUMNS)].stack()) <= {0, 1}


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_label_is_carried_but_never_consulted(seed: int) -> None:
    """D-113: zerar ou inverter o rotulo nao muda regra nem decisao."""
    original = holdout(seed)
    decided_columns = list(RULE_COLUMNS) + ["rules_fired", "predicted"]

    reference = build_baseline(original, thresholds(seed)).predictions
    zeroed = original.assign(**{LABEL: 0})
    inverted = original.assign(**{LABEL: 1 - original[LABEL]})

    for relabeled in (zeroed, inverted):
        predictions = build_baseline(relabeled, thresholds(seed)).predictions

        assert predictions[decided_columns].equals(reference[decided_columns])
        assert list(predictions[LABEL]) == list(relabeled[LABEL])


def test_reading_from_csv_decides_the_same_as_from_memory(tmp_path: Path) -> None:
    """O comando isolado le CSV e o orquestrador passa a memoria: mesma decisao.

    Os atributos e os limiares saem arredondados em quatro casas, e e isso que
    garante que uma sessao encostada no limiar caia do mesmo lado pelos dois
    caminhos.
    """
    write_csv(holdout(1), tmp_path / "holdout.csv")
    write_csv(thresholds(1), tmp_path / "thresholds.csv")

    from_csv = build_baseline(
        pd.read_csv(tmp_path / "holdout.csv"), pd.read_csv(tmp_path / "thresholds.csv")
    ).predictions
    from_memory = build_baseline(holdout(1), thresholds(1)).predictions

    write_csv(from_csv, tmp_path / "from_csv.csv")
    write_csv(from_memory, tmp_path / "from_memory.csv")

    assert (tmp_path / "from_csv.csv").read_bytes() == (
        tmp_path / "from_memory.csv"
    ).read_bytes()


def test_the_decision_is_deterministic() -> None:
    first = build_baseline(holdout(1), thresholds(1)).predictions
    second = build_baseline(holdout(1), thresholds(1)).predictions

    assert first.equals(second)


# O cronometro (D-106).


def test_the_timer_discards_the_warmup_and_times_ten_rounds() -> None:
    calls = []

    median_nanoseconds(lambda: calls.append(1))

    assert len(calls) == DISCARDED_ROUNDS + TIMED_ROUNDS


def test_the_timing_row_derives_both_units() -> None:
    """800 sessoes em 800 microssegundos: 1 por sessao, um milhao por segundo."""
    row = timing_row("rules", 800, 800_000)

    assert row["microseconds_per_session"] == 1.0
    assert row["sessions_per_second"] == 1_000_000.0
    assert row["timed_rounds"] == TIMED_ROUNDS


def test_the_baseline_timing_is_one_row_about_the_whole_holdout() -> None:
    timing = build_baseline(holdout(1), thresholds(1)).timing

    assert tuple(timing.columns) == TIMING_COLUMNS
    assert len(timing) == 1
    assert timing["mechanism"].iloc[0] == "rules"
    assert timing["sessions"].iloc[0] == len(holdout(1))
    assert timing["median_nanoseconds"].iloc[0] > 0
