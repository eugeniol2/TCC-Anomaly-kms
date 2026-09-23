"""Testes do M6, do M7 e do M8: perfil, conjunto de sessoes e limiares.

Sao os tres modulos do aquecimento e da calibracao. O que eles produzem nao e
resultado nenhum — e a base contra a qual o periodo avaliado sera lido — e por
isso os testes aqui guardam sobretudo **o que nao pode vazar de um periodo para
outro**.

Quatro invariantes valem mais que as outras:

1. **O perfil sai so das semanas 1 e 2** (D-044). Perfil que enxergasse a
   semana 3 acoplaria a janela horaria ao limiar, que a D-054 mantem separados.
2. **Toda sessao das semanas 1 e 2 e tipica, por construcao** (D-073). A janela
   e o minimo e o maximo daquelas sessoes, entao nenhuma delas pode cair fora.
   E o teste que pega inversao de sinal ou janela lida do periodo errado.
3. **O `sessions.csv` do aquecimento nao tem rotulo** (D-063). Se a coluna nao
   existe, o M8 nao pode usa-la por engano.
4. **O limiar sai so da semana 3** (D-043), e nunca do holdout.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pytest

from src.attack.build import build_attack
from src.attack.parameters import AttackSpecification
from src.audit_logger.build import build_log
from src.calibration.build import COLUMNS as THRESHOLD_COLUMNS
from src.calibration.build import build_thresholds, calibration_period
from src.calibration.parameters import PERCENTILE, THRESHOLD_ATTRIBUTES
from src.dataset.build import ATTRIBUTES, IDENTIFIERS, LABEL, build_dataset
from src.globals.experiment import SEEDS
from src.globals.phases import (
    CALIBRATION_WEEK,
    EVALUATED,
    PROFILE_WEEKS,
    WARMUP,
    day_after_last_of,
    first_day_of,
)
from src.globals.tables import MULTIVALUE_SEPARATOR
from src.historical_profiles.build import COLUMNS as PROFILE_COLUMNS
from src.historical_profiles.build import build_profiles, hour_of_day
from src.kms.build import build_outcomes
from src.population.build import Population, build_population
from src.population.parameters import KeyRepositorySpecification
from src.traffic.build import build_traffic
from src.traffic.parameters import TrafficSpecification

REPOSITORY_SPECIFICATION = KeyRepositorySpecification()
TRAFFIC_SPECIFICATION = TrafficSpecification()
ATTACK_SPECIFICATION = AttackSpecification()

SAMPLE_SEEDS = (1, 7, 15, 23, 30)
OPERATOR_COUNT = 44


@lru_cache(maxsize=None)
def population(seed: int) -> Population:
    return build_population(seed, REPOSITORY_SPECIFICATION)


@lru_cache(maxsize=None)
def traffic(seed: int) -> pd.DataFrame:
    tables = population(seed)

    return build_traffic(seed, tables.operators, tables.keys, TRAFFIC_SPECIFICATION)


@lru_cache(maxsize=None)
def warmup_log(seed: int) -> pd.DataFrame:
    tables = population(seed)
    outcomes = build_outcomes(traffic(seed), tables.keys, tables.operators, WARMUP)

    return build_log(traffic(seed), outcomes, WARMUP)


@lru_cache(maxsize=None)
def profiles(seed: int) -> pd.DataFrame:
    return build_profiles(warmup_log(seed))


@lru_cache(maxsize=None)
def warmup_sessions(seed: int) -> pd.DataFrame:
    return build_dataset(warmup_log(seed), profiles(seed), WARMUP)


@lru_cache(maxsize=None)
def thresholds(seed: int) -> pd.DataFrame:
    return build_thresholds(warmup_sessions(seed))


@lru_cache(maxsize=None)
def evaluated_sessions(seed: int, sigma: float) -> pd.DataFrame:
    tables = population(seed)
    campaign = build_attack(
        seed, sigma, tables.operators, tables.keys, traffic(seed),
        TRAFFIC_SPECIFICATION, ATTACK_SPECIFICATION,
    )
    outcomes = build_outcomes(
        campaign.requests, tables.keys, tables.operators, EVALUATED
    )
    log = build_log(campaign.requests, outcomes, EVALUATED)

    return build_dataset(log, profiles(seed), EVALUATED, campaign.compromised)


def within_profile_weeks(sessions: pd.DataFrame) -> pd.DataFrame:
    first_week, last_week = PROFILE_WEEKS

    opens = pd.Timestamp(first_day_of(first_week))
    closes = pd.Timestamp(day_after_last_of(last_week))
    instants = pd.to_datetime(sessions["opened_at"])

    return sessions[(instants >= opens) & (instants < closes)]


# M6: o perfil historico.


@pytest.mark.parametrize("seed", SEEDS)
def test_every_operator_has_a_profile(seed: int) -> None:
    """Operador sem perfil deixaria o M7 sem contra o que comparar.

    Nesta escala nao acontece — todo operador abre sessao nas duas primeiras
    semanas —, e o teste existe para que deixar de acontecer seja visivel se a
    escala da populacao ou o ritmo mudarem (D-006, D-058).
    """
    assert tuple(profiles(seed).columns) == PROFILE_COLUMNS
    assert len(profiles(seed)) == OPERATOR_COUNT
    assert profiles(seed)["operator_id"].is_unique


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_profile_ignores_the_calibration_week(seed: int) -> None:
    """D-044 e D-054: a janela sai das semanas 1 e 2, nunca da semana 3.

    Conferido pelo efeito, e nao pela leitura do codigo. A semana 3 traz
    horarios novos, entao uma janela calculada sobre o aquecimento inteiro e
    **mais larga ou igual** para todo operador, e estritamente mais larga para
    alguns. Se as duas versoes coincidissem em todos, o recorte nao estaria
    acontecendo.
    """
    log = warmup_log(seed)
    instants = pd.to_datetime(log["timestamp"])

    week_three = log[instants >= pd.Timestamp(first_day_of(CALIBRATION_WEEK))]
    assert len(week_three) > 0

    produced = profiles(seed).set_index("operator_id")
    relabelled = log.assign(
        timestamp=instants.where(
            instants < pd.Timestamp(first_day_of(CALIBRATION_WEEK)),
            instants - pd.Timedelta(days=7),
        ).dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    widened = build_profiles(relabelled).set_index("operator_id")

    narrower = produced["window_closes_at"] - produced["window_opens_at"]
    wider = widened["window_closes_at"] - widened["window_opens_at"]

    assert (wider >= narrower - 1e-9).all()
    assert (wider > narrower + 1e-9).any()


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_observed_addresses_are_a_subset_of_the_generated_ones(seed: int) -> None:
    """O perfil observa, nao inventa.

    E a diferenca entre os dois conjuntos que produz origem inedita legitima no
    periodo avaliado: um endereco habitual pouco usado pode faltar no
    aquecimento (D-040).
    """
    generated = {
        row.operator_id: set(row.usual_ips.split(MULTIVALUE_SEPARATOR))
        for row in population(seed).operators.itertuples()
    }

    for row in profiles(seed).itertuples():
        seen = set(row.observed_ips.split(MULTIVALUE_SEPARATOR))

        assert seen
        assert seen <= generated[row.operator_id]


def test_some_generated_address_is_missing_from_some_profile() -> None:
    """Se todo endereco aparecesse, `new_source_ip` nasceria morto.

    A D-040 encomendou o decaimento geometrico justamente para que o quarto
    endereco tenha ~91 % de chance de faltar no aquecimento.
    """
    incomplete = 0

    for seed in SAMPLE_SEEDS:
        generated = {
            row.operator_id: set(row.usual_ips.split(MULTIVALUE_SEPARATOR))
            for row in population(seed).operators.itertuples()
        }
        for row in profiles(seed).itertuples():
            seen = set(row.observed_ips.split(MULTIVALUE_SEPARATOR))
            incomplete += len(seen) < len(generated[row.operator_id])

    assert incomplete > 0


# M7: o conjunto de sessoes.


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_warmup_dataset_carries_no_label(seed: int) -> None:
    """D-063: coluna que nao existe no arquivo nao pode ser usada por engano."""
    assert tuple(warmup_sessions(seed).columns) == IDENTIFIERS + ATTRIBUTES
    assert LABEL not in warmup_sessions(seed).columns


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_evaluated_dataset_carries_exactly_the_campaign_as_positives(
    seed: int,
) -> None:
    """Cada sessao comprometida e uma positiva, e sao 58 (D-061, D-081)."""
    sessions = evaluated_sessions(seed, 0.5)

    assert tuple(sessions.columns) == IDENTIFIERS + ATTRIBUTES + (LABEL,)
    assert int(sessions[LABEL].sum()) == ATTACK_SPECIFICATION.campaign_sessions
    assert set(sessions[LABEL]) == {0, 1}


def test_the_warmup_refuses_a_label_and_the_evaluated_demands_one() -> None:
    """A assimetria da D-063, imposta na assinatura e nao na disciplina."""
    log = warmup_log(1)

    with pytest.raises(ValueError, match="exige"):
        build_dataset(log, profiles(1), EVALUATED, None)

    with pytest.raises(ValueError, match="nao recebe rotulo"):
        build_dataset(log, profiles(1), WARMUP, pd.DataFrame({"session_id": []}))


@pytest.mark.parametrize("seed", SEEDS)
def test_no_session_of_the_profile_weeks_is_atypical(seed: int) -> None:
    """A invariante que prova que a janela foi lida do periodo certo.

    A janela e o minimo e o maximo das sessoes das semanas 1 e 2 (D-073), entao
    **nenhuma delas pode cair fora**, e nenhum endereco delas pode ser inedito.
    Janela lida do periodo errado, ou comparacao invertida, quebra isto na hora.
    """
    inside = within_profile_weeks(warmup_sessions(seed))

    assert len(inside) > 0
    assert (inside["atypical_hour"] == 0).all()
    assert (inside["new_source_ip"] == 0).all()


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_one_row_per_session_and_no_identifier_among_the_attributes(
    seed: int,
) -> None:
    """D-015: identificador fica no arquivo, fora da lista de atributos."""
    sessions = warmup_sessions(seed)
    log = warmup_log(seed)

    assert len(sessions) == log["session_id"].nunique()
    assert sessions["session_id"].is_unique
    assert set(IDENTIFIERS).isdisjoint(set(ATTRIBUTES))


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_rate_attributes_are_finite_and_consistent(seed: int) -> None:
    """Sessao de duracao zero nao pode produzir taxa infinita.

    O piso de um segundo da `SHORTEST_MEASURABLE_MINUTES` existe para isso, e
    e a unica coisa entre o conjunto e um `inf` que quebraria o treino.
    """
    sessions = warmup_sessions(seed)

    for attribute in ATTRIBUTES:
        assert sessions[attribute].notna().all()
        assert (sessions[attribute] >= 0).all()

    assert sessions["requests_per_minute"].map(float).map(
        lambda rate: rate != float("inf")
    ).all()
    assert (sessions["failures_per_event"] <= 1).all()
    assert (sessions["denials_per_event"] <= sessions["failures_per_event"]).all()
    assert (sessions["distinct_keys"] <= sessions["events"]).all()


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_opening_instant_matches_the_first_event_of_the_session(
    seed: int,
) -> None:
    """`opened_at` e o primeiro evento, e e o que o M8 usa para achar a semana 3."""
    log = warmup_log(seed)
    expected = log.groupby("session_id")["timestamp"].min()

    produced = warmup_sessions(seed).set_index("session_id")["opened_at"]

    assert produced.sort_index().equals(expected.sort_index())


# M8: os limiares.


@pytest.mark.parametrize("seed", SEEDS)
def test_thresholds_cover_the_six_magnitude_rules(seed: int) -> None:
    """Seis, e nao oito: as duas de historico ja sao binarias (D-080)."""
    produced = thresholds(seed)

    assert tuple(produced.columns) == THRESHOLD_COLUMNS
    assert tuple(produced["attribute"]) == THRESHOLD_ATTRIBUTES
    assert (produced["percentile"] == PERCENTILE).all()
    assert (produced["threshold"] > 0).all()


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_calibration_reads_the_third_week_and_nothing_else(seed: int) -> None:
    """D-043: o limiar sai da semana 3, que e limpa e anterior ao ataque."""
    selected = calibration_period(warmup_sessions(seed))
    instants = pd.to_datetime(selected["opened_at"])

    opens = pd.Timestamp(first_day_of(CALIBRATION_WEEK))
    closes = pd.Timestamp(day_after_last_of(CALIBRATION_WEEK))

    assert len(selected) > 0
    assert len(selected) < len(warmup_sessions(seed))
    assert (instants >= opens).all()
    assert (instants < closes).all()
    assert (thresholds(seed)["sessions"] == len(selected)).all()


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_each_rule_fires_on_about_one_percent_of_the_calibration_week(
    seed: int,
) -> None:
    """O percentil 99 fixa o alarme falso por construcao, e o teste confere.

    Nao e resultado de deteccao: a semana 3 e limpa, entao toda marcacao ali e
    falso positivo por definicao, e o numero e propriedade do percentil.
    """
    calibration = calibration_period(warmup_sessions(seed))
    limits = thresholds(seed).set_index("attribute")["threshold"]

    for attribute in THRESHOLD_ATTRIBUTES:
        fired = (calibration[attribute] > limits[attribute]).mean()

        assert fired <= 0.02


@pytest.mark.parametrize("seed", (1, 30))
def test_the_same_warmup_gives_the_same_thresholds(seed: int) -> None:
    """Nenhum dos tres modulos sorteia nada: os tres sao deterministicos."""
    assert build_profiles(warmup_log(seed)).equals(profiles(seed))
    assert build_dataset(warmup_log(seed), profiles(seed), WARMUP).equals(
        warmup_sessions(seed)
    )
    assert build_thresholds(warmup_sessions(seed)).equals(thresholds(seed))


def test_an_evaluated_dataset_has_no_calibration_week() -> None:
    """Passar o conjunto errado ao M8 falha, em vez de calibrar sobre o ataque."""
    with pytest.raises(ValueError, match="fase avaliada"):
        build_thresholds(evaluated_sessions(1, 0.5))
