"""Testes do M9: a particao do conjunto avaliado em treino e holdout.

A particao nao produz resultado nenhum, mas decide contra o que todo resultado
vai ser medido. Por isso os testes aqui guardam quatro coisas:

1. **Cada sessao cai de um lado so, e todas caem** (D-018, D-061). E o que
   torna estrutural a ausencia do vazamento que a D-029 procura.
2. **A contagem de positivas e a declarada** (D-070, D-081): 23 das 58.
3. **A divisao e a mesma nas onze condicoes de uma semente** (D-102). As
   legitimas sao as mesmas sessoes; as do atacante, as mesmas posicoes da
   campanha.
4. **O sorteio nao olha o conteudo.** Trocar os atributos ou a ordem das linhas
   nao muda quem vai para onde.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pytest
from numpy.random import PCG64, Generator, SeedSequence

from src.scenario_engine.attack.build import build_attack
from src.scenario_engine.attack.parameters import AttackSpecification
from src.audit_logger.build import build_log
from src.dataset_generator.dataset.build import ATTRIBUTES, LABEL, build_dataset
from src.shared.phases import EVALUATED, WARMUP
from src.shared.rng import ATTACK, POPULATION, TRAFFIC, stream
from src.dataset_generator.historical_profiles.build import build_profiles
from src.kms.build import build_outcomes
from src.dataset_generator.partition.build import build_partition, holdout_count
from src.pipeline.population import Population, build_population
from src.kms.repository.parameters import KeyRepositorySpecification
from src.scenario_engine.traffic.build import build_traffic
from src.scenario_engine.traffic.parameters import TrafficSpecification

REPOSITORY_SPECIFICATION = KeyRepositorySpecification()
TRAFFIC_SPECIFICATION = TrafficSpecification()
ATTACK_SPECIFICATION = AttackSpecification()

SAMPLE_SEEDS = (1, 15, 30)
SAMPLE_SIGMAS = (0.0, 0.5, 1.0)

CAMPAIGN_SESSIONS = 58
HOLDOUT_POSITIVES = 23


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

    return build_dataset(
        log, build_profiles(warmup_log(seed)), EVALUATED, campaign.compromised
    )


def legitimate_keys(side: pd.DataFrame) -> set[tuple[str, str]]:
    """As sessoes legitimas de um lado, pelo par que nao muda com sigma."""
    legitimate = side[side[LABEL] == 0]

    return set(zip(legitimate["operator_id"], legitimate["opened_at"]))


def campaign_positions(sessions: pd.DataFrame, side: pd.DataFrame) -> set[int]:
    """Em que posicoes da campanha estao as sessoes do atacante de um lado."""
    campaign = sessions[sessions[LABEL] == 1].sort_values("opened_at")
    position_of = {session: rank for rank, session in enumerate(campaign["session_id"])}

    return {position_of[session] for session in side.loc[side[LABEL] == 1, "session_id"]}


# Cada sessao de um lado so, e na proporcao declarada.


@pytest.mark.parametrize("sigma", SAMPLE_SIGMAS)
@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_every_session_falls_on_exactly_one_side(seed: int, sigma: float) -> None:
    sessions = evaluated_sessions(seed, sigma)
    partition = build_partition(seed, sessions)

    train = set(partition.train["session_id"])
    holdout = set(partition.holdout["session_id"])

    assert train.isdisjoint(holdout)
    assert train | holdout == set(sessions["session_id"])


@pytest.mark.parametrize("sigma", SAMPLE_SIGMAS)
@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_each_class_is_split_in_the_declared_share(seed: int, sigma: float) -> None:
    """D-070 dentro de cada classe: 23 das 58 positivas, e 40 % das negativas."""
    sessions = evaluated_sessions(seed, sigma)
    holdout = build_partition(seed, sessions).holdout

    negatives = int((sessions[LABEL] == 0).sum())

    assert int(holdout[LABEL].sum()) == HOLDOUT_POSITIVES
    assert int((holdout[LABEL] == 0).sum()) == holdout_count(negatives)


def test_the_holdout_count_is_the_nearest_integer() -> None:
    assert holdout_count(CAMPAIGN_SESSIONS) == HOLDOUT_POSITIVES
    assert holdout_count(10) == 4
    assert holdout_count(3) == 1
    assert holdout_count(0) == 0


# A mesma divisao nas onze condicoes (D-102).


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_legitimate_holdout_is_the_same_in_every_condition(seed: int) -> None:
    holdouts = [
        legitimate_keys(build_partition(seed, evaluated_sessions(seed, sigma)).holdout)
        for sigma in SAMPLE_SIGMAS
    ]

    assert holdouts[0] == holdouts[1] == holdouts[2]


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_attack_holdout_is_the_same_campaign_positions(seed: int) -> None:
    """A k-esima sessao do atacante cai do mesmo lado em todo sigma."""
    positions = []

    for sigma in SAMPLE_SIGMAS:
        sessions = evaluated_sessions(seed, sigma)
        holdout = build_partition(seed, sessions).holdout
        positions.append(campaign_positions(sessions, holdout))

    assert positions[0] == positions[1] == positions[2]


def test_different_seeds_give_different_partitions() -> None:
    """O sorteio depende da semente: nao e a mesma divisao para as 30 replicas."""
    sessions = evaluated_sessions(1, 0.5)

    first = build_partition(1, sessions).holdout
    second = build_partition(2, sessions).holdout

    assert set(first["session_id"]) != set(second["session_id"])


# O sorteio nao olha o conteudo.


def test_the_attributes_do_not_decide_the_side() -> None:
    """Zerar os atributos nao muda quem vai para o holdout."""
    sessions = evaluated_sessions(1, 0.5)
    blanked = sessions.assign(**{attribute: 0 for attribute in ATTRIBUTES})

    original = build_partition(1, sessions).holdout
    without_content = build_partition(1, blanked).holdout

    assert list(original["session_id"]) == list(without_content["session_id"])


def test_the_row_order_does_not_decide_the_side() -> None:
    sessions = evaluated_sessions(1, 0.5)
    shuffled = sessions.sample(frac=1.0, random_state=0)

    assert build_partition(1, sessions).holdout.equals(
        build_partition(1, shuffled).holdout
    )


def test_the_partition_is_deterministic() -> None:
    sessions = evaluated_sessions(1, 0.5)

    first = build_partition(1, sessions)
    second = build_partition(1, sessions)

    assert first.train.equals(second.train)
    assert first.holdout.equals(second.holdout)


# O formato da saida.


def test_both_sides_keep_the_columns_and_the_order_of_the_sessions_file() -> None:
    """Mesmas colunas do `sessions.csv`, e em ordem cronologica.

    Se as positivas fossem agrupadas no fim do arquivo, a posicao da linha
    anunciaria o rotulo.
    """
    sessions = evaluated_sessions(1, 0.5)
    partition = build_partition(1, sessions)

    for side in partition:
        assert list(side.columns) == list(sessions.columns)
        assert side["session_id"].is_monotonic_increasing


def test_the_warmup_sessions_are_refused() -> None:
    """So o periodo avaliado tem rotulo, e so ele se particiona."""
    warmup = build_dataset(warmup_log(1), build_profiles(warmup_log(1)), WARMUP)

    with pytest.raises(ValueError, match="rotulo"):
        build_partition(1, warmup)


# O quarto fluxo nao moveu os tres primeiros.


@pytest.mark.parametrize("subsystem", (POPULATION, TRAFFIC, ATTACK))
@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_partition_stream_did_not_move_the_older_ones(
    seed: int, subsystem: int
) -> None:
    """Os tres fluxos da D-003 sao os de quando havia so tres (D-102).

    Se o quarto fluxo tivesse deslocado os outros, todo dado gerado antes da
    particao mudaria sem aviso.
    """
    three_streams = SeedSequence(seed).spawn(3)
    before = Generator(PCG64(three_streams[subsystem])).random(8)

    assert list(stream(seed, subsystem).random(8)) == list(before)
