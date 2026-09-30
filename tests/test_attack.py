"""Testes do M3: a campanha de ataque sob credencial comprometida.

Tres invariantes valem mais que as outras, e as tres protegem o desenho
experimental em vez do codigo:

1. **Convergencia em sigma 1.** Toda dimensao interpolada vale exatamente o que
   o M2 usaria para aquele administrador. Se isso deixar de valer, sigma 1
   deixa de ser o piso declarado da varredura sem que nada acuse.
2. **O mesmo administrador nas onze condicoes** (D-011). E o que pareia as
   condicoes e permite atribuir a sigma a diferenca observada.
3. **Nada no `requests.csv` denuncia o atacante.** Nem coluna, nem formato de
   identificador, nem posicao na faixa de numeracao (D-063, D-083).

A primeira e barata e exata: compara dois objetos, sem gerar sessao nenhuma.
As outras duas precisam do arquivo montado.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pytest

from src.scenario_engine.attack.build import (
    COMPROMISED_COLUMNS,
    RUN_COLUMNS,
    AttackOutput,
    administrators_of,
    build_attack,
)
from src.scenario_engine.attack.parameters import AttackSpecification
from src.scenario_engine.attack.stealth import stealth_of
from src.shared.experiment import SEEDS, SIGMAS
from src.shared.phases import EVALUATED, WARMUP, belongs_to
from src.scenario_engine.traffic.build import COLUMNS as REQUEST_COLUMNS
from src.scenario_engine.traffic.build import build_traffic
from src.scenario_engine.traffic.operators import read_operators
from src.scenario_engine.traffic.parameters import TrafficSpecification
from src.scenario_engine.traffic.regimes import REGIMES
from src.pipeline.population import Population, build_population
from src.kms.repository.parameters import KeyRepositorySpecification

REPOSITORY_SPECIFICATION = KeyRepositorySpecification()
TRAFFIC_SPECIFICATION = TrafficSpecification()
ATTACK_SPECIFICATION = AttackSpecification()

CUSTODY = REGIMES["occasional_custody"]

SAMPLE_SEEDS = (1, 7, 15, 23, 30)
"""Sementes das conferencias caras, que montam a campanha inteira."""

SAMPLE_SIGMAS = (0.0, 0.5, 1.0)


@lru_cache(maxsize=None)
def population(seed: int) -> Population:
    return build_population(seed, REPOSITORY_SPECIFICATION)


@lru_cache(maxsize=None)
def traffic(seed: int) -> pd.DataFrame:
    tables = population(seed)

    return build_traffic(
        seed, tables.operators, tables.keys, TRAFFIC_SPECIFICATION
    )


@lru_cache(maxsize=None)
def attack(seed: int, sigma: float) -> AttackOutput:
    tables = population(seed)

    return build_attack(
        seed, sigma, tables.operators, tables.keys, traffic(seed),
        TRAFFIC_SPECIFICATION, ATTACK_SPECIFICATION,
    )


def compromised_rows_of(output: AttackOutput) -> pd.DataFrame:
    marked = set(output.compromised["session_id"])

    return output.requests[output.requests["session_id"].isin(marked)]


# Convergencia: o piso da varredura. Puro, sem gerar sessao.


def test_sigma_one_reproduces_the_traffic_parameters_exactly() -> None:
    """D-082: em sigma 1 nao ha aproximacao, ha igualdade.

    Igualdade exata e nao `pytest.approx` de proposito. A interpolacao usa a
    forma de dois pesos justamente para que o extremo saia exato; trocar por
    `ostensive + sigma * (furtive - ostensive)` devolveria
    0,005000000000000004 e este teste acusaria.
    """
    furtive = stealth_of(
        1.0, CUSTODY, TRAFFIC_SPECIFICATION, ATTACK_SPECIFICATION
    )

    assert furtive.seconds_between_requests == CUSTODY.seconds_between_requests
    assert furtive.requests_range == CUSTODY.requests_range
    assert furtive.distinct_keys_range == TRAFFIC_SPECIFICATION.distinct_keys_range
    assert furtive.stale_scope_rate == TRAFFIC_SPECIFICATION.stale_scope_rate
    assert (
        furtive.absent_identifier_rate
        == TRAFFIC_SPECIFICATION.absent_identifier_rate
    )
    assert furtive.atypical_hour_chance == 0.0
    assert furtive.novel_address_chance == 0.0


def test_sigma_zero_reproduces_the_ostensive_parameters_exactly() -> None:
    ostensive = stealth_of(
        0.0, CUSTODY, TRAFFIC_SPECIFICATION, ATTACK_SPECIFICATION
    )

    assert (
        ostensive.seconds_between_requests
        == ATTACK_SPECIFICATION.ostensive_request_interval
    )
    assert ostensive.requests_range == ATTACK_SPECIFICATION.ostensive_requests_range
    assert ostensive.atypical_hour_chance == 1.0
    assert ostensive.novel_address_chance == 1.0


@pytest.mark.parametrize("sigma", SIGMAS)
def test_every_dimension_moves_monotonically_toward_the_furtive_end(
    sigma: float,
) -> None:
    """Sigma maior nunca deixa uma dimensao mais ostensiva.

    Sem isto a varredura deixaria de ser uma escala: duas condicoes poderiam
    trocar de ordem numa dimensao e a leitura "quanto maior sigma, mais dificil
    de pegar" perderia o sentido.
    """
    current = stealth_of(sigma, CUSTODY, TRAFFIC_SPECIFICATION, ATTACK_SPECIFICATION)
    ostensive = stealth_of(0.0, CUSTODY, TRAFFIC_SPECIFICATION, ATTACK_SPECIFICATION)
    furtive = stealth_of(1.0, CUSTODY, TRAFFIC_SPECIFICATION, ATTACK_SPECIFICATION)

    assert (
        furtive.seconds_between_requests
        >= current.seconds_between_requests
        >= ostensive.seconds_between_requests
    )
    assert (
        ostensive.requests_range[1]
        >= current.requests_range[1]
        >= furtive.requests_range[1]
    )
    assert (
        ostensive.stale_scope_rate
        >= current.stale_scope_rate
        >= furtive.stale_scope_rate
    )
    assert current.atypical_hour_chance == pytest.approx(1.0 - sigma)


def test_a_batch_regime_is_refused() -> None:
    """So um operador de ritmo humano pode ser personificado.

    O administrador comprometido e sempre `occasional_custody` (D-010), entao
    receber um regime de lote aqui significa que alguem trocou o alvo, e falhar
    e melhor que interpolar uma janela horaria que o lote nao tem.
    """
    with pytest.raises(ValueError, match="ritmo de chegada"):
        stealth_of(
            0.5, REGIMES["periodic_batch"], TRAFFIC_SPECIFICATION,
            ATTACK_SPECIFICATION,
        )


# O pareamento entre condicoes, que e o que a comparacao de sigma assume.


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_same_admin_is_compromised_across_every_sigma(seed: int) -> None:
    """D-011: sigma nao pode trocar o alvo, ou nada seria atribuivel a sigma."""
    chosen = {
        attack(seed, sigma).run["compromised_admin"].iloc[0] for sigma in SIGMAS
    }

    assert len(chosen) == 1


@pytest.mark.parametrize("seed", SEEDS)
def test_the_compromised_admin_is_an_administrator_of_this_seed(seed: int) -> None:
    """O alvo sai dos oito administradores da propria populacao (D-010, D-036)."""
    tables = population(seed)
    operators = read_operators(tables.operators)

    holders = {operator.operator_id for operator in administrators_of(operators)}
    chosen = attack(seed, 0.5).run["compromised_admin"].iloc[0]

    assert len(holders) == 8
    assert chosen in holders


def test_the_target_varies_between_seeds() -> None:
    """Se a semente nao mudasse o alvo, o sorteio seria decorativo."""
    chosen = {attack(seed, 0.5).run["compromised_admin"].iloc[0] for seed in SEEDS}

    assert len(chosen) > 1


# Forma dos arquivos, e o que eles nao podem carregar.


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
@pytest.mark.parametrize("sigma", SAMPLE_SIGMAS)
def test_requests_keep_the_seven_columns_and_none_marks_the_attacker(
    seed: int, sigma: float
) -> None:
    """D-066 e D-063: o arquivo mesclado e indistinguivel em forma do do M2."""
    output = attack(seed, sigma)
    forbidden = {"compromised", "label", "is_attack", "sigma", "outcome"}

    assert tuple(output.requests.columns) == REQUEST_COLUMNS
    assert forbidden.isdisjoint(set(output.requests.columns))


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
@pytest.mark.parametrize("sigma", SAMPLE_SIGMAS)
def test_the_campaign_has_the_same_size_in_every_condition(
    seed: int, sigma: float
) -> None:
    """D-081: sigma move comportamento, nunca a contagem de positivas."""
    output = attack(seed, sigma)

    assert len(output.compromised) == ATTACK_SPECIFICATION.campaign_sessions
    assert tuple(output.compromised.columns) == COMPROMISED_COLUMNS
    assert tuple(output.run.columns) == RUN_COLUMNS


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
@pytest.mark.parametrize("sigma", SAMPLE_SIGMAS)
def test_the_session_identifier_does_not_announce_the_label(
    seed: int, sigma: float
) -> None:
    """D-083: renumerar tudo e o que impede o identificador de virar rotulo.

    Se as sessoes do atacante ficassem no fim da faixa, bastaria abrir o
    arquivo e olhar o numero. O teste exige que elas estejam espalhadas: a
    primeira comprometida cai na metade inicial e a ultima na metade final.
    """
    output = attack(seed, sigma)

    order = {
        session: position
        for position, session in enumerate(output.requests["session_id"].unique())
    }
    positions = sorted(order[session] for session in output.compromised["session_id"])
    total = len(order)

    assert positions[0] < total / 2
    assert positions[-1] > total / 2
    assert output.compromised["session_id"].is_unique


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
@pytest.mark.parametrize("sigma", SAMPLE_SIGMAS)
def test_the_campaign_stays_inside_the_evaluated_weeks(
    seed: int, sigma: float
) -> None:
    """D-048: ataque no aquecimento contaminaria perfil e limiares."""
    output = attack(seed, sigma)

    assert belongs_to(EVALUATED, output.requests["timestamp"]).all()
    assert not belongs_to(WARMUP, output.requests["timestamp"]).any()


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
@pytest.mark.parametrize("sigma", SAMPLE_SIGMAS)
def test_every_compromised_session_belongs_to_the_compromised_admin(
    seed: int, sigma: float
) -> None:
    """O atacante nao tem identidade propria: age sob a credencial do alvo."""
    output = attack(seed, sigma)
    target = output.run["compromised_admin"].iloc[0]

    assert set(compromised_rows_of(output)["operator_id"]) == {target}


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
@pytest.mark.parametrize("sigma", SAMPLE_SIGMAS)
def test_the_legitimate_traffic_of_the_evaluated_weeks_survives_intact(
    seed: int, sigma: float
) -> None:
    """O M3 acrescenta, nunca altera nem descarta o que o M2 gerou.

    Comparado sem `event_id` e sem `session_id`, que sao renumerados de
    proposito (D-083). O resto de cada linha tem de sobreviver igual.
    """
    output = attack(seed, sigma)
    loose = ["event_id", "session_id"]

    expected = traffic(seed)[belongs_to(EVALUATED, traffic(seed)["timestamp"])]
    marked = set(output.compromised["session_id"])
    produced = output.requests[~output.requests["session_id"].isin(marked)]

    assert len(produced) == len(expected)
    assert (
        produced.drop(columns=loose).reset_index(drop=True)
        .equals(expected.drop(columns=loose).reset_index(drop=True))
    )


# Determinismo, e o que separa os dois extremos.


@pytest.mark.parametrize("seed", (1, 30))
def test_the_same_input_gives_the_same_campaign(seed: int) -> None:
    tables = population(seed)

    first = build_attack(
        seed, 0.5, tables.operators, tables.keys, traffic(seed),
        TRAFFIC_SPECIFICATION, ATTACK_SPECIFICATION,
    )
    second = build_attack(
        seed, 0.5, tables.operators, tables.keys, traffic(seed),
        TRAFFIC_SPECIFICATION, ATTACK_SPECIFICATION,
    )

    assert first.requests.equals(second.requests)
    assert first.compromised.equals(second.compromised)


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_ostensive_end_breaks_the_legitimate_ceiling_and_the_furtive_does_not(
    seed: int,
) -> None:
    """O acoplamento que a D-080 registrou, agora conferido dos dois lados.

    Em sigma 0 a campanha e mais larga que qualquer sessao legitima daquela
    semente; em sigma 1 ela cabe dentro do que o trafego legitimo produz, que e
    o que faz o piso ser piso.

    **O teto sai da propria semente, e nao de um numero fixo** (D-098). Ele era
    15 enquanto a amplitude legitima tinha teto rigido, e fixa-lo aqui faria o
    teste falhar por mudanca legitima da distribuicao em vez de por regressao,
    que foi exatamente o que aconteceu quando a cauda entrou.
    """
    legitimate = traffic(seed)
    ceiling = legitimate.groupby("session_id")["key_id"].nunique().max()

    ostensive = compromised_rows_of(attack(seed, 0.0))
    furtive = compromised_rows_of(attack(seed, 1.0))

    widest_ostensive = ostensive.groupby("session_id")["key_id"].nunique().median()
    widest_furtive = furtive.groupby("session_id")["key_id"].nunique().max()

    assert widest_ostensive > ceiling
    assert widest_furtive <= ceiling
