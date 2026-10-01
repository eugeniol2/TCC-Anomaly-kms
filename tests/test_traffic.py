"""Testes do M2: determinismo e as invariantes de que os modulos seguintes dependem.

O M4 decide o desfecho comparando o escopo da chave com os escopos do operador.
Se o M2 enderecar chave errada em volume errado, o desfecho muda, o rotulo muda
junto, e nao ha erro em lugar nenhum, so um resultado diferente. E isso que
estes testes guardam.

Duas invariantes valem mais que as outras. A primeira e que **nenhuma coluna
diz o desfecho** (D-013): se o gerador o escrevesse, o rotulo seria inventado em
vez de derivado da politica. A segunda e que **os dois desvios da D-056
existem**: sem eles, negacao por politica e identificador inexistente so
poderiam vir do atacante, e cada um viraria separador trivial.

As invariantes rodam nas 30 sementes da grade. Construir o trafego custa cerca
de 0,4 s por semente, entao a suite deste modulo leva alguns segundos, e o cache
garante uma construcao por semente, nao uma por teste.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.shared.experiment import SEEDS
from src.shared.tables import MULTIVALUE_SEPARATOR
from src.pipeline.population import Population, build_population
from src.entities.kms.repository.parameters import KeyRepositorySpecification
from src.entities.scenario_engine.traffic.build import COLUMNS, build_traffic
from src.entities.scenario_engine.traffic.operations import OPERATIONS
from src.entities.scenario_engine.attack.parameters import AttackSpecification
from src.entities.scenario_engine.traffic.parameters import (
    BUSINESS_WEEKDAYS,
    IDENTIFIER_HEX_DIGITS,
    IDENTIFIER_PREFIX,
    LONG_SESSION_CHANCE,
    TrafficSpecification,
)
from src.entities.scenario_engine.traffic.regimes import REGIMES, ScheduledRhythm

SPECIFICATION = TrafficSpecification()
REPOSITORY_SPECIFICATION = KeyRepositorySpecification()

REFERENCE_SEED = 1
REFERENCE_DIRECTORY = Path(__file__).parent / "reference"
REFERENCE_DIGEST = REFERENCE_DIRECTORY / "seed-01-requests.sha256"

DETERMINISM_SEEDS = (1, 15, 30)
"""Sementes que sao construidas duas vezes.

As invariantes cobrem as 30; a repeticao cobre tres. Reconstruir as 30 dobraria
o custo da suite para provar de novo o que o `SeedSequence` garante por
construcao.
"""


@lru_cache(maxsize=None)
def population(seed: int) -> Population:
    """Populacao de uma semente, reaproveitada entre os testes."""
    return build_population(seed, REPOSITORY_SPECIFICATION)


@lru_cache(maxsize=None)
def traffic(seed: int) -> pd.DataFrame:
    """Trafego de uma semente, reaproveitado entre os testes."""
    tables = population(seed)

    return build_traffic(seed, tables.operators, tables.keys, SPECIFICATION)


def csv_text(frame: pd.DataFrame) -> str:
    """O CSV exatamente como `write_csv` o grava."""
    return frame.to_csv(index=False, lineterminator="\n")


def scopes_by_operator(operators: pd.DataFrame) -> dict[str, set[str]]:
    """De cada operador para os escopos que ele detem."""
    return {
        row.operator_id: set(row.scopes.split(MULTIVALUE_SEPARATOR))
        for row in operators.itertuples()
    }


def addresses_by_operator(operators: pd.DataFrame) -> dict[str, list[str]]:
    """De cada operador para suas origens habituais, na ordem da tabela."""
    return {
        row.operator_id: row.usual_ips.split(MULTIVALUE_SEPARATOR)
        for row in operators.itertuples()
    }


def regime_by_operator(operators: pd.DataFrame) -> dict[str, str]:
    """De cada operador para o regime dele."""
    return dict(zip(operators["operator_id"], operators["regime"]))


def session_openings(requests: pd.DataFrame) -> pd.DataFrame:
    """A primeira requisicao de cada sessao, que e quando ela abriu."""
    opening = requests.groupby("session_id", sort=False).first()
    opening["moment"] = pd.to_datetime(opening["timestamp"])

    return opening


# Determinismo: a propriedade que sustenta a reprodutibilidade do experimento.


@pytest.mark.parametrize("seed", DETERMINISM_SEEDS)
def test_same_seed_produces_the_same_traffic(seed: int) -> None:
    tables = population(seed)

    first = build_traffic(seed, tables.operators, tables.keys, SPECIFICATION)
    second = build_traffic(seed, tables.operators, tables.keys, SPECIFICATION)

    assert first.equals(second)


@pytest.mark.parametrize("seed", DETERMINISM_SEEDS)
def test_different_seeds_produce_different_traffic(seed: int) -> None:
    other = 2 if seed == 1 else 1

    assert not traffic(seed).equals(traffic(other))


def test_reference_output_has_not_changed() -> None:
    """Detector de mudanca, nao teste de correcao.

    Guarda o resumo criptografico em vez do CSV inteiro porque o arquivo tem
    cerca de 78 mil linhas: versiona-lo pesaria mais que o repositorio de
    codigo. A contrapartida e que a falha diz **que** mudou, nao **o que**
    mudou, e o arquivo se regenera com
    `python -m src.main --sementes 1 --ate scenario_engine`, em `data/seed-01/requests.csv`.

    Quando falhar, confirme se a mudanca era intencional, registre a decisao e
    atualize o valor de referencia.
    """
    digest = hashlib.sha256(csv_text(traffic(REFERENCE_SEED)).encode()).hexdigest()

    assert digest == REFERENCE_DIGEST.read_text().strip()


# Forma do arquivo: o contrato com o M4 e o M5.


@pytest.mark.parametrize("seed", SEEDS)
def test_columns_are_exactly_the_agreed_ones(seed: int) -> None:
    assert tuple(traffic(seed).columns) == COLUMNS


@pytest.mark.parametrize("seed", SEEDS)
def test_no_column_carries_the_outcome(seed: int) -> None:
    """O gerador emite tentativas, nunca desfechos (D-013).

    Se o desfecho nascesse aqui, o rotulo seria inventado em vez de derivado da
    politica, e o argumento central do trabalho cairia por circularidade. Quem
    decide e o M4.
    """
    forbidden = {"outcome", "status", "authorized", "denied", "result", "label"}

    assert forbidden.isdisjoint(set(traffic(seed).columns))


@pytest.mark.parametrize("seed", SEEDS)
def test_events_are_chronological_and_numbered_in_order(seed: int) -> None:
    requests = traffic(seed)
    expected = [f"event_{number:06d}" for number in range(1, len(requests) + 1)]

    assert list(requests["event_id"]) == expected
    assert requests["timestamp"].is_monotonic_increasing


@pytest.mark.parametrize("seed", SEEDS)
def test_every_session_has_one_operator_and_one_origin(seed: int) -> None:
    """A sessao e a unidade de analise (D-061): ela nao pode ser de dois donos."""
    grouped = traffic(seed).groupby("session_id")

    assert (grouped["operator_id"].nunique() == 1).all()
    assert (grouped["source_ip"].nunique() == 1).all()


# Fidelidade ao que as decisoes fixaram.


@pytest.mark.parametrize("seed", SEEDS)
def test_traffic_covers_exactly_the_simulated_period(seed: int) -> None:
    moments = pd.to_datetime(traffic(seed)["timestamp"])
    first = datetime.combine(SPECIFICATION.first_day, datetime.min.time())
    limit = first + timedelta(days=SPECIFICATION.day_count + 1)

    assert moments.min() >= first
    assert moments.max() < limit


@pytest.mark.parametrize("seed", SEEDS)
def test_every_operator_produces_traffic(seed: int) -> None:
    """Nenhum operador fica mudo no log.

    Dirigir a atividade por propriedade da chave deixaria cerca de 5,6 % dos
    operadores sem nada a acessar (D-042). Ela e dirigida por escopo, e todo
    escopo tem detentor, entao todos aparecem.
    """
    tables = population(seed)

    assert set(traffic(seed)["operator_id"]) == set(tables.operators["operator_id"])


@pytest.mark.parametrize("seed", SEEDS)
def test_source_address_is_always_one_of_the_habitual_ones(seed: int) -> None:
    """O M2 nao inventa origem de rede.

    Origem inedita no periodo avaliado precisa vir de um endereco habitual
    pouco usado que faltou no aquecimento, e nao de um endereco novo em folha
    (D-040). Endereco forjado aqui seria marcador perfeito do atacante.
    """
    requests = traffic(seed)
    habitual = addresses_by_operator(population(seed).operators)

    is_habitual = [
        address in habitual[operator]
        for operator, address in zip(requests["operator_id"], requests["source_ip"])
    ]

    assert all(is_habitual)


@pytest.mark.parametrize("seed", SEEDS)
def test_operations_come_from_the_agreed_set(seed: int) -> None:
    assert set(traffic(seed)["operation"]) <= set(OPERATIONS)


@pytest.mark.parametrize("seed", SEEDS)
def test_session_size_sits_in_the_regime_range_but_is_not_capped_by_it(
    seed: int,
) -> None:
    """A faixa do regime e o comprimento tipico, e nao um teto (D-097).

    O piso continua rigido (a sessao nao encolhe abaixo do minimo do regime),
    a grande maioria cai dentro da faixa, e **alguma a ultrapassa**. As tres
    coisas juntas sao a forma da cauda; qualquer uma sozinha nao e.
    """
    requests = traffic(seed)
    regime_of = regime_by_operator(population(seed).operators)

    size = requests.groupby("session_id").size()
    regime = requests.groupby("session_id")["operator_id"].first().map(regime_of)

    for name, specification in REGIMES.items():
        lowest, highest = specification.requests_range
        chosen = size[regime == name]

        assert chosen.min() >= lowest

        beyond = chosen > highest
        assert beyond.mean() < 2 * LONG_SESSION_CHANCE
        assert beyond.any(), (
            f"nenhuma sessao de `{name}` passou de {highest} eventos: "
            "o teto voltou a ser rigido, e a regra `events` volta a separar "
            "as classes sozinha em sigma baixo (D-097)"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_the_longest_legitimate_session_reaches_the_ostensive_range(
    seed: int,
) -> None:
    """A sessao legitima mais longa alcanca a faixa do atacante ostensivo.

    E a propriedade que a D-097 comprou, e a razao de ela existir: enquanto o
    teto legitimo era 40 e o atacante sorteava em (40, 90), as duas classes
    **nao se sobrepunham**, e o `events` separava por aritmetica de faixa: F1
    0,982 em sigma 0,0, com falso positivo zero por construcao.
    """
    ostensive_lowest, _ = AttackSpecification().ostensive_requests_range
    longest = traffic(seed).groupby("session_id").size().max()

    assert longest >= ostensive_lowest


@pytest.mark.parametrize("seed", SEEDS)
def test_distinct_keys_are_typical_in_range_and_sometimes_exceed_it(
    seed: int,
) -> None:
    """Amplitude da sessao legitima, contra a varredura ampla do atacante.

    O teto vale sobre as chaves do alcance; os desvios da D-056 enderecam fora
    dele e por isso ficam de fora da contagem.

    **A faixa e a amplitude tipica, e nao um teto** (D-098). Enquanto era teto,
    nenhuma sessao legitima passava de 12 chaves, e o `distinct_keys` separava
    as classes sozinho ate sigma 0,5: F1 0,879, e **identico** em 0,0, 0,2 e
    0,5, que e a assinatura de um separador que nao responde a sigma nenhum.

    Chamava-se `..._widen_on_long_sessions` ate 27/09, e o nome prometia o que
    o teste nunca conferiu: ele nao olha se a sessao longa e a que alarga, e
    ela nao e (D-100). Confere so que alguma sessao passa do tipico, e que
    passar e raro.
    """
    requests = traffic(seed)
    tables = population(seed)

    scopes = scopes_by_operator(tables.operators)
    scope_by_key = dict(zip(tables.keys["key_id"], tables.keys["scope"]))

    in_reach = [
        scope_by_key.get(key) in scopes[operator]
        for operator, key in zip(requests["operator_id"], requests["key_id"])
    ]
    reached = requests[in_reach]

    distinct = reached.groupby("session_id")["key_id"].nunique()
    _, highest = SPECIFICATION.distinct_keys_range

    assert distinct.min() >= 1

    beyond = distinct > highest
    assert beyond.mean() < 2 * LONG_SESSION_CHANCE
    assert beyond.any(), (
        f"nenhuma sessao passou de {highest} chaves distintas: o teto voltou "
        "a ser rigido, e o `distinct_keys` volta a separar as classes sozinho "
        "ate sigma 0,5 (D-098)"
    )


# Os dois desvios da D-056, sem os quais a falha vira marcador do atacante.


@pytest.mark.parametrize("seed", SEEDS)
def test_both_legitimate_failure_paths_occur(seed: int) -> None:
    """Escopo obsoleto e identificador inexistente existem no trafego limpo.

    Se nenhum deles ocorresse, `denied > 0` e `unknown > 0` passariam a
    significar atacante, que e o separador trivial que a D-028 procura.
    """
    requests = traffic(seed)
    tables = population(seed)

    scopes = scopes_by_operator(tables.operators)
    scope_by_key = dict(zip(tables.keys["key_id"], tables.keys["scope"]))

    unknown = sum(key not in scope_by_key for key in requests["key_id"])
    stale = sum(
        key in scope_by_key and scope_by_key[key] not in scopes[operator]
        for operator, key in zip(requests["operator_id"], requests["key_id"])
    )

    assert unknown > 0
    assert stale > 0


@pytest.mark.parametrize("seed", SEEDS)
def test_deviation_rates_stay_near_the_configured_ones(seed: int) -> None:
    """Banda larga de proposito: o teste guarda a ordem de grandeza.

    Taxa muito acima faria a falha deixar de ser excecao; muito abaixo a faria
    sumir em algumas sementes. Os valores exatos sao provisorios ate a
    calibracao sobre o log de aquecimento.
    """
    requests = traffic(seed)
    tables = population(seed)

    scopes = scopes_by_operator(tables.operators)
    scope_by_key = dict(zip(tables.keys["key_id"], tables.keys["scope"]))

    unknown = sum(key not in scope_by_key for key in requests["key_id"]) / len(requests)
    stale = sum(
        key in scope_by_key and scope_by_key[key] not in scopes[operator]
        for operator, key in zip(requests["operator_id"], requests["key_id"])
    ) / len(requests)

    assert SPECIFICATION.absent_identifier_rate * 0.5 < unknown < SPECIFICATION.absent_identifier_rate * 2
    assert (
        SPECIFICATION.stale_scope_rate * 0.5 < stale < SPECIFICATION.stale_scope_rate * 2
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_absent_identifier_matches_the_repository_format(seed: int) -> None:
    """Guarda a constante duplicada em `traffic.parameters`.

    A fronteira entre modulos e o arquivo, entao o M2 nao importa o formato do
    M1: ele o repete. Se o M1 mudar o formato do identificador, este teste
    falha antes que o identificador forjado fique distinguivel de um real, e o
    que faria o atributo de formato separar as classes sozinho.
    """
    requests = traffic(seed)
    existing = set(population(seed).keys["key_id"])

    forged = {key for key in requests["key_id"] if key not in existing}
    width = len(IDENTIFIER_PREFIX) + IDENTIFIER_HEX_DIGITS

    assert forged
    assert all(key.startswith(IDENTIFIER_PREFIX) for key in forged)
    assert all(len(key) == width for key in forged)
    assert all(int(key[len(IDENTIFIER_PREFIX) :], 16) >= 0 for key in forged)


# Ritmo: o que as convencoes do projeto mandam conferir antes de passar ao M3.


@pytest.mark.parametrize("seed", SEEDS)
def test_batch_sessions_open_at_the_scheduled_hours(seed: int) -> None:
    """Servico automatizado apresenta picos periodicos de volume.

    E o primeiro item da lista de conferencia. O pico sai do calendario de
    horas fixas do regime, e e o que distingue maquina de pessoa no log.
    """
    rhythm = REGIMES["periodic_batch"].rhythm

    assert isinstance(rhythm, ScheduledRhythm)

    opening = session_openings(traffic(seed))
    regime_of = regime_by_operator(population(seed).operators)
    batch = opening[opening["operator_id"].map(regime_of) == "periodic_batch"]

    minutes = batch["moment"].dt.hour * 60 + batch["moment"].dt.minute
    distances = [
        min(abs(minute - hour * 60) for hour in rhythm.hours) for minute in minutes
    ]

    assert max(distances) <= rhythm.jitter_minutes


@pytest.mark.parametrize("seed", SEEDS)
def test_people_only_open_sessions_on_business_days(seed: int) -> None:
    """Fim de semana e do lote, nao da pessoa.

    E o outro lado do pico periodico: o servico cobre os sete dias, e e essa
    diferenca que o M7 vai medir sem consultar o perfil.
    """
    opening = session_openings(traffic(seed))
    regime_of = regime_by_operator(population(seed).operators)
    regime = opening["operator_id"].map(regime_of)

    for name in ("routine", "occasional_custody"):
        weekdays = set(opening[regime == name]["moment"].dt.weekday)

        assert weekdays <= BUSINESS_WEEKDAYS

    weekend = set(opening[regime == "periodic_batch"]["moment"].dt.weekday)

    assert weekend - BUSINESS_WEEKDAYS


@pytest.mark.parametrize("seed", SEEDS)
def test_people_open_sessions_inside_their_working_window(seed: int) -> None:
    """Janela estreita e o que da ao M6 uma faixa horaria habitual para extrair.

    Chegada uniforme nas 24 horas deixaria o atributo de hora atipica
    degenerado: nenhuma hora seria atipica se todas fossem igualmente comuns.
    """
    opening = session_openings(traffic(seed))
    regime_of = regime_by_operator(population(seed).operators)
    regime = opening["operator_id"].map(regime_of)

    for name in ("routine", "occasional_custody"):
        rhythm = REGIMES[name].rhythm
        hours = opening[regime == name]["moment"].dt.hour

        assert hours.min() >= rhythm.opens_at_hour
        assert hours.max() < rhythm.closes_at_hour


@pytest.mark.parametrize("seed", SEEDS)
def test_administrator_rhythm_is_more_irregular_than_the_routine_one(seed: int) -> None:
    """Administrador apresenta ritmo irregular, padrao humano.

    E o segundo item da lista de conferencia. A binomial negativa produz
    variancia maior que a media; a Poisson as iguala. A comparacao entre os
    dois regimes e mais estavel que um limiar absoluto sobre um deles.
    """
    opening = session_openings(traffic(seed))
    regime_of = regime_by_operator(population(seed).operators)
    opening["regime"] = opening["operator_id"].map(regime_of)
    opening["day"] = opening["moment"].dt.date

    dispersions = {}

    for name in ("routine", "occasional_custody"):
        subset = opening[opening["regime"] == name]
        per_day = subset.groupby(["operator_id", "day"]).size()
        dispersions[name] = per_day.var() / per_day.mean()

    assert dispersions["occasional_custody"] > dispersions["routine"]
