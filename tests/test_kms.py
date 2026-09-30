"""Testes do M4 e do M5: o desfecho e o registro.

O M4 e a peca de que depende o argumento central do trabalho: se o desfecho
nao viesse da politica, o rotulo seria inventado. Por isso os testes aqui
guardam menos o codigo e mais a **derivacao**: que cada desfecho corresponda
ao que as tabelas estaticas dizem, e nao ao que o gerador quis.

Tres invariantes valem mais que as outras:

1. **A ordem da D-077.** Requisicao fora de escopo e para chave desabilitada
   devolve `denied_by_policy`, nao `disabled_key`. Autorizacao antes de estado.
2. **Os quatro desfechos ocorrem em trafego legitimo.** Sem isso, qualquer um
   deles viraria marcador perfeito do atacante quando o M3 existir.
3. **O log nao carrega escopo, perfil, proprietario nem rotulo** (D-064).

O M4 nao recebe semente: ele e deterministico por construcao, e o teste de
repeticao confere isso em vez de supor.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
import pytest

from src.audit_logger.build import COLUMNS as LOG_COLUMNS
from src.audit_logger.build import build_log
from src.shared.experiment import SEEDS
from src.shared.phases import EVALUATED, WARMUP, belongs_to
from src.shared.tables import MULTIVALUE_SEPARATOR
from src.kms.build import build_outcomes, requests_of_phase
from src.kms.policy import (
    DENIED_BY_POLICY,
    DISABLED_KEY,
    OUTCOMES,
    SUCCESS,
    UNKNOWN_KEY,
)
from src.kms.policy import COLUMNS as OUTCOME_COLUMNS
from src.kms.policy import Repository, outcome_of, read_repository
from src.pipeline.population import Population, build_population
from src.kms.repository.parameters import KeyRepositorySpecification
from src.scenario_engine.traffic.build import build_traffic
from src.scenario_engine.traffic.parameters import TrafficSpecification

REPOSITORY_SPECIFICATION = KeyRepositorySpecification()
TRAFFIC_SPECIFICATION = TrafficSpecification()

DETERMINISM_SEEDS = (1, 15, 30)


@lru_cache(maxsize=None)
def population(seed: int) -> Population:
    return build_population(seed, REPOSITORY_SPECIFICATION)


@lru_cache(maxsize=None)
def traffic(seed: int) -> pd.DataFrame:
    tables = population(seed)

    return build_traffic(seed, tables.operators, tables.keys, TRAFFIC_SPECIFICATION)


@lru_cache(maxsize=None)
def outcomes(seed: int) -> pd.DataFrame:
    tables = population(seed)

    return build_outcomes(traffic(seed), tables.keys, tables.operators, WARMUP)


@lru_cache(maxsize=None)
def log(seed: int) -> pd.DataFrame:
    return build_log(traffic(seed), outcomes(seed), WARMUP)


def repository_of(seed: int) -> Repository:
    tables = population(seed)

    return read_repository(tables.keys, tables.operators)


# A ordem de avaliacao, que e a D-077.


def test_nonexistent_identifier_wins_over_everything() -> None:
    """Identificador que nao existe nao tem escopo para comparar."""
    repository = Repository(
        scope_by_key={"k_real": "scope_01"},
        disabled=frozenset({"k_real"}),
        scopes_by_operator={"admin_01": frozenset({"scope_01"})},
    )

    assert outcome_of("admin_01", "k_ausente", repository) == UNKNOWN_KEY


def test_policy_wins_over_disabled_state() -> None:
    """O caso ambiguo da D-077: fora de escopo **e** desabilitada.

    Responder `disabled_key` a quem nao detem o escopo confirmaria que a chave
    existe e revelaria em que estado ela esta. Nega-se sem qualificar.
    """
    repository = Repository(
        scope_by_key={"k_alheia": "scope_09"},
        disabled=frozenset({"k_alheia"}),
        scopes_by_operator={"user_01": frozenset({"scope_01"})},
    )

    assert outcome_of("user_01", "k_alheia", repository) == DENIED_BY_POLICY


def test_disabled_only_reaches_keys_the_operator_may_use() -> None:
    """Consequencia da ordem: `disabled_key` tem leitura unica.

    Com autorizacao antes de estado, o desfecho significa sempre "voce podia
    pedir, mas a chave esta inativa", e nunca mistura autorizacao com
    disponibilidade.
    """
    repository = Repository(
        scope_by_key={"k_minha": "scope_01"},
        disabled=frozenset({"k_minha"}),
        scopes_by_operator={"user_01": frozenset({"scope_01"})},
    )

    assert outcome_of("user_01", "k_minha", repository) == DISABLED_KEY


def test_reachable_and_active_succeeds() -> None:
    repository = Repository(
        scope_by_key={"k_minha": "scope_01"},
        disabled=frozenset(),
        scopes_by_operator={"user_01": frozenset({"scope_01"})},
    )

    assert outcome_of("user_01", "k_minha", repository) == SUCCESS


@pytest.mark.parametrize("seed", SEEDS)
def test_every_outcome_is_derived_from_the_static_tables(seed: int) -> None:
    """Nenhum desfecho e afirmado: todos se reconstroem das duas tabelas.

    Este e o teste que sustenta a D-013. Se ele passar, o rotulo do M7 sera
    derivado da politica em vez de inventado pelo gerador.
    """
    tables = population(seed)
    repository = repository_of(seed)
    of_phase = requests_of_phase(traffic(seed), WARMUP)

    expected = [
        outcome_of(row.operator_id, row.key_id, repository)
        for row in of_phase.itertuples()
    ]

    assert list(outcomes(seed)["outcome"]) == expected
    assert len(tables.keys) == REPOSITORY_SPECIFICATION.total_keys


# Determinismo, sem semente: o KMS nao sorteia nada.


@pytest.mark.parametrize("seed", DETERMINISM_SEEDS)
def test_the_same_input_gives_the_same_outcomes(seed: int) -> None:
    tables = population(seed)

    first = build_outcomes(traffic(seed), tables.keys, tables.operators, WARMUP)
    second = build_outcomes(traffic(seed), tables.keys, tables.operators, WARMUP)

    assert first.equals(second)


# Forma dos arquivos: o contrato com o M5 e com o M6.


@pytest.mark.parametrize("seed", SEEDS)
def test_outcomes_have_exactly_two_columns(seed: int) -> None:
    """D-078: `event_id` e `outcome`, nada mais.

    Coluna de diagnostico aqui seria coluna que o M5 teria de lembrar de
    descartar, e o escopo violado e o que a D-064 mantem fora do log.
    """
    assert tuple(outcomes(seed).columns) == OUTCOME_COLUMNS


@pytest.mark.parametrize("seed", SEEDS)
def test_log_has_exactly_the_eight_agreed_columns(seed: int) -> None:
    assert tuple(log(seed).columns) == LOG_COLUMNS


@pytest.mark.parametrize("seed", SEEDS)
def test_log_carries_neither_authorization_boundary_nor_label(seed: int) -> None:
    """D-064 e D-063: nada que deixe o modelo aprender a politica ou o rotulo.

    Escopo, perfil ou proprietario poriam o modelo em condicao de reconstruir
    a fronteira de autorizacao. O rotulo viaja fora do log, em
    `compromised_sessions.csv`, e o M7 o junta so na fase avaliada.
    """
    forbidden = {"scope", "profile", "owner", "scopes", "regime",
                 "compromised", "label", "is_attack"}

    assert forbidden.isdisjoint(set(log(seed).columns))


@pytest.mark.parametrize("seed", SEEDS)
def test_no_event_is_lost_or_duplicated_in_the_join(seed: int) -> None:
    """A juncao e por `event_id`, e tem de casar um para um."""
    of_phase = requests_of_phase(traffic(seed), WARMUP)

    assert len(log(seed)) == len(of_phase)
    assert log(seed)["event_id"].is_unique
    assert set(log(seed)["event_id"]) == set(of_phase["event_id"])


# O que o M6, o M7 e o M8 vao encontrar.


@pytest.mark.parametrize("seed", SEEDS)
def test_the_warmup_covers_exactly_the_ruler_weeks(seed: int) -> None:
    """O aquecimento nao pode vazar para o periodo avaliado.

    Se vazasse, o perfil historico e os limiares veriam dados que o atacante
    tocara, e o baseline nasceria calibrado contra o que deveria detectar.
    """
    inside = belongs_to(WARMUP, log(seed)["timestamp"])
    evaluated = belongs_to(EVALUATED, log(seed)["timestamp"])

    assert inside.all()
    assert not evaluated.any()


@pytest.mark.parametrize("seed", SEEDS)
def test_the_four_outcomes_occur_in_clean_traffic(seed: int) -> None:
    """Se um desfecho so aparecesse com atacante, ele o marcaria sozinho.

    E o terceiro item da lista de conferencia do projeto, e so pode ser
    verificado depois que o M4 existe: e ele quem produz a negacao.
    """
    present = set(log(seed)["outcome"])

    assert present == set(OUTCOMES)


@pytest.mark.parametrize("seed", SEEDS)
def test_policy_denial_reaches_many_operators(seed: int) -> None:
    """Negacao concentrada em poucos operadores seria quase tao ruim quanto nenhuma.

    A D-056 pede taxas altas o bastante para que praticamente todo operador
    acumule ao menos uma falha, e baixas o bastante para que a falha nao seja
    ela propria um marcador.
    """
    denied = log(seed)[log(seed)["outcome"] == DENIED_BY_POLICY]
    share = len(denied) / len(log(seed))

    assert denied["operator_id"].nunique() >= 20
    assert 0.001 < share < 0.02


@pytest.mark.parametrize("seed", SEEDS)
def test_denied_keys_are_real_and_outside_the_operator_scopes(seed: int) -> None:
    """A negacao vem da politica, nao de um sinalizador do gerador."""
    tables = population(seed)
    scope_by_key = dict(zip(tables.keys["key_id"], tables.keys["scope"]))
    scopes_by_operator = {
        row.operator_id: set(row.scopes.split(MULTIVALUE_SEPARATOR))
        for row in tables.operators.itertuples()
    }

    denied = log(seed)[log(seed)["outcome"] == DENIED_BY_POLICY]

    for row in denied.itertuples():
        assert row.key_id in scope_by_key
        assert scope_by_key[row.key_id] not in scopes_by_operator[row.operator_id]


@pytest.mark.parametrize("seed", SEEDS)
def test_unknown_keys_are_absent_from_the_repository(seed: int) -> None:
    """O identificador inexistente tem o formato de um real, e nao colide."""
    existing = set(population(seed).keys["key_id"])

    unknown = log(seed)[log(seed)["outcome"] == UNKNOWN_KEY]

    assert len(unknown) > 0
    assert existing.isdisjoint(set(unknown["key_id"]))
    assert unknown["key_id"].str.match(r"^k_[0-9a-f]{12}$").all()
