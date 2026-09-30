"""Testes do M1: determinismo e as invariantes de que os modulos seguintes dependem.

O rotulo do experimento deriva destas duas tabelas. O M4 decide o desfecho de
cada requisicao comparando o escopo da chave com os escopos do operador, entao
tabela incoerente produz desfecho errado, que produz rotulo errado, sem erro em
lugar nenhum. E isso que estes testes guardam.

As invariantes rodam nas 30 sementes da grade, nao numa so: falha especifica de
semente e exatamente o que passa despercebido.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.population import Population, build_population
from src.entities.scenario_engine.population.operators import holders_by_scope
from src.entities.scenario_engine.population.profiles import PROFILES, Profile
from src.entities.kms.repository.parameters import KeyRepositorySpecification
from src.shared.experiment import SEEDS
from src.shared.tables import MULTIVALUE_SEPARATOR

SPECIFICATION = KeyRepositorySpecification()

REFERENCE_SEED = 1
REFERENCE_TABLES = ("keys", "operators")
REFERENCE_DIRECTORY = Path(__file__).parent / "reference"

IDENTIFIER_SPACE = 2**48


@lru_cache(maxsize=None)
def population(seed: int) -> Population:
    """Populacao de uma semente, reaproveitada entre os testes."""
    return build_population(seed, SPECIFICATION)


def csv_text(frame: pd.DataFrame) -> str:
    """O CSV exatamente como `write_csv` o grava."""
    return frame.to_csv(index=False, lineterminator="\n")


def reference_path(table: str) -> Path:
    """O arquivo de referencia daquela tabela."""
    return REFERENCE_DIRECTORY / f"seed-{REFERENCE_SEED:02d}-{table}.csv"


def profile_of(operator_id: str) -> Profile:
    """O perfil de um operador, deduzido do prefixo do identificador."""
    prefix = operator_id.split("_")[0]

    return next(profile for profile in PROFILES if profile.id_prefix == prefix)


def multivalues(cell: str) -> list[str]:
    """Os valores de uma celula multivalorada."""
    return cell.split(MULTIVALUE_SEPARATOR)


# Determinismo: a propriedade que sustenta a reprodutibilidade do experimento.


@pytest.mark.parametrize("seed", SEEDS)
def test_same_seed_produces_the_same_population(seed: int) -> None:
    first = build_population(seed, SPECIFICATION)
    second = build_population(seed, SPECIFICATION)

    assert first.operators.equals(second.operators)
    assert first.keys.equals(second.keys)


def test_different_seeds_produce_different_populations() -> None:
    """Fluxo mal derivado faria replicas distintas produzirem dados iguais (D-003)."""
    assert not population(1).operators.equals(population(2).operators)
    assert not population(1).keys.equals(population(2).keys)


# Coerencia entre as tabelas: o que quebraria o rotulo sem levantar erro.


@pytest.mark.parametrize("seed", SEEDS)
def test_every_scope_has_a_holder(seed: int) -> None:
    """Escopo sem detentor deixaria chaves fora do alcance de toda credencial."""
    holders = holders_by_scope(population(seed).operators)

    assert len(holders) == SPECIFICATION.scope_count


@pytest.mark.parametrize("seed", SEEDS)
def test_the_key_table_has_no_owner(seed: int) -> None:
    """A chave nao tem dono (D-099), e a ausencia precisa ser guardada.

    Ate 24/09 havia uma coluna `owner`, e um teste conferindo que o dono detinha
    o escopo da chave. Ela nunca foi consumida por modulo nenhum: quem alcanca
    uma chave e quem detem o escopo dela. O teste que existia guardava a
    coerencia de um campo inerte; este guarda que ele nao volte.
    """
    assert "owner" not in population(seed).keys.columns


# Escala e forma da populacao, conforme a Tabela 1 da proposta (D-035, D-041).


@pytest.mark.parametrize("seed", SEEDS)
def test_operator_count_per_profile(seed: int) -> None:
    counts = population(seed).operators["profile"].value_counts()

    for profile in PROFILES:
        assert counts[profile.name] == profile.operators


@pytest.mark.parametrize("seed", SEEDS)
def test_scope_count_per_profile(seed: int) -> None:
    for operator in population(seed).operators.itertuples():
        expected = profile_of(operator.operator_id).scopes_each

        assert len(multivalues(operator.scopes)) == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_address_count_within_the_profile_range(seed: int) -> None:
    for operator in population(seed).operators.itertuples():
        lowest, highest = profile_of(operator.operator_id).addresses_range

        assert lowest <= len(multivalues(operator.usual_ips)) <= highest


@pytest.mark.parametrize("seed", SEEDS)
def test_addresses_are_distinct_across_the_population(seed: int) -> None:
    """Endereco repetido faria dois operadores compartilharem origem habitual."""
    addresses = [
        address
        for operator in population(seed).operators.itertuples()
        for address in multivalues(operator.usual_ips)
    ]

    assert len(addresses) == len(set(addresses))


@pytest.mark.parametrize("seed", SEEDS)
def test_key_total_matches_the_specification(seed: int) -> None:
    assert len(population(seed).keys) == SPECIFICATION.total_keys


@pytest.mark.parametrize("seed", SEEDS)
def test_no_scope_is_empty(seed: int) -> None:
    """A reparticao entre escopos e desigual; o piso impede que algum fique vazio.

    Escopo vazio deixaria seus detentores sem nada a acessar dentro dele (D-037).
    """
    keys_per_scope = population(seed).keys["scope"].value_counts()

    assert len(keys_per_scope) == SPECIFICATION.scope_count
    assert keys_per_scope.min() >= SPECIFICATION.scope_floor


# Armadilhas catalogadas na revisao: atributo que separa as classes sozinho.


@pytest.mark.parametrize("seed", SEEDS)
def test_key_ids_are_unique(seed: int) -> None:
    identifiers = population(seed).keys["key_id"]

    assert identifiers.is_unique


@pytest.mark.parametrize("seed", SEEDS)
def test_key_ids_span_the_identifier_space(seed: int) -> None:
    """D-009: identificador sequencial daria progressao aritmetica.

    300 sorteios uniformes em 48 bits cobrem quase todo o espaco. Um esquema
    sequencial teria amplitude na ordem de 300.
    """
    values = [int(identifier[2:], 16) for identifier in population(seed).keys["key_id"]]
    span = max(values) - min(values)

    assert span > 0.9 * IDENTIFIER_SPACE


def test_key_rows_are_not_grouped_by_scope() -> None:
    """Sem o embaralhamento, a posicao da linha carregaria o escopo."""
    scopes = population(REFERENCE_SEED).keys["scope"].tolist()

    assert scopes != sorted(scopes)


# Deteccao de mudanca: nao e teste de correcao.


@pytest.mark.parametrize("table", REFERENCE_TABLES)
def test_reference_output_has_not_changed(table: str) -> None:
    """Detector de mudanca, nao teste de correcao.

    Nao tem verdade propria: a referencia e o que o gerador produzia quando os
    arquivos foram gravados. Ele pega a classe de mudanca que nenhuma
    invariante pega, a que e valida mas diferente: reordenar duas chamadas do
    `build_population`, por exemplo, mantem todas as invariantes e produz
    outros dados.

    Falha sempre que o gerador mudar, inclusive de proposito. Quando falhar, a
    pergunta certa e: eu pretendia mudar os dados? Se sim, abra entrada no
    registro de decisoes e regenere com `python -m tests.test_population`, para
    que o diff da referencia entre no mesmo commit. Se nao, e um defeito.
    """
    produced = csv_text(getattr(population(REFERENCE_SEED), table)).splitlines()
    expected = reference_path(table).read_text(encoding="utf-8").splitlines()

    assert len(produced) == len(expected), (
        f"a tabela tem {len(produced)} linhas e a referencia tem {len(expected)}"
    )

    differing = [
        number
        for number, (left, right) in enumerate(zip(produced, expected), start=1)
        if left != right
    ]

    assert not differing, (
        f"{len(differing)} de {len(expected)} linhas diferem da referencia. "
        f"A primeira e a linha {differing[0]}:\n"
        f"  gerado:     {produced[differing[0] - 1]}\n"
        f"  referencia: {expected[differing[0] - 1]}\n"
        f"Use `git diff {reference_path(table).name}` para ver a mudanca inteira."
    )


def write_reference() -> None:
    """Regrava os arquivos de referencia com o que o gerador produz agora.

    Rode apenas depois de confirmar que a mudanca no gerador era intencional e
    de registra-la em `decisoes.md`. O diff dos arquivos deve entrar no mesmo
    commit da mudanca que o causou.
    """
    REFERENCE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for table in REFERENCE_TABLES:
        frame = getattr(build_population(REFERENCE_SEED, SPECIFICATION), table)
        reference_path(table).write_text(csv_text(frame), encoding="utf-8", newline="")
        print(f"gravado {reference_path(table)}")


if __name__ == "__main__":
    write_reference()
