"""Testes do observador: o que a tela mostra é o que o pipeline produz.

O risco que este arquivo guarda não é o de o viewer quebrar: ele quebraria
ruidosamente, e em cima de uma apresentação. É o de ele **divergir em
silêncio**.

O `steps.py` não chama `build_population`: ele chama os internos dela, um a um,
porque precisa mostrar o intermediário de cada etapa. Isso é uma **segunda
cópia da ordem de execução**, e a ordem faz parte do resultado: todas as
chamadas consomem sorteios do mesmo fluxo, então trocar duas de lugar muda a
saída inteira com a mesma semente. É o mesmo risco que a D-091 tirou do
pipeline, e que aqui não dá para tirar: o observador precisa dos passos
separados justamente porque é isso que ele existe para mostrar.

O que dá para fazer é **acusar**. Se alguém reordenar duas chamadas dentro de
um módulo, o digest de referência acusa a mudança do pipeline e estes testes
acusam a do viewer. Sem eles, a tela passaria a mostrar uma população que não é
a do experimento, de forma plausível e sem aviso, que é o pior tipo de erro
para um trabalho que vai ser defendido com esta tela aberta.

**Não cobre a aparência.** Nada aqui renderiza Streamlit: o que se verifica é
que os dados por trás dos quadros são os do pipeline, e que os quadros montam
sem erro. Layout se confere olhando.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.attack.build import build_attack
from src.attack.parameters import AttackSpecification
from src.audit_logger.build import build_log
from src.calibration.build import build_thresholds
from src.dataset.build import build_dataset
from src.globals.phases import EVALUATED, WARMUP
from src.historical_profiles.build import build_profiles
from src.kms.build import build_outcomes
from src.population.build import build_population
from src.population.parameters import KeyRepositorySpecification
from src.traffic.build import build_traffic
from src.traffic.parameters import TrafficSpecification
from src.viewer.frames import (
    frames_da_fase_1,
    frames_da_fase_2,
    frames_da_fase_3,
)
from src.viewer.steps import (
    FASES,
    attack_steps,
    population_steps,
    traffic_steps,
    warmup_steps,
)

REPOSITORY = KeyRepositorySpecification()
TRAFFIC = TrafficSpecification()
ATTACK = AttackSpecification()

SAMPLE_SEEDS = (1, 7, 30)
"""Tres sementes e o suficiente: a divergencia de ordem seria sistematica.

Um erro de ordem muda a saida de **toda** semente, entao ele aparece na
primeira. As tres existem para o caso de a divergencia depender de um sorteio
que so acontece as vezes.
"""

SIGMA = 0.5


def por_funcao(passos: list) -> dict:
    """Os passos indexados pelo nome da funcao que cada um observa."""
    return {passo.funcao: passo for passo in passos}


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_viewer_rebuilds_the_population_the_pipeline_produces(
    seed: int,
) -> None:
    """A ordem reconstruida no `steps.py` bate com a de `build_population`.

    Conferido pela saida e nao pela leitura do codigo: se as chamadas
    estivessem em ordem diferente, os sorteios sairiam do fluxo noutra ordem e
    as tabelas seriam outras: validas, plausiveis e erradas.
    """
    produzida = build_population(seed, REPOSITORY)
    observada = por_funcao(population_steps(seed, REPOSITORY))

    operadores = observada["build_operators_covering_pool"].saida
    chaves = observada["shuffle_rows"].saida

    assert produzida.operators.equals(operadores)
    assert produzida.keys.equals(chaves)


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_viewer_rebuilds_the_traffic_the_pipeline_produces(
    seed: int,
) -> None:
    """O mesmo para o M2, que e o modulo com mais passos observados."""
    tabelas = build_population(seed, REPOSITORY)
    produzido = build_traffic(seed, tabelas.operators, tabelas.keys, TRAFFIC)

    foco = tabelas.operators["operator_id"].iloc[0]
    observado = por_funcao(
        traffic_steps(seed, tabelas.operators, tabelas.keys, TRAFFIC, foco)
    )

    final = observado["chronological + with_event_ids"].para_descrever()

    assert produzido.equals(final)


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_viewer_rebuilds_the_ruler_the_pipeline_produces(
    seed: int,
) -> None:
    """Perfis e limiares, que sao a regua e o que o baseline leva congelado."""
    tabelas = build_population(seed, REPOSITORY)
    requests = build_traffic(seed, tabelas.operators, tabelas.keys, TRAFFIC)

    outcomes = build_outcomes(requests, tabelas.keys, tabelas.operators, WARMUP)
    log = build_log(requests, outcomes, WARMUP)
    perfis = build_profiles(log)
    limiares = build_thresholds(build_dataset(log, perfis, WARMUP))

    foco = tabelas.operators["operator_id"].iloc[0]
    observado = por_funcao(
        warmup_steps(seed, tabelas.operators, tabelas.keys, requests, foco)
    )

    assert perfis.equals(observado["profile_of"].para_descrever())
    assert limiares.equals(observado["build_thresholds"].saida)


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_viewer_rebuilds_the_campaign_the_pipeline_produces(
    seed: int,
) -> None:
    """A campanha, que e o unico passo observado que depende de sigma."""
    tabelas = build_population(seed, REPOSITORY)
    requests = build_traffic(seed, tabelas.operators, tabelas.keys, TRAFFIC)

    produzida = build_attack(
        seed, SIGMA, tabelas.operators, tabelas.keys, requests, TRAFFIC, ATTACK
    )

    warmup_log = build_log(
        requests,
        build_outcomes(requests, tabelas.keys, tabelas.operators, WARMUP),
        WARMUP,
    )
    perfis = build_profiles(warmup_log)

    observada = por_funcao(
        attack_steps(
            seed, SIGMA, tabelas.operators, tabelas.keys, requests, perfis
        )
    )

    assert produzida.requests.equals(observada["build_attack"].para_descrever())


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_every_frame_carries_real_tables_and_none_is_empty(seed: int) -> None:
    """Os onze quadros montam, e nenhum painel implementado vem vazio.

    Painel vazio nao levanta erro: ele renderiza uma tabela sem linhas, e numa
    apresentacao isso passa por 'ainda nao implementado'. Aqui ele falha.
    """
    tabelas = build_population(seed, REPOSITORY)
    requests = build_traffic(seed, tabelas.operators, tabelas.keys, TRAFFIC)
    foco = tabelas.operators["operator_id"].iloc[0]

    outcomes = build_outcomes(requests, tabelas.keys, tabelas.operators, WARMUP)
    log = build_log(requests, outcomes, WARMUP)
    perfis = build_profiles(log)
    sessoes = build_dataset(log, perfis, WARMUP)
    limiares = build_thresholds(sessoes)

    campanha = build_attack(
        seed, SIGMA, tabelas.operators, tabelas.keys, requests, TRAFFIC, ATTACK
    )
    log_avaliado = build_log(
        campanha.requests,
        build_outcomes(
            campanha.requests, tabelas.keys, tabelas.operators, EVALUATED
        ),
        EVALUATED,
    )
    sessoes_avaliadas = build_dataset(
        log_avaliado, perfis, EVALUATED, campanha.compromised
    )

    quadros = (
        frames_da_fase_1(
            seed,
            tabelas.operators,
            tabelas.keys,
            requests,
            population_steps(seed, REPOSITORY)
            + traffic_steps(
                seed, tabelas.operators, tabelas.keys, TRAFFIC, foco
            ),
        )
        + frames_da_fase_2(
            requests,
            tabelas.operators,
            tabelas.keys,
            outcomes,
            log,
            perfis,
            sessoes,
            limiares,
            warmup_steps(
                seed, tabelas.operators, tabelas.keys, requests, foco
            ),
        )
        + frames_da_fase_3(
            SIGMA,
            requests,
            perfis,
            campanha.requests,
            campanha.compromised,
            campanha.run,
            log_avaliado,
            sessoes_avaliadas,
            attack_steps(
                seed, SIGMA, tabelas.operators, tabelas.keys, requests, perfis
            ),
        )
    )

    assert [quadro.numero for quadro in quadros] == list(range(1, 12))

    for quadro in quadros:
        for painel in quadro.entradas + quadro.saidas:
            is_tabela = isinstance(painel.dado, pd.DataFrame)

            if is_tabela:
                assert len(painel.dado) > 0, (
                    f"quadro {quadro.numero} ({quadro.entidade}): "
                    f"o painel `{painel.nome}` veio vazio"
                )


def test_every_decision_variable_reads_a_real_parameter() -> None:
    """Nenhuma variavel da tela traz valor escrito a mao.

    O viewer existe para mostrar o gerador, entao um numero digitado no
    `decisions.py` seria pior que nao mostrar nada: ele continuaria plausivel
    depois de o parametro mudar.
    """
    tabelas = build_population(1, REPOSITORY)
    quadros = frames_da_fase_1(
        1, tabelas.operators, tabelas.keys,
        build_traffic(1, tabelas.operators, tabelas.keys, TRAFFIC),
        population_steps(1, REPOSITORY),
    )

    variaveis = [
        variavel for quadro in quadros for variavel in quadro.variaveis
    ]

    assert len(variaveis) > 0

    for variavel in variaveis:
        assert variavel.valor is not None
        assert str(variavel.valor) != ""
        assert variavel.significado != ""


def test_the_three_phases_declare_what_they_are() -> None:
    """As tres fases tem nome, objetivo e contagem de execucoes.

    Campo vazio aqui vira bloco em branco no topo da tela, e o titulo da fase
    e o que casa o viewer com o diagrama da arquitetura.
    """
    assert len(FASES) == 3

    for fase in FASES:
        assert fase.nome
        assert fase.modulos
        assert fase.execucoes
        assert fase.objetivo
        assert fase.descricao
