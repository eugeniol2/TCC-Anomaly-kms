"""Testes do viewer: os quadros leem o que o pipeline gravou, e as curvas batem com o gerador.

Desde a D-121 o viewer nao roda o pipeline: ele le os arquivos que o comando unico
gravou. Por isso nao existe mais uma segunda copia da ordem de execucao para
conferir. O que resta guardar sao duas coisas: que os treze quadros montam com
arquivos de verdade, sem painel vazio, e que as curvas de teoria da pagina
Comportamentos descrevem o gerador que existe.

**Nao cobre a aparencia.** Nada aqui renderiza Streamlit. Layout se confere olhando.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.metrics.evaluation.build import MECHANISMS
from src.metrics.importance.build import read_importance_run, run_importance
from src.pipeline.build import Specifications, run_seed_branch, run_sigma_branch
from src.pipeline.experiment import write_evaluation
from src.pipeline.population import build_population
from src.entities.dataset_generator.dataset.build import ATTRIBUTES, LABEL
from src.entities.kms.repository.parameters import KeyRepositorySpecification
from src.entities.policy_engine.baseline.build import RULE_ATTRIBUTES, RULE_COLUMNS
from src.entities.policy_engine.baseline.parameters import MINIMUM_RULES_FIRED
from src.entities.policy_engine.calibration.parameters import THRESHOLD_ATTRIBUTES
from src.entities.scenario_engine.attack.parameters import AttackSpecification
from src.entities.scenario_engine.population.profiles import PROFILES
from src.entities.scenario_engine.traffic.build import build_traffic
from src.entities.scenario_engine.traffic.calendar import daily_session_count
from src.entities.scenario_engine.traffic.parameters import PRIMARY_ADDRESS_SHARE, TrafficSpecification
from src.entities.scenario_engine.traffic.regimes import REGIMES
from src.entities.scenario_engine.traffic.sessions import address_weights, draw_request_count, request_instants
from src.viewer.behaviors import (
    ORDEM,
    Pagina,
    barras_da_exponencial,
    metodos_da_sessao_do_regime,
    metodos_do_calendario,
    montar_pagina,
    secao_do_excesso,
    tabela_do_atacante,
    tamanhos_da_sessao,
)
from src.viewer.data import read_grid, read_run, read_seed, read_thresholds
from src.viewer.formatting import valor_escrito
from src.viewer.frames import FASES, QUADROS, Execucao, quadros_da_fase
from src.viewer.results import (
    RECORTES,
    matriz_de_confusao,
    secao_da_importancia,
    tabela_da_comparacao,
    tabela_do_tempo,
)
from src.viewer.rules import (
    frase_do_tempo,
    secao_ao_longo_de_sigma,
    secao_do_corte,
    secao_do_limiar,
    secao_qual_regra_dispara,
    secao_regras_por_sessao,
    tabela_da_matriz,
    tabela_das_regras,
    tabela_dos_limiares_nas_sementes,
)
from src.viewer.theory import teoria_da_geometrica

REPOSITORY = KeyRepositorySpecification()
TRAFFIC = TrafficSpecification()
ATTACK = AttackSpecification()

SAMPLE_SEEDS = (1, 7, 30)
SEED = 1
SIGMA = 0.5

TEST_CONFIGURATION = {
    "random_forest": {
        "n_estimators": 10, "max_depth": None, "min_samples_leaf": 1,
        "max_features": "sqrt", "class_weight": None,
    },
    "xgboost": {
        "n_estimators": 10, "max_depth": 3, "learning_rate": 0.1,
        "subsample": 1.0, "colsample_bytree": 1.0, "class_weight": "balanced",
    },
}
"""Uma configuracao pequena, so para os testes rodarem rapido. A de verdade sai da 902."""


@pytest.fixture(scope="module")
def data_root(tmp_path_factory) -> Path:
    """Uma raiz de dados com uma execucao inteira e a avaliacao dela, como o comando grava."""
    root = tmp_path_factory.mktemp("data")
    branch = run_seed_branch(SEED, root, Specifications(models=TEST_CONFIGURATION))
    run_sigma_branch(branch, SIGMA)
    write_evaluation(root, (SEED,), (SIGMA,))

    return root


def execucao_de(root: Path, com_grade: bool = True) -> Execucao:
    return Execucao(
        SEED, SIGMA, read_seed(root, SEED), read_run(root, SEED, SIGMA),
        read_grid(root) if com_grade else None, TEST_CONFIGURATION,
    )


def todos_os_quadros(execucao: Execucao) -> list:
    return [quadro for fase in FASES for quadro in quadros_da_fase(fase, execucao)]


def test_every_frame_carries_real_tables_and_none_is_empty(data_root: Path) -> None:
    """Os treze quadros montam com os arquivos gravados, e nenhum painel vem vazio.

    Painel vazio nao levanta erro: renderiza uma tabela sem linhas, e numa
    apresentacao isso passa por "nao implementado". Aqui ele falha.
    """
    quadros = todos_os_quadros(execucao_de(data_root))

    assert [quadro.numero for quadro in quadros] == list(range(1, len(QUADROS) + 1))

    for quadro in quadros:
        assert not quadro.aviso, quadro.entidade

        for painel in quadro.entradas + quadro.saidas:
            if isinstance(painel.dado, pd.DataFrame):
                assert len(painel.dado) > 0, f"{quadro.entidade}: `{painel.nome}` vazio"


def test_the_phases_cover_every_frame_once() -> None:
    numeros = [numero for fase in FASES for numero in fase.quadros]

    assert sorted(numeros) == list(range(1, len(QUADROS) + 1))


def test_the_evaluation_frame_warns_when_the_grid_is_missing(data_root: Path) -> None:
    ultimo = quadros_da_fase(FASES[-1], execucao_de(data_root, com_grade=False))[-1]

    assert "src.main" in ultimo.aviso


def test_missing_data_points_to_the_command(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="src.main"):
        read_seed(tmp_path, SEED)


def test_every_decision_variable_has_a_value_and_a_meaning(data_root: Path) -> None:
    """Nenhuma variavel da tela vem vazia."""
    variaveis = [
        variavel
        for quadro in todos_os_quadros(execucao_de(data_root))
        for variavel in quadro.variaveis
    ]

    assert len(variaveis) > 0

    for variavel in variaveis:
        assert str(variavel.valor) != ""
        assert variavel.significado != ""


def test_the_results_tables_are_written_as_in_the_text(data_root: Path) -> None:
    """Virgula decimal, e os nomes dos mecanismos por extenso."""
    grade = read_grid(data_root)
    comparacao = tabela_da_comparacao(grade.comparison)
    tempo = tabela_do_tempo(grade.timing)

    assert set(comparacao["modelo"]) == {"Random Forest", "XGBoost"}
    assert all("," in valor for valor in comparacao["F1 do modelo"])
    assert list(tempo["mecanismo"]) == ["Regras", "Random Forest", "XGBoost"]


# As curvas de teoria: a unica parte da tela que **descreve** o gerador em vez
# de chama-lo.
#
# Nao ha como evitar. Densidade nao se obtem de um sorteador sem sortear
# muito, e amostrar a cada carga da pagina seria lento e ruidoso, entao o
# grafico deriva a forma fechada enquanto o gerador sorteia: duas
# implementacoes do mesmo modelo, que podem divergir.
#
# O que estes testes fazem e o que a duplicacao permite. Sortear bastante uma
# vez, aqui, e exigir que o histograma bata com a curva. Se alguem trocar a
# distribuicao no gerador e esquecer a teoria, a tela passaria a explicar um
# modelo que nao existe mais, de forma plausivel e sem aviso, numa
# apresentacao. E o mesmo risco que os testes acima guardam para os dados.

AMOSTRAS = 200_000

TOLERANCIA = 0.004
"""Folga absoluta entre a curva e o histograma.

Quatro milesimos: com 200 mil amostras o erro padrao de uma proporcao fica
abaixo de 0,0012, entao a folga e cerca de tres desvios. Frouxa o bastante para
nao falhar por acaso, apertada o bastante para pegar troca de distribuicao: a
diferenca entre Poisson e Pascal na mesma media passa de 0,05 na primeira
barra.
"""

SEMENTE_DA_AMOSTRA = 20260924
"""Semente das amostras destes testes, fixa para nao falharem por sorte.

Fora da faixa das replicas e das preparatorias: estes sorteios nao pertencem
ao experimento, so conferem a tela.
"""


def grafico_de_contagem(pagina: Pagina, regime: str) -> pd.DataFrame:
    """O grafico de sessoes por dia util que a aba desenha para um regime."""
    comportamento = next(
        comportamento for comportamento in pagina.comportamentos
        if comportamento.regime == regime
    )
    secao = next(
        secao for secao in comportamento.secoes
        if secao.titulo.startswith("Quantas sessões por dia útil")
    )

    return secao.dados.set_index("sessões no dia")


def test_the_rhythm_charts_match_the_generator() -> None:
    """As barras de Poisson e Pascal sao as que `daily_session_count` produz.

    As curvas moram na pagina Comportamentos desde 26/09. Antes estavam nas
    teorias das variaveis do M2, que foram removidas para nao haver duas
    explicacoes do mesmo numero.
    """
    _, pagina = pagina_da_semente(SAMPLE_SEEDS[0])
    rotina = REGIMES["routine"].rhythm
    custodia = REGIMES["occasional_custody"].rhythm

    casos = (
        (grafico_de_contagem(pagina, "routine"), "Poisson", rotina),
        (
            grafico_de_contagem(pagina, "occasional_custody"),
            f"Pascal, n = {custodia.dispersion}",
            custodia,
        ),
    )

    for curva, coluna, ritmo in casos:

        rng = np.random.default_rng(SEMENTE_DA_AMOSTRA)
        sorteado = np.array([
            daily_session_count(rng, ritmo) for _ in range(AMOSTRAS)
        ])

        for contagem, esperado in curva[coluna].items():
            medido = (sorteado == contagem).mean()

            assert abs(medido - esperado) < TOLERANCIA, (
                f"{coluna}: em {contagem} sessoes a curva diz {esperado:.4f} "
                f"e o gerador produz {medido:.4f}"
            )


def test_the_address_chart_calls_the_generator_function() -> None:
    """O grafico de origem **chama** `address_weights`, nao repete a formula.

    A formula era reimplementada ali ate 24/09. Enquanto foi, mudar o
    decaimento no gerador deixaria a tela desenhando o antigo sem nada acusar.
    Igualdade exata, e nao tolerancia: aqui nao ha amostragem, sao as duas
    chamando a mesma funcao.
    """
    for quantidade in (2, 3, 4):
        desenhado = teoria_da_geometrica(
            PRIMARY_ADDRESS_SHARE, quantidade
        ).dados["chance de ser usado"].to_numpy()

        assert np.array_equal(desenhado, address_weights(quantidade))


# A aba de comportamentos.


def pagina_da_semente(seed: int) -> tuple[pd.DataFrame, Pagina]:
    """O trafego do M2 e a aba montada sobre ele."""
    tabelas = build_population(seed, REPOSITORY)
    requests = build_traffic(seed, tabelas.operators, tabelas.keys, TRAFFIC)

    return requests, montar_pagina(requests, tabelas.operators, TRAFFIC, ATTACK)


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_behavior_page_measures_every_session_of_the_traffic(seed: int) -> None:
    """A aba mede o `requests.csv` inteiro, sem perder sessao no caminho.

    As medicoes agrupam por sessao e por regime. Um operador sem regime, ou
    uma sessao que caisse fora do agrupamento, sumiria das contagens sem erro
    nenhum, e os graficos continuariam plausiveis.
    """
    requests, pagina = pagina_da_semente(seed)

    medidas = pagina.quadro["sessões medidas"].sum()

    assert medidas == requests["session_id"].nunique()
    assert tuple(pagina.quadro["regime"]) == ORDEM


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_every_behavior_chart_has_data_and_no_gap(seed: int) -> None:
    """Todo grafico da aba tem linhas, e nenhuma serie tem buraco.

    Buraco numa serie vira barra ausente, que se le como 'zero' na tela.
    """
    _, pagina = pagina_da_semente(seed)

    secoes = list(pagina.gerais)

    for comportamento in pagina.comportamentos:
        secoes += comportamento.secoes

    graficos = [secao.dados for secao in secoes if secao.dados is not None]

    assert graficos

    for dados in graficos:
        assert not dados.empty
        assert dados.notna().all().all()


@pytest.mark.parametrize("seed", SAMPLE_SEEDS)
def test_the_share_charts_add_up_to_one_per_regime(seed: int) -> None:
    """Hora do dia e dia da semana: cada serie e a distribuicao de um regime."""
    _, pagina = pagina_da_semente(seed)

    por_regime = [
        secao for secao in pagina.gerais
        if secao.dados is not None and set(ORDEM) <= set(secao.dados.columns)
    ]

    assert len(por_regime) == 2

    for secao in por_regime:
        somas = secao.dados[list(ORDEM)].sum()

        assert np.allclose(somas, 1.0)


def test_the_attacker_section_exists_only_where_the_attacker_goes() -> None:
    """So o regime que o atacante personifica ganha a secao do atacante."""
    _, pagina = pagina_da_semente(SAMPLE_SEEDS[0])

    com_atacante = [
        comportamento.regime
        for comportamento in pagina.comportamentos
        if any("atacante" in secao.titulo for secao in comportamento.secoes)
    ]

    assert com_atacante == ["occasional_custody"]


def test_the_regime_tabs_only_list_methods_that_differ() -> None:
    """Metodo igual nos tres regimes vai para a visao geral, nao para as abas.

    Ate 27/09 cada aba repetia cinco linhas identicas as das outras duas. Se
    alguem devolver uma linha comum a tabela do regime, este teste acusa.
    """
    tabelas = [
        metodos_do_calendario(REGIMES[nome].rhythm)
        + metodos_da_sessao_do_regime(REGIMES[nome], perfil.name)
        for nome, perfil in zip(ORDEM, sorted(PROFILES, key=lambda p: ORDEM.index(p.regime)))
    ]

    em_todas = set(tabelas[0]).intersection(*tabelas[1:])

    assert not em_todas, f"linhas iguais nos tres regimes: {em_todas}"


@pytest.mark.parametrize("sigma", (0.0, 0.3, 1.0))
def test_the_sigma_table_matches_the_attack_at_every_sigma(sigma: float) -> None:
    """A coluna do sigma escolhido e a que `stealth_of` resolve.

    Nas pontas ela coincide com a coluna vizinha: em 0,0 com a ostensiva, em
    1,0 com a legitima. Se a tabela interpolasse por conta propria, a
    igualdade em 1,0 poderia falhar por arredondamento, que e o defeito que a
    D-082 cuida no gerador.
    """
    regime = REGIMES["occasional_custody"]
    tabela = tabela_do_atacante(sigma, regime, TRAFFIC, ATTACK)
    escolhida = tabela.columns[2]

    if sigma == 0.0:
        assert tabela[escolhida].equals(tabela["σ 0,0 (ostensivo)"])

    if sigma == 1.0:
        assert tabela[escolhida].equals(tabela["σ 1,0 (legítimo)"])


def test_the_size_chart_overlaps_only_as_sigma_grows() -> None:
    """Em sigma 0 as duas distribuicoes quase nao se tocam; em 1 coincidem."""
    regime = REGIMES["occasional_custody"]

    def sobreposicao(sigma: float) -> float:
        dados = tamanhos_da_sessao(sigma, regime, TRAFFIC, ATTACK)
        return float(np.minimum(dados["sessão do atacante"], dados["sessão legítima"]).sum())

    assert sobreposicao(0.0) < 0.1
    assert sobreposicao(1.0) > 0.9


TOLERANCIA_DO_EXCESSO = 0.01
"""Folga da curva do excesso, maior que a das outras curvas.

So uma sessao em vinte se estende, entao das 200 mil sorteadas cerca de 10 mil
entram no histograma, e o erro padrao de uma barra chega a 0,0025. Um centesimo
e cerca de quatro desvios: nao falha por acaso, e ainda pega troca de
distribuicao, que mexe a primeira barra em mais que isso.
"""


def test_the_extension_chart_matches_the_generator() -> None:
    """A geometrica desenhada e a que `draw_request_count` soma a sessao.

    A faixa de um valor so isola o excesso: tudo que passar dele veio da cauda.
    """
    tipico = 20
    rng = np.random.default_rng(SEMENTE_DA_AMOSTRA)
    sorteado = np.array([
        draw_request_count(rng, (tipico, tipico), TRAFFIC) for _ in range(AMOSTRAS)
    ])

    estendidas = sorteado[sorteado > tipico]
    excesso = estendidas - tipico

    assert abs(len(estendidas) / AMOSTRAS - TRAFFIC.long_session_chance) < TOLERANCIA

    curva = secao_do_excesso(TRAFFIC).dados.set_index("requisições a mais")["chance"]

    for requisicoes, esperado in curva.items():
        medido = (excesso == requisicoes).mean()

        assert abs(medido - esperado) < TOLERANCIA_DO_EXCESSO, (
            f"em {requisicoes} a mais a curva diz {esperado:.4f} e o gerador "
            f"produz {medido:.4f}"
        )


@pytest.mark.parametrize("regime", ("routine", "periodic_batch", "occasional_custody"))
def test_the_interval_bars_match_the_generator(regime: str) -> None:
    """As barras da exponencial sao as que `request_instants` produz.

    Os instantes saem com microssegundos, entao aqui nao ha o arredondamento
    do log: o que se compara e o sorteio do gerador, faixa por faixa.
    """
    media = REGIMES[regime].seconds_between_requests
    rng = np.random.default_rng(SEMENTE_DA_AMOSTRA)

    instantes = request_instants(rng, datetime(2026, 1, 5), AMOSTRAS + 1, media)
    intervalos = np.diff(np.array(instantes, dtype="datetime64[us]")) / np.timedelta64(1, "s")

    barras = barras_da_exponencial(media)
    largura = barras["segundos até a próxima"].iloc[1] - barras["segundos até a próxima"].iloc[0]

    for inicio, esperado in zip(barras["segundos até a próxima"], barras["chance"]):
        medido = ((intervalos >= inicio) & (intervalos < inicio + largura)).mean()

        assert abs(medido - esperado) < TOLERANCIA, (
            f"{regime}: na faixa de {inicio} s a barra diz {esperado:.4f} e o "
            f"gerador produz {medido:.4f}"
        )


# A pagina Regras: o baseline por dentro, lido dos arquivos gravados.


def legitimas_e_do_atacante(predicoes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    is_do_atacante = predicoes[LABEL] == 1

    return predicoes[~is_do_atacante], predicoes[is_do_atacante]


def test_the_rules_table_shows_this_seed_thresholds(data_root: Path) -> None:
    """Oito linhas, e cada regra de grandeza com o limiar que o M8 gravou para a semente."""
    arquivos = read_seed(data_root, SEED)
    tabela = tabela_das_regras(arquivos.thresholds)
    por_atributo = arquivos.thresholds.set_index("attribute")["threshold"]

    assert len(tabela) == len(RULE_ATTRIBUTES)

    for atributo, quando in zip(RULE_ATTRIBUTES, tabela["acende quando"]):
        is_de_grandeza = atributo in THRESHOLD_ATTRIBUTES

        if is_de_grandeza:
            assert quando == f"{atributo} > {valor_escrito(float(por_atributo[atributo]))}"
        else:
            assert quando == f"{atributo} = 1"


def test_the_threshold_chart_splits_exactly_at_the_threshold(data_root: Path) -> None:
    """As duas cores somam o aquecimento inteiro, e a de cima e o que a regra pega."""
    arquivos = read_seed(data_root, SEED)
    por_atributo = arquivos.thresholds.set_index("attribute")["threshold"]

    for atributo in THRESHOLD_ATTRIBUTES:
        dados = secao_do_limiar(atributo, arquivos.sessions, arquivos.thresholds).dados
        valores = arquivos.sessions[atributo]
        acima = (valores > por_atributo[atributo]).mean()

        assert dados["até o limiar"].sum() + dados["acima do limiar"].sum() == pytest.approx(1.0)
        assert dados["acima do limiar"].sum() == pytest.approx(acima), atributo


def test_the_cut_the_baseline_uses_reproduces_its_decision(data_root: Path) -> None:
    """No corte de verdade, o grafico da o alarme falso e a revocacao da decisao gravada."""
    predicoes = read_run(data_root, SEED, SIGMA).predictions_rules
    legitimas, do_atacante = legitimas_e_do_atacante(predicoes)
    por_corte = secao_do_corte(predicoes).dados.set_index("corte (regras)")

    no_corte = por_corte.loc[MINIMUM_RULES_FIRED]

    assert no_corte["alarme falso"] == pytest.approx(legitimas["predicted"].mean())
    assert no_corte["revocação"] == pytest.approx(do_atacante["predicted"].mean())
    assert por_corte["alarme falso"].is_monotonic_decreasing
    assert por_corte["revocação"].is_monotonic_decreasing


def test_the_rule_charts_match_the_rule_columns(data_root: Path) -> None:
    """Quantas regras acendem e qual acende saem das colunas do `predictions_rules.csv`."""
    predicoes = read_run(data_root, SEED, SIGMA).predictions_rules
    legitimas, do_atacante = legitimas_e_do_atacante(predicoes)

    por_sessao = secao_regras_por_sessao(predicoes).dados

    assert por_sessao["sessões legítimas"].sum() == pytest.approx(1.0)
    assert por_sessao["sessões do atacante"].sum() == pytest.approx(1.0)

    taxas = secao_qual_regra_dispara(predicoes).dados.set_index("regra")

    for atributo, coluna in zip(RULE_ATTRIBUTES, RULE_COLUMNS):
        assert taxas.loc[atributo, "sessões legítimas"] == pytest.approx(legitimas[coluna].mean())
        assert taxas.loc[atributo, "sessões do atacante"] == pytest.approx(do_atacante[coluna].mean())

    matriz = tabela_da_matriz(predicoes).set_index("medida")["valor"]
    verdadeiros_positivos = int(((predicoes[LABEL] == 1) & (predicoes["predicted"] == 1)).sum())

    assert matriz["verdadeiros positivos"] == str(verdadeiros_positivos)


def test_the_rules_page_reads_the_grid_and_skips_missing_seeds(data_root: Path) -> None:
    """A curva por sigma e o tempo saem da grade; os limiares, so das sementes que rodaram."""
    grade = read_grid(data_root)
    curva = secao_ao_longo_de_sigma(grade.metrics).dados

    assert list(curva.columns) == ["σ", "F1", "revocação", "especificidade"]
    assert not curva.isna().any().any()
    assert "µs" in frase_do_tempo(grade.timing)

    limiares = tabela_dos_limiares_nas_sementes(read_thresholds(data_root, SAMPLE_SEEDS))

    assert list(limiares["atributo"]) == list(THRESHOLD_ATTRIBUTES)
    assert set(limiares["sementes"]) == {1}


def test_the_importance_section_has_every_attribute_and_mechanism(data_root: Path) -> None:
    """A secao da pagina Resultados le a importancia gravada, um atributo por linha."""
    run = read_importance_run(data_root, SEED, SIGMA)
    importancia = run_importance(run, TEST_CONFIGURATION)

    dados = secao_da_importancia(importancia, SIGMA).dados

    assert list(dados.columns) == ["atributo", "Regras", "Random Forest", "XGBoost"]
    assert list(dados["atributo"]) == list(ATTRIBUTES)
    assert not dados.isna().any().any()


def test_the_confusion_matrix_adds_up_to_the_metrics(data_root: Path) -> None:
    """As quatro celulas somam as sessoes do recorte, cada linha soma 100 %, e as contagens
    sao as do `metrics.csv`."""
    metricas = read_grid(data_root).metrics

    for recorte in RECORTES.values():
        for mecanismo in MECHANISMS:
            celulas = matriz_de_confusao(metricas, SIGMA, recorte, mecanismo).set_index("célula")
            is_da_linha = (
                (metricas["sigma"] == SIGMA)
                & (metricas["scope"] == recorte)
                & (metricas["mechanism"] == mecanismo)
            )
            gravada = metricas[is_da_linha].iloc[0]

            linha_do_ataque = celulas.loc[["verdadeiros positivos", "falsos negativos"]]
            linha_da_legitima = celulas.loc[["falsos positivos", "verdadeiros negativos"]]

            assert celulas["contagem"].sum() == gravada["sessions"]
            assert celulas.loc["verdadeiros positivos", "contagem"] == gravada["true_positives"]
            assert celulas.loc["falsos positivos", "contagem"] == gravada["false_positives"]
            assert linha_do_ataque["fração da verdade"].sum() == pytest.approx(1.0)
            assert linha_da_legitima["fração da verdade"].sum() == pytest.approx(1.0)
