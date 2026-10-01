from __future__ import annotations

import sys
from pathlib import Path

RAIZ_DO_PROJETO = Path(__file__).resolve().parents[2]

if str(RAIZ_DO_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_DO_PROJETO))

import altair as alt
import pandas as pd
import streamlit as st

from src.entities.models.build import read_configuration
from src.entities.policy_engine.calibration.parameters import THRESHOLD_ATTRIBUTES
from src.entities.scenario_engine.attack.parameters import AttackSpecification
from src.entities.scenario_engine.traffic.parameters import TrafficSpecification
from src.entities.scenario_engine.traffic.regimes import REGIMES
from src.metrics.evaluation.build import MECHANISMS
from src.shared import layout
from src.shared.experiment import HYPERPARAMETER_SEARCH_SEED, SEEDS, SIGMAS
from src.viewer.behaviors import (
    Comportamento,
    Pagina,
    montar_pagina,
    tabela_do_atacante,
    tamanhos_da_sessao,
)
from src.viewer.data import (
    GridFiles,
    RunFiles,
    SeedFiles,
    figure_paths,
    read_grid,
    read_importance,
    read_run,
    read_seed,
    read_thresholds,
)
from src.viewer.decisions import como_tabela
from src.viewer.formatting import com_virgula, porcento
from src.viewer.frames import FASES, Execucao, Fase, Painel, Quadro, quadros_da_fase
from src.viewer.results import (
    NOMES,
    RECORTES,
    leitura_da_matriz,
    matriz_de_confusao,
    numero,
    secao_da_importancia,
    tabela_da_comparacao,
    tabela_do_tempo,
)
from src.viewer.rules import (
    frase_do_tempo,
    resumo_das_regras,
    secao_ao_longo_de_sigma,
    secao_do_corte,
    secao_do_limiar,
    secao_qual_regra_dispara,
    secao_regras_por_sessao,
    tabela_da_matriz,
    tabela_das_regras,
    tabela_dos_limiares_nas_sementes,
)
from src.viewer.theory import Teoria

RAIZ_DOS_DADOS = RAIZ_DO_PROJETO / layout.DEFAULT_ROOT

ALTURA_DA_TABELA = 320
ALTURA_DAS_VARIAVEIS = 420
ALTURA_DO_GRAFICO = 260

DOMINIO_DA_QUEDA = [-0.05, 1.0]
ESPESSURA_DA_BARRA = 14
COR_DA_LINHA_DE_BASE = "#8a8f98"
ROTULO_COM_VIRGULA = "replace(format(datum.value, '.1f'), '.', ',')"

ALTURA_DA_MATRIZ = 200
FUNDO_DA_PAGINA_CLARO = "#ffffff"
FUNDO_DA_PAGINA_ESCURO = "#0e1117"
CELULA_VAZIA_CLARA = "#eef2f6"
CELULA_VAZIA_ESCURA = "#1c2530"
TEXTO_SOBRE_CLARO = "#1f2328"
TEXTO_SOBRE_ESCURO = "#ffffff"
PESO_DO_FUNDO_CLARO = 0.45
PESO_DO_FUNDO_ESCURO = 0.35

TODOS = "todos"
EXEMPLO_DE_FILTRO = 'outcome != "success"'

SLOTS_CLARO = ("#2a78d6", "#eb6834", "#1baf7a")
SLOTS_ESCURO = ("#3987e5", "#d95926", "#199e70")

FASE_ESCOLHIDA = "fase_escolhida"
SEMENTE_ESCOLHIDA = "semente_escolhida"
SIGMA_ESCOLHIDO = "sigma_escolhido"
PASSO_ESCOLHIDO = "passo_escolhido"
EMPILHAR = "empilhar"
SIGMA_DO_ATACANTE = "sigma_do_atacante"
ATRIBUTO_DO_LIMIAR = "atributo_do_limiar"
SIGMA_DA_IMPORTANCIA = "sigma_da_importancia"
SIGMA_DA_MATRIZ = "sigma_da_matriz"
RECORTE_DA_MATRIZ = "recorte_da_matriz"
PERSISTE = "session"
SIGMA_PADRAO = 0.5


@st.cache_data(show_spinner="lendo o ramo da semente...")
def semente(seed: int) -> SeedFiles:
    return read_seed(RAIZ_DOS_DADOS, seed)


@st.cache_data(show_spinner="lendo a execução...")
def execucao(seed: int, sigma: float) -> RunFiles:
    return read_run(RAIZ_DOS_DADOS, seed, sigma)


@st.cache_data(show_spinner="lendo a avaliação...")
def grade() -> GridFiles | None:
    try:
        return read_grid(RAIZ_DOS_DADOS)
    except FileNotFoundError:
        return None


@st.cache_data
def configuracao() -> dict[str, dict]:
    busca = layout.preparation_directory(RAIZ_DOS_DADOS, HYPERPARAMETER_SEARCH_SEED)

    try:
        return read_configuration(busca / layout.CONFIGURATION)
    except FileNotFoundError:
        return {}


@st.cache_data(show_spinner="lendo a importância...")
def importancia() -> pd.DataFrame | None:
    try:
        return read_importance(RAIZ_DOS_DADOS)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="lendo os limiares das sementes...")
def limiares_das_sementes() -> pd.DataFrame:
    return read_thresholds(RAIZ_DOS_DADOS, SEEDS)


@st.cache_data(show_spinner="medindo os regimes...")
def comportamentos(seed: int) -> Pagina:
    arquivos = semente(seed)

    return montar_pagina(
        arquivos.requests, arquivos.operators, TrafficSpecification(), AttackSpecification()
    )


def exigir_dados(seed: int) -> None:
    """Para a pagina com uma instrucao clara quando o comando ainda nao rodou."""
    existe = layout.seed_directory(RAIZ_DOS_DADOS, seed).exists()

    if not existe:
        st.error(f"Não há dados da semente {seed} em {RAIZ_DOS_DADOS}. "
                 "Rode antes: `python -m src.main`")
        st.stop()


def controles_do_filtro(quadro: pd.DataFrame, chave: str) -> tuple[str, str]:
    """O seletor de operador e a caixa de expressao, lado a lado."""
    coluna_operador, coluna_expressao = st.columns([1, 3])
    escolhido = TODOS

    if "operator_id" in quadro.columns:
        operadores = [TODOS] + sorted(quadro["operator_id"].unique())
        escolhido = coluna_operador.selectbox("Operador", operadores, key=f"op_{chave}")

    expressao = coluna_expressao.text_input(
        "Filtro", key=f"expr_{chave}", placeholder=EXEMPLO_DE_FILTRO,
        help="Expressão do pandas: `and`, `or`, `not in`, `>`, `==`. "
             "Exemplo: `distinct_keys > 12 and events < 20`",
    )

    return escolhido, expressao


def filtrar(quadro: pd.DataFrame, chave: str) -> pd.DataFrame:
    """Um operador e uma expressao do pandas, acima da tabela.

    Expressao invalida vira aviso, nunca tela quebrada.
    """
    escolhido, expressao = controles_do_filtro(quadro, chave)
    filtrado = quadro if escolhido == TODOS else quadro[quadro["operator_id"] == escolhido]

    if expressao.strip():
        try:
            filtrado = filtrado.query(expressao)
        except Exception as erro:
            st.warning(f"expressão inválida: {erro}", icon=":material/error:")

    if len(filtrado) != len(quadro):
        st.caption(f"**{len(filtrado)}** de {len(quadro)} linhas "
                   f"({porcento(len(filtrado) / len(quadro))})")

    return filtrado


def mostrar_painel(painel: Painel, chave: str) -> None:
    tamanho = painel.tamanho()
    st.markdown(f"**`{painel.nome}`**" + (f"  ·  {tamanho}" if tamanho else ""))

    if painel.legenda:
        st.caption(painel.legenda)

    if isinstance(painel.dado, pd.DataFrame):
        st.dataframe(filtrar(painel.dado, chave), use_container_width=True,
                     hide_index=True, height=ALTURA_DA_TABELA)
    else:
        st.code(str(painel.dado))


def mostrar_lado(titulo: str, paineis: tuple[Painel, ...], chave: str) -> None:
    """As entradas ou as saidas, empilhadas e em largura cheia."""
    st.subheader(titulo)

    for posicao, painel in enumerate(paineis):
        mostrar_painel(painel, f"{chave}_{posicao}")


def is_tema_escuro() -> bool:
    return st.context.theme.get("type") == "dark"


def slots_do_tema() -> tuple[str, ...]:
    return SLOTS_ESCURO if is_tema_escuro() else SLOTS_CLARO


def mostrar_grafico(teoria: Teoria) -> None:
    """Barras para dominio discreto, linha para continuo."""
    eixo_x = teoria.dados.columns[0]
    series = list(teoria.dados.columns[1:])
    cores = list(slots_do_tema()[:len(series)])

    if teoria.forma == "barras":
        st.bar_chart(
            data=teoria.dados, x=eixo_x, y=series, color=cores,
            x_label=teoria.rotulo_x, y_label=teoria.rotulo_y,
            height=ALTURA_DO_GRAFICO, use_container_width=True,
            stack=False, horizontal=teoria.horizontal,
        )
    else:
        st.line_chart(
            data=teoria.dados, x=eixo_x, y=series, color=cores,
            x_label=teoria.rotulo_x, y_label=teoria.rotulo_y,
            height=ALTURA_DO_GRAFICO, use_container_width=True,
        )


def mostrar_secao(teoria: Teoria) -> None:
    st.markdown(f"### {teoria.titulo}")
    st.markdown(teoria.texto)

    if teoria.dados is not None:
        mostrar_grafico(teoria)

    if teoria.leitura:
        st.caption(f"**O que ler no gráfico.** {teoria.leitura}")


def grafico_de_importancia(dados: pd.DataFrame, mecanismo: str, cor: str) -> alt.LayerChart:
    """As oito barras de um mecanismo, no eixo comum aos tres e na ordem dos atributos.
    """
    ordem = list(dados["atributo"])
    tabela = pd.DataFrame({
        "atributo": dados["atributo"],
        "queda": dados[mecanismo],
        "queda do F1": dados[mecanismo].map(numero),
    })

    barras = alt.Chart(tabela).mark_bar(
        color=cor, size=ESPESSURA_DA_BARRA, cornerRadiusEnd=3,
    ).encode(
        x=alt.X("queda:Q", scale=alt.Scale(domain=DOMINIO_DA_QUEDA, nice=False),
                title="queda do F1",
                axis=alt.Axis(labelExpr=ROTULO_COM_VIRGULA)),
        y=alt.Y("atributo:N", sort=ordem, title=None),
        tooltip=[alt.Tooltip("atributo:N"), alt.Tooltip("queda do F1:N")],
    )
    linha_de_base = alt.Chart(pd.DataFrame({"queda": [0.0]})).mark_rule(
        color=COR_DA_LINHA_DE_BASE,
    ).encode(x="queda:Q")

    return alt.layer(barras, linha_de_base).properties(title=mecanismo, height=ALTURA_DO_GRAFICO)


def mostrar_importancia(teoria: Teoria) -> None:
    """Um painel por mecanismo, lado a lado, e os numeros num expansor embaixo."""
    st.markdown(f"### {teoria.titulo}")
    st.markdown(teoria.texto)

    mecanismos = list(teoria.dados.columns[1:])
    colunas = st.columns(len(mecanismos))

    for coluna, mecanismo, cor in zip(colunas, mecanismos, slots_do_tema()):
        coluna.altair_chart(grafico_de_importancia(teoria.dados, mecanismo, cor),
                            use_container_width=True)

    st.caption(f"**O que ler no gráfico.** {teoria.leitura}")

    with st.expander("Os números"):
        numeros = teoria.dados.copy()

        for mecanismo in mecanismos:
            numeros[mecanismo] = numeros[mecanismo].map(numero)

        st.dataframe(numeros, use_container_width=True, hide_index=True)


def misturar(cor: str, fundo: str, peso_do_fundo: float) -> str:
    """Uma cor a caminho do fundo: o peso diz quanto do fundo entra na mistura."""
    canais = []

    for inicio in (1, 3, 5):
        da_cor = int(cor[inicio:inicio + 2], 16)
        do_fundo = int(fundo[inicio:inicio + 2], 16)
        canal = round((1 - peso_do_fundo) * da_cor + peso_do_fundo * do_fundo)
        canais.append(f"{canal:02x}")

    return "#" + "".join(canais)


def grafico_da_matriz(celulas: pd.DataFrame, titulo: str, cor: str) -> alt.LayerChart:
    """A matriz 2 x 2 de um mecanismo: verdade nas linhas, decisao nas colunas."""
    is_escuro = is_tema_escuro()
    fundo = FUNDO_DA_PAGINA_ESCURO if is_escuro else FUNDO_DA_PAGINA_CLARO
    vazia = CELULA_VAZIA_ESCURA if is_escuro else CELULA_VAZIA_CLARA
    texto = TEXTO_SOBRE_ESCURO if is_escuro else TEXTO_SOBRE_CLARO
    peso_do_fundo = PESO_DO_FUNDO_ESCURO if is_escuro else PESO_DO_FUNDO_CLARO
    cheia = misturar(cor, fundo, peso_do_fundo)

    base = alt.Chart(celulas).encode(
        x=alt.X("decisão:N", sort=["alerta", "sem alerta"], title="decisão",
                axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("verdade:N", sort=["ataque", "legítima"], title="verdade"),
    )
    quadros = base.mark_rect(stroke=fundo, strokeWidth=2, cornerRadius=3).encode(
        color=alt.Color("fração da verdade:Q", legend=None,
                        scale=alt.Scale(domain=[0, 1], range=[vazia, cheia])),
        tooltip=[alt.Tooltip("célula:N"), alt.Tooltip("rótulo:N", title="sessões")],
    )
    textos = base.mark_text(fontSize=13, fontWeight="bold", color=texto).encode(
        text="rótulo:N",
    )

    return alt.layer(quadros, textos).properties(title=titulo, height=ALTURA_DA_MATRIZ)


def mostrar_matrizes(metricas: pd.DataFrame, sigma: float, recorte: str) -> None:
    """Uma matriz por mecanismo, lado a lado, somadas nas sementes."""
    colunas = st.columns(len(MECHANISMS))

    for coluna, mecanismo, cor in zip(colunas, MECHANISMS, slots_do_tema()):
        celulas = matriz_de_confusao(metricas, sigma, recorte, mecanismo)
        coluna.altair_chart(grafico_da_matriz(celulas, NOMES[mecanismo], cor),
                            use_container_width=True)

    st.caption(leitura_da_matriz(metricas, sigma, recorte))


def mostrar_variaveis(quadro: Quadro) -> None:
    """As variaveis de decisao, num expansor acima das tabelas."""
    if not quadro.variaveis:
        return

    with st.expander(f"Variáveis de decisão: {len(quadro.variaveis)} que regem esta entidade"):
        st.dataframe(
            como_tabela(quadro.variaveis), use_container_width=True, hide_index=True,
            height=ALTURA_DAS_VARIAVEIS,
            column_config={
                "variável": st.column_config.TextColumn(width="medium"),
                "valor": st.column_config.TextColumn(width="medium"),
                "o que é": st.column_config.TextColumn(width="large"),
            },
        )

        for variavel in quadro.variaveis:
            if variavel.teoria:
                with st.popover(f"Teoria · {variavel.nome}", use_container_width=True):
                    mostrar_secao(variavel.teoria)


def mostrar_quadro(quadro: Quadro, titular: bool) -> None:
    """Titulo, uma descricao curta, as variaveis e as tabelas."""
    if titular:
        st.title(quadro.entidade)
    else:
        st.markdown(f"## {quadro.entidade}")

    st.markdown(quadro.resumo)

    if quadro.aviso:
        st.warning(quadro.aviso, icon=":material/info:")

    mostrar_variaveis(quadro)
    mostrar_lado("Input", quadro.entradas, f"entra{quadro.numero}")
    mostrar_lado("Output", quadro.saidas, f"sai{quadro.numero}")


def escolher_fase() -> Fase:
    st.sidebar.title("Fase")
    rotulos = []

    for fase in FASES:
        rotulos.append(fase.nome)

    st.session_state.setdefault(FASE_ESCOLHIDA, rotulos[0])

    escolhido = st.sidebar.radio("Fase", rotulos, key=FASE_ESCOLHIDA,
                                 label_visibility="collapsed", persist_state=PERSISTE)

    return FASES[rotulos.index(escolhido)]


def escolher_semente() -> int:
    st.session_state.setdefault(SEMENTE_ESCOLHIDA, SEEDS[0])

    return int(st.sidebar.number_input(
        "Semente", min_value=SEEDS[0], max_value=SEEDS[-1],
        key=SEMENTE_ESCOLHIDA, persist_state=PERSISTE,
    ))


def escolher_sigma() -> float:
    st.session_state.setdefault(SIGMA_ESCOLHIDO, SIGMA_PADRAO)

    return float(st.sidebar.select_slider(
        "Furtividade σ", options=list(SIGMAS), key=SIGMA_ESCOLHIDO,
        persist_state=PERSISTE,
        format_func=com_virgula,
        help="0,0 é ostensivo; 1,0 é indistinguível de uma sessão legítima.",
    ))


def andar(rotulos: list[str], passos: int) -> None:
    """Callback dos botoes: move a selecao antes do redesenho, sem sair da faixa."""
    atual = rotulos.index(st.session_state[PASSO_ESCOLHIDO])
    st.session_state[PASSO_ESCOLHIDO] = rotulos[min(max(atual + passos, 0), len(rotulos) - 1)]


def navegacao(quadros: list[Quadro]) -> tuple[int, bool]:
    """Seletor de passo, botoes e progresso. O seletor e a unica fonte do passo."""
    total = len(quadros)
    rotulos = []

    for posicao, quadro in enumerate(quadros, start=1):
        rotulos.append(f"{posicao} de {total}  ·  {quadro.entidade}")

    st.sidebar.divider()
    st.sidebar.subheader("Passo")

    if st.session_state.get(PASSO_ESCOLHIDO) not in rotulos:
        st.session_state[PASSO_ESCOLHIDO] = rotulos[0]

    escolhido = st.sidebar.selectbox("Ir para", rotulos, key=PASSO_ESCOLHIDO,
                                     label_visibility="collapsed", persist_state=PERSISTE)
    atual = rotulos.index(escolhido) + 1

    atras, adiante = st.sidebar.columns(2)
    atras.button("◀ Voltar", key="voltar", use_container_width=True,
                 disabled=atual <= 1, on_click=andar, args=(rotulos, -1))
    adiante.button("Avançar ▶", key="avancar", use_container_width=True,
                   disabled=atual >= total, on_click=andar, args=(rotulos, +1))

    st.sidebar.progress(atual / total, text=f"Passo {atual} de {total}")
    st.sidebar.caption(f"**{quadros[atual - 1].entidade}**  ·  `{quadros[atual - 1].modulos}`")

    st.session_state.setdefault(EMPILHAR, False)
    empilhar = st.sidebar.checkbox("Manter os anteriores na tela", key=EMPILHAR,
                                   persist_state=PERSISTE)

    return atual, empilhar


def pagina_do_pipeline() -> None:
    fase = escolher_fase()
    st.sidebar.divider()
    st.sidebar.subheader("Controles")
    seed = escolher_semente()
    exigir_dados(seed)

    is_avaliado = fase is FASES[-1]
    sigma = escolher_sigma() if is_avaliado else SIGMA_PADRAO

    contexto = Execucao(
        seed, sigma, semente(seed),
        execucao(seed, sigma) if is_avaliado else None,
        grade() if is_avaliado else None,
        configuracao(),
    )
    quadros = quadros_da_fase(fase, contexto)
    atual, empilhar = navegacao(quadros)
    primeiro = 1 if empilhar else atual

    for numero in range(primeiro, atual + 1):
        mostrar_quadro(quadros[numero - 1], numero == primeiro)

        if numero < atual:
            st.divider()


def mostrar_sigma_do_atacante(regime: str) -> None:
    """O slider de sigma, e o que o atacante faz naquela condicao."""
    st.markdown("### O atacante em cada σ")
    st.session_state.setdefault(SIGMA_DO_ATACANTE, SIGMA_PADRAO)
    sigma = float(st.select_slider(
        "Furtividade σ", options=list(SIGMAS), key=SIGMA_DO_ATACANTE,
        persist_state=PERSISTE,
        format_func=com_virgula,
    ))

    regime_escolhido = REGIMES[regime]
    trafego = TrafficSpecification()
    ataque = AttackSpecification()

    tabela = tabela_do_atacante(sigma, regime_escolhido, trafego, ataque)
    tamanhos = tamanhos_da_sessao(sigma, regime_escolhido, trafego, ataque)

    st.dataframe(tabela, use_container_width=True, hide_index=True)
    mostrar_grafico(Teoria(
        titulo="", texto="", dados=tamanhos,
        rotulo_x="requisições na sessão", rotulo_y="fração das sessões",
    ))
    st.caption("**O que ler no gráfico.** Quanto mais as duas cores se sobrepõem, "
               "menos o tamanho da sessão sozinho denuncia o atacante.")


def mostrar_comportamento(comportamento: Comportamento) -> None:
    st.caption(f"regime `{comportamento.regime}`  ·  perfil `{comportamento.perfil}`  ·  "
               f"{comportamento.operadores} operadores nesta semente")

    with st.container(border=True):
        st.markdown(comportamento.resumo)

    for secao in comportamento.secoes:
        mostrar_secao(secao)
        st.divider()

    if comportamento.personificado:
        mostrar_sigma_do_atacante(comportamento.regime)


def pagina_dos_comportamentos() -> None:
    st.sidebar.subheader("Controles")
    seed = escolher_semente()
    exigir_dados(seed)

    st.title("Comportamento dos operadores")
    st.markdown("Como cada um dos três regimes usa o KMS: quando abre sessão, quanto "
                "ela dura, o que pede e de onde vem.")

    pagina = comportamentos(seed)
    nomes_das_abas = ["Visão geral"]

    for comportamento in pagina.comportamentos:
        nomes_das_abas.append(comportamento.regime)

    abas = st.tabs(nomes_das_abas)

    with abas[0]:
        st.dataframe(pagina.quadro, use_container_width=True, hide_index=True)

        for secao in pagina.gerais:
            mostrar_secao(secao)

    for aba, comportamento in zip(abas[1:], pagina.comportamentos):
        with aba:
            mostrar_comportamento(comportamento)


def escolher_atributo_do_limiar() -> str:
    st.session_state.setdefault(ATRIBUTO_DO_LIMIAR, THRESHOLD_ATTRIBUTES[0])

    return st.selectbox("Atributo de grandeza", THRESHOLD_ATTRIBUTES,
                        key=ATRIBUTO_DO_LIMIAR, persist_state=PERSISTE)


def mostrar_tabela(quadro: pd.DataFrame) -> None:
    st.dataframe(quadro, use_container_width=True, hide_index=True)


def pagina_das_regras() -> None:
    st.sidebar.subheader("Controles")
    seed = escolher_semente()
    exigir_dados(seed)
    sigma = escolher_sigma()

    arquivos = semente(seed)
    predicoes = execucao(seed, sigma).predictions_rules

    st.title("O baseline de regras")
    st.markdown(resumo_das_regras(seed, sigma, predicoes))

    st.subheader("As oito regras")
    mostrar_tabela(tabela_das_regras(arquivos.thresholds))
    st.divider()

    st.subheader("De onde sai cada limiar")
    atributo = escolher_atributo_do_limiar()
    mostrar_secao(secao_do_limiar(atributo, arquivos.sessions, arquivos.thresholds))
    st.divider()

    mostrar_secao(secao_regras_por_sessao(predicoes))
    st.divider()
    mostrar_secao(secao_qual_regra_dispara(predicoes))
    st.divider()
    mostrar_secao(secao_do_corte(predicoes))
    st.divider()

    st.subheader("A matriz de confusão desta execução")
    mostrar_tabela(tabela_da_matriz(predicoes))

    tabelas = grade()

    if tabelas is not None:
        st.divider()
        mostrar_secao(secao_ao_longo_de_sigma(tabelas.metrics))
        st.markdown(frase_do_tempo(tabelas.timing))

    st.divider()
    st.subheader("Os limiares nas 30 sementes")
    st.markdown("Cada semente tem o seu conjunto, tirado do próprio aquecimento. Limiar "
                "muito instável entre sementes seria fragilidade do baseline (D-043).")
    mostrar_tabela(tabela_dos_limiares_nas_sementes(limiares_das_sementes()))


def escolher_sigma_da_secao(chave: str) -> float:
    """Um σ proprio da secao, na pagina: cada secao dos Resultados olha o seu."""
    st.session_state.setdefault(chave, SIGMA_PADRAO)

    return float(st.select_slider(
        "Furtividade σ", options=list(SIGMAS), key=chave,
        persist_state=PERSISTE, format_func=com_virgula,
    ))


def escolher_recorte_da_matriz() -> str:
    """O holdout inteiro ou so os administradores."""
    nomes = list(RECORTES)
    st.session_state.setdefault(RECORTE_DA_MATRIZ, nomes[0])
    escolhido = st.radio("Recorte", nomes, key=RECORTE_DA_MATRIZ, horizontal=True,
                         persist_state=PERSISTE)

    return RECORTES[escolhido]


def pagina_dos_resultados() -> None:
    st.title("Resultados")
    tabelas = grade()

    if tabelas is None:
        st.error(f"Não há tabelas da avaliação em {RAIZ_DOS_DADOS}. "
                 "Rode antes: `python -m src.main`")
        st.stop()

    st.markdown("Mediana das 30 sementes em cada σ. A diferença é o F1 do modelo menos "
                "o das regras, semente a semente; o teste é o Wilcoxon pareado "
                "bilateral, com a correção de Holm só sobre as condições mantidas.")

    figuras = figure_paths(RAIZ_DOS_DADOS)

    if "f1_sigma" in figuras:
        st.image(str(figuras["f1_sigma"]), use_container_width=True)

    st.subheader("Comparação por σ")
    st.dataframe(tabela_da_comparacao(tabelas.comparison),
                 use_container_width=True, hide_index=True)

    st.subheader("Matriz de confusão")
    coluna_do_sigma, coluna_do_recorte = st.columns(2)

    with coluna_do_sigma:
        sigma_da_matriz = escolher_sigma_da_secao(SIGMA_DA_MATRIZ)

    with coluna_do_recorte:
        recorte = escolher_recorte_da_matriz()

    mostrar_matrizes(tabelas.metrics, sigma_da_matriz, recorte)

    if "roc" in figuras:
        st.subheader("Curva ROC")
        st.image(str(figuras["roc"]), use_container_width=True)

    st.subheader("Tempo de inferência")
    st.dataframe(tabela_do_tempo(tabelas.timing), use_container_width=True, hide_index=True)

    st.subheader("Importância por permutação")
    tabela_da_importancia = importancia()

    if tabela_da_importancia is None:
        st.info("Ainda não há `importance.csv` nesta raiz. Rode: `python -m src.main`")
        return

    sigma = escolher_sigma_da_secao(SIGMA_DA_IMPORTANCIA)
    mostrar_importancia(secao_da_importancia(tabela_da_importancia, sigma))


def main() -> None:
    st.set_page_config(page_title="Pipeline em tres fases", layout="wide")
    paginas = [
        st.Page(pagina_do_pipeline, title="Pipeline", default=True),
        st.Page(pagina_dos_comportamentos, title="Comportamentos"),
        st.Page(pagina_das_regras, title="Regras"),
        st.Page(pagina_dos_resultados, title="Resultados"),
    ]
    st.navigation(paginas, position="top").run()


if __name__ == "__main__":
    main()
