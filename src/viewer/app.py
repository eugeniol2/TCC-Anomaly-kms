"""Visualizador do pipeline: um passo por vez, com os arquivos de verdade.

    streamlit run src/viewer/app.py

Le o que o comando unico gravou em `data/` (D-121). Tres paginas: **Pipeline**,
os treze passos com o que entra e o que sai de cada entidade; **Comportamentos**,
os tres regimes em detalhe e o atacante em cada sigma; e **Resultados**, a
comparacao por sigma, o tempo e as figuras.
"""

from __future__ import annotations

import sys
from pathlib import Path

# O streamlit poe a pasta do script no sys.path, nao a raiz do projeto, entao
# `import src...` falharia. Aqui a raiz entra a mao, antes dos imports do projeto.
RAIZ_DO_PROJETO = Path(__file__).resolve().parents[2]

if str(RAIZ_DO_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_DO_PROJETO))

import pandas as pd
import streamlit as st

from src.models.build import read_configuration
from src.scenario_engine.attack.parameters import AttackSpecification
from src.scenario_engine.traffic.parameters import TrafficSpecification
from src.scenario_engine.traffic.regimes import REGIMES
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
    read_run,
    read_seed,
)
from src.viewer.decisions import como_tabela
from src.viewer.formatting import porcento
from src.viewer.frames import FASES, Execucao, Fase, Painel, Quadro, quadros_da_fase
from src.viewer.results import tabela_da_comparacao, tabela_do_tempo
from src.viewer.theory import Teoria

# ATENCAO: nada de docstring de constante neste arquivo. O streamlit tem
# "magic": uma string solta no nivel do modulo e renderizada na pagina.

RAIZ_DOS_DADOS = RAIZ_DO_PROJETO / layout.DEFAULT_ROOT

# Alturas em pixels: das tabelas, das variaveis de decisao e dos graficos.
ALTURA_DA_TABELA = 320
ALTURA_DAS_VARIAVEIS = 420
ALTURA_DO_GRAFICO = 260

TODOS = "todos"
EXEMPLO_DE_FILTRO = 'outcome != "success"'

# Os tres primeiros tons da paleta de referencia, um conjunto por tema, validados
# juntos para daltonismo. A ordem nao se embaralha.
SLOTS_CLARO = ("#2a78d6", "#eb6834", "#1baf7a")
SLOTS_ESCURO = ("#3987e5", "#d95926", "#199e70")

# Chaves dos controles. `persist_state` guarda o valor ao trocar de pagina.
FASE_ESCOLHIDA = "fase_escolhida"
SEMENTE_ESCOLHIDA = "semente_escolhida"
SIGMA_ESCOLHIDO = "sigma_escolhido"
PASSO_ESCOLHIDO = "passo_escolhido"
EMPILHAR = "empilhar"
SIGMA_DO_ATACANTE = "sigma_do_atacante"
PERSISTE = "session"
SIGMA_PADRAO = 0.5


# Os dados, lidos do disco uma vez e guardados.


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


# Os paineis: uma tabela, com filtro, ou um valor.


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


# As teorias: o grafico de uma distribuicao, com o texto.


def slots_do_tema() -> tuple[str, ...]:
    is_escuro = st.context.theme.get("type") == "dark"

    return SLOTS_ESCURO if is_escuro else SLOTS_CLARO


def mostrar_grafico(teoria: Teoria) -> None:
    """Barras para dominio discreto, linha para continuo."""
    series = list(teoria.dados.columns[1:])
    comum = {
        "data": teoria.dados, "x": teoria.dados.columns[0], "y": series,
        "color": list(slots_do_tema()[:len(series)]),
        "x_label": teoria.rotulo_x, "y_label": teoria.rotulo_y,
        "height": ALTURA_DO_GRAFICO, "use_container_width": True,
    }

    if teoria.forma == "barras":
        st.bar_chart(stack=False, horizontal=teoria.horizontal, **comum)
    else:
        st.line_chart(**comum)


def mostrar_secao(teoria: Teoria) -> None:
    st.markdown(f"### {teoria.titulo}")
    st.markdown(teoria.texto)

    if teoria.dados is not None:
        mostrar_grafico(teoria)

    if teoria.leitura:
        st.caption(f"**O que ler no gráfico.** {teoria.leitura}")


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


# A barra lateral e a navegacao.


def escolher_fase() -> Fase:
    st.sidebar.title("Fase")
    rotulos = [fase.nome for fase in FASES]
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
        format_func=lambda valor: f"{valor:.1f}".replace(".", ","),
        help="0,0 é ostensivo; 1,0 é indistinguível de uma sessão legítima.",
    ))


def andar(rotulos: list[str], passos: int) -> None:
    """Callback dos botoes: move a selecao antes do redesenho, sem sair da faixa."""
    atual = rotulos.index(st.session_state[PASSO_ESCOLHIDO])
    st.session_state[PASSO_ESCOLHIDO] = rotulos[min(max(atual + passos, 0), len(rotulos) - 1)]


def navegacao(quadros: list[Quadro]) -> tuple[int, bool]:
    """Seletor de passo, botoes e progresso. O seletor e a unica fonte do passo."""
    total = len(quadros)
    rotulos = [f"{posicao} de {total}  ·  {quadro.entidade}"
               for posicao, quadro in enumerate(quadros, start=1)]

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


# As tres paginas.


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
        format_func=lambda valor: f"{valor:.1f}".replace(".", ","),
    ))

    especificacoes = (REGIMES[regime], TrafficSpecification(), AttackSpecification())
    st.dataframe(tabela_do_atacante(sigma, *especificacoes),
                 use_container_width=True, hide_index=True)
    mostrar_grafico(Teoria(
        titulo="", texto="", dados=tamanhos_da_sessao(sigma, *especificacoes),
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
    abas = st.tabs(["Visão geral"] + [c.regime for c in pagina.comportamentos])

    with abas[0]:
        st.dataframe(pagina.quadro, use_container_width=True, hide_index=True)

        for secao in pagina.gerais:
            mostrar_secao(secao)

    for aba, comportamento in zip(abas[1:], pagina.comportamentos):
        with aba:
            mostrar_comportamento(comportamento)


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

    if "roc" in figuras:
        st.subheader("Curva ROC")
        st.image(str(figuras["roc"]), use_container_width=True)

    st.subheader("Tempo de inferência")
    st.dataframe(tabela_do_tempo(tabelas.timing), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Pipeline em tres fases", layout="wide")
    paginas = [
        st.Page(pagina_do_pipeline, title="Pipeline", default=True),
        st.Page(pagina_dos_comportamentos, title="Comportamentos"),
        st.Page(pagina_dos_resultados, title="Resultados"),
    ]
    st.navigation(paginas, position="top").run()


# O streamlit roda o arquivo como `__main__`; importado, ele nao abre a navegacao.
if __name__ == "__main__":
    main()
