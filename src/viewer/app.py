"""Visualizador do pipeline: um passo por clique, com os dados reais.

    streamlit run src/viewer/app.py

Nao e simulacao nem mock. Cada passo chama a funcao de verdade do M1 ou do
M2, com a semente escolhida, e mostra o que ela devolveu.
"""

from __future__ import annotations

import sys
from pathlib import Path

# O streamlit poe a pasta do script no sys.path, nao a raiz do projeto, entao
# `import src...` falharia. O pytest resolve isso pelo pythonpath do
# pyproject.toml, que so vale para ele. Aqui a raiz entra a mao, antes dos
# imports do projeto.
RAIZ_DO_PROJETO = Path(__file__).resolve().parents[2]

if str(RAIZ_DO_PROJETO) not in sys.path:
    sys.path.insert(0, str(RAIZ_DO_PROJETO))

import pandas as pd
import streamlit as st

from src.population.build import build_population
from src.population.parameters import KeyRepositorySpecification
from src.traffic.parameters import TrafficSpecification
from src.viewer.steps import (
    FASES,
    Fase,
    Step,
    population_steps,
    traffic_steps,
)

TITULO = "Como os dados sao gerados, passo a passo"


@st.cache_data(show_spinner="rodando o M1...")
def tabelas_do_m1(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As duas tabelas estaticas, para o M2 consumir."""
    populacao = build_population(seed, KeyRepositorySpecification())

    return populacao.operators, populacao.keys


@st.cache_data(show_spinner="rodando o M1 passo a passo...")
def passos_do_m1(seed: int) -> list[Step]:
    return population_steps(seed, KeyRepositorySpecification())


@st.cache_data(show_spinner="rodando o M2 passo a passo...")
def passos_do_m2(seed: int, foco: str) -> list[Step]:
    operators, keys = tabelas_do_m1(seed)

    return traffic_steps(seed, operators, keys, TrafficSpecification(), foco)


def mostrar(passo: Step, numero: int) -> None:
    """Um passo na tela: o que entrou, o que a funcao faz, o que saiu."""
    st.subheader(f"{numero}. `{passo.funcao}`", divider="gray")
    st.caption(f"{passo.fase}  ·  **{passo.modulo}**  ·  {passo.legenda}")

    st.markdown(passo.explicacao)

    if passo.entrada:
        entrada = "  ·  ".join(f"**{nome}** = {valor}"
                               for nome, valor in passo.entrada.items())
        st.markdown(f"recebe &nbsp; {entrada}")

    if isinstance(passo.saida, pd.DataFrame):
        st.dataframe(passo.saida, use_container_width=True, hide_index=True)
        estatisticas(passo)
    else:
        st.code("\n".join(str(item) for item in passo.saida))


def estatisticas(passo: Step) -> None:
    """O `describe` da tabela, recolhido para nao competir com ela.

    Descreve sempre a tabela **completa**, mesmo quando o passo exibe so as
    linhas do operador em foco: estatistica de recorte engana.

    Com `include="all"` ele cobre tambem as colunas de texto, que e a maior
    parte do que este pipeline produz: `count`, `unique`, `top` e `freq`.
    """
    frame = passo.para_descrever()

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return

    recortado = passo.completa is not None
    titulo = "estatisticas da tabela  ·  describe()"

    if recortado:
        titulo += f"  ·  sobre as {len(frame)} linhas completas, nao o recorte"

    with st.expander(titulo):
        st.dataframe(frame.describe(include="all").transpose(),
                     use_container_width=True)
        st.caption(
            "`unique` e quantos valores distintos a coluna tem; `top` e o mais "
            "frequente e `freq` e quantas vezes ele aparece."
        )


def escolher_fase() -> Fase:
    """Seletor das tres fases da arquitetura, no topo da barra lateral.

    A fase vem primeiro porque ela decide o que a tela mostra: as duas
    pendentes nao tem passo nenhum, so a descricao do que farao.
    """
    st.sidebar.title("Fase")

    rotulos = [
        fase.nome if fase.implementada else f"{fase.nome}  (pendente)"
        for fase in FASES
    ]
    escolhido = st.sidebar.radio("Fase", rotulos, index=0, label_visibility="collapsed")

    return FASES[rotulos.index(escolhido)]


def barra_lateral() -> tuple[int, str]:
    """Semente e operador em foco. Mudar qualquer um reinicia a contagem."""
    st.sidebar.divider()
    st.sidebar.subheader("Controles")

    seed = st.sidebar.number_input("Semente", min_value=1, max_value=30, value=1)

    operators, _ = tabelas_do_m1(int(seed))
    nomes = sorted(operators["operator_id"])
    padrao = nomes.index("admin_02") if "admin_02" in nomes else 0
    foco = st.sidebar.selectbox("Operador em foco", nomes, index=padrao)

    st.sidebar.caption(
        "A semente determina tudo. Trocar de semente e sortear outra populacao "
        "e outro calendario — e a mesma semente sempre devolve os mesmos dados."
    )

    return int(seed), str(foco)


def controle_de_passos(passos: list[Step]) -> tuple[int, bool]:
    """Navegacao na barra lateral, para nao rolar a tela a cada avanco.

    Devolve o passo atual e se os anteriores devem continuar visiveis.
    """
    total = len(passos)

    if "atual" not in st.session_state:
        st.session_state.atual = 1

    st.session_state.atual = min(st.session_state.atual, total)

    st.sidebar.divider()
    st.sidebar.subheader("Navegacao")

    if st.sidebar.button("Proximo passo", type="primary", use_container_width=True,
                         disabled=st.session_state.atual >= total):
        st.session_state.atual += 1

    voltar, reiniciar = st.sidebar.columns(2)

    if voltar.button("Voltar", use_container_width=True,
                     disabled=st.session_state.atual <= 1):
        st.session_state.atual -= 1
    if reiniciar.button("Reiniciar", use_container_width=True):
        st.session_state.atual = 1

    atual = st.session_state.atual
    st.sidebar.progress(atual / total, text=f"passo {atual} de {total}")
    st.sidebar.caption(passos[atual - 1].fase)

    empilhar = st.sidebar.toggle(
        "Manter os anteriores na tela", value=False,
        help="Desligado, cada passo aparece sozinho e nao e preciso rolar.",
    )

    st.sidebar.caption("**" + passos[atual - 1].funcao + "**")

    return atual, empilhar


def fase_pendente(fase: Fase) -> None:
    """O que a fase fara, para quem a seleciona antes de ela existir."""
    st.header(fase.nome)
    st.caption(f"`{fase.modulos}`  ·  {fase.execucoes}")

    st.info("Esta fase ainda nao foi implementada.", icon=":material/schedule:")
    st.markdown(fase.descricao)

    st.divider()
    st.caption(
        "O que existe hoje e a Fase 1, que produz o `requests.csv` das sete "
        "semanas. E dele que esta fase partira."
    )


def main() -> None:
    st.set_page_config(page_title=TITULO, layout="wide")
    st.title(TITULO)

    fase = escolher_fase()

    if not fase.implementada:
        fase_pendente(fase)
        return

    seed, foco = barra_lateral()

    passos = passos_do_m1(seed) + passos_do_m2(seed, foco)
    atual, empilhar = controle_de_passos(passos)

    primeiro = 1 if empilhar else atual

    for numero in range(primeiro, atual + 1):
        mostrar(passos[numero - 1], numero)


main()
