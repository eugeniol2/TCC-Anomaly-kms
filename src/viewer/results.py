"""A pagina Resultados: a comparacao por sigma, o tempo, a importancia e as figuras.

Le as tabelas que o M12 gravou na raiz de dados e as deixa no formato do
trabalho, com virgula decimal. Nao calcula nada: o calculo e do M12.
"""

from __future__ import annotations

import pandas as pd

from src.entities.dataset_generator.dataset.build import ATTRIBUTES
from src.metrics.evaluation.build import ADMINISTRATORS, ALL_SESSIONS, MECHANISMS
from src.viewer.formatting import com_virgula
from src.viewer.theory import Teoria

NOMES = {"rules": "Regras", "random_forest": "Random Forest", "xgboost": "XGBoost"}


def numero(valor: float, casas: int = 3) -> str:
    """Um numero com virgula, ou um traco quando nao existe."""
    return "—" if pd.isna(valor) else com_virgula(valor, casas)


def situacao(linha: pd.Series) -> str:
    """Excluida pela arvore rasa, significativa ou nao (D-028, D-116)."""
    if linha["excluded"]:
        return "excluída (árvore rasa ≥ 0,95)"

    if linha["significant"]:
        return "significativa"

    return "não significativa"


def p_valor(valor: float) -> str:
    if pd.isna(valor):
        return "—"

    return "< 0,001" if valor < 0.001 else numero(valor)


def inteiro_com_milhar(valor: float) -> str:
    """Um inteiro com ponto de milhar, como no texto: 1.234.567."""
    com_virgula_de_milhar = f"{valor:,.0f}"

    return com_virgula_de_milhar.replace(",", ".")


def com_intervalo(
    meios: pd.Series, baixos: pd.Series, altos: pd.Series, casas: int = 3
) -> list[str]:
    """A mediana com o intervalo interquartilico ao lado, uma linha por vez."""
    escritos = []

    for meio, baixo, alto in zip(meios, baixos, altos):
        escritos.append(
            f"{numero(meio, casas)} ({numero(baixo, casas)} a {numero(alto, casas)})"
        )

    return escritos


def tabela_da_comparacao(comparison: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por sigma e modelo, como vai para o texto."""
    return pd.DataFrame({
        "σ": comparison["sigma"].map(com_virgula),
        "modelo": comparison["model"].map(NOMES),
        "F1 do modelo": comparison["median_f1_model"].map(numero),
        "F1 das regras": comparison["median_f1_rules"].map(numero),
        "diferença (IQR)": com_intervalo(
            comparison["median_difference"],
            comparison["q1_difference"],
            comparison["q3_difference"],
        ),
        "p corrigido": comparison["p_holm"].map(p_valor),
        "situação": comparison.apply(situacao, axis=1),
        "F1 do modelo, administradores": comparison["median_f1_model_administrators"].map(numero),
        "F1 das regras, administradores": comparison["median_f1_rules_administrators"].map(numero),
    })


MENOR_QUEDA_LEGIVEL = 0.01
# Abaixo de um centesimo de F1 a queda e ruido das permutacoes, e nao dependencia.


def mais_importante(dados: pd.DataFrame, mecanismo: str) -> str:
    """O atributo de que o mecanismo mais depende, com a queda escrita ao lado.

    Quando nenhuma queda chega a um centesimo de F1, nenhum atributo e indispensavel
    sozinho, e a frase diz isso em vez de apontar o primeiro da lista.
    """
    linha = dados.loc[dados[mecanismo].idxmax()]
    is_nenhum = linha[mecanismo] < MENOR_QUEDA_LEGIVEL

    if is_nenhum:
        return "nenhum atributo sozinho"

    return f"`{linha['atributo']}` ({numero(linha[mecanismo])})"


def secao_da_importancia(importancia: pd.DataFrame, sigma: float) -> Teoria:
    """Quanto o F1 de cada mecanismo cai ao embaralhar cada atributo, neste sigma (D-124)."""
    deste_sigma = importancia[importancia["sigma"] == sigma]
    medianas = deste_sigma.groupby(["attribute", "mechanism"])["f1_drop"].median()
    por_atributo = medianas.unstack("mechanism").reindex(list(ATTRIBUTES))
    sementes = deste_sigma["seed"].nunique()

    dados = pd.DataFrame({"atributo": list(ATTRIBUTES)})

    for mecanismo in MECHANISMS:
        dados[NOMES[mecanismo]] = por_atributo[mecanismo].to_numpy()

    return Teoria(
        titulo=f"De que atributo cada mecanismo depende, em σ {com_virgula(sigma)}",
        texto=(
            f"Quanto o F1 cai, na mediana das {sementes} sementes, quando um atributo do "
            "holdout é embaralhado e os outros ficam como estão. Quanto maior a barra, "
            "mais o mecanismo depende **daquele atributo sozinho**.\n\n"
            "Barra perto de zero não quer dizer que o atributo é ignorado: quer dizer que "
            "ele **não é indispensável**, porque outro carrega a mesma informação. Em σ "
            "0,0 o atacante se denuncia por vários atributos ao mesmo tempo, e embaralhar "
            "um só não custa nada ao Random Forest. Pela mesma razão, atributos que andam "
            "juntos dividem a importância: `events`, `duration_minutes` e "
            "`requests_per_minute`, e no atacante `distinct_keys` com as falhas.\n\n"
            "É **diagnóstico, e não SHAP**: não explica decisão individual (D-059). Entrou "
            "depois de ver os resultados, no lugar da remedição que saiu (D-114, D-124)."
        ),
        dados=dados,
        leitura=(
            f"Neste σ, as regras dependem mais de: {mais_importante(dados, NOMES['rules'])}; "
            f"o Random Forest, de: {mais_importante(dados, NOMES['random_forest'])}; "
            f"o XGBoost, de: {mais_importante(dados, NOMES['xgboost'])}. Mova o σ: de 0,4 "
            "a 0,7 os modelos dependem quase só de `distinct_keys`; em 1,0, o pouco que "
            "lhes sobra é o ritmo da sessão, que separa administrador de usuário, e não "
            "o atacante do administrador (D-118)."
        ),
    )


CELULAS_DA_MATRIZ = (
    ("ataque", "alerta", "true_positives", "verdadeiros positivos"),
    ("ataque", "sem alerta", "false_negatives", "falsos negativos"),
    ("legítima", "alerta", "false_positives", "falsos positivos"),
    ("legítima", "sem alerta", "true_negatives", "verdadeiros negativos"),
)
"""As quatro células: a verdade, a decisão, a coluna do `metrics.csv` e o nome."""

RECORTES = {"holdout inteiro": ALL_SESSIONS, "só administradores": ADMINISTRATORS}


def matriz_de_confusao(
    metricas: pd.DataFrame, sigma: float, recorte: str, mecanismo: str
) -> pd.DataFrame:
    """As quatro células de um mecanismo num σ, somadas nas sementes.

    A fração é sobre a linha, a verdade: na linha do ataque ela é a revocação, na da
    legítima é o alarme falso e a especificidade. Colorir pela contagem deixaria só
    os verdadeiros negativos acesos, porque há dezenas de legítimas por ataque.
    """
    is_do_sigma = metricas["sigma"] == sigma
    is_do_recorte = metricas["scope"] == recorte
    is_do_mecanismo = metricas["mechanism"] == mecanismo
    linhas = metricas[is_do_sigma & is_do_recorte & is_do_mecanismo]

    ataques = int(linhas["true_positives"].sum() + linhas["false_negatives"].sum())
    legitimas = int(linhas["false_positives"].sum() + linhas["true_negatives"].sum())
    celulas = []

    for verdade, decisao, coluna, nome in CELULAS_DA_MATRIZ:
        contagem = int(linhas[coluna].sum())
        da_mesma_verdade = ataques if verdade == "ataque" else legitimas
        fracao = contagem / da_mesma_verdade

        celulas.append({
            "verdade": verdade,
            "decisão": decisao,
            "célula": nome,
            "contagem": contagem,
            "fração da verdade": fracao,
            "rótulo": f"{inteiro_com_milhar(contagem)}  ·  {com_virgula(fracao * 100)} %",
        })

    return pd.DataFrame(celulas)


def leitura_da_matriz(metricas: pd.DataFrame, sigma: float, recorte: str) -> str:
    """O que somam as matrizes, e como se lê cada linha."""
    is_da_condicao = (metricas["sigma"] == sigma) & (metricas["scope"] == recorte)
    is_das_regras = metricas["mechanism"] == "rules"
    linhas = metricas[is_da_condicao & is_das_regras]

    sementes = linhas["seed"].nunique()
    ataques = int(linhas["positives"].sum())
    legitimas = int(linhas["sessions"].sum()) - ataques

    return (
        f"Somadas as {sementes} sementes: **{inteiro_com_milhar(ataques)} sessões do "
        f"atacante** e **{inteiro_com_milhar(legitimas)} legítimas**. Cada linha soma "
        "100 %: na do ataque, a célula de alerta é a **revocação**; na da legítima, a "
        "célula de alerta é o **alarme falso**, e a outra é a especificidade. A cor "
        "segue a fração, não a contagem."
    )


def tabela_do_tempo(timing: pd.DataFrame) -> pd.DataFrame:
    """Microssegundos por sessao de cada mecanismo, com o intervalo interquartilico."""
    return pd.DataFrame({
        "mecanismo": timing["mechanism"].map(NOMES),
        "µs por sessão (IQR)": com_intervalo(
            timing["median_microseconds"],
            timing["q1_microseconds"],
            timing["q3_microseconds"],
            casas=2,
        ),
        "sessões por segundo": timing["median_sessions_per_second"].map(inteiro_com_milhar),
        "execuções": timing["runs"],
    })
