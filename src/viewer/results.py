"""A pagina Resultados: a comparacao por sigma, o tempo e as figuras.

Le as tabelas que o M12 gravou na raiz de dados e as deixa no formato do
trabalho, com virgula decimal. Nao calcula nada: o calculo e do M12.
"""

from __future__ import annotations

import pandas as pd

from src.viewer.formatting import com_virgula

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
