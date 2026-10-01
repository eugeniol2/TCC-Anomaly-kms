from __future__ import annotations

import numpy as np
import pandas as pd

from src.entities.dataset_generator.dataset.build import LABEL
from src.entities.policy_engine.baseline.build import MECHANISM, RULE_ATTRIBUTES, RULE_COLUMNS
from src.entities.policy_engine.baseline.parameters import MINIMUM_RULES_FIRED
from src.entities.policy_engine.calibration.parameters import (
    THRESHOLD_ATTRIBUTES,
    THRESHOLD_PERCENTILE,
)
from src.formulas.classification import confusion, rates
from src.metrics.evaluation.build import ALL_SESSIONS
from src.viewer.behaviors import frequencia_medida
from src.viewer.formatting import com_virgula, porcento, valor_escrito
from src.viewer.theory import Teoria

O_QUE_CADA_REGRA_VE = {
    "events": "requisições demais na sessão",
    "duration_minutes": "sessão longa demais",
    "requests_per_minute": "ritmo rápido demais",
    "distinct_keys": "chaves distintas demais",
    "failures_per_event": "falhas demais por requisição",
    "denials_per_event": "negações demais por requisição",
    "atypical_hour": "aberta fora da janela habitual",
    "new_source_ip": "origem de rede nunca vista",
}

FAIXAS_DO_HISTOGRAMA = 30


def limiar_de(atributo: str, limiares: pd.DataFrame) -> float:
    """O limiar daquele atributo, como o `thresholds.csv` da semente o grava."""
    por_atributo = limiares.set_index("attribute")["threshold"]

    return float(por_atributo[atributo])


def legitimas_e_do_atacante(predicoes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """As sessões do holdout separadas pelo rótulo, que a página lê e o baseline não."""
    is_do_atacante = predicoes[LABEL] == 1

    return predicoes[~is_do_atacante], predicoes[is_do_atacante]


def resumo_das_regras(seed: int, sigma: float, predicoes: pd.DataFrame) -> str:
    legitimas, do_atacante = legitimas_e_do_atacante(predicoes)

    return (
        "O baseline **não aprende nada**. São oito regras, uma por atributo da sessão. "
        f"Seis comparam o atributo contra um **limiar**, o percentil {THRESHOLD_PERCENTILE} das "
        "sessões do aquecimento daquela semente; duas leem o **perfil histórico** do "
        f"operador. A sessão vira **alerta quando {MINIMUM_RULES_FIRED} ou mais regras "
        "acendem**. Os limiares ficam congelados desde o fim do aquecimento, e o rótulo "
        "do holdout vai junto na saída, mas nunca é consultado (D-033, D-113).\n\n"
        f"Semente **{seed}**, σ **{com_virgula(sigma)}**: {len(predicoes)} sessões no "
        f"holdout, {len(legitimas)} legítimas e {len(do_atacante)} do atacante."
    )


def tabela_das_regras(limiares: pd.DataFrame) -> pd.DataFrame:
    """As oito regras, com o limiar desta semente nas seis de grandeza."""
    linhas = []

    for atributo in RULE_ATTRIBUTES:
        is_de_grandeza = atributo in THRESHOLD_ATTRIBUTES

        if is_de_grandeza:
            limiar = valor_escrito(limiar_de(atributo, limiares))
            tipo = f"grandeza: percentil {THRESHOLD_PERCENTILE} do aquecimento"
            dispara = f"{atributo} > {limiar}"
        else:
            tipo = "histórico: o perfil do operador"
            dispara = f"{atributo} = 1"

        linhas.append({
            "regra": O_QUE_CADA_REGRA_VE[atributo],
            "tipo": tipo,
            "acende quando": dispara,
        })

    return pd.DataFrame(linhas)


def bordas_do_histograma(valores: pd.Series) -> np.ndarray:
    """Uma barra por valor nas contagens; trinta faixas iguais nos contínuos."""
    menor = float(valores.min())
    maior = float(valores.max())
    is_contagem = pd.api.types.is_integer_dtype(valores)
    is_constante = menor == maior

    if is_contagem or is_constante:
        return np.arange(menor, maior + 2) - 0.5

    return np.linspace(menor, maior, FAIXAS_DO_HISTOGRAMA + 1)


def secao_do_limiar(atributo: str, sessoes: pd.DataFrame, limiares: pd.DataFrame) -> Teoria:
    """O aquecimento daquele atributo, separado no limiar que sai dele."""
    valores = sessoes[atributo]
    limiar = limiar_de(atributo, limiares)
    total = len(valores)

    bordas = bordas_do_histograma(valores)
    ate_o_limiar, _ = np.histogram(valores[valores <= limiar], bins=bordas)
    acima_do_limiar, _ = np.histogram(valores[valores > limiar], bins=bordas)
    centros = (bordas[:-1] + bordas[1:]) / 2
    passam = float((valores > limiar).mean())

    dados = pd.DataFrame({
        atributo: centros.round(4),
        "até o limiar": ate_o_limiar / total,
        "acima do limiar": acima_do_limiar / total,
    })

    return Teoria(
        titulo=f"O limiar de `{atributo}`",
        texto=(
            f"O limiar é o percentil {THRESHOLD_PERCENTILE} das **{total} sessões** das quatro "
            f"semanas de aquecimento desta semente, todas legítimas: "
            f"**{valor_escrito(limiar)}**. A regra acende quando a sessão passa "
            f"**estritamente** dele, o que acontece em **{porcento(passam, 2)}** do "
            "tráfego limpo."
        ),
        dados=dados,
        rotulo_x=atributo,
        rotulo_y="fração do aquecimento",
        leitura=(
            "A segunda cor é a ponta mais alta do aquecimento: é ali que a regra começa "
            "a acender. O limiar sai só do tráfego limpo, antes de existir ataque, e "
            "não muda depois (D-033, D-046)."
        ),
    )


def secao_regras_por_sessao(predicoes: pd.DataFrame) -> Teoria:
    """Quantas das oito regras acendem em cada sessão, por classe."""
    legitimas, do_atacante = legitimas_e_do_atacante(predicoes)
    eixo = np.arange(0, len(RULE_ATTRIBUTES) + 1)

    dados = pd.DataFrame({
        "regras acesas": eixo,
        "sessões legítimas": frequencia_medida(legitimas["rules_fired"], eixo),
        "sessões do atacante": frequencia_medida(do_atacante["rules_fired"], eixo),
    })

    nenhuma = float((legitimas["rules_fired"] == 0).mean())
    alertas_legitimos = float((legitimas["rules_fired"] >= MINIMUM_RULES_FIRED).mean())
    alertas_do_atacante = float((do_atacante["rules_fired"] >= MINIMUM_RULES_FIRED).mean())

    return Teoria(
        titulo="Quantas regras acendem por sessão",
        texto=(
            "Cada sessão do holdout acende de 0 a 8 regras. O baseline **alerta a partir "
            f"de {MINIMUM_RULES_FIRED}**, e qualquer combinação serve: duas de grandeza, "
            "duas de histórico ou uma de cada (D-075).\n\n"
            "Cada cor soma 100 % da sua classe. **As barras das legítimas são as mesmas "
            "em todo σ**: as sessões legítimas do holdout e os limiares não mudam com σ "
            "(D-102, D-043). Só as do atacante se movem."
        ),
        dados=dados,
        rotulo_x="regras acesas na sessão",
        rotulo_y="fração da classe",
        leitura=(
            f"As legítimas ficam quase todas em 0: **{porcento(nenhuma)}** não acendem "
            f"regra nenhuma, e só **{porcento(alertas_legitimos, 2)}** chegam a "
            f"{MINIMUM_RULES_FIRED}. Das sessões do atacante, neste σ, "
            f"**{porcento(alertas_do_atacante)}** chegam lá."
        ),
    )


def secao_qual_regra_dispara(predicoes: pd.DataFrame) -> Teoria:
    """Em que fração das sessões de cada classe cada regra acende."""
    legitimas, do_atacante = legitimas_e_do_atacante(predicoes)
    linhas = []
    mais_acesa = RULE_ATTRIBUTES[0]
    maior_taxa = -1.0

    for atributo, coluna in zip(RULE_ATTRIBUTES, RULE_COLUMNS):
        taxa_do_atacante = float(do_atacante[coluna].mean())
        linhas.append({
            "regra": atributo,
            "sessões legítimas": float(legitimas[coluna].mean()),
            "sessões do atacante": taxa_do_atacante,
        })

        is_a_mais_acesa = taxa_do_atacante > maior_taxa

        if is_a_mais_acesa:
            mais_acesa = atributo
            maior_taxa = taxa_do_atacante

    return Teoria(
        titulo="Qual regra acende",
        texto=(
            "Em que fração das sessões de cada classe cada regra acende, neste σ. As seis "
            "de grandeza acendem perto de 1 % das legítimas, que é o que o percentil "
            f"{THRESHOLD_PERCENTILE} promete; numa semente só, com algumas centenas de "
            "legítimas no holdout, a taxa oscila em torno disso. `atypical_hour` acende em "
            "mais, porque a janela habitual de cada operador é estreita."
        ),
        dados=pd.DataFrame(linhas),
        rotulo_x="regra",
        rotulo_y="fração da classe",
        horizontal=True,
        leitura=(
            f"Nas sessões do atacante, a que mais acende é `{mais_acesa}`, em "
            f"**{porcento(maior_taxa)}** delas. Mova o σ na barra lateral: quanto mais "
            "furtivo o atacante, menos regras acendem nele."
        ),
    )


def secao_do_corte(predicoes: pd.DataFrame) -> Teoria:
    """O que cada corte custaria: alarme falso e revocação, de 1 a 8 regras."""
    legitimas, do_atacante = legitimas_e_do_atacante(predicoes)
    linhas = []

    for corte in range(1, len(RULE_ATTRIBUTES) + 1):
        linhas.append({
            "corte (regras)": corte,
            "alarme falso": float((legitimas["rules_fired"] >= corte).mean()),
            "revocação": float((do_atacante["rules_fired"] >= corte).mean()),
        })

    dados = pd.DataFrame(linhas)
    por_corte = dados.set_index("corte (regras)")
    no_corte = por_corte.loc[MINIMUM_RULES_FIRED]
    com_uma = por_corte.loc[1]

    return Teoria(
        titulo=f"Por que {MINIMUM_RULES_FIRED} regras",
        texto=(
            f"O corte em {MINIMUM_RULES_FIRED} foi fixado **antes de qualquer "
            "resultado**, olhando só o alarme falso no tráfego limpo: quando ele foi "
            "escolhido, uma regra bastando marcava cerca de 13 % das sessões legítimas "
            "(D-075). Escolher pelo F1 daria ao baseline o treino que ele não tem.\n\n"
            "O gráfico mostra o que cada corte custaria **nesta execução**. Ele usa o "
            "rótulo do holdout, que o baseline nunca consulta: é leitura do que já se "
            "sabe, e não ajuste."
        ),
        dados=dados,
        rotulo_x="regras exigidas para o alerta",
        rotulo_y="fração da classe",
        forma="linha",
        leitura=(
            f"Em {MINIMUM_RULES_FIRED}: alarme falso de "
            f"**{porcento(no_corte['alarme falso'], 2)}** e revocação de "
            f"**{porcento(no_corte['revocação'])}**. Com uma regra só, o alarme falso "
            f"iria a **{porcento(com_uma['alarme falso'])}**."
        ),
    )


def tabela_da_matriz(predicoes: pd.DataFrame) -> pd.DataFrame:
    """A matriz de confusão desta execução e as taxas que saem dela."""
    contagens = confusion(predicoes[LABEL], predicoes["predicted"])
    taxas = rates(contagens)

    linhas = [
        ("verdadeiros positivos", str(contagens["true_positives"])),
        ("falsos positivos", str(contagens["false_positives"])),
        ("verdadeiros negativos", str(contagens["true_negatives"])),
        ("falsos negativos", str(contagens["false_negatives"])),
        ("F1", com_virgula(taxas["f1"], 3)),
        ("precisão", com_virgula(taxas["precision"], 3)),
        ("revocação", com_virgula(taxas["recall"], 3)),
        ("especificidade", com_virgula(taxas["specificity"], 3)),
        ("acurácia", com_virgula(taxas["accuracy"], 3)),
    ]

    return pd.DataFrame(linhas, columns=["medida", "valor"])


def secao_ao_longo_de_sigma(metricas: pd.DataFrame) -> Teoria:
    """F1, revocação e especificidade do baseline, na mediana das sementes, por σ."""
    is_do_baseline = metricas["mechanism"] == MECHANISM
    is_do_holdout_inteiro = metricas["scope"] == ALL_SESSIONS
    do_baseline = metricas[is_do_baseline & is_do_holdout_inteiro]

    medianas = do_baseline.groupby("sigma")[["f1", "recall", "specificity"]].median()
    dados = pd.DataFrame({
        "σ": medianas.index,
        "F1": medianas["f1"].to_numpy(),
        "revocação": medianas["recall"].to_numpy(),
        "especificidade": medianas["specificity"].to_numpy(),
    })

    primeira = dados.iloc[0]
    ultima = dados.iloc[-1]
    sementes = do_baseline["seed"].nunique()

    return Teoria(
        titulo="O baseline ao longo de σ",
        texto=(
            f"Mediana das {sementes} sementes em cada σ, no holdout inteiro. O σ mede a "
            "furtividade do atacante: em 0,0 ele é ostensivo nas cinco dimensões, em 1,0 "
            "é indistinguível do administrador que personifica."
        ),
        dados=dados,
        rotulo_x="σ",
        rotulo_y="mediana das sementes",
        forma="linha",
        leitura=(
            f"A especificidade fica em **{com_virgula(primeira['especificidade'], 3)}** "
            "em todo σ: as sessões legítimas e os limiares são os mesmos nas onze "
            "condições de uma semente (D-102), e só o atacante muda. O que cai é a "
            f"revocação, de **{com_virgula(primeira['revocação'], 3)}** em σ "
            f"{com_virgula(primeira['σ'])} para **{com_virgula(ultima['revocação'], 3)}** "
            f"em σ {com_virgula(ultima['σ'])}: o atacante furtivo acende menos regras."
        ),
    )


def frase_do_tempo(tempo: pd.DataFrame) -> str:
    """O tempo de decisão do baseline, da tabela de tempo da avaliação."""
    do_baseline = tempo[tempo["mechanism"] == MECHANISM].iloc[0]

    return (
        f"O baseline decide em **{com_virgula(do_baseline['median_microseconds'], 2)} µs** "
        f"por sessão, na mediana das {int(do_baseline['runs'])} execuções: compara seis "
        "atributos com seis limiares, lê dois atributos que já chegam como 0 ou 1, e "
        "conta quantas regras acenderam."
    )


def tabela_dos_limiares_nas_sementes(limiares: pd.DataFrame) -> pd.DataFrame:
    """Mediana, faixa e variação de cada limiar entre as sementes."""
    linhas = []

    for atributo in THRESHOLD_ATTRIBUTES:
        valores = limiares.loc[limiares["attribute"] == atributo, "threshold"]
        has_varias_sementes = len(valores) > 1
        variacao = "—"

        if has_varias_sementes:
            coeficiente = valores.std(ddof=1) / valores.mean()
            variacao = porcento(coeficiente)

        linhas.append({
            "atributo": atributo,
            "sementes": len(valores),
            "mediana": valor_escrito(round(float(valores.median()), 4)),
            "mínimo": valor_escrito(round(float(valores.min()), 4)),
            "máximo": valor_escrito(round(float(valores.max()), 4)),
            "variação entre sementes": variacao,
        })

    return pd.DataFrame(linhas)
