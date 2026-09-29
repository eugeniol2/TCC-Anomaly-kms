"""M9: o conjunto avaliado dividido em treino e holdout.

Quatro propriedades vem de decisao e nao de conveniencia:

**Por sessao, e nunca por evento** (D-018, D-061). Cada linha do `sessions.csv` e
uma sessao inteira, entao nenhuma sessao pode cair dos dois lados: o vazamento
que a D-029 procura nao tem por onde acontecer.

**Estratificada por classe** (D-018), e exatamente: cada classe e dividida a
parte, com `HOLDOUT_SHARE` de cada uma no holdout. Sao 23 das 58 positivas.

**A mesma divisao nas onze condicoes de sigma de uma semente** (D-102). O sorteio
sai do fluxo `PARTITION`, que depende so da semente. As sessoes legitimas das
semanas 5 a 8 sao as mesmas em todo sigma, mas a D-083 as renumera em cada
condicao, entao elas sao ordenadas por **operador e instante de abertura**, que
nao mudam. As 58 do atacante sao ordenadas pela **posicao cronologica na
campanha**. Com a mesma ordem e a mesma quantidade em cada classe, o mesmo
sorteio escolhe as mesmas posicoes: a mesma sessao legitima vai para o holdout
em sigma 0,0 e em sigma 1,0, e a k-esima sessao do atacante cai sempre do mesmo
lado.

**Nao escolhe nada olhando o conteudo.** O sorteio ve so a posicao e a classe,
nunca um atributo: a particao nao tem como favorecer um mecanismo.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.random import Generator

from src.dataset.build import LABEL
from src.dataset.parameters import LABEL_PRESENT
from src.globals.rng import PARTITION, stream
from src.partition.parameters import HOLDOUT_SHARE

STABLE_KEY = ("opened_at", "operator_id", "tie")
"""Como uma sessao e reconhecida entre as onze condicoes de sigma.

O `session_id` nao serve: a D-083 renumera as sessoes em cada condicao. Operador
e instante de abertura nao mudam, e so se repetem em 2 casos em ~113 mil sessoes
legitimas nas 30 sementes; `tie` desempata esses casos pela ordem de aparicao.
"""


class Partition(NamedTuple):
    """Os dois lados da divisao, com as colunas do `sessions.csv`."""

    train: pd.DataFrame
    holdout: pd.DataFrame


def holdout_count(total: int) -> int:
    """Quantas sessoes de uma classe vao para o holdout.

    O inteiro mais proximo de `total x HOLDOUT_SHARE`. Com 0,4 o meio exato
    nunca ocorre (seria `4 x total = 10k + 5`, par contra impar), mas a regra
    fica escrita para o caso de a fracao mudar: meio para cima, e nao o meio
    para o par do `round` do Python, que mudaria de lado conforme a paridade.
    """
    return math.floor(total * HOLDOUT_SHARE + 0.5)


def in_stable_order(sessions: pd.DataFrame) -> pd.DataFrame:
    """As sessoes de uma classe numa ordem que nao depende de sigma.

    A ordem do arquivo e cronologica pelo `session_id`, e e ela que decide o
    desempate: duas sessoes do mesmo operador abertas no mesmo segundo recebem
    `tie` 0 e 1 na ordem em que aparecem.
    """
    by_file = sessions.sort_values("session_id", kind="mergesort")
    tie = by_file.groupby(["operator_id", "opened_at"], sort=False).cumcount()

    ordered = by_file.assign(tie=tie).sort_values(list(STABLE_KEY), kind="mergesort")

    return ordered.drop(columns="tie").reset_index(drop=True)


def holdout_mask(rng: Generator, total: int) -> np.ndarray:
    """Quais posicoes de uma classe vao para o holdout.

    Uma permutacao inteira, e as primeiras posicoes dela. Depende so de quantas
    sessoes a classe tem, e nunca do que ha nelas.
    """
    chosen = rng.permutation(total)[: holdout_count(total)]
    mask = np.zeros(total, dtype=bool)
    mask[chosen] = True

    return mask


def split_class(rng: Generator, sessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uma classe dividida em treino e holdout."""
    ordered = in_stable_order(sessions)
    mask = holdout_mask(rng, len(ordered))

    return ordered[~mask], ordered[mask]


def check_label(sessions: pd.DataFrame) -> None:
    """So o conjunto avaliado tem rotulo, e so ele se particiona."""
    has_label = LABEL in sessions.columns

    if not has_label:
        raise ValueError(
            "o `sessions.csv` recebido nao tem rotulo; "
            "e o do aquecimento? O M9 particiona so o periodo avaliado"
        )


def in_file_order(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Junta as classes de um lado e volta a ordem cronologica do arquivo.

    Sem isso as positivas ficariam todas no fim do arquivo, e a posicao da linha
    anunciaria o rotulo, que e o que a D-083 evitou ao renumerar as sessoes.
    """
    joined = pd.concat(parts, ignore_index=True)

    return joined.sort_values("session_id", kind="mergesort", ignore_index=True)[columns]


def build_partition(seed: int, sessions: pd.DataFrame) -> Partition:
    """Do conjunto avaliado de uma condicao aos dois lados da divisao.

    A ordem dos sorteios faz parte do resultado: primeiro as legitimas, depois
    as do atacante, sempre do mesmo fluxo.
    """
    check_label(sessions)

    rng = stream(seed, PARTITION)
    is_positive = sessions[LABEL] == LABEL_PRESENT

    legitimate_train, legitimate_holdout = split_class(rng, sessions[~is_positive])
    attack_train, attack_holdout = split_class(rng, sessions[is_positive])

    columns = list(sessions.columns)

    return Partition(
        train=in_file_order([legitimate_train, attack_train], columns),
        holdout=in_file_order([legitimate_holdout, attack_holdout], columns),
    )
