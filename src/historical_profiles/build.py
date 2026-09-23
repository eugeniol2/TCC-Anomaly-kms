"""M6: o que cada operador fez nas semanas 1 e 2, e que fica congelado ali.

O `historical_profile` **nao se confunde com o `profile`** (D-034). `profile` e o
tipo do operador na tabela de populacao — Usuario Legitimo, Servico
Automatizado, Administrador — e e dado do gerador. O que este modulo produz e
**observacao de comportamento**: a janela horaria em que aquele operador abriu
sessao e os enderecos de onde ele veio, lidos do log.

A diferenca sustenta o argumento do trabalho. O perfil observado e o que um
analista teria; o tipo do operador e o que so o gerador sabe.

Duas propriedades vem de decisoes e nao de conveniencia:

**Sai das semanas 1 e 2, e nunca e recalculado** (D-044). Perfil movel
absorveria o comportamento do atacante durante o periodo avaliado, e o baseline
passaria a comparar o atacante contra ele mesmo. O mesmo arquivo serve a
calibracao da semana 3 e as semanas 4 a 7.

**A janela e do menor ao maior horario observado** (D-073). Sem percentil, sem
descarte, sem parametro. O corte de 95 % da D-031 foi abandonado porque nao era
entregavel nesta escala: com mediana de 16 sessoes por perfil, 5 % da 0,8
sessao, e o arredondamento manda descartar zero em quatro perfis de cada cinco.
"""

from __future__ import annotations

import pandas as pd

from src.globals.phases import PROFILE_WEEKS, day_after_last_of, first_day_of
from src.globals.tables import MULTIVALUE_SEPARATOR

COLUMNS = (
    "operator_id",
    "window_opens_at",
    "window_closes_at",
    "observed_ips",
    "session_count",
)
"""As colunas de `historical_profiles.csv` (D-086).

`observed_ips` **nao** e `usual_ips` de `operators.csv`, e o nome difere de
proposito. Aquela e a lista que o gerador sorteou; esta e o subconjunto dela
que apareceu no log das semanas 1 e 2. A diferenca entre as duas e exatamente o
que produz origem inedita legitima no periodo avaliado (D-040), entao chama-las
igual apagaria o mecanismo.

`session_count` nao alimenta regra nenhuma. Esta ali porque a fragilidade que a
D-073 deixou em aberto — perfil pequeno produz janela estreita e dispara mais —
so e mensuravel se a contagem estiver em disco.
"""


def profile_period(log: pd.DataFrame) -> pd.DataFrame:
    """Apenas as semanas 1 e 2, que sao as que constroem o perfil.

    O log de aquecimento cobre tres semanas: a terceira calibra os limiares e
    **nao** entra aqui. Calcular a janela sobre a semana de calibracao daria a
    cobertura por construcao, que e a degeneracao que a D-054 evita.
    """
    first_week, last_week = PROFILE_WEEKS

    opens = pd.Timestamp(first_day_of(first_week))
    closes = pd.Timestamp(day_after_last_of(last_week))

    instants = pd.to_datetime(log["timestamp"])

    return log[(instants >= opens) & (instants < closes)]


def session_openings(log: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por sessao: quem abriu, quando e de onde.

    O instante da sessao e o do **primeiro evento**, e a origem e a do primeiro
    tambem. Origem e uma so por sessao desde o M2, entao tomar a primeira nao
    perde nada e deixa explicito qual e a regra.
    """
    moments = pd.to_datetime(log["timestamp"])
    dated = log.assign(moment=moments)

    grouped = dated.sort_values("moment", kind="stable").groupby("session_id")

    return pd.DataFrame({
        "operator_id": grouped["operator_id"].first(),
        "source_ip": grouped["source_ip"].first(),
        "moment": grouped["moment"].min(),
    })


HOUR_DECIMALS = 4
"""Casas decimais da hora fracionaria, cerca de 0,36 segundo de resolucao.

Existe para o CSV ficar legivel — 8,7364 em vez de 8,736388888888889 — e o
arredondamento **tem de acontecer aqui dentro**, nao em quem chama.

Foi um defeito antes de ser uma constante. O M6 arredondava as pontas da janela
e o M7 comparava a hora sem arredondar, de modo que a sessao que **define** a
ponta caia fora da propria janela por 1,1 x 10⁻⁵: a ponta subia para 8,7364 e a
hora continuava 8,73638… Marcava 37 das 993 sessoes das semanas 1 e 2 na
semente 1, sempre as extremas, e inflava `atypical_hour` em todos os periodos
sem que nada acusasse.
Com o arredondamento dentro desta funcao, os dois lados quantizam pelo mesmo
caminho por construcao, e nao por duas lembrancas coincidirem.
"""


def hour_of_day(moments: pd.Series) -> pd.Series:
    """A hora do dia como fracao, para a janela nao perder os minutos.

    Uma sessao das 09:50 e das 09:10 tem a mesma hora inteira e horarios
    diferentes. Truncar para a hora alargaria toda janela em ate uma hora, e a
    largura da janela e o que decide quantas sessoes legitimas disparam.
    """
    fractional = moments.dt.hour + moments.dt.minute / 60 + moments.dt.second / 3600

    return fractional.round(HOUR_DECIMALS)


def profile_of(openings: pd.DataFrame) -> pd.DataFrame:
    """A janela e as origens de cada operador, do log das semanas 1 e 2."""
    hours = hour_of_day(openings["moment"])
    dated = openings.assign(hour=hours)

    grouped = dated.groupby("operator_id")

    addresses = grouped["source_ip"].apply(
        lambda seen: MULTIVALUE_SEPARATOR.join(sorted(set(seen)))
    )

    return pd.DataFrame({
        "operator_id": grouped.size().index,
        "window_opens_at": grouped["hour"].min().values,
        "window_closes_at": grouped["hour"].max().values,
        "observed_ips": addresses.values,
        "session_count": grouped.size().values,
    })


def build_profiles(log: pd.DataFrame) -> pd.DataFrame:
    """Do log do aquecimento ao perfil historico de cada operador.

    Recebe o log inteiro da fase `warmup` e recorta as semanas 1 e 2 aqui
    dentro, em vez de exigir que quem chama ja recorte: a fatia e propriedade
    do M6, e deixa-la do lado de fora seria mais uma coisa a lembrar.
    """
    openings = session_openings(profile_period(log))
    profiles = profile_of(openings)

    return profiles.sort_values("operator_id", ignore_index=True)[list(COLUMNS)]
