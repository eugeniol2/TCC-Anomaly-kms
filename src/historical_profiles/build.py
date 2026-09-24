"""M6: o que cada operador fez no aquecimento, e que fica congelado ali.

O `historical_profile` **nao se confunde com o `profile`** (D-034). `profile` e o
tipo do operador na tabela de populacao (Usuario Legitimo, Servico
Automatizado, Administrador) e e dado do gerador. O que este modulo produz e
**observacao de comportamento**: a janela horaria em que aquele operador abriu
sessao e os enderecos de onde ele veio, lidos do log.

A diferenca sustenta o argumento do trabalho. O perfil observado e o que um
analista teria; o tipo do operador e o que so o gerador sabe.

Duas propriedades vem de decisoes e nao de conveniencia:

**Sai das quatro semanas de aquecimento, e nunca e recalculado** (D-044,
D-096). Perfil movel absorveria o comportamento do atacante durante o periodo
avaliado, e o baseline passaria a comparar o atacante contra ele mesmo.

**A janela e do menor ao maior horario observado** (D-073). Sem percentil, sem
descarte, sem parametro. O corte de 95 % da D-031 foi abandonado porque nao era
entregavel nesta escala: com mediana de 16 sessoes por perfil, 5 % da 0,8
sessao, e o arredondamento manda descartar zero em quatro perfis de cada cinco.
"""

from __future__ import annotations

import pandas as pd

from src.globals.phases import WARMUP, belongs_to
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
que apareceu no log do aquecimento. A diferenca entre as duas e exatamente o
que produz origem inedita legitima no periodo avaliado (D-040), entao chama-las
igual apagaria o mecanismo.

`session_count` nao alimenta regra nenhuma. Esta ali porque a fragilidade que a
D-073 deixou em aberto (perfil pequeno produz janela estreita e dispara mais)
so e mensuravel se a contagem estiver em disco.
"""


def ruler_period(log: pd.DataFrame) -> pd.DataFrame:
    """O aquecimento inteiro, que desde a D-096 **e** a regua.

    Nao ha mais recorte dentro dele. Ate 23/09 o perfil saia das semanas 1 e 2
    e os limiares da semana 3, e a divisao custava alarme falso sem comprar
    nada: a regua via metade dos dados que podia ver.

    A funcao sobrevive a divisao como **guarda**: ela recusa silenciosamente o
    que nao for do aquecimento, para um log do periodo avaliado nunca virar
    perfil. Perfil que enxergasse o periodo avaliado absorveria o atacante, e
    o desvio que deveria denuncia-lo viraria a normalidade dele (D-044).
    """
    return log[belongs_to(WARMUP, log["timestamp"])]


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


HOUR_FORMAT = "%H:%M:%S"
"""Como a hora do dia e escrita: `09:00:45`, de largura fixa e com zero a esquerda.

**A largura fixa e o que torna a comparacao possivel como texto.** Com zero a
esquerda, a ordem alfabetica coincide com a cronologica: `"09:00:45"` vem
antes de `"14:23:01"` como string e como hora. E a mesma propriedade que faz o
`requests.csv` poder ser ordenado pelo `timestamp` sem converter nada.

Esta coluna foi **fracao de hora** ate 23/09, `9.0125` em vez de `09:00:45`
(D-095). A troca e de legibilidade, e tambem de robustez: a fracao precisa de
arredondamento, e foi dele que nasceu o defeito da D-090: o M6 arredondava a
ponta da janela e o M7 comparava sem arredondar, de modo que a sessao que
**define** a ponta caia fora dela por 1,1 x 10⁻⁵, marcando 37 das 993 sessoes
da regua de entao, as semanas 1 e 2, na semente 1.
Formatar para `HH:MM:SS` e **exato**: o `timestamp` tem resolucao de segundo, e
nada se perde no caminho. Aquela classe de defeito deixa de existir em vez de
ser contida.
"""


def hour_of_day(moments: pd.Series) -> pd.Series:
    """A hora do dia como texto `HH:MM:SS`, sem a data.

    Sem a data de proposito: a janela habitual e sobre a **hora**, nao sobre o
    dia. Duas sessoes das 09:10, em dias diferentes, sao o mesmo ponto dentro
    da janela.

    Nao trunca para a hora cheia: uma sessao das 09:50 e outra das 09:10 tem a
    mesma hora inteira e horarios diferentes, e truncar alargaria toda janela
    em ate uma hora, e a largura da janela e o que decide quantas sessoes
    legitimas disparam.
    """
    return moments.dt.strftime(HOUR_FORMAT)


def window_width_hours(profiles: pd.DataFrame) -> pd.Series:
    """A largura da janela em horas, para quem precisa do numero.

    As pontas sao texto `HH:MM:SS`, que compara e se le, mas nao subtrai.
    Quem quer a largura converte aqui, e so aqui: espalhar a conversao
    seria convidar cada chamador a inventar a sua.
    """
    opens = pd.to_timedelta(profiles["window_opens_at"])
    closes = pd.to_timedelta(profiles["window_closes_at"])

    return (closes - opens).dt.total_seconds() / 3600


def profile_of(openings: pd.DataFrame) -> pd.DataFrame:
    """A janela e as origens de cada operador, do log do aquecimento."""
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

    Recebe o log da fase `warmup` e usa **todas** as quatro semanas dele: o
    aquecimento e a regua, e nao ha recorte a fazer (D-096).
    """
    openings = session_openings(ruler_period(log))
    profiles = profile_of(openings)

    return profiles.sort_values("operator_id", ignore_index=True)[list(COLUMNS)]
