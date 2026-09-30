"""M7: do log de auditoria ao conjunto que os modelos classificam.

Uma linha por sessao (D-061). Nao existe tamanho de janela: nem parametro, nem
sobreposicao, nem resto descartado. Por isso a particao por sessao deixa de
ser regra e vira estrutura: nenhuma linha atravessa fronteira de sessao, entao o
vazamento que a D-029 procura nao pode existir.

**Oito atributos** (D-074, com a troca da D-080), cobrindo as cinco dimensoes
que sigma interpola. Dimensao sem atributo correspondente seria dimensao inerte:
o atacante variaria aquele comportamento e nada mediria a diferenca.

| Atributo | Dimensao de sigma |
|---|---|
| `events` | nenhuma: e o companheiro das razoes |
| `duration_minutes` | taxa |
| `requests_per_minute` | taxa |
| `distinct_keys` | chaves distintas |
| `failures_per_event` | falhas de autorizacao |
| `denials_per_event` | falhas, o sinal de enumeracao |
| `atypical_hour` | horario |
| `new_source_ip` | origem de rede |

**O rotulo entra so na fase avaliada** (D-063). No aquecimento o
`sessions.csv` nao tem coluna de rotulo nenhuma: ele alimenta so o M8, e
calibracao por percentil nao usa rotulo. Coluna que nao existe no arquivo nao
pode ser usada por engano.
"""

from __future__ import annotations

import pandas as pd

from src.entities.dataset_generator.dataset.parameters import (
    LABEL_ABSENT,
    LABEL_PRESENT,
    SHORTEST_MEASURABLE_MINUTES,
)
from src.shared.phases import EVALUATED
from src.shared.tables import MULTIVALUE_SEPARATOR
from src.entities.dataset_generator.historical_profiles.build import hour_of_day
from src.entities.kms.policy import DENIED_BY_POLICY, SUCCESS

IDENTIFIERS = ("session_id", "operator_id", "opened_at")
"""Preservados no conjunto, **nunca expostos como atributo** (D-015).

Ficam porque a analise e por operador e, dentro do operador, por sessao, e
porque sem eles nao ha como auditar um falso positivo. Quem separa
identificador de atributo e o M9 e o M11, e a lista `ATTRIBUTES` abaixo e o
contrato: o que nao esta nela nao entra no modelo.

`opened_at` esta aqui por necessidade do M8, que calibra sobre o **aquecimento**
e precisa poder recusar um conjunto que nao seja dele. Sem ele o M8 teria de
reabrir o `log.csv`, e passaria a depender de dois arquivos para responder uma
pergunta de calendario.
**Ele nao pode virar atributo.** O que o horario tem de informativo ja esta em
`atypical_hour`, que compara contra o historico do operador; o instante cru
deixaria o modelo aprender o calendario do experimento (que as semanas 5 a 8
concentram o ataque) em vez de aprender comportamento.
"""

ATTRIBUTES = (
    "events",
    "duration_minutes",
    "requests_per_minute",
    "distinct_keys",
    "failures_per_event",
    "denials_per_event",
    "atypical_hour",
    "new_source_ip",
)
"""Os oito atributos, e nada alem deles.

Ficam de fora `event_id`, e as formas cruas de `key_id` e `source_ip`:
identificador exposto e a armadilha de Bin Sarhan e Altwaijry (2023), que
chegaram a 100 % em tudo com atributos vindos de URLs.
"""

LABEL = "compromised"


def require_label(sessions: pd.DataFrame) -> None:
    """Recusa um conjunto sem rotulo, como o do aquecimento.

    A particao, o baseline e os modelos so trabalham sobre o periodo avaliado, o
    unico que tem rotulo (D-063).
    """
    has_label = LABEL in sessions.columns

    if not has_label:
        raise ValueError(
            "o conjunto recebido nao tem rotulo; e o do aquecimento? "
            "So o periodo avaliado tem rotulo"
        )


def columns_of(phase: str) -> tuple[str, ...]:
    """As colunas do `sessions.csv` daquela fase.

    No aquecimento sao os identificadores e os oito atributos; na fase
    avaliada, mais o rotulo. A diferenca esta no arquivo, e nao num parametro
    que alguem possa esquecer de passar adiante.
    """
    is_evaluated = phase == EVALUATED

    if is_evaluated:
        return IDENTIFIERS + ATTRIBUTES + (LABEL,)

    return IDENTIFIERS + ATTRIBUTES


def count_failures(outcomes: pd.Series) -> int:
    """Quantos desfechos da sessao nao foram sucesso."""
    return (outcomes != SUCCESS).sum()


def count_denials(outcomes: pd.Series) -> int:
    """Quantos desfechos da sessao foram negacao por politica."""
    return (outcomes == DENIED_BY_POLICY).sum()


def per_session(log: pd.DataFrame) -> pd.DataFrame:
    """As grandezas brutas de cada sessao, antes de virarem atributo.

    O `apply` roda a funcao uma vez por sessao, com os desfechos daquela sessao.
    """
    dated = log.assign(moment=pd.to_datetime(log["timestamp"]))
    grouped = dated.groupby("session_id", sort=True)

    return pd.DataFrame({
        "operator_id": grouped["operator_id"].first(),
        "source_ip": grouped["source_ip"].first(),
        "opened_at": grouped["moment"].min(),
        "closed_at": grouped["moment"].max(),
        "events": grouped.size(),
        "distinct_keys": grouped["key_id"].nunique(),
        "failures": grouped["outcome"].apply(count_failures),
        "denials": grouped["outcome"].apply(count_denials),
    })


def with_rate_attributes(sessions: pd.DataFrame) -> pd.DataFrame:
    """Duracao, taxa e as duas razoes de falha.

    Contagem vira taxa, ou vem acompanhada da contagem de eventos: com sessoes
    de 6 a 40 eventos, contagem bruta confunde sessao longa com sessao intensa.
    `distinct_keys` e a excecao deliberada da D-080, e o companheiro dela e o
    `events`, que esta na lista.
    """
    spans = (sessions["closed_at"] - sessions["opened_at"]).dt.total_seconds() / 60
    measurable = spans.clip(lower=SHORTEST_MEASURABLE_MINUTES)

    return sessions.assign(
        duration_minutes=spans.round(4),
        requests_per_minute=(sessions["events"] / measurable).round(4),
        failures_per_event=(sessions["failures"] / sessions["events"]).round(4),
        denials_per_event=(sessions["denials"] / sessions["events"]).round(4),
    )


def as_address_set(joined: str) -> frozenset[str]:
    """As origens observadas, de texto unido por separador para conjunto."""
    return frozenset(str(joined).split(MULTIVALUE_SEPARATOR)) # |


def read_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """Indexa o perfil por operador, com as origens ja como conjunto."""
    indexed = profiles.set_index("operator_id")
    addresses = indexed["observed_ips"].apply(as_address_set)

    return indexed.assign(seen_addresses=addresses)


def with_history_attributes(
    sessions: pd.DataFrame, profiles: pd.DataFrame
) -> pd.DataFrame:
    """Os dois atributos binarios, lidos contra o perfil do aquecimento.

    Ambos comparam a sessao com o historico **daquele operador**, nunca com um
    limiar global: e o que a D-041 exige ao sortear a quantidade de origens por
    operador em faixas sobrepostas.

    Operador ausente do perfil nao existe nesta escala, porque todo operador abre
    sessao nas quatro semanas de aquecimento, e o `KeyError` que isso
    levantaria e preferivel a um padrao silencioso que decidisse por conta
    propria se a sessao e atipica.
    """
    history = read_profiles(profiles)

    hours = hour_of_day(sessions["opened_at"])
    opens = sessions["operator_id"].map(history["window_opens_at"])
    closes = sessions["operator_id"].map(history["window_closes_at"])

    # Comparacao de texto, e nao de numero. As tres pontas sao `HH:MM:SS`
    # de largura fixa, entao a ordem alfabetica e a cronologica, e a
    # sessao que definiu a ponta bate com ela exatamente, que e o que a
    # fracao de hora nao conseguia garantir (D-090, D-095).
    is_inside = (hours >= opens) & (hours <= closes)

    is_known = []

    for row in sessions.itertuples():
        seen_addresses = history.loc[row.operator_id, "seen_addresses"]
        is_known.append(row.source_ip in seen_addresses)

    return sessions.assign(
        atypical_hour=(~is_inside).astype(int),
        new_source_ip=(~pd.Series(is_known, index=sessions.index)).astype(int),
    )


def with_label(sessions: pd.DataFrame, compromised: pd.DataFrame) -> pd.DataFrame:
    """Marca as sessoes que a campanha abriu (D-063, D-084).

    Presenca na lista e o rotulo. O arquivo tem uma coluna so, entao a juncao e
    um teste de pertinencia e nao um `merge` que poderia duplicar linha.
    """
    marked = frozenset(compromised["session_id"])
    belongs = sessions.index.isin(marked)

    return sessions.assign(
        compromised=pd.Series(belongs, index=sessions.index).map(
            {True: LABEL_PRESENT, False: LABEL_ABSENT}
        )
    )


def build_dataset(
    log: pd.DataFrame,
    profiles: pd.DataFrame,
    phase: str,
    compromised: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Do log de uma fase ao conjunto de sessoes daquela fase.

    `compromised` e obrigatorio na fase avaliada e recusado no aquecimento. A
    assimetria e a da D-063: no aquecimento nao ha rotulo para juntar, e aceitar
    o argumento ali abriria caminho para o arquivo de rotulo influenciar a
    calibracao dos limiares, que precisa ser cega.
    """
    is_evaluated = phase == EVALUATED

    if is_evaluated and compromised is None:
        raise ValueError("a fase avaliada exige `compromised_sessions.csv`")

    if not is_evaluated and compromised is not None:
        raise ValueError(
            "o aquecimento nao recebe rotulo: ele alimenta so o M8, e "
            "calibracao por percentil nao usa rotulo"
        )

    sessions = with_rate_attributes(per_session(log))
    described = with_history_attributes(sessions, profiles)

    if is_evaluated:
        described = with_label(described, compromised)

    dated = described.assign(
        opened_at=described["opened_at"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    )

    return dated.reset_index()[list(columns_of(phase))]
