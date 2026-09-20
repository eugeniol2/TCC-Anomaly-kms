"""Parametros que governam a geracao do trafego legitimo.

Todo numero que o M2 usa mora aqui. Nenhum fica escondido dentro de uma funcao
nem dentro de uma chamada de construtor: e este arquivo que se le para saber o
que o gerador faz, e e nele que se mexe para mudar.

A referencia `D-xxx` de cada grupo aponta a entrada de `decisoes.md` que fixou
aqueles valores e diz por que nao poderiam ser outros.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# ── Calendario ─────────────────────────────────────────── D-067, D-069

FIRST_DAY = date(2026, 1, 5)     # segunda-feira; fixa, nunca `date.today()`
WEEK_COUNT = 7                   # 2 de perfil + 1 de calibracao + 4 avaliadas
DAYS_PER_WEEK = 7
BUSINESS_WEEKDAYS = frozenset({0, 1, 2, 3, 4})   # segunda a sexta

# ── Origem de rede ─────────────────────────────────────── D-040, D-058

PRINCIPAL_ADDRESS_SHARE = 0.80   # geometrica truncada; as demais decaem 1 - p

# ── Formato do identificador de chave ──────────────────── D-009, D-056

IDENTIFIER_SPACE = 2**48         # espelhado do M1; um teste guarda a copia
IDENTIFIER_PREFIX = "k_"
IDENTIFIER_DIGITS = 12

# ── Servico automatizado: lote periodico ───────────────── D-058

BATCH_HOURS = (2, 8, 14, 20)     # todos os dias, inclusive fim de semana
BATCH_JITTER_MINUTES = 10        # desvio em torno da hora cheia
BATCH_REQUESTS_RANGE = (20, 40)
BATCH_REQUEST_INTERVAL = 2.0     # segundos, media do exponencial

# ── Usuario legitimo: esporadico ───────────────────────── D-058

SPORADIC_SESSIONS_PER_BUSINESS_DAY = 1.5   # Poisson: variancia igual a media
SPORADIC_OPENS_AT = 8
SPORADIC_CLOSES_AT = 18
SPORADIC_REQUESTS_RANGE = (6, 20)
SPORADIC_REQUEST_INTERVAL = 45.0

# ── Administrador: custodia ocasional ──────────────────── D-058

CUSTODY_SESSIONS_PER_BUSINESS_DAY = 2.0
CUSTODY_OPENS_AT = 9
CUSTODY_CLOSES_AT = 19
CUSTODY_DISPERSION = 2           # Pascal; ausente seria Poisson. SEM ENTRADA
CUSTODY_REQUESTS_RANGE = (8, 25)
CUSTODY_REQUEST_INTERVAL = 90.0

# ── Falhas em trafego legitimo ─────────────────────────── D-056

DEFAULT_STALE_SCOPE_RATE = 0.005   # escopo obsoleto; provisorio ate calibracao
DEFAULT_MISTYPED_RATE = 0.003      # identificador errado; idem

# ── Amplitude da sessao ────────────────────────────────── D-058

DEFAULT_DISTINCT_KEYS_RANGE = (3, 12)   # limitado pelo alcance e pelo tamanho


# ── Como os valores acima se encaixam ──────────────────────────────────


@dataclass(frozen=True)
class ScheduledRhythm:
    """Lotes em horas fixas, todos os dias do calendario."""

    hours: tuple[int, ...]
    jitter_minutes: int


@dataclass(frozen=True)
class ArrivalRhythm:
    """Chegadas aleatorias em dias uteis, dentro de uma janela de horario.

    `dispersion` ausente significa Poisson, onde a variancia iguala a media.
    Presente, e o parametro da Pascal, que produz variancia maior.
    """

    sessions_per_business_day: float
    opens_at: int
    closes_at: int
    dispersion: int | None = None


@dataclass(frozen=True)
class Regime:
    """Como um regime abre sessoes e o que acontece dentro delas."""

    rhythm: ScheduledRhythm | ArrivalRhythm
    requests_range: tuple[int, int]
    seconds_between_requests: float


REGIMES: dict[str, Regime] = {
    "periodic_batch": Regime(
        rhythm=ScheduledRhythm(BATCH_HOURS, BATCH_JITTER_MINUTES),
        requests_range=BATCH_REQUESTS_RANGE,
        seconds_between_requests=BATCH_REQUEST_INTERVAL,
    ),
    "sporadic": Regime(
        rhythm=ArrivalRhythm(
            SPORADIC_SESSIONS_PER_BUSINESS_DAY, SPORADIC_OPENS_AT, SPORADIC_CLOSES_AT
        ),
        requests_range=SPORADIC_REQUESTS_RANGE,
        seconds_between_requests=SPORADIC_REQUEST_INTERVAL,
    ),
    "occasional_custody": Regime(
        rhythm=ArrivalRhythm(
            CUSTODY_SESSIONS_PER_BUSINESS_DAY,
            CUSTODY_OPENS_AT,
            CUSTODY_CLOSES_AT,
            CUSTODY_DISPERSION,
        ),
        requests_range=CUSTODY_REQUESTS_RANGE,
        seconds_between_requests=CUSTODY_REQUEST_INTERVAL,
    ),
}
"""O comportamento de cada regime, e nao de cada perfil.

A coluna `regime` existe para que o M2 dependa do comportamento e nao do rotulo
do perfil. Este dicionario e o unico lugar do modulo que menciona os tres nomes:
todo o resto recebe um `Regime` ja resolvido e nao sabe qual e.
"""


@dataclass(frozen=True)
class TrafficSpecification:
    """Os parametros de uma execucao, ja reunidos.

    Contrato entre a linha de comando, que pode sobrescrever alguns deles, e a
    construcao das requisicoes, que os consome.
    """

    first_day: date = FIRST_DAY
    week_count: int = WEEK_COUNT
    stale_scope_rate: float = DEFAULT_STALE_SCOPE_RATE
    mistyped_rate: float = DEFAULT_MISTYPED_RATE
    distinct_keys_range: tuple[int, int] = DEFAULT_DISTINCT_KEYS_RANGE

    @property
    def day_count(self) -> int:
        """Dias corridos do periodo simulado."""
        return self.week_count * DAYS_PER_WEEK
