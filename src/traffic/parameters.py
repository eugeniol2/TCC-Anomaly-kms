"""Parametros que governam a geracao do trafego legitimo.

Todo numero que o M2 usa mora aqui. Nenhum fica escondido dentro de uma funcao
nem dentro de uma chamada de construtor: e este arquivo que se le para saber o
que o gerador faz, e e nele que se mexe para mudar.

A referencia `D-xxx` de cada grupo aponta a entrada de `decisoes.md` que fixou
aqueles valores e diz por que nao poderiam ser outros.

Como os valores dos tres regimes se encaixam e assunto de `regimes.py`, que
importa deste. Aqui ficam os valores; la, a forma que eles preenchem.
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

# ── Administrador: custodia ocasional ──────────────────── D-058, D-071

CUSTODY_SESSIONS_PER_BUSINESS_DAY = 2.0
CUSTODY_OPENS_AT = 9
CUSTODY_CLOSES_AT = 19
CUSTODY_DISPERSION = 2           # n da Pascal; var/media 1,94, ausente seria 1,0
CUSTODY_REQUESTS_RANGE = (8, 25)
CUSTODY_REQUEST_INTERVAL = 90.0

# ── Falhas em trafego legitimo ─────────────────────────── D-056

DEFAULT_STALE_SCOPE_RATE = 0.005   # escopo obsoleto; provisorio ate calibracao
DEFAULT_MISTYPED_RATE = 0.003      # identificador errado; idem

# ── Amplitude da sessao ────────────────────────────────── D-058

DEFAULT_DISTINCT_KEYS_RANGE = (3, 12)   # limitado pelo alcance e pelo tamanho


# ── Como os valores acima se encaixam ──────────────────────────────────


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
