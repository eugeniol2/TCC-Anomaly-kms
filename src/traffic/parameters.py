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

from src.globals.phases import DAYS_PER_WEEK, FIRST_DAY, WEEK_COUNT

# ── Calendario ─────────────────────────────────────────── D-067, D-069
#
# O ancora e a duracao moram em `globals/phases.py`, e nao aqui, porque nao
# sao do M2: o M4 em diante precisa deles para saber de que fatia de tempo
# esta tratando, e importar do vizinho quebraria a fronteira entre modulos.
# Reexportados para quem ja os pedia daqui.

BUSINESS_WEEKDAYS = frozenset({0, 1, 2, 3, 4})   # dias em que pessoa abre sessao

# ── Origem de rede ─────────────────────────────────────── D-040, D-058

PRIMARY_ADDRESS_SHARE = 0.80     # chance de a sessao vir do endereco principal

# ── Formato do identificador de chave ──────────────────── D-009, D-056
# Usado para forjar identificador que nao existe no repositorio.
IDENTIFIER_SPACE = 2**48         # quantos identificadores o formato comporta
IDENTIFIER_PREFIX = "k_"         # o que vem antes dos digitos
IDENTIFIER_DIGITS = 12           # digitos hexadecimais; espelho do M1

# ── Servico automatizado: lote periodico ───────────────── D-058

BATCH_HOURS = (2, 8, 14, 20)     # horas do lote, todos os dias
BATCH_JITTER_MINUTES = 10        # desvio em torno da hora cheia, random de 10 minutos.
BATCH_REQUESTS_RANGE = (20, 40)  # requisicoes por sessao, sorteado na faixa
BATCH_REQUEST_INTERVAL = 2.0     # segundos entre requisicoes, media do exponencial

# ── Usuario legitimo: esporadico ───────────────────────── D-058

SPORADIC_SESSIONS_PER_BUSINESS_DAY = 1.5   # Poisson: variancia igual a media
SPORADIC_OPENS_AT = 8                      # hora em que a janela de inicio abre
SPORADIC_CLOSES_AT = 18                    # e em que fecha
SPORADIC_REQUESTS_RANGE = (6, 20)          # requisicoes por sessao
SPORADIC_REQUEST_INTERVAL = 45.0           # segundos entre requisicoes

# ── Administrador: custodia ocasional ──────────────────── D-058, D-071

CUSTODY_SESSIONS_PER_BUSINESS_DAY = 2.0   # media; a Pascal abaixo e que dispersa
CUSTODY_OPENS_AT = 9                      # hora em que a janela de inicio abre
CUSTODY_CLOSES_AT = 19                    # e em que fecha
CUSTODY_DISPERSION = 2                    # n da Pascal; var/media 1,94
CUSTODY_REQUESTS_RANGE = (8, 25)          # requisicoes por sessao
CUSTODY_REQUEST_INTERVAL = 90.0           # segundos entre requisicoes

# ── Mistura de operacoes por perfil ────────────────────── D-055

# Fracao das requisicoes de um operador daquele perfil. Nenhuma celula e zero:
# operacao privativa marcaria o perfil por construcao. O atacante sorteia da
# linha do administrador que personifica, entao a mistura nao o denuncia.

USER_OPERATION_MIX = {           # quem so consome dado: decifra na maior parte
    "Decrypt": 0.55,
    "Encrypt": 0.25,
    "DescribeKey": 0.12,
    "ExportKeyMaterial": 0.08,
}
SERVICE_OPERATION_MIX = {        # aplicacao em lote: decifra e cifra em volume
    "Decrypt": 0.45,
    "Encrypt": 0.35,
    "DescribeKey": 0.10,
    "ExportKeyMaterial": 0.10,
}
ADMIN_OPERATION_MIX = {          # custodia: inspeciona e recupera material
    "Decrypt": 0.35,
    "Encrypt": 0.15,
    "DescribeKey": 0.30,
    "ExportKeyMaterial": 0.20,
}

# ── Falhas em trafego legitimo ─────────────────────────── D-056

DEFAULT_STALE_SCOPE_RATE = 0.005   # fracao que aponta para escopo obsoleto
DEFAULT_ABSENT_IDENTIFIER_RATE = 0.003   # fracao que pede identificador inexistente

# ── Amplitude da sessao ────────────────────────────────── D-058

DEFAULT_DISTINCT_KEYS_RANGE = (3, 12)   # chaves distintas por sessao, se couberem


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
    absent_identifier_rate: float = DEFAULT_ABSENT_IDENTIFIER_RATE
    distinct_keys_range: tuple[int, int] = DEFAULT_DISTINCT_KEYS_RANGE

    @property
    def day_count(self) -> int:
        """Dias corridos do periodo simulado."""
        return self.week_count * DAYS_PER_WEEK
