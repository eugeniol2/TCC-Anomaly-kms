"""Todos os numeros que governam a campanha de ataque (M3).

Cada dimensao de sigma tem **dois extremos**: o ostensivo, que vale em sigma 0,
e o furtivo, que vale em sigma 1. O furtivo nao e escolhido aqui — ele e
importado do M2, porque em sigma 1 a sessao comprometida precisa sair da
**mesma distribuicao** que uma sessao legitima daquele administrador (D-082).
Repetir o valor aqui criaria duas fontes para o mesmo numero, e bastaria alguem
calibrar o M2 sem lembrar deste arquivo para os dois extremos se descolarem.

Por isso este arquivo declara apenas os extremos **ostensivos** e o tamanho da
campanha. O furtivo e uma importacao, e a convergencia em sigma 1 deixa de
depender de coincidencia numerica.
"""

from __future__ import annotations

from dataclasses import dataclass

# Tamanho da campanha (D-081). Fixo nas 11 condicoes de sigma.

CAMPAIGN_SESSIONS = 58
"""Sessoes que o atacante abre nas semanas 5 a 8.

Sao 1,5x as 39 sessoes legitimas que o administrador alvo tem na mediana do
periodo avaliado. Com a particao 60/40 da D-070, rendem **23 positivas no
holdout** — a contagem absoluta que a armadilha de Athapaththu et al. cobra.
"""

# Extremos ostensivos, que valem em sigma 0 (D-082).

OSTENSIVE_REQUEST_INTERVAL = 6.0
"""Segundos entre requisicoes numa varredura. O legitimo do admin e 90,0."""

OSTENSIVE_REQUESTS_RANGE = (40, 90)
"""Requisicoes por sessao em sigma 0. O legitimo do admin e 8 a 25."""

OSTENSIVE_DISTINCT_KEYS_RANGE = (30, 70)
"""Chaves distintas por sessao em sigma 0. O legitimo e 3 a 12.

Bem acima do teto de 15 que a D-080 mediu no trafego limpo, e de proposito:
sigma 0 e ostensivo e deve ser facil de pegar.
"""

OSTENSIVE_OPENS_AT = 0
OSTENSIVE_CLOSES_AT = 6
"""Faixa horaria da sessao fora da janela do operador — madrugada."""

OSTENSIVE_STALE_SCOPE_RATE = 0.30
"""Fracao das requisicoes que cai fora do escopo em sigma 0.

E o que a enumeracao produz: quem varre nao sabe onde a fronteira esta, entao
esbarra nela o tempo todo. O legitimo e 0,5 % (D-056).
"""

OSTENSIVE_ABSENT_IDENTIFIER_RATE = 0.10
"""Fracao que pede identificador que nao existe. O legitimo e 0,3 %."""


@dataclass(frozen=True)
class AttackSpecification:
    """Os parametros da campanha, reunidos para viajar por parametro.

    Existe pela mesma razao que a `TrafficSpecification`: o teste troca um
    valor sem editar este arquivo, e a funcao recebe o que precisa em vez de
    ler modulo global.
    """

    campaign_sessions: int = CAMPAIGN_SESSIONS
    ostensive_request_interval: float = OSTENSIVE_REQUEST_INTERVAL
    ostensive_requests_range: tuple[int, int] = OSTENSIVE_REQUESTS_RANGE
    ostensive_distinct_keys_range: tuple[int, int] = OSTENSIVE_DISTINCT_KEYS_RANGE
    ostensive_opens_at: int = OSTENSIVE_OPENS_AT
    ostensive_closes_at: int = OSTENSIVE_CLOSES_AT
    ostensive_stale_scope_rate: float = OSTENSIVE_STALE_SCOPE_RATE
    ostensive_absent_identifier_rate: float = OSTENSIVE_ABSENT_IDENTIFIER_RATE
 