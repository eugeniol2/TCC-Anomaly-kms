"""Perfis comportamentais da populacao: quantos operadores cada um tem, quantos
escopos cada operador detem, quantas origens de rede habituais usa e sob que
regime exerce a recuperacao de material (D-035).

Os tres nomes descrevem **papel**, nunca legitimidade. O primeiro chamou-se
`legitimate_user` ate 23/09 e foi renomeado para `end_user` (D-092): o nome
antigo sugeria "nao e o atacante", e nao e isso que ele diz. **Os tres perfis
sao legitimos** — o atacante nao tem perfil proprio, ele age sob a credencial
de um `administrator` real. Quem diz se a sessao e maliciosa e a coluna
`compromised` do `sessions.csv`, e so ela.

A operacao de recuperacao de material e exercida por todos os perfis legitimos.
Se fosse privativa de parte deles, o proprio exercicio da operacao funcionaria
como marcador de perfil e os operadores incapazes de exportar formariam uma
populacao negativa por construcao.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """Uma linha da Tabela 1: quantos operadores, quantos escopos, que regime."""

    name: str
    id_prefix: str
    operators: int
    scopes_each: int
    addresses_range: tuple[int, int]
    regime: str


PROFILES = (
    Profile(
        name="end_user",
        id_prefix="user",
        operators=30,
        scopes_each=1,
        addresses_range=(2, 4),
        regime="routine",
    ),
    Profile(
        name="automated_service",
        id_prefix="service",
        operators=6,
        scopes_each=2,
        addresses_range=(1, 2),
        regime="periodic_batch",
    ),
    Profile(
        name="administrator",
        id_prefix="admin",
        operators=8,
        scopes_each=4,
        addresses_range=(2, 4),
        regime="occasional_custody",
    ),
)
"""As faixas de endereco se sobrepoem de proposito (D-041).

Faixa fixa por perfil deixaria a contagem de enderecos indicar o perfil. Faixa
sorteada por operador da a cada um a sua propria taxa base de origem de rede
nova, o que impede o baseline de tratar "origem nova" com um limiar global em
vez de comparar contra o historico daquele operador.

Servico automatizado fica mais baixo por rodar de um ou dois hosts, e porque
nunca e personificado pelo atacante.
"""


def total_operators() -> int:
    """Soma dos operadores de todos os perfis legitimos."""
    return sum(profile.operators for profile in PROFILES)


def total_scope_assignments() -> int:
    """Quantas atribuicoes de escopo a populacao inteira distribui.

    Limita quantos escopos podem existir. Acima de cerca de um terco deste
    numero, a cobertura completa por sorteio deixa de ser provavel, e acima
    dele e impossivel: sobraria escopo sem nenhum detentor.
    """
    return sum(profile.operators * profile.scopes_each for profile in PROFILES)
