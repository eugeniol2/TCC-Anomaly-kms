"""Parametros que governam a construcao do repositorio de chaves.

Todo numero que o M1 usa mora aqui. E este arquivo que se le para saber o que o
gerador faz, e e nele que se mexe para mudar.

A escala da populacao (quantos operadores de cada perfil, quantos escopos e
quantas origens de rede) mora em `profiles.py`, junto da estrutura que a
descreve.

A referencia `D-xxx` de cada valor aponta a entrada de `decisoes.md` que o fixou
e diz por que nao poderia ser outro.
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Tamanho do repositorio ─────────────────────────────── D-035

DEFAULT_KEYS = 300
"""Chaves no repositorio.

Precisa ser grande o bastante para que a enumeracao conduzida pelo atacante
encontre chaves fora do alcance da credencial e produza negacoes por politica.
"""

DEFAULT_SCOPES = 12
"""Escopos existentes.

Doze permite atribuir 1, 2 e 4 escopos aos tres perfis mantendo sobreposicao
parcial entre operadores, sem que nenhum alcance o repositorio inteiro.
"""

# ── Estado das chaves ──────────────────────────────────── D-038

DEFAULT_DISABLED_RATE = 0.05
"""Fracao que nasce desabilitada.

Precisa existir chave desabilitada ja no inicio do aquecimento para que o
desfecho "erro por chave desabilitada" ocorra em trafego legitimo. Se surgisse
so de operacoes `DisableKey` durante a simulacao, seria rara demais no periodo
avaliado e viraria marcador do atacante.
"""

# ── Desigualdade entre escopos ─────────────────────────── D-037

DEFAULT_SCOPE_FLOOR = 3
"""Piso de chaves por escopo.

A reparticao e deliberadamente desigual, e sem piso um escopo pequeno poderia
ficar vazio, deixando seus detentores sem nada a acessar.
"""

DEFAULT_CONCENTRATION = 2.0
"""Concentracao da Dirichlet que reparte as chaves entre escopos.

Abaixo de 1 concentra demais: em varredura sobre 30 sementes, 0,7 produziu ate
128 chaves num unico escopo, quase metade do repositorio. Acima de 3 uniformiza,
e escopos de tamanho parecido fariam o total de chaves distintas acessadas
variar pouco entre operadores legitimos. Em 2,0 os escopos vao de cerca de 8 a
53 chaves.
"""


# ── Como os valores acima se encaixam ──────────────────────────────────


@dataclass(frozen=True)
class KeyRepositorySpecification:
    """Os parametros de uma execucao, ja reunidos.

    Contrato entre a linha de comando, que pode sobrescrever qualquer um deles,
    e a construcao das chaves, que os consome.
    """

    total_keys: int = DEFAULT_KEYS
    scope_count: int = DEFAULT_SCOPES
    disabled_rate: float = DEFAULT_DISABLED_RATE
    scope_floor: int = DEFAULT_SCOPE_FLOOR
    concentration: float = DEFAULT_CONCENTRATION
