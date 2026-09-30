"""Como os números aparecem na tela: a convenção do texto do trabalho.

Vírgula decimal e espaço antes do símbolo de porcentagem, como no relatório e
na monografia. Moram aqui, e não em cada página, porque até 28/09 a tabela de
variáveis mostrava `0.05` e `0.5%` pelo `str()` do Python, enquanto a página
Comportamentos já escrevia `0,05` e `0,5 %`: a mesma tela com duas convenções.
"""

from __future__ import annotations

from typing import Any


def com_virgula(valor: float, casas: int = 1) -> str:
    """Número com vírgula decimal, que é a convenção do texto do trabalho."""
    return f"{valor:.{casas}f}".replace(".", ",")


def porcento(fracao: float, casas: int = 1) -> str:
    """Fração escrita em porcentagem, com espaço antes do símbolo."""
    return f"{com_virgula(fracao * 100, casas)} %"


def is_faixa_de_inteiros(valor: Any) -> bool:
    """Um par de inteiros, como `(3, 12)`, que a tabela escreve `3 a 12`."""
    is_par = isinstance(valor, tuple) and len(valor) == 2

    if not is_par:
        return False

    for ponta in valor:
        is_inteiro = isinstance(ponta, int)

        if not is_inteiro:
            return False

    return True


def valor_escrito(valor: Any) -> str:
    """Um valor de variável como a tabela o mostra.

    Número real sai com vírgula e sem zeros à direita (`0,05`, `15`); faixa de
    dois inteiros sai como `3 a 12`; lista sai separada por vírgula. O resto,
    texto já escrito, passa como veio.
    """
    is_real = isinstance(valor, float)

    if is_real:
        return f"{valor:g}".replace(".", ",")

    is_faixa = is_faixa_de_inteiros(valor)

    if is_faixa:
        return f"{valor[0]} a {valor[1]}"

    is_lista = isinstance(valor, (tuple, list))

    if is_lista:
        itens = []

        for item in valor:
            itens.append(str(item))

        return ", ".join(itens)

    return str(valor)
