"""A grade experimental: quais sementes e quais condicoes de furtividade.

Uma execucao e identificada pelo par (semente, sigma); a semente sozinha
identifica uma varredura inteira de sigma. Dentro de uma varredura, a populacao,
as chaves, o administrador comprometido e o calendario de sessoes legitimas sao
os mesmos nas 11 condicoes: so o comportamento do atacante muda. E esse
pareamento que permite atribuir a sigma a diferenca observada.

A grade foi fixada antes da primeira execucao e antes de qualquer resultado ter
sido observado, o que e o que sustenta o enquadramento confirmatorio do trabalho
(D-004, D-039).

Nenhum modulo importa isto ainda. O orquestrador das 330 execucoes vai, e este e
o unico lugar onde os valores da grade sao escritos.
"""

from __future__ import annotations

SEEDS = tuple(range(1, 31))
"""As 30 replicas. Cada semente varre as 11 condicoes de sigma.

Lista consecutiva de proposito: e publicavel de forma verificavel e nao permite
suspeita de garimpo de sementes favoraveis. A objecao usual contra sementes
sequenciais nao se aplica aqui, porque os fluxos nao usam a semente como estado
interno: saem de `SeedSequence.spawn`, cuja mistura com efeito avalanche faz
sementes vizinhas produzirem estados iniciais sem correlacao (D-039).
"""

SIGMAS = tuple(step / 10 for step in range(11))
"""Furtividade do atacante, de 0,0 a 1,0 em passo de 0,1.

Em 0,0 o atacante e ostensivo em todas as cinco dimensoes comportamentais: taxa
de requisicoes elevada, varredura ampla de chaves, horario atipico, origem de
rede nao habitual e falhas de autorizacao frequentes. Em 1,0 cada dimensao vale
**exatamente** o que o M2 usaria para o operador personificado, e a sessao
comprometida passa a sair da mesma distribuicao que uma legitima (D-082).

O que **nao** e invariante e o objetivo da campanha. O que sigma preserva e o
numero de sessoes, fixo em 58 nas onze condicoes (D-081); o atacante furtivo
simplesmente consegue menos. O inverso (objetivo fixo e sessoes variaveis)
faria a proporcao de anomalias mudar junto com sigma, e a comparacao entre
condicoes confundiria furtividade com desbalanceamento.
"""

TOTAL_RUNS = len(SEEDS) * len(SIGMAS)
"""330 execucoes."""

HYPERPARAMETER_SEARCH_SEED = 902
"""Semente da execucao preparatoria de hiperparametros, em sigma 0,5.

Reservada, fora da faixa das replicas: a populacao que ajudou a escolher a
configuracao nao pode reaparecer entre as 330 avaliadas, senao o desempenho do
modelo naquela replica vem inflado.

Os limiares do baseline nao usam semente reservada: saem do aquecimento da
propria execucao, um conjunto por semente (D-043).

A faixa 9xx deixa espaco para preparacoes futuras sem risco de colisao, caso a
escala das replicas seja revista para alem de 30 (D-047).
"""

REHEARSAL_SEED = 903
"""Semente do ensaio: o pipeline inteiro, antes das 330, numa semente reservada.

Confere o gerador de ponta a ponta, da populacao a avaliacao, antes de gastar as
execucoes de verdade (D-107). Chamava-se `EXCLUSION_CRITERION_SEED` ate 28/09,
quando tambem calibrava o limiar X de exclusao por trivialidade da D-052; a D-107
tirou o X, e o nome passou a dizer o que a semente faz.

Reservada e distinta da 902 de proposito: reaproveitar o holdout da 902 conflitaria
com a D-045, que o reserva de qualquer papel na selecao.

ORDEM: a 903 roda antes das 330, nao junto. Se acusar defeito do gerador, o achado
precisa vir antes de gastar as execucoes, e o conserto invalidaria as ja feitas.
"""

RESERVED_SEEDS = (HYPERPARAMETER_SEARCH_SEED, REHEARSAL_SEED)
"""As sementes das preparacoes. Nenhuma pertence a `SEEDS`."""
