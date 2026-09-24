"""M1 population: gera as duas tabelas estaticas do experimento.

Le apenas a semente e escreve `operators.csv` e `keys.csv`. Deterministico:
mesma semente, mesma saida byte a byte.

    python -m src.population --seed 42

Modelo de dominio: 44 operadores em tres perfis legitimos (30 Usuario
Legitimo, 6 Servico Automatizado e 8 Administrador) e um repositorio de 300
chaves distribuidas de forma desigual entre 12 escopos. Um dos 8 administradores
e o personificado pelo atacante em cada execucao; os outros 7 sao o grupo de
contraste (D-035, D-036).
"""
