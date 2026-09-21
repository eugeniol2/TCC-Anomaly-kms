"""Os passos do pipeline, na ordem em que o M1 e o M2 os executam.

Este modulo nao e do pipeline: ele **observa** o pipeline. Chama as funcoes
reais, com os dados reais, e guarda o que cada uma devolveu para que o app
possa mostrar entrada e saida lado a lado. Nada aqui e simulado.

A regra que o mantem seguro: o observador se adapta ao codigo, nunca o
contrario. Se algo for dificil de exibir, o jeito e o app se virar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.globals.rng import POPULATION, TRAFFIC, stream
from src.globals.tables import shuffle_rows
from src.population.keys import build_keys, disable_random_sample, split_keys_by_scope
from src.population.operators import build_operators_covering_pool, holders_by_scope
from src.population.parameters import KeyRepositorySpecification
from src.population.scopes import scope_pool
from src.traffic.build import chronological, plan_sessions, request_rows, with_event_ids
from src.traffic.operators import read_operators
from src.traffic.parameters import TrafficSpecification
from src.traffic.repository import build_repository, keys_by_scope, reach_of


@dataclass(frozen=True)
class Fase:
    """Uma das tres fases da arquitetura, implementada ou nao."""

    nome: str
    modulos: str
    execucoes: str
    descricao: str
    implementada: bool


FASES = (
    Fase(
        nome="Fase 1 · Preparação dos dados",
        modulos="M1 · M2",
        execucoes="30 execuções, uma por semente",
        descricao=(
            "Constrói o mundo estático — quem existe e o que existe — e gera as "
            "sete semanas de tráfego legítimo. Nada aqui depende de σ, porque o "
            "atacante ainda não entrou."
        ),
        implementada=True,
    ),
    Fase(
        nome="Fase 2 · Aquecimento e calibração",
        modulos="M4 · M5 · M6 · M7 · M8",
        execucoes="30 execuções, uma por semente",
        descricao=(
            "Extrai de tráfego limpo as duas referências contra as quais tudo "
            "será medido: o perfil histórico, das semanas 1 e 2, e os limiares do "
            "baseline, da semana 3. As duas ficam congeladas daqui em diante."
        ),
        implementada=False,
    ),
    Fase(
        nome="Fase 3 · Ataque, treino e comparação",
        modulos="M3 · M4 · M5 · M7 · M9 · M10 · M11 · M12",
        execucoes="330 execuções, 11 condições de σ por semente",
        descricao=(
            "Injeta a campanha nas semanas 4 a 7, monta o dataset avaliado, "
            "particiona por sessão e compara o baseline de regras contra os dois "
            "modelos supervisionados. É a única fase que depende de σ, e por isso "
            "a única que roda 330 vezes."
        ),
        implementada=False,
    ),
)
"""As tres fases da arquitetura, na ordem de execucao."""

FASE_1 = FASES[0].nome


@dataclass
class Step:
    """Um passo do pipeline, com o que ele recebeu e o que devolveu."""

    fase: str
    modulo: str
    funcao: str
    explicacao: str
    saida: Any
    legenda: str = ""
    entrada: dict[str, Any] = field(default_factory=dict)

    completa: Any = None
    """A tabela inteira, quando `saida` e um recorte.

    Alguns passos exibem so as linhas do operador em foco, para caber na tela
    e para acompanhar um sujeito so. O `describe` tem de falar do arquivo
    inteiro, nao do recorte, senao as estatisticas enganam.
    """

    def para_descrever(self) -> Any:
        return self.saida if self.completa is None else self.completa


def population_steps(seed: int, specification: KeyRepositorySpecification) -> list[Step]:
    """Refaz o `build_population`, guardando cada resultado intermediario."""
    rng = stream(seed, POPULATION)
    passos: list[Step] = []

    pool = scope_pool(specification.scope_count)
    passos.append(Step(
        fase=FASE_1,
        modulo="M1",
        funcao="scope_pool",
        explicacao="Cria os nomes dos escopos. Nao sorteia nada — e so uma lista.",
        entrada={"quantity": specification.scope_count},
        saida=pool,
        legenda=f"{len(pool)} escopos",
    ))

    operators = build_operators_covering_pool(rng, pool)
    passos.append(Step(
        fase=FASE_1,
        modulo="M1",
        funcao="build_operators_covering_pool",
        explicacao=(
            "Sorteia a populacao: quantos escopos e quantas origens de rede cada "
            "operador tem, conforme o perfil. Repete ate todo escopo ter ao menos "
            "um detentor — sem isso sobrariam chaves que ninguem alcanca."
        ),
        entrada={"pool": f"{len(pool)} escopos"},
        saida=operators,
        legenda=f"{len(operators)} operadores",
    ))

    scope_holders = holders_by_scope(operators)
    passos.append(Step(
        fase=FASE_1,
        modulo="M1",
        funcao="holders_by_scope",
        explicacao="Indice inverso: de cada escopo para quem o detem.",
        entrada={"operators": f"{len(operators)} linhas"},
        saida=pd.DataFrame([
            {"escopo_de_chaves": escopo, "numero_operadores": len(quem)}
            for escopo, quem in sorted(scope_holders.items())
        ]),
        legenda="quantos detentores por escopo",
    ))

    scope_sizes = split_keys_by_scope(rng, pool, specification)
    passos.append(Step(
        fase=FASE_1,
        modulo="M1",
        funcao="split_keys_by_scope",
        explicacao=(
            "Decide quantas chaves cada escopo recebe. Deliberadamente desigual "
            "(D-037): escopos de tamanho uniforme fariam o total de chaves "
            "distintas variar pouco entre operadores, e o atacante ficaria "
            "destacavel por esse atributo sozinho."
        ),
        entrada={"total_keys": specification.total_keys,
                 "scope_floor": specification.scope_floor,
                 "concentration": specification.concentration},
        saida=pd.DataFrame([
            {"escopo_de_chaves": escopo, "numero_chaves": quantas}
            for escopo, quantas in sorted(scope_sizes.items())
        ]),
        legenda=f"soma {sum(scope_sizes.values())}",
    ))

    keys_in_scope_order = build_keys(rng, scope_sizes, scope_holders)
    passos.append(Step(
        fase=FASE_1,
        modulo="M1",
        funcao="build_keys",
        explicacao=(
            "Sorteia os identificadores e atribui proprietario. Identificador "
            "aleatorio, nunca sequencial (D-009): com sequencia, o atacante "
            "enumerando produziria progressao aritmetica e qualquer atributo de "
            "distancia separaria as classes sozinho."
        ),
        entrada={"scope_sizes": f"{len(scope_sizes)} escopos"},
        saida=keys_in_scope_order,
        legenda=f"{len(keys_in_scope_order)} chaves, em ordem de escopo",
    ))

    keys_with_status = disable_random_sample(
        rng, keys_in_scope_order, specification.disabled_rate
    )
    desabilitadas = int((keys_with_status["status"] == "disabled").sum())
    passos.append(Step(
        fase=FASE_1,
        modulo="M1",
        funcao="disable_random_sample",
        explicacao=(
            "Desabilita uma fracao das chaves (D-038). Precisa existir chave "
            "desabilitada desde o inicio para que o desfecho `disabled_key` "
            "ocorra em trafego legitimo, em vez de virar marcador do atacante."
        ),
        entrada={"disabled_rate": specification.disabled_rate},
        saida=keys_with_status,
        legenda=f"{desabilitadas} desabilitadas",
    ))

    keys = shuffle_rows(rng, keys_with_status)
    passos.append(Step(
        fase=FASE_1,
        modulo="M1",
        funcao="shuffle_rows",
        explicacao=(
            "Embaralha as linhas antes de gravar, para que a ordem do arquivo nao "
            "revele o escopo. E por isso que o M2 ordena de novo ao ler: o alcance "
            "de um operador nao pode depender da ordem de gravacao."
        ),
        entrada={"keys": f"{len(keys_with_status)} linhas em ordem de escopo"},
        saida=keys,
        legenda="keys.csv",
    ))

    return passos


def traffic_steps(
    seed: int,
    operators_table: pd.DataFrame,
    keys_table: pd.DataFrame,
    specification: TrafficSpecification,
    operador_foco: str,
) -> list[Step]:
    """Refaz o `build_traffic`, guardando cada resultado intermediario."""
    rng = stream(seed, TRAFFIC)
    passos: list[Step] = []

    operators = read_operators(operators_table)
    passos.append(Step(
        fase=FASE_1,
        modulo="M2",
        funcao="read_operators",
        explicacao=(
            "Converte cada linha do CSV num `Operator`. As colunas `scopes` e "
            "`usual_ips` vem grudadas com `|`, porque CSV e tabela plana, e aqui "
            "viram tuplas. A ordem das linhas e preservada: ela participa do "
            "sorteio mais adiante."
        ),
        entrada={"operators.csv": f"{len(operators_table)} linhas"},
        saida=pd.DataFrame([o._asdict() for o in operators]),
        legenda=f"{len(operators)} objetos Operator",
    ))

    por_escopo = keys_by_scope(keys_table)
    passos.append(Step(
        fase=FASE_1,
        modulo="M2",
        funcao="keys_by_scope",
        explicacao=(
            "Indice inverso das chaves: de cada escopo para as chaves dele. A "
            "tabela responde 'qual o escopo desta chave?'; o M2 precisa do "
            "contrario, porque operador detem escopo, nao chave (D-042)."
        ),
        entrada={"keys.csv": f"{len(keys_table)} linhas"},
        saida=pd.DataFrame([
            {"escopo_de_chaves": escopo, "numero_chaves": len(chaves)}
            for escopo, chaves in sorted(por_escopo.items())
        ]),
        legenda=f"{len(por_escopo)} escopos",
    ))

    foco = next(o for o in operators if o.operator_id == operador_foco)
    alcance = reach_of(foco, por_escopo)
    passos.append(Step(
        fase=FASE_1,
        modulo="M2",
        funcao="reach_of",
        explicacao=(
            f"O {operador_foco} detem **escopos**, nao chaves. Esta funcao soma "
            "as chaves de cada escopo dele, e o total e o que ele **alcanca**.\n\n"
            "Alcancar nao e possuir: a coluna `owner` da tabela de chaves e "
            "metadado e **nenhum modulo a consulta** (D-042). Um operador pode "
            "alcancar dezenas de chaves e nao ser dono de nenhuma — e cerca de "
            "5,6 % deles nao sao donos de chave alguma.\n\n"
            "O resultado e o `in_reach`; o que sobra do repositorio vira "
            "`out_of_reach`, alvo do desvio de escopo obsoleto da D-056."
        ),
        entrada={"operador": operador_foco, "escopos que detem": list(foco.scopes)},
        saida=pd.DataFrame([
            {"escopo_de_chaves": escopo,
             "numero_chaves": len(por_escopo.get(escopo, []))}
            for escopo in foco.scopes
        ] + [
            {"escopo_de_chaves": "ALCANCE TOTAL", "numero_chaves": len(alcance)},
            {"escopo_de_chaves": "das quais ele possui",
             "numero_chaves": int((keys_table["owner"] == operador_foco).sum())},
        ]),
        legenda=f"alcanca {len(alcance)} de {len(keys_table)} chaves",
    ))

    repositorio = build_repository(keys_table, operators)
    passos.append(Step(
        fase=FASE_1,
        modulo="M2",
        funcao="build_repository",
        explicacao=(
            "Monta os tres recortes do repositorio para cada operador, de uma "
            "vez. Cada recorte produz um desfecho diferente no log: `in_reach` "
            "vira `success`, `out_of_reach` vira `denied_by_policy`, e `existing` "
            "serve para forjar identificador inexistente sem colidir."
        ),
        entrada={"operadores": len(operators), "chaves": len(keys_table)},
        saida=pd.DataFrame([
            {"operador": nome,
             "in_reach": len(visao.in_reach),
             "out_of_reach": len(visao.out_of_reach),
             "existing": len(visao.existing)}
            for nome, visao in repositorio.items()
        ]),
        legenda=f"{len(repositorio)} operadores",
    ))

    planejadas = plan_sessions(rng, operators, specification)
    do_foco = [s for s in planejadas if s.operator.operator_id == operador_foco]
    passos.append(Step(
        fase=FASE_1,
        modulo="M2",
        funcao="plan_sessions",
        explicacao=(
            "Decide **quando** cada operador abre sessao, pelo regime dele: lote "
            "em horas fixas, ou chegadas aleatorias em dia util. O identificador "
            "sai da posicao no tempo, nao do operador, para nao carregar perfil."
        ),
        entrada={"semanas": specification.week_count,
                 "operadores": len(operators)},
        saida=pd.DataFrame([
            {"session_id": s.session_id,
             "operador": s.operator.operator_id,
             "inicio": s.start}
            for s in do_foco
        ]),
        completa=pd.DataFrame([
            {"session_id": s.session_id,
             "operador": s.operator.operator_id,
             "inicio": s.start}
            for s in planejadas
        ]),
        legenda=f"{len(planejadas)} sessoes no total, {len(do_foco)} do {operador_foco}",
    ))

    linhas = request_rows(rng, planejadas, repositorio, specification)
    passos.append(Step(
        fase=FASE_1,
        modulo="M2",
        funcao="request_rows",
        explicacao=(
            "Preenche cada sessao: escolhe a origem de rede, quantas requisicoes, "
            "os instantes, as operacoes e as chaves. Emite **tentativas**, nunca "
            "desfechos — quem decide se foi autorizada e o M4 (D-013)."
        ),
        entrada={"sessoes": len(planejadas)},
        saida=pd.DataFrame([
            linha for linha in linhas
            if linha["session_id"] == do_foco[0].session_id
        ]),
        completa=pd.DataFrame(linhas),
        legenda=(f"{len(linhas)} requisicoes no total; abaixo, a primeira "
                 f"sessao do {operador_foco}"),
    ))

    requests = with_event_ids(chronological(pd.DataFrame(linhas)))
    passos.append(Step(
        fase=FASE_1,
        modulo="M2",
        funcao="chronological + with_event_ids",
        explicacao=(
            "Ordena no tempo, que e como um log de auditoria se le, e numera os "
            "eventos nessa ordem. Nenhuma das duas sorteia nada.\n\n"
            f"Abaixo, so as linhas do **{operador_foco}**. Repare que os "
            "`event_id` **saltam**: a numeracao e global e cronologica, entao "
            "entre dois eventos dele existem os de todos os outros operadores "
            "que agiram no meio. O arquivo de verdade vem intercalado."
        ),
        entrada={"linhas": len(linhas)},
        saida=requests[requests["operator_id"] == operador_foco].head(20),
        completa=requests,
        legenda=(f"requests.csv, {len(requests)} linhas; abaixo, 20 do "
                 f"{operador_foco}"),
    ))

    return passos
