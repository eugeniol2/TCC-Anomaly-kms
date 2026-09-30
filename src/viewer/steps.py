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

from src.attack.build import build_attack, draw_compromised_admin
from src.attack.parameters import AttackSpecification
from src.attack.stealth import stealth_of
from src.audit_logger.build import build_log
from src.calibration.build import build_thresholds
from src.calibration.parameters import PERCENTILE, THRESHOLD_ATTRIBUTES
from src.calibration.build import ruler_period as ruler_sessions
from src.dataset.build import (
    ATTRIBUTES,
    LABEL,
    build_dataset,
    per_session,
    with_rate_attributes,
)
from src.globals.phases import EVALUATED, WARMUP
from src.globals.rng import ATTACK, POPULATION, TRAFFIC, stream
from src.historical_profiles.build import (
    profile_of,
    ruler_period,
    session_openings,
)
from src.kms.build import build_outcomes, requests_of_phase
from src.kms.policy import read_repository
from src.partition.build import Partition, build_partition
from src.traffic.regimes import REGIMES
from src.globals.tables import shuffle_rows
from src.population.keys import build_keys, disable_random_sample, split_keys_by_scope
from src.population.operators import build_operators_covering_pool
from src.population.parameters import KeyRepositorySpecification
from src.population.scopes import scope_pool
from src.traffic.build import chronological, plan_sessions, request_rows, with_event_ids
from src.traffic.operators import read_operators
from src.traffic.parameters import TrafficSpecification
from src.traffic.repository import build_repository, keys_by_scope, reach_of
from src.viewer.behaviors import linhas_do_atacante
from src.viewer.formatting import com_virgula, porcento


@dataclass(frozen=True)
class Fase:
    """Uma das tres fases da arquitetura, implementada ou nao."""

    nome: str
    modulos: str
    implementada: bool
    pendencia: str = ""
    """O que ainda falta na fase, quando ela esta so parcialmente implementada.

    A fase 3 e o caso: a campanha, o conjunto avaliado e a particao existem,
    mas nada que os consome. Marca-la como nao implementada esconderia metade do que ja
    roda; marca-la como pronta mentiria sobre a outra metade.
    """


FASES = (
    Fase(
        nome="Fase 1 · Preparação dos dados",
        modulos="M1 · M2",
        implementada=True,
    ),
    Fase(
        nome="Fase 2 · Aquecimento e calibração",
        modulos="M4 · M5 · M6 · M7 · M8",
        implementada=True,
    ),
    Fase(
        nome="Fase 3 · Ataque, treino e comparação",
        modulos="M3 · M4 · M5 · M7 · M9 · M10 · M11 · M12",
        implementada=True,
        pendencia=(
            "Os passos abaixo param no `train.csv` e no `holdout.csv`, que são a "
            "**entrada** do baseline e dos modelos. Os dois já decidem, e a avaliação "
            "está escrita, mas a tela não mostra resultado: nenhum F1 das 30 "
            "réplicas foi calculado."
        ),
    ),
)
"""As tres fases da arquitetura, na ordem de execucao."""

FASE_1 = FASES[0].nome
FASE_2 = FASES[1].nome
FASE_3 = FASES[2].nome


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
        explicacao="Cria os nomes dos escopos. Nao sorteia nada: e so uma lista.",
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
            "um detentor, porque sem isso sobrariam chaves que ninguem alcanca."
        ),
        entrada={"pool": f"{len(pool)} escopos"},
        saida=operators,
        legenda=f"{len(operators)} operadores",
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

    keys_in_scope_order = build_keys(rng, scope_sizes)
    passos.append(Step(
        fase=FASE_1,
        modulo="M1",
        funcao="build_keys",
        explicacao=(
            "Sorteia os identificadores. Aleatorio, nunca sequencial (D-009): "
            "com sequencia, o atacante enumerando produziria progressao "
            "aritmetica e qualquer atributo de distancia separaria as classes "
            "sozinho.\n\n"
            "Tres colunas, e nenhuma diz de quem a chave e (D-099): quem alcanca "
            "uma chave e quem detem o escopo dela, e o escopo e de varios."
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
            "revele o escopo. E por isso que o Scenario Engine ordena de novo ao ler: o alcance "
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
            "tabela responde 'qual o escopo desta chave?'; o Scenario Engine precisa do "
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
            "**A chave nao tem dono** (D-099). O escopo dela e detido por varios "
            "operadores ao mesmo tempo, entao alcance e o unico criterio, aqui "
            "e no KMS, que autoriza pela mesma regra. Ate 24/09 a tabela tinha "
            "uma coluna `owner`, e nenhum modulo a consultava.\n\n"
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
            "serve para forjar uma chave inexistente sem colidir com uma real."
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
            "desfechos: quem decide se foi autorizada e o KMS (D-013)."
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


def warmup_steps(
    seed: int,
    operators: pd.DataFrame,
    keys: pd.DataFrame,
    requests: pd.DataFrame,
    foco: str,
) -> list[Step]:
    """Refaz a fase 2 (M4, M5, M6, M7 e M8) guardando cada intermediario.

    Nenhum destes modulos sorteia nada: os cinco sao deterministicos por
    construcao. Nao ha `stream` aqui, e e por isso que o M4 recebe a semente
    apenas para escolher a pasta.
    """
    passos: list[Step] = []

    repositorio = read_repository(keys, operators)
    do_foco = sorted(repositorio.scopes_by_operator[foco])
    passos.append(Step(
        fase=FASE_2,
        modulo="M4",
        funcao="read_repository",
        explicacao=(
            "Monta os tres indices contra os quais cada requisicao sera julgada: "
            "o escopo de cada chave, o conjunto de chaves desabilitadas e os "
            "escopos de cada operador. **Tudo sai das duas tabelas da Populacao**, e o "
            "que faz o desfecho ser derivado da politica em vez de inventado."
        ),
        entrada={"keys": len(keys), "operators": len(operators)},
        saida=pd.DataFrame([
            {"indice": "scope_by_key", "tamanho": len(repositorio.scope_by_key)},
            {"indice": "disabled", "tamanho": len(repositorio.disabled)},
            {"indice": "scopes_by_operator", "tamanho": len(repositorio.scopes_by_operator)},
        ]),
        legenda=f"{foco} detem os escopos {', '.join(do_foco)}",
    ))

    da_fase = requests_of_phase(requests, WARMUP)
    passos.append(Step(
        fase=FASE_2,
        modulo="M4",
        funcao="requests_of_phase",
        explicacao=(
            "Recorta as semanas 1 a 4. O `requests.csv` tem as sete, e as quatro "
            "ultimas pertencem ao ramo de sigma, porque o aquecimento e anterior ao "
            "ataque e tem de ficar limpo (D-048)."
        ),
        entrada={"requests": len(requests), "fase": WARMUP},
        saida=da_fase.head(10),
        completa=da_fase,
        legenda=f"{len(da_fase)} de {len(requests)} requisicoes",
    ))

    outcomes = build_outcomes(requests, keys, operators, WARMUP)
    contagem = outcomes["outcome"].value_counts().rename_axis("outcome")
    passos.append(Step(
        fase=FASE_2,
        modulo="M4",
        funcao="build_outcomes",
        explicacao=(
            "O KMS julga cada requisicao, **nesta ordem**: chave inexistente, "
            "depois fora de escopo, depois chave desabilitada, depois "
            "sucesso (D-077). Autorizacao antes de estado: quem nao detem o "
            "escopo recebe negacao, e nao a informacao de que a chave existe.\n\n"
            "Duas colunas so, `event_id` e `outcome`: o diagnostico de qual "
            "regra disparou ficaria no arquivo e o Audit Logger teria de lembrar de "
            "descarta-lo (D-078)."
        ),
        entrada={"requisicoes": len(da_fase)},
        saida=contagem.reset_index(name="eventos"),
        completa=outcomes,
        legenda=f"{len(outcomes)} desfechos, os quatro tipos presentes",
    ))

    log = build_log(requests, outcomes, WARMUP)
    passos.append(Step(
        fase=FASE_2,
        modulo="M5",
        funcao="build_log",
        explicacao=(
            "Casa tentativa com desfecho por `event_id` e fecha as **oito "
            "colunas** do log (D-064). O `key_id` e o **requisitado**, nao o "
            "resolvido, para que a chave inexistente seja observavel.\n\n"
            "Nao ha escopo, perfil nem proprietario aqui: qualquer um deles "
            "deixaria o modelo reconstruir a fronteira de autorizacao e aprender "
            "a politica em vez do comportamento. E nao ha rotulo: ele viaja em "
            "outro arquivo (D-063)."
        ),
        entrada={"requests": len(da_fase), "outcomes": len(outcomes)},
        saida=log[log["operator_id"] == foco].head(12),
        completa=log,
        legenda=f"log.csv, {len(log)} eventos; abaixo, 12 do {foco}",
    ))

    do_perfil = ruler_period(log)
    passos.append(Step(
        fase=FASE_2,
        modulo="M6",
        funcao="ruler_period",
        explicacao=(
            "E guarda, nao recorte: confere que o log recebido e mesmo o do "
            "aquecimento e devolve-o inteiro. A regua sao as **quatro semanas**, "
            "e nao ha divisao dentro dela (D-096).\n\n"
            "Ate 23/09 aqui se recortavam as semanas 1 e 2, porque a 3 calibrava "
            "os limiares e a D-054 queria evitar o acoplamento. As seis regras de "
            "grandeza nao consultam o perfil, entao nao havia acoplamento, so um "
            "perfil estimado da metade dos dados."
        ),
        entrada={"log": len(log)},
        saida=do_perfil.head(8),
        completa=do_perfil,
        legenda=f"{len(do_perfil)} eventos, o aquecimento inteiro",
    ))

    aberturas = session_openings(do_perfil)
    passos.append(Step(
        fase=FASE_2,
        modulo="M6",
        funcao="session_openings",
        explicacao=(
            "Uma linha por sessao: quem abriu, quando e de onde. O instante e o "
            "do **primeiro evento**, e a origem e a dele, porque origem e uma so por "
            "sessao desde o Scenario Engine."
        ),
        entrada={"eventos": len(do_perfil)},
        saida=aberturas[aberturas["operator_id"] == foco],
        completa=aberturas,
        legenda=f"{len(aberturas)} sessoes no perfil; abaixo, as do {foco}",
    ))

    perfis = profile_of(aberturas)
    passos.append(Step(
        fase=FASE_2,
        modulo="M6",
        funcao="profile_of",
        explicacao=(
            "A janela vai do **menor ao maior** horario observado, sem percentil "
            "e sem descarte (D-073). O corte de 95 % foi abandonado porque nao era "
            "entregavel na escala de entao: com a regua de duas semanas a mediana "
            "era de 16 sessoes por perfil, 5 % davam 0,8 sessao, e o arredondamento "
            "mandava descartar zero. A regua de quatro semanas dobrou esse numero, "
            "mas a forma ja tinha sido decidida, e minimo a maximo e livre de "
            "distribuicao, o que o percentil nao e.\n\n"
            "`observed_ips` **nao** e `usual_ips`. Aquela e a lista que a Populacao "
            "sorteou; esta e o subconjunto que apareceu no log, e a diferenca "
            "entre as duas e o que produz origem inedita legitima depois (D-040)."
        ),
        entrada={"sessoes": len(aberturas)},
        saida=perfis[perfis["operator_id"] == foco],
        completa=perfis,
        legenda=f"historical_profiles.csv, {len(perfis)} operadores",
    ))

    brutas = per_session(log)
    passos.append(Step(
        fase=FASE_2,
        modulo="M7",
        funcao="per_session",
        explicacao=(
            "Agrupa o log por sessao e conta o que e contavel: eventos, chaves "
            "distintas, falhas e negacoes. Ainda sao grandezas brutas: nenhuma "
            "delas e atributo ainda."
        ),
        entrada={"eventos": len(log)},
        saida=brutas[brutas["operator_id"] == foco].head(8),
        completa=brutas,
        legenda=f"{len(brutas)} sessoes no aquecimento",
    ))

    com_taxas = with_rate_attributes(brutas)
    passos.append(Step(
        fase=FASE_2,
        modulo="M7",
        funcao="with_rate_attributes",
        explicacao=(
            "Contagem vira taxa. Com sessoes de comprimento variavel, contagem "
            "bruta confunde sessao longa com sessao intensa.\n\n"
            "`distinct_keys` e a excecao deliberada (D-080): a razao "
            "`chaves / eventos` tem teto em 1,0 e a regra correspondente nascia "
            "morta. O companheiro dela e o `events`, que esta na lista ao lado."
        ),
        entrada={"sessoes": len(brutas)},
        saida=com_taxas[com_taxas["operator_id"] == foco].head(8),
        completa=com_taxas,
        legenda="duracao, taxa e as duas razoes de falha",
    ))

    sessoes = build_dataset(log, perfis, WARMUP)
    passos.append(Step(
        fase=FASE_2,
        modulo="M7",
        funcao="build_dataset",
        explicacao=(
            "Os dois atributos binarios, lidos contra o perfil **daquele "
            "operador** e nunca contra um limiar global. Nenhuma sessao do "
            "aquecimento e atipica, porque a janela e o minimo e o maximo delas "
            "proprias, e nenhuma cai fora do que ela mesma delimitou. E a "
            "invariante que pegou o erro de arredondamento da D-090.\n\n"
            "**Sem coluna de rotulo.** O conjunto do aquecimento alimenta so a "
            "Calibracao, e calibracao por percentil nao usa rotulo: se a coluna nao "
            "existe, ninguem a usa por engano (D-063)."
        ),
        entrada={"log": len(log), "perfis": len(perfis)},
        saida=sessoes[sessoes["operator_id"] == foco].head(8),
        completa=sessoes,
        legenda=f"sessions.csv, {len(sessoes)} sessoes x 8 atributos",
    ))

    do_aquecimento = ruler_sessions(sessoes)
    passos.append(Step(
        fase=FASE_2,
        modulo="M8",
        funcao="ruler_period",
        explicacao=(
            "A mesma guarda dos Perfis historicos, do outro lado da regua: confere que as sessoes "
            "sao as do aquecimento e devolve-as inteiras. O aquecimento e limpo e "
            "anterior ao ataque, e e por isso que os limiares podem ser os mesmos "
            "nas 11 condicoes de sigma (D-043).\n\n"
            "**O limiar sai do mesmo periodo que o perfil** (D-096). Ate 23/09 "
            "saia da semana 3 sozinha, que era um terco dos dados disponiveis."
        ),
        entrada={"sessoes": len(sessoes)},
        saida=do_aquecimento.head(8),
        completa=do_aquecimento,
        legenda=f"{len(do_aquecimento)} sessoes, o arquivo inteiro",
    ))

    limiares = build_thresholds(sessoes)
    passos.append(Step(
        fase=FASE_2,
        modulo="M8",
        funcao="build_thresholds",
        explicacao=(
            f"Percentil {PERCENTILE} de cada grandeza, sobre o aquecimento. "
            f"**{len(THRESHOLD_ATTRIBUTES)} regras, nao {len(ATTRIBUTES)}**: "
            "`atypical_hour` e `new_source_ip` ja vem binarias do Dataset Generator e "
            "disparam quando valem 1: percentil sobre uma coluna de zeros e uns "
            "daria 0 ou 1 e nao significaria nada.\n\n"
            "O baseline **nao recebe treino**: chega ao periodo avaliado com "
            "estes numeros congelados, e nunca ve um rotulo (D-033)."
        ),
        entrada={"sessoes do aquecimento": len(do_aquecimento)},
        saida=limiares,
        legenda="thresholds.csv, um conjunto por semente",
    ))

    return passos


def contagem_por_lado(particao: Partition) -> pd.DataFrame:
    """Quantas sessoes de cada classe caem em cada lado, contadas (D-023)."""
    linhas = []

    for classe, rotulo in (("legitimas", 0), ("do atacante", 1)):
        treino = int((particao.train[LABEL] == rotulo).sum())
        holdout = int((particao.holdout[LABEL] == rotulo).sum())
        linhas.append({"classe": classe, "treino": treino, "holdout": holdout})

    return pd.DataFrame(linhas)


def attack_steps(
    seed: int,
    sigma: float,
    operators: pd.DataFrame,
    keys: pd.DataFrame,
    requests: pd.DataFrame,
    perfis: pd.DataFrame,
) -> list[Step]:
    """Refaz a fase 3 ate onde ela existe: M3, M4, M5, M7 e M9 sobre sigma.

    O operador em foco aqui e sempre o administrador comprometido: acompanhar
    outro nao mostraria nada que a fase 2 ja nao tenha mostrado.
    """
    rng = stream(seed, ATTACK)
    passos: list[Step] = []

    lidos = read_operators(operators)
    alvo = draw_compromised_admin(rng, lidos)
    passos.append(Step(
        fase=FASE_3,
        modulo="M3",
        funcao="draw_compromised_admin",
        explicacao=(
            "**Primeiro sorteio do fluxo de ataque**, antes de qualquer coisa "
            "que dependa de sigma. E o que garante que as 11 condicoes da mesma "
            "semente compartilhem o alvo, isolando o efeito de sigma (D-011).\n\n"
            "O atacante nao tem identidade propria: age sob a credencial deste "
            "administrador, o que elimina deteccao por controle de acesso e "
            "deixa o comportamento como unico sinal."
        ),
        entrada={"administradores": 8},
        saida=operators[operators["operator_id"] == alvo.operator_id],
        legenda=f"comprometido nesta semente: {alvo.operator_id}",
    ))

    trafego = TrafficSpecification()
    legitimo = REGIMES[alvo.regime]
    furtividade = stealth_of(sigma, legitimo, trafego, AttackSpecification())
    passos.append(Step(
        fase=FASE_3,
        modulo="M3",
        funcao="stealth_of",
        explicacao=(
            f"Interpola as cinco dimensoes entre o ostensivo e o furtivo, em "
            f"sigma **{com_virgula(sigma)}**. A coluna da direita e o que o "
            "**trafego legitimo usaria** "
            "para este administrador: ela nao foi escolhida, foi importada.\n\n"
            "Em sigma 1 as duas colunas coincidem e a campanha chama as mesmas "
            "funcoes do Scenario Engine: a sessao comprometida sai da mesma "
            "distribuicao que uma "
            "legitima, e nenhum mecanismo pode separa-las. E o piso declarado "
            "da varredura (D-082).\n\n"
            "**As duas primeiras linhas dizem o tipico, nao o teto** (D-097, "
            "D-098). A cauda e a mesma nos dois lados: ela vem da especificacao "
            "de trafego, que o Scenario Engine e a campanha compartilham, entao a convergencia "
            "exata em sigma 1 vale com ela inclusive. Enquanto a faixa era teto, "
            "as duas classes nao se sobrepunham e `events` separava sozinho com "
            "F1 0,982 em sigma 0."
        ),
        entrada={"sigma": sigma, "regime": alvo.regime},
        saida=pd.DataFrame(
            [
                {"dimensao": nome, "neste sigma": deste, "no legitimo": no_legitimo}
                for (nome, deste), (_, no_legitimo) in zip(
                    linhas_do_atacante(furtividade),
                    linhas_do_atacante(
                        stealth_of(1.0, legitimo, trafego, AttackSpecification())
                    ),
                )
            ]
        ),
        legenda=f"sigma {com_virgula(sigma)}: 0 e ostensivo, 1 e indistinguivel",
    ))

    campanha = build_attack(
        seed, sigma, operators, keys, requests, trafego, AttackSpecification()
    )
    marcadas = set(campanha.compromised["session_id"])
    do_atacante = campanha.requests[campanha.requests["session_id"].isin(marcadas)]
    passos.append(Step(
        fase=FASE_3,
        modulo="M3",
        funcao="build_attack",
        explicacao=(
            "Mescla **58 sessoes comprometidas** as semanas 5 a 8 do trafego "
            "legitimo. O numero e o mesmo nas 11 condicoes (D-081): o que sigma "
            "move e o comportamento dentro delas, nunca quantas sao, porque se movesse "
            "as duas coisas, a proporcao de anomalias mudaria junto e a "
            "comparacao entre condicoes confundiria furtividade com "
            "desbalanceamento.\n\n"
            "Sessoes e eventos sao **renumerados em ordem cronologica** (D-083). "
            "Sem isso as comprometidas ficariam no fim da faixa numerica e o "
            "identificador anunciaria o rotulo."
        ),
        entrada={"legitimo": len(requests), "sigma": sigma},
        saida=do_atacante.head(12),
        completa=campanha.requests,
        legenda=(f"{len(campanha.requests)} requisicoes; abaixo, 12 das "
                 f"{len(marcadas)} sessoes do atacante"),
    ))

    passos.append(Step(
        fase=FASE_3,
        modulo="M3",
        funcao="compromised_sessions",
        explicacao=(
            "O rotulo, **fora do log** (D-063). Uma coluna so: presenca na lista "
            "e o rotulo, e quem e o administrador ja esta no `run.csv`.\n\n"
            "Repare nos numeros: eles estao espalhados entre as sessoes "
            "legitimas, nao agrupados no fim. E o que a renumeracao do passo "
            "anterior garante."
        ),
        entrada={"sessoes da campanha": len(marcadas)},
        saida=campanha.compromised.head(12),
        completa=campanha.compromised,
        legenda=f"compromised_sessions.csv, {len(campanha.compromised)} linhas",
    ))

    outcomes = build_outcomes(campanha.requests, keys, operators, EVALUATED)
    log = build_log(campanha.requests, outcomes, EVALUATED)
    do_atacante_log = log[log["session_id"].isin(marcadas)]
    passos.append(Step(
        fase=FASE_3,
        modulo="M4 · M5",
        funcao="build_outcomes + build_log",
        explicacao=(
            "**As mesmas funcoes da fase 2**, agora sobre as semanas 5 a 8. O "
            "KMS nao sabe que ha atacante: ele avalia a politica da chave, e as "
            "requisicoes do atacante sao negadas pelas mesmas regras que negam "
            "as legitimas.\n\n"
            "E isso que faz o rotulo ser **derivado da politica** em vez de "
            "inventado pelo gerador, que e o argumento central do trabalho."
        ),
        entrada={"requisicoes": len(campanha.requests)},
        saida=do_atacante_log["outcome"].value_counts().rename_axis(
            "outcome").reset_index(name="eventos do atacante"),
        completa=log,
        legenda=f"log.csv, {len(log)} eventos",
    ))

    sessoes = build_dataset(log, perfis, EVALUATED, campanha.compromised)
    positivas = int(sessoes["compromised"].sum())
    passos.append(Step(
        fase=FASE_3,
        modulo="M7",
        funcao="build_dataset",
        explicacao=(
            "O conjunto que os modelos vao classificar. Mesmo codigo da fase 2, "
            "mais a juncao do rotulo, que so acontece aqui.\n\n"
            "Cada sessao comprometida e **uma** positiva (D-061). A proporcao "
            "abaixo e contada no dado, nao herdada do parametro do gerador, "
            "como o metodo exige."
        ),
        entrada={"log": len(log), "rotulos": len(campanha.compromised)},
        saida=sessoes[sessoes["compromised"] == 1].head(10),
        completa=sessoes,
        legenda=(f"sessions.csv, {len(sessoes)} sessoes, {positivas} positivas "
                 f"({porcento(positivas / len(sessoes), 2)}); abaixo, 10 comprometidas"),
    ))

    particao = build_partition(seed, sessoes)
    passos.append(Step(
        fase=FASE_3,
        modulo="M9",
        funcao="build_partition",
        explicacao=(
            "Cada classe e dividida **a parte**, 40 % para o holdout (D-070), e a "
            "sessao vai inteira para um lado so (D-061).\n\n"
            "O sorteio sai da semente e nao de sigma (D-102): as legitimas sao "
            "reconhecidas por operador e instante de abertura, que nao mudam entre "
            "condicoes, e as do atacante pela posicao na campanha. Por isso **as "
            "mesmas sessoes legitimas** estao no holdout em todo sigma, e o que muda "
            "de um ponto da curva ao outro e so o atacante.\n\n"
            "**E aqui que a tela para.** O baseline e os modelos ja decidem sobre "
            "este holdout, mas as decisoes nao aparecem aqui: ver acerto e um "
            "momento que o registro marca."
        ),
        entrada={"sessoes": len(sessoes), "semente": seed},
        saida=contagem_por_lado(particao),
        completa=particao.holdout,
        legenda=f"holdout.csv, {len(particao.holdout)} sessoes; acima, a contagem por classe",
    ))

    return passos
