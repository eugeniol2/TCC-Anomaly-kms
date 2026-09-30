"""As variáveis de decisão que regem cada entidade do pipeline.

Este módulo responde a pergunta que a tela precisa responder antes de mostrar
qualquer tabela: **que números fazem este dado ser o que é?**

Cada variável traz o nome como ele aparece no código, o valor em vigor e uma
linha dizendo o que ele é. O *porquê* não mora aqui: ele está na caixa "Por
que é assim" de cada quadro, e na entrada do registro. Misturar os dois faria a
tabela virar texto corrido, e tabela com parágrafo dentro não se lê.

O campo `decisao` guarda a entrada do registro que fixou o valor. Ele **não
aparece na tela**: `D-040` não significa nada para quem está vendo a
apresentação. Fica no código porque ali ele é útil, já que diz de onde o número veio
para quem for mexer neste arquivo.

**Os valores são lidos dos próprios objetos de parâmetro**, nunca copiados. Um
número escrito à mão aqui ficaria desatualizado na primeira recalibração, e a
tela passaria a explicar um gerador que não existe mais.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.scenario_engine.attack.parameters import AttackSpecification
from src.scenario_engine.attack.stealth import stealth_of
from src.audit_logger.build import COLUMNS as LOG_COLUMNS
from src.policy_engine.calibration.parameters import PERCENTILE, THRESHOLD_ATTRIBUTES
from src.dataset_generator.dataset.build import ATTRIBUTES, IDENTIFIERS, LABEL
from src.dataset_generator.dataset.parameters import SHORTEST_MEASURABLE_MINUTES
from src.shared.experiment import SEEDS, SIGMAS
from src.shared.phases import EVALUATED_WEEKS, RULER_WEEKS
from src.dataset_generator.historical_profiles.build import HOUR_FORMAT
from src.kms.policy import COLUMNS as OUTCOME_COLUMNS, OUTCOMES
from src.dataset_generator.partition.build import holdout_count
from src.dataset_generator.partition.parameters import HOLDOUT_SHARE
from src.kms.repository.parameters import KeyRepositorySpecification
from src.scenario_engine.population.profiles import PROFILES
from src.scenario_engine.traffic.parameters import (
    IDENTIFIER_DIGITS,
    IDENTIFIER_PREFIX,
    PRIMARY_ADDRESS_SHARE,
    TrafficSpecification,
)
from src.scenario_engine.traffic.regimes import REGIMES
from src.viewer.behaviors import linhas_do_atacante
from src.viewer.formatting import com_virgula, porcento, valor_escrito
from src.viewer.theory import (
    Teoria,
    teoria_da_cauda,
    teoria_da_dirichlet,
)

COLUNAS = ("variável", "valor", "o que é")

MODELOS = ("Random Forest", "XGBoost")
"""Os dois modelos supervisionados (D-051). A contagem de comparacoes da tela sai
daqui, e nao de um 22 escrito a mao."""

REGRAS_DE_GRANDEZA = len(THRESHOLD_ATTRIBUTES)
REGRAS_DE_PERFIL = len(ATTRIBUTES) - REGRAS_DE_GRANDEZA


@dataclass(frozen=True)
class Variavel:
    """Um número, ou uma lista, que poderia ter sido outro."""

    nome: str
    valor: Any
    significado: str
    decisao: str
    teoria: Teoria | None = None
    """A distribuicao por tras do valor, quando ha uma.

    So algumas variaveis tem: as que nao sao um numero escolhido, e sim a
    forma de uma distribuicao. O app mostra cada uma num expansor abaixo
    da tabela, com o grafico."""


def como_tabela(variaveis: tuple[Variavel, ...]) -> pd.DataFrame:
    """As variáveis numa tabela de três colunas, para o app mostrar de uma vez."""
    linhas = [
        {
            "variável": variavel.nome,
            "valor": valor_escrito(variavel.valor),
            "o que é": variavel.significado,
        }
        for variavel in variaveis
    ]

    return pd.DataFrame(linhas, columns=list(COLUNAS))


def variaveis_do_m1(
    repositorio: KeyRepositorySpecification, chaves: pd.DataFrame | None = None
) -> tuple[Variavel, ...]:
    """A escala da população e a forma do repositório de chaves."""
    por_perfil = ", ".join(
        f"{perfil.operators} {perfil.name}" for perfil in PROFILES
    )
    escopos_por_perfil = "; ".join(
        f"{perfil.name}: {perfil.scopes_each}" for perfil in PROFILES
    )
    enderecos_por_perfil = "; ".join(
        f"{perfil.name}: {perfil.addresses_range[0]} a {perfil.addresses_range[1]}"
        for perfil in PROFILES
    )

    return (
        Variavel("PROFILES", por_perfil,
                 "Quantos operadores de cada perfil.", "D-035"),
        Variavel("scopes_each", escopos_por_perfil,
                 "Quantos escopos cada perfil detém, o que define o alcance dele.",
                 "D-035"),
        Variavel("addresses_range", enderecos_por_perfil,
                 "Quantas origens de rede habituais cada operador tem, sorteado "
                 "na faixa.",
                 "D-041"),
        Variavel("total_keys", repositorio.total_keys,
                 "Chaves no repositório.", "D-035"),
        Variavel("scope_count", repositorio.scope_count,
                 "Escopos entre os quais as chaves se dividem.", "D-035"),
        Variavel("concentration", repositorio.concentration,
                 "Concentração da Dirichlet que reparte as chaves entre escopos; "
                 "quanto menor, mais desigual.",
                 "D-037",
                 teoria_da_dirichlet(chaves, repositorio.concentration)
                 if chaves is not None else None),
        Variavel("scope_floor", repositorio.scope_floor,
                 "Mínimo de chaves por escopo.", "D-037"),
        Variavel("disabled_rate", porcento(repositorio.disabled_rate, 0),
                 "Fração das chaves que nasce desabilitada, sorteada por chave.",
                 "D-038"),
        Variavel("formato do key_id",
                 f"{IDENTIFIER_PREFIX} + {IDENTIFIER_DIGITS} hexadecimais, aleatório",
                 "Como o identificador de chave é gerado. Nunca sequencial.",
                 "D-009"),
    )


def variaveis_do_m2(trafego: TrafficSpecification) -> tuple[Variavel, ...]:
    """O que vale para o tráfego inteiro, qualquer que seja o regime.

    **O que é de um regime só não mora aqui**, e sim na página Comportamentos
    (`behaviors.py`): horas do lote, média e lei da contagem diária, janela,
    faixa de requisições, passo entre requisições e mistura de operações.
    Mostrados nos dois lugares, eles eram duas explicações do mesmo número, e
    a da página Comportamentos é a completa, com o dado medido ao lado.
    """
    lote = REGIMES["periodic_batch"]
    # Uma linha por regime: os tres numa celula so ficavam longos demais
    # para ler, e a coluna de valor cortava o ultimo.
    regimes = tuple(
        Variavel(f"regime {perfil.regime}", perfil.name,
                 f"Regime do perfil {perfil.name}. O ritmo, o tamanho e o "
                 "passo dele estão na página Comportamentos.",
                 "D-007, D-058")
        for perfil in PROFILES
    )

    return regimes + (
        Variavel("week_count", trafego.week_count,
                 f"Semanas simuladas: {RULER_WEEKS} de régua (o perfil e os "
                 f"limiares, do mesmo período) e {EVALUATED_WEEKS} avaliadas.",
                 "D-096"),
        Variavel("first_day", trafego.first_day,
                 "Primeiro dia simulado. Segunda-feira, fixa.", "D-067"),
        Variavel("LONG_SESSION_CHANCE", trafego.long_session_chance,
                 "Fração das sessões que se estendem além do típico.", "D-097",
                 teoria_da_cauda(
                     lote.requests_range,
                     trafego.long_session_chance,
                     trafego.long_session_excess,
                     AttackSpecification().ostensive_requests_range[0],
                 )),
        Variavel("LONG_SESSION_EXCESS", trafego.long_session_excess,
                 "Requisições a mais na sessão que se estende, média da "
                 "geométrica.", "D-097"),
        Variavel("DEFAULT_DISTINCT_KEYS_RANGE", trafego.distinct_keys_range,
                 "Chaves distintas que uma sessão toca, se couberem no alcance. "
                 "É a amplitude **típica**: uma sessão em vinte passa dela, "
                 "sorteada à parte da cauda do comprimento.",
                 "D-058, D-098, D-100"),
        Variavel("PRIMARY_ADDRESS_SHARE", PRIMARY_ADDRESS_SHARE,
                 "Chance de a sessão vir do endereço principal; os demais decaem "
                 "geometricamente. A explicação está na página Comportamentos.",
                 "D-040"),
        Variavel("DEFAULT_STALE_SCOPE_RATE", porcento(trafego.stale_scope_rate),
                 "Fração das requisições que aponta para escopo obsoleto.",
                 "D-056"),
        Variavel("DEFAULT_ABSENT_IDENTIFIER_RATE",
                 porcento(trafego.absent_identifier_rate),
                 "Fração das requisições que pede uma chave inexistente: um key_id "
                 "que não corresponde a chave nenhuma.", "D-076"),
    )


def variaveis_do_kms() -> tuple[Variavel, ...]:
    """O KMS quase não tem número: o que ele tem é ordem."""
    return (
        Variavel("ordem de avaliação",
                 "1 chave inexistente, 2 fora de escopo, "
                 "3 chave desabilitada, 4 sucesso",
                 "Em que ordem o KMS testa cada condição. O primeiro caso que se "
                 "aplica decide.",
                 "D-077"),
        Variavel("OUTCOMES", OUTCOMES, "Os quatro desfechos possíveis.", "D-064"),
        Variavel("colunas do outcomes.csv", OUTCOME_COLUMNS,
                 "O que o KMS devolve ao Audit Logger.", "D-078"),
    )


def variaveis_do_audit_logger() -> tuple[Variavel, ...]:
    """O que vai para o log, e o que fica de fora dele de propósito."""
    return (
        Variavel("colunas do log.csv", LOG_COLUMNS,
                 f"As {len(LOG_COLUMNS)} colunas do log. Sem escopo nem perfil.",
                 "D-064"),
        Variavel("key_id registrado", "o requisitado, não o resolvido",
                 "Qual identificador vai para o log.", "D-064"),
        Variavel("rótulo no log", "não existe",
                 "Onde a marca de comprometimento fica: em arquivo separado.",
                 "D-063"),
    )


def variaveis_do_m4_m5() -> tuple[Variavel, ...]:
    """As duas juntas, para o passo da fase 3 que roda as duas de novo."""
    return variaveis_do_kms() + variaveis_do_audit_logger()


def variaveis_do_m6() -> tuple[Variavel, ...]:
    """O perfil histórico não tem parâmetro de forma, e isso é a decisão."""
    return (
        Variavel("RULER_WEEKS", RULER_WEEKS,
                 "Semanas que constroem a régua: o perfil e os limiares.",
                 "D-096"),
        Variavel("forma da janela horária", "do mínimo ao máximo observado",
                 "Como a janela é delimitada. Sem percentil e sem descarte.",
                 "D-073"),
        Variavel("HOUR_FORMAT", HOUR_FORMAT,
                 "Como a hora da janela é escrita. Texto de largura fixa, que "
                 "compara na ordem certa e não precisa de arredondamento.",
                 "D-095"),
        Variavel("observed_ips",
                 f"só os IPs que aparecem no log das semanas 1 a {RULER_WEEKS}",
                 "Os IPs de origem que o operador usou de fato na régua. É um "
                 "subconjunto dos usual_ips que a População sorteou, não a "
                 "lista inteira.",
                 "D-040"),
        Variavel("recálculo", f"nunca depois da semana {RULER_WEEKS}",
                 "Quando o perfil é atualizado: uma vez, no fim da régua, e "
                 "congelado dali em diante.", "D-044, D-096"),
    )


def variaveis_do_m7() -> tuple[Variavel, ...]:
    """Os oito atributos, e o que fica de fora deles."""
    return (
        Variavel("unidade de análise", "a sessão",
                 "O que vira uma linha. Não existe tamanho de janela.", "D-061"),
        Variavel("ATTRIBUTES", ", ".join(ATTRIBUTES),
                 f"Os {len(ATTRIBUTES)} atributos que os modelos recebem.", "D-074, D-080"),
        Variavel("distinct_keys", "contagem bruta, não razão",
                 "Como as chaves distintas entram: contadas, acompanhadas de "
                 "`events`.",
                 "D-080"),
        Variavel("IDENTIFIERS", IDENTIFIERS,
                 "Colunas que ficam no arquivo e não entram como atributo.",
                 "D-015, D-088"),
        Variavel("SHORTEST_MEASURABLE_MINUTES", SHORTEST_MEASURABLE_MINUTES,
                 "Piso da duração ao calcular a taxa, equivalente a um segundo.",
                 "D-088"),
        Variavel(LABEL, "só na fase avaliada",
                 "A coluna do rótulo: diz se a sessão é do atacante. Só existe "
                 "no sessions.csv das semanas 5 a 8.", "D-063"),
    )


def variaveis_do_m8() -> tuple[Variavel, ...]:
    """O percentil, e de onde ele sai."""
    return (
        Variavel("período dos limiares", f"as {RULER_WEEKS} semanas do aquecimento",
                 "De onde sai o percentil. O mesmo período do perfil: a régua "
                 "é uma só.",
                 "D-096"),
        Variavel("PERCENTILE", PERCENTILE,
                 "Percentil de cada grandeza que vira limiar.", "D-031, D-043"),
        Variavel("THRESHOLD_ATTRIBUTES", ", ".join(THRESHOLD_ATTRIBUTES),
                 f"As {REGRAS_DE_GRANDEZA} regras de grandeza. As outras "
                 f"{REGRAS_DE_PERFIL} já vêm binárias.",
                 "D-080"),
        Variavel("comparação", "> limiar, estrita",
                 "Como a regra decide se disparou.", "D-080"),
        Variavel("limiares por semente", f"{len(SEEDS)} conjuntos",
                 f"Um por semente, compartilhado pelas {len(SIGMAS)} condições "
                 "de σ.", "D-043"),
        Variavel("treino do baseline", "nenhum, em momento nenhum",
                 "Se o baseline aprende com rótulo em alguma etapa.",
                 "D-033, D-046"),
    )


def dimensoes_do_atacante(
    sigma: float, ataque: AttackSpecification, trafego: TrafficSpecification
) -> tuple[Variavel, ...]:
    """As dimensões de σ, dos dois extremos e deste σ.

    Os nomes e os valores saem de `linhas_do_atacante`, a mesma função da
    tabela do atacante na página Comportamentos. Até 28/09 esta tabela tinha
    nomes próprios ("taxa, intervalo", "falhas de autorização") e não tinha a
    linha da chave inexistente.
    """
    regime = REGIMES["occasional_custody"]
    ostensivo, deste, legitimo = (
        linhas_do_atacante(stealth_of(valor, regime, trafego, ataque))
        for valor in (0.0, sigma, 1.0)
    )

    return tuple(
        Variavel(nome, f"{zero}  →  {um}",
                 f"Do ostensivo (σ 0,0) ao legítimo (σ 1,0). Neste σ: {meio}.",
                 "D-082")
        for (nome, zero), (_, meio), (_, um) in zip(ostensivo, deste, legitimo)
    )


def variaveis_do_m3(
    sigma: float, ataque: AttackSpecification, trafego: TrafficSpecification
) -> tuple[Variavel, ...]:
    """A campanha: quantas sessões, quem, e o que σ move."""
    administradores = next(
        perfil.operators for perfil in PROFILES if perfil.name == "administrator"
    )
    condicoes = len(SIGMAS)

    return (
        Variavel("σ desta execução", com_virgula(sigma),
                 "0,0 é ostensivo; 1,0 é indistinguível de uma sessão legítima.",
                 "D-004"),
        Variavel("campaign_sessions", ataque.campaign_sessions,
                 f"Sessões que o atacante abre, o mesmo número nas {condicoes} "
                 "condições.",
                 "D-081"),
        Variavel("positivas no holdout", holdout_count(ataque.campaign_sessions),
                 f"Quantas das {ataque.campaign_sessions} sessões comprometidas "
                 "caem no holdout.", "D-081, D-070"),
    ) + dimensoes_do_atacante(sigma, ataque, trafego) + (
        Variavel("o que σ move", "o proveito, não o esforço",
                 "O atacante furtivo abre tantas sessões quanto o ostensivo e "
                 "consegue menos.",
                 "D-081"),
        Variavel("administrador comprometido",
                 f"sorteado por semente, entre {administradores}",
                 f"Quem tem a credencial comprometida. O mesmo nas {condicoes} "
                 "condições.",
                 "D-010, D-011"),
        Variavel("mistura de operações", "a do administrador personificado",
                 "De onde saem as operações do atacante. Não é dimensão de σ.",
                 "D-055"),
    )


def variaveis_do_m9(holdout: pd.DataFrame) -> tuple[Variavel, ...]:
    """A partição: quanto vai para cada lado, e o que a divisão preserva."""
    treino = 1 - HOLDOUT_SHARE

    return (
        Variavel("HOLDOUT_SHARE", porcento(HOLDOUT_SHARE, 0),
                 f"A fração de cada classe que vai para o holdout. O treino "
                 f"fica com {porcento(treino, 0)}.",
                 "D-070"),
        Variavel("estratificação", "por classe, e por sessão",
                 "Cada classe é dividida à parte, e nenhuma sessão é cortada "
                 "ao meio.",
                 "D-018, D-061"),
        Variavel("positivas no holdout", int(holdout[LABEL].sum()),
                 "Contagem absoluta, que é o número que importa.",
                 "D-070, D-081"),
        Variavel("o mesmo sorteio nos σ", f"nas {len(SIGMAS)} condições",
                 "As mesmas sessões legítimas vão para o holdout em todo σ da "
                 "semente. Só o atacante muda de um ponto da curva ao outro.",
                 "D-102"),
    )


def variaveis_do_m10() -> tuple[Variavel, ...]:
    """O baseline de regras: quantas regras, e quando alerta."""
    return (
        Variavel("regras do baseline", f"{len(ATTRIBUTES)}, uma por atributo",
                 f"{REGRAS_DE_GRANDEZA} comparam contra limiar; {REGRAS_DE_PERFIL} "
                 "leem o perfil histórico.",
                 "D-080"),
        Variavel("ponto de operação", "duas ou mais regras disparadas",
                 "Quando o baseline emite alerta.", "D-075"),
        Variavel("treino", "nenhum",
                 "Os limiares chegam prontos do aquecimento e ficam congelados.",
                 "D-033, D-046"),
        Variavel("rótulo", "vai junto, não decide",
                 "O `compromised` do holdout é copiado para a saída, ao lado da "
                 "decisão, e não é consultado.",
                 "D-113"),
    )


def variaveis_do_m11(configuracao: dict[str, dict]) -> tuple[Variavel, ...]:
    """Os modelos: quais, com que configuração, e como treinam."""
    escolhidas = tuple(
        Variavel(f"configuração {modelo}",
                 "; ".join(f"{nome}={valor}" for nome, valor in sorted(parametros.items())),
                 "Escolhida pela busca na 902, uma entre as que empataram no topo.",
                 "D-103, D-117")
        for modelo, parametros in configuracao.items()
    )

    return (
        Variavel("modelos", MODELOS,
                 "Os dois modelos supervisionados da comparação.",
                 "D-051, D-026"),
    ) + escolhidas + (
        Variavel("treino", "uma vez, no treino",
                 "Cada modelo treina uma vez e decide sobre o holdout, cortando em 0,5.",
                 "D-105, D-114"),
        Variavel("reamostragem", "nenhuma, nem SMOTE",
                 "Se a proporção entre classes é alterada antes do treino. O peso "
                 "por classe não é reamostragem.",
                 "D-024, D-105"),
    )


def variaveis_do_m12() -> tuple[Variavel, ...]:
    """A avaliação: as métricas, o teste e a trivialidade."""
    return (
        Variavel("desfecho primário", "F1, sempre com a matriz de confusão",
                 "A métrica que responde a pergunta de pesquisa.", "D-021"),
        Variavel("desfecho secundário", "o mesmo F1, só nas sessões de administradores",
                 "Onde o atalho de reconhecer o papel some.", "D-119"),
        Variavel("métricas secundárias",
                 "acurácia, precisão, revocação, especificidade, ROC AUC",
                 "O que mais é reportado em toda condição. A AUC só nos modelos.",
                 "D-022, D-119"),
        Variavel("teste estatístico",
                 "Wilcoxon pareado bilateral, correção de Holm, α de 0,05",
                 "Sobre as condições que a trivialidade mantiver × "
                 f"{len(MODELOS)} modelos; na grade, 16 comparações.",
                 "D-111, D-116, D-120"),
        Variavel("trivialidade", "árvore de profundidade 1, F1 ≥ 0,95 exclui",
                 "Treina no treino e mede no holdout, sem peso por classe.",
                 "D-028, D-116"),
        Variavel("tempo de inferência", "só a decisão, mediana de 10, um núcleo",
                 "Métrica de primeira classe, em microssegundos por sessão.", "D-025, D-106"),
        Variavel("grade", f"{len(SEEDS)} sementes × {len(SIGMAS)} σ = "
                          f"{len(SEEDS) * len(SIGMAS)} execuções",
                 "Quantas execuções a comparação usa.", "D-004, D-039"),
    )
