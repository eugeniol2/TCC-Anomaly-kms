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

from src.attack.parameters import AttackSpecification
from src.calibration.parameters import PERCENTILE, THRESHOLD_ATTRIBUTES
from src.dataset.build import ATTRIBUTES
from src.dataset.parameters import SHORTEST_MEASURABLE_MINUTES
from src.globals.experiment import SEEDS, SIGMAS
from src.globals.phases import EVALUATED_WEEKS, RULER_WEEKS, WEEK_COUNT
from src.historical_profiles.build import HOUR_FORMAT
from src.kms.policy import OUTCOMES
from src.population.parameters import KeyRepositorySpecification
from src.population.profiles import PROFILES
from src.traffic.parameters import (
    PRIMARY_ADDRESS_SHARE,
    ADMIN_OPERATION_MIX,
    TrafficSpecification,
)
from src.traffic.regimes import REGIMES
from src.viewer.theory import (
    Teoria,
    teoria_da_cauda,
    teoria_da_dirichlet,
    teoria_da_exponencial,
    teoria_da_geometrica,
    teoria_da_pascal,
    teoria_da_poisson,
)

COLUNAS = ("variável", "valor", "o que é")


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
            "valor": str(variavel.valor),
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
        Variavel("disabled_rate", f"{repositorio.disabled_rate:.0%}",
                 "Fração das chaves que nasce desabilitada, sorteada por chave.",
                 "D-038"),
        Variavel("formato do key_id", "k_ + 12 hexadecimais, aleatório",
                 "Como o identificador de chave é gerado. Nunca sequencial.",
                 "D-009"),
    )


def variaveis_do_m2(trafego: TrafficSpecification) -> tuple[Variavel, ...]:
    """O ritmo de cada regime e os dois desvios que produzem falha legítima."""
    lote = REGIMES["periodic_batch"]
    rotina = REGIMES["routine"]
    custodia = REGIMES["occasional_custody"]

    return (
        Variavel("profile → regime",
                 "end_user → routine; "
                 "automated_service → periodic_batch; "
                 "administrator → occasional_custody",
                 "O regime é derivado do perfil, um para um. O M2 consulta o "
                 "regime, nunca o perfil.",
                 "D-012"),
        Variavel("week_count", trafego.week_count,
                 f"Semanas simuladas: {RULER_WEEKS} de régua (o perfil e os "
                 f"limiares, do mesmo período) e {EVALUATED_WEEKS} avaliadas.",
                 "D-096"),
        Variavel("first_day", trafego.first_day,
                 "Primeiro dia simulado. Segunda-feira, fixa.", "D-067"),
        Variavel("BATCH_HOURS", lote.rhythm.hours,
                 "Horas em que o automated_service roda, todos os dias.",
                 "D-058"),
        Variavel("BATCH_JITTER_MINUTES", lote.rhythm.jitter_minutes,
                 "Desvio em torno da hora cheia do automated_service.",
                 "D-058"),
        Variavel("ROUTINE_SESSIONS_PER_BUSINESS_DAY",
                 rotina.rhythm.sessions_per_business_day,
                 "Sessões por dia útil do end_user, em média. Poisson.",
                 "D-058",
                 teoria_da_poisson(
                     rotina.rhythm.sessions_per_business_day)),
        Variavel("CUSTODY_SESSIONS_PER_BUSINESS_DAY",
                 custodia.rhythm.sessions_per_business_day,
                 "Sessões por dia útil do administrator, em média.", "D-058"),
        Variavel("CUSTODY_DISPERSION", custodia.rhythm.dispersion,
                 "Parâmetro da Pascal que dá ritmo irregular ao administrator.",
                 "D-071",
                 teoria_da_pascal(
                     custodia.rhythm.sessions_per_business_day,
                     custodia.rhythm.dispersion)),
        Variavel("janelas de horário",
                 f"automated_service: qualquer hora; end_user: "
                 f"{rotina.rhythm.opens_at}–{rotina.rhythm.closes_at} h; "
                 f"administrator: "
                 f"{custodia.rhythm.opens_at}–{custodia.rhythm.closes_at} h",
                 "Em que faixa do dia cada regime abre sessão.", "D-058"),
        Variavel("requisições por sessão, típico",
                 f"end_user: {rotina.requests_range}; "
                 f"automated_service: {lote.requests_range}; "
                 f"administrator: {custodia.requests_range}",
                 "Faixa típica de requisições por sessão. **Não é teto**: acima "
                 "dela vem a cauda da D-097.", "D-058"),
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
        Variavel("segundos entre requisições",
                 f"end_user: {rotina.seconds_between_requests}; "
                 f"automated_service: {lote.seconds_between_requests}; "
                 f"administrator: {custodia.seconds_between_requests}",
                 "Média do intervalo exponencial dentro da sessão.", "D-067",
                 teoria_da_exponencial({
                     "end_user": rotina.seconds_between_requests,
                     "automated_service": lote.seconds_between_requests,
                     "administrator": custodia.seconds_between_requests,
                 })),
        Variavel("DEFAULT_DISTINCT_KEYS_RANGE", trafego.distinct_keys_range,
                 "Chaves distintas que uma sessão toca, se couberem no alcance. "
                 "É a amplitude **típica**: a sessão que se estende também se "
                 "alarga, pela mesma cauda.", "D-058, D-098"),
        Variavel("PRIMARY_ADDRESS_SHARE", PRIMARY_ADDRESS_SHARE,
                 "Chance de a sessão vir do endereço principal; os demais decaem "
                 "geometricamente.",
                 "D-040",
                 teoria_da_geometrica(PRIMARY_ADDRESS_SHARE, 4)),
        Variavel("DEFAULT_STALE_SCOPE_RATE", f"{trafego.stale_scope_rate:.1%}",
                 "Fração das requisições que aponta para escopo obsoleto.",
                 "D-056"),
        Variavel("DEFAULT_ABSENT_IDENTIFIER_RATE",
                 f"{trafego.absent_identifier_rate:.1%}",
                 "Fração que pede identificador que não existe.", "D-076"),
        Variavel("ADMIN_OPERATION_MIX", ADMIN_OPERATION_MIX,
                 "Mistura de operações do administrador. Nenhuma célula é zero.",
                 "D-055"),
    )


def variaveis_do_m4_m5() -> tuple[Variavel, ...]:
    """O KMS quase não tem número: o que ele tem é ordem."""
    return (
        Variavel("ordem de avaliação",
                 "1 identificador inexistente, 2 fora de escopo, "
                 "3 chave desabilitada, 4 sucesso",
                 "Em que ordem o KMS testa cada condição. O primeiro caso que se "
                 "aplica decide.",
                 "D-077"),
        Variavel("OUTCOMES", OUTCOMES, "Os quatro desfechos possíveis.", "D-064"),
        Variavel("colunas do outcomes.csv", "event_id, outcome",
                 "O que o KMS devolve ao Audit Logger.", "D-078"),
        Variavel("colunas do log.csv",
                 "event_id, session_id, operator_id, timestamp, "
                 "source_ip, operation, key_id, outcome",
                 "As oito colunas do log. Sem escopo, perfil nem proprietário.",
                 "D-064"),
        Variavel("key_id registrado", "o requisitado, não o resolvido",
                 "Qual identificador vai para o log.", "D-064"),
        Variavel("rótulo no log", "não existe",
                 "Onde a marca de comprometimento fica: em arquivo separado.",
                 "D-063"),
    )


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
        Variavel("origens registradas", "só as que aparecem no log",
                 "Subconjunto das habituais que o M1 sorteou, não a lista inteira.",
                 "D-040"),
        Variavel("recálculo", "nunca depois da semana 2",
                 "Quando o perfil é atualizado.", "D-044"),
    )


def variaveis_do_m7() -> tuple[Variavel, ...]:
    """Os oito atributos, e o que fica de fora deles."""
    return (
        Variavel("unidade de análise", "a sessão",
                 "O que vira uma linha. Não existe tamanho de janela.", "D-061"),
        Variavel("ATTRIBUTES", ", ".join(ATTRIBUTES),
                 "Os oito atributos que os modelos recebem.", "D-074, D-080"),
        Variavel("distinct_keys", "contagem bruta, não razão",
                 "Como as chaves distintas entram: contadas, acompanhadas de "
                 "`events`.",
                 "D-080"),
        Variavel("identificadores preservados",
                 "session_id, operator_id, opened_at",
                 "Ficam no arquivo e não entram como atributo.", "D-015, D-088"),
        Variavel("SHORTEST_MEASURABLE_MINUTES", f"{SHORTEST_MEASURABLE_MINUTES:.6f}",
                 "Piso da duração ao calcular a taxa, equivalente a um segundo.",
                 "D-088"),
        Variavel("rótulo", "só na fase avaliada",
                 "Quando a coluna `compromised` existe.", "D-063"),
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
                 "As seis regras de grandeza. As outras duas já vêm binárias.",
                 "D-080"),
        Variavel("comparação", "> limiar, estrita",
                 "Como a regra decide se disparou.", "D-080"),
        Variavel("limiares por semente", f"{len(SEEDS)} conjuntos",
                 "Um por semente, compartilhado pelas 11 condições de σ.", "D-043"),
        Variavel("treino do baseline", "nenhum, em momento nenhum",
                 "Se o baseline aprende com rótulo em alguma etapa.",
                 "D-033, D-046"),
    )


def variaveis_do_m3(
    sigma: float, ataque: AttackSpecification, trafego: TrafficSpecification
) -> tuple[Variavel, ...]:
    """As cinco dimensões de σ, com os dois extremos lado a lado."""
    custodia = REGIMES["occasional_custody"]

    return (
        Variavel("σ desta execução", sigma,
                 "0,0 é ostensivo; 1,0 é indistinguível de uma sessão legítima.",
                 "D-004"),
        Variavel("campaign_sessions", ataque.campaign_sessions,
                 "Sessões que o atacante abre, o mesmo número nas 11 condições.",
                 "D-081"),
        Variavel("positivas no holdout", 23,
                 "Quantas sessões comprometidas caem no holdout.", "D-081, D-070"),
        Variavel("taxa, intervalo",
                 f"{ataque.ostensive_request_interval} s  →  "
                 f"{custodia.seconds_between_requests} s",
                 "Segundos entre requisições, do ostensivo ao furtivo.", "D-082"),
        Variavel("taxa, requisições por sessão",
                 f"{ataque.ostensive_requests_range}  →  {custodia.requests_range}",
                 "Tamanho da sessão, do ostensivo ao furtivo.", "D-082"),
        Variavel("chaves distintas",
                 f"{ataque.ostensive_distinct_keys_range}  →  "
                 f"{trafego.distinct_keys_range}",
                 "Amplitude da varredura, do ostensivo ao furtivo. A faixa "
                 "legítima é o típico, não um teto: a sessão que se estende "
                 "também se alarga.",
                 "D-082, D-080, D-098"),
        Variavel("horário", "madrugada, qualquer dia  →  dia útil, 09–19 h",
                 "Quando a sessão abre. É probabilidade, não grandeza: σ regula a "
                 "chance de a sessão ter a marca.",
                 "D-082"),
        Variavel("origem de rede", "endereço nunca visto  →  um dos habituais",
                 "De onde a sessão parte. Também probabilidade.", "D-082"),
        Variavel("falhas de autorização",
                 f"{ataque.ostensive_stale_scope_rate:.0%}  →  "
                 f"{trafego.stale_scope_rate:.1%}",
                 "Fração fora de escopo, do ostensivo ao furtivo.", "D-082"),
        Variavel("o que σ move", "o proveito, não o esforço",
                 "O atacante furtivo abre tantas sessões quanto o ostensivo e "
                 "consegue menos.",
                 "D-081"),
        Variavel("administrador comprometido", "sorteado por semente, entre 8",
                 "Quem tem a credencial comprometida. O mesmo nas 11 condições.",
                 "D-010, D-011"),
        Variavel("mistura de operações", "a do administrador personificado",
                 "De onde saem as operações do atacante. Não é dimensão de σ.",
                 "D-055"),
    )


def variaveis_pendentes_do_m9() -> tuple[Variavel, ...]:
    """A partição: decidida, não implementada."""
    return (
        Variavel("partição", "60 % treino, 40 % holdout",
                 "Como o conjunto avaliado se divide.", "D-070"),
        Variavel("estratificação", "por classe, e por sessão",
                 "O que a partição preserva, e o que ela nunca corta ao meio.",
                 "D-061"),
        Variavel("positivas no holdout", 23,
                 "Contagem absoluta, que é o número que importa.",
                 "D-006, D-081"),
    )


def variaveis_pendentes_do_m10_m11() -> tuple[Variavel, ...]:
    """O baseline e os modelos: decididos, não implementados."""
    return (
        Variavel("regras do baseline", "oito, uma por atributo",
                 "Seis comparam contra limiar; duas leem o perfil histórico.",
                 "D-080"),
        Variavel("ponto de operação", "duas ou mais regras disparadas",
                 "Quando o baseline emite alerta.", "D-075"),
        Variavel("especificidade do baseline", "99,29 % contra tráfego limpo",
                 "Alarme falso medido sem atacante: 0,67 % das sessões, nas 30 "
                 "sementes.", "D-080, D-075"),
        Variavel("modelos", "Random Forest, XGBoost",
                 "Os dois modelos supervisionados da comparação.",
                 "D-051, D-026"),
        Variavel("reamostragem", "nenhuma, nem SMOTE",
                 "Se a proporção entre classes é alterada antes do treino.",
                 "D-024"),
        Variavel("hiperparâmetros", "uma vez, na preparatória de semente 902",
                 "Onde a configuração dos modelos é escolhida.", "D-032, D-045"),
    )


def variaveis_pendentes_do_m12() -> tuple[Variavel, ...]:
    """A avaliação: decidida, não implementada."""
    return (
        Variavel("desfecho primário", "F1, sempre com a matriz de confusão",
                 "A métrica que responde a pergunta de pesquisa.", "D-021"),
        Variavel("métricas secundárias",
                 "acurácia, precisão, revocação, especificidade",
                 "O que mais é reportado em toda condição.", "D-022"),
        Variavel("teste estatístico",
                 "Wilcoxon pareado, correção de Holm, 22 comparações",
                 "11 condições de σ × 2 modelos, cada um contra o baseline.",
                 "D-027, D-051"),
        Variavel("grade", f"{len(SEEDS)} sementes × {len(SIGMAS)} σ = "
                          f"{len(SEEDS) * len(SIGMAS)} execuções",
                 "Quantas execuções a comparação usa.", "D-004, D-039"),
        Variavel("trivialidade", "árvore de profundidade 1, F1 ≥ 0,95 exclui",
                 "O critério que decide se uma condição entra na comparação.",
                 "D-028"),
        Variavel("remedição", "remover o atributo dominante e retreinar",
                 "O que se faz depois de identificar trivialidade.", "D-050"),
        Variavel("tempo de inferência", "com tabela própria",
                 "Métrica de primeira classe, não nota de rodapé.", "D-023"),
        Variavel("semanas", f"{WEEK_COUNT}, em {RULER_WEEKS} + {EVALUATED_WEEKS}",
                 "O período simulado de cada execução.", "D-096"),
    )
