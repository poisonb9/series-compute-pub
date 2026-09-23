# -*- coding: utf-8 -*-
"""Camada de DIMENSIONAMENTO: converte fracao de exposicao em quantidade negociavel.

POR QUE ESTA CAMADA EXISTE, E POR QUE ELA FICA ACIMA DO MOTOR
--------------------------------------------------------------
`src/position_sizing.py` produz uma FRACAO. `src/backtester.py` consome uma QUANTIDADE.
Ninguem nunca ligou os dois, e a auditoria de execucao ao vivo de 2026-08-14 marcou isso
como LETAL: o modulo de Kelly de-ratado com cap de alavancagem foi construido, auditado
(`position_sizing_engine_7`, pack 039, GO_AUDITED) e nao tem chamador vivo.

A prescricao obvia -- ligar `final_fraction` direto em `BacktestConfig.position_size` --
esta ERRADA e o erro seria SILENCIOSO: a validacao `position_size > 0` aceita `0.01` sem
reclamar, e o motor produziria escala errada sem levantar nada.

Decisao de projeto, 2026-08-14: a conversao vive ACIMA do motor. O `Backtester` continua
INVARIANTE A CAPITAL e devolve resultado comparavel entre estrategias; `metrics.py` e
`monte_carlo.py` -- engines protegidos -- continuam lendo o numero com o significado que
sempre tiveram. NENHUM engine protegido e tocado por este modulo.

A SEMANTICA, FIXADA POR DECISAO DE PROJETO EM 2026-08-14
------------------------------------------------------
`final_fraction` e NOCIONAL, nao orcamento de perda. O Kelly continuo `f* = mean/variance`
e, por definicao, a fracao da riqueza ALOCADA -- por isso passa de 1.0 sem nada estar
errado, e por isso o teto se chama `max_leverage`.

    exposicao_usd = fracao * capital
    quantidade    = exposicao_usd / preco

⚠️ NOCIONAL NAO E PROTECAO CONTRA RUINA. Exposicao de 1.25x com um stop a 10% perde 12.5%
da conta num trade. Por isso esta camada devolve os DOIS numeros lado a lado -- exposicao e
perda-no-stop -- para que ninguem precise adivinhar qual dos dois um numero solto e.

QUEM CHAMA ISTO, E A LACUNA DECLARADA
--------------------------------------
⛔ HOJE, NINGUEM. Fora de `tests/`, este modulo NAO TEM CHAMADOR VIVO -- e isso e
exatamente o defeito que as auditorias de 2026-08-14 acharam em `deflated_sharpe_ratio`,
`src/position_sizing.py` e `src/risk_engine.py`: peca construida, auditada e nao soldada.
Escrever esta camada sem declarar o consumidor a tornaria a QUARTA peca orfa do projeto.

O consumidor PRETENDIDO, e o que o bloqueia:

    consumidor ... a etapa entre "o backtest aprovou uma candidata" e "quanto arriscar
                   em dolares" -- que HOJE nao existe como codigo em lugar nenhum
                   (auditoria de execucao ao vivo, 14/08: item 4, LETAL)
    bloqueio ..... essa etapa precisa de um CAPITAL e de uma DISTANCIA DE STOP por
                   trade. O `Backtester` nao expoe nenhum dos dois, e expo-los
                   significaria alterar engine protegido (`CLAUDE.md` secao 52).

Enquanto esse bloco nao existir, este modulo e' uma peca PRONTA E DECLARADAMENTE
DESLIGADA -- nao uma peca esquecida. A diferenca esta neste paragrafo, e ele deve ser
apagado no dia em que um chamador de producao aparecer.

O QUE ISTO NAO FAZ
------------------
- nao roda estrategia, nao le dado, nao decide entrar ou sair, nao emite veredito;
- nao gasta tentativa, nao toca o holdout, nao escreve estado canonico;
- NAO envia ordem, e nao existe caminho tecnico daqui para uma ordem.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Dimensionamento", "dimensionar"]


@dataclass(frozen=True)
class Dimensionamento:
    """Uma posicao dimensionada, com as DUAS grandezas separadas e nomeadas.

    `exposicao_usd` e quanto capital fica exposto (nocional). `perda_no_stop_usd` e quanto
    se perde se o stop for atingido -- e `None` quando nenhuma distancia de stop foi dada,
    porque sem stop a perda NAO e definida e fingir um numero seria pior que omiti-lo."""

    fracao_nocional: float
    capital_usd: float
    preco: float
    quantidade: float
    exposicao_usd: float
    alavancagem: float
    perda_no_stop_usd: float | None
    perda_no_stop_fracao: float | None
    motivo_recusa: str = ""

    @property
    def recusada(self) -> bool:
        return bool(self.motivo_recusa)


def _recusa(motivo: str, fracao: float, capital: float, preco: float) -> Dimensionamento:
    """Falha fechado: toda recusa devolve quantidade ZERO e diz por que."""
    return Dimensionamento(
        fracao_nocional=fracao, capital_usd=capital, preco=preco,
        quantidade=0.0, exposicao_usd=0.0, alavancagem=0.0,
        perda_no_stop_usd=None, perda_no_stop_fracao=None, motivo_recusa=motivo,
    )


def dimensionar(
    fracao_nocional: float,
    capital_usd: float,
    preco: float,
    *,
    distancia_do_stop: float | None = None,
) -> Dimensionamento:
    """Converte uma fracao NOCIONAL em quantidade, e diz a perda-no-stop quando ela existe.

    `fracao_nocional` e o `final_fraction` de `position_size()`. `distancia_do_stop` e a
    distancia RELATIVA ate o stop (0.10 = 10% abaixo da entrada), nao um preco.

    Falha fechado em toda entrada invalida: capital nao positivo, preco nao positivo,
    fracao negativa ou nao finita devolvem quantidade ZERO com o motivo declarado.
    """
    for nome, valor in (("fracao_nocional", fracao_nocional), ("capital_usd", capital_usd), ("preco", preco)):
        if valor != valor or valor in (float("inf"), float("-inf")):
            return _recusa("%s nao e finito" % nome, fracao_nocional, capital_usd, preco)
    if fracao_nocional < 0.0:
        return _recusa("fracao_nocional negativa", fracao_nocional, capital_usd, preco)
    if capital_usd <= 0.0:
        return _recusa("capital_usd deve ser > 0", fracao_nocional, capital_usd, preco)
    if preco <= 0.0:
        return _recusa("preco deve ser > 0", fracao_nocional, capital_usd, preco)

    exposicao = fracao_nocional * capital_usd
    quantidade = exposicao / preco

    perda_usd: float | None = None
    perda_fracao: float | None = None
    if distancia_do_stop is not None:
        if distancia_do_stop != distancia_do_stop or distancia_do_stop <= 0.0:
            return _recusa("distancia_do_stop deve ser > 0 quando informada",
                           fracao_nocional, capital_usd, preco)
        perda_usd = exposicao * distancia_do_stop
        perda_fracao = perda_usd / capital_usd

    return Dimensionamento(
        fracao_nocional=fracao_nocional, capital_usd=capital_usd, preco=preco,
        quantidade=quantidade, exposicao_usd=exposicao,
        alavancagem=exposicao / capital_usd,
        perda_no_stop_usd=perda_usd, perda_no_stop_fracao=perda_fracao,
    )
