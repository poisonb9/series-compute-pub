#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Peneira de viabilidade EX-ANTE das specs (binHR, Lopez de Prado cap. 15).

Dado o desenho de um detector -- alvo, stop e frequencia de sinal -- calcula a
PRECISAO MINIMA exigida para atingir um Sharpe alvo. Se a precisao exigida for
irrealista, a spec e' ineconomica **antes de qualquer backtest**.

    E[X] = p*pi_mais + (1-p)*pi_menos
    sd[X] = (pi_mais - pi_menos) * raiz(p*(1-p))
    theta = E[X]/sd[X] * raiz(n)

⭐ Por que importa: e' a unica ferramenta da serie que REDUZ `N`, e
   `MinBTL ~ 2*ln[N]`. Descartar spec ineconomica antes de testa-la alivia o
   deficit de amostra sem precisar de mais dado.

⭐ VERIFICADO POR VALOR contra o exemplo do livro:
   n=260, pi-=-0.01, pi+=0.005, alvo SR=2  ->  livro diz p=0.72; este
   codigo devolve 0.7222.

⛔ LIMITE DECLARADO: o modelo pressupoe apostas IID e payoff BINARIO (toca
   alvo ou toca stop). Spec com saida por tempo ou por invalidacao logica nao
   se encaixa sem adaptacao. Serve como TRIAGEM, nao veredito.

⛔ O CUSTO ENTRA AQUI: a Medicao 1 mediu spread de 0,35 a 0,68 bps por par, e
   3x isso na janela 21-22h UTC. O custo e' subtraido do ganho e somado a'
   perda -- ignora-lo torna a peneira otimista.
"""
import argparse
import math
import sys

for _f in (sys.stdout, sys.stderr):
    try: _f.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

# spread medido em 23/09 (Medicao 1), em bps, por par
SPREAD_BPS = {"EURUSD": 0.35, "GBPUSD": 0.68, "USDJPY": 0.43}


def sharpe(p, pi_menos, pi_mais, n):
    e = p * pi_mais + (1 - p) * pi_menos
    sd = (pi_mais - pi_menos) * math.sqrt(p * (1 - p))
    return (e / sd) * math.sqrt(n) if sd > 0 else float("inf")


def precisao_exigida(pi_menos, pi_mais, n, alvo):
    # ⛔ o piso NAO pode ser 0.5: com R alto (alvo >> stop), a precisao exigida
    #    e' MENOR que 50% e a busca devolveria 0.500 como se fosse a resposta.
    lo, hi = 0.0001, 0.999999
    for _ in range(200):
        mid = (lo + hi) / 2
        if sharpe(mid, pi_menos, pi_mais, n) < alvo:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stop-atr", type=float, required=True,
                    help="distancia do stop em multiplos de ATR (Medicao 3)")
    ap.add_argument("--ptlim", type=float, default=4.0,
                    help="multiplo de ATR do alvo (decisao 24; padrao 4.0)")
    ap.add_argument("--atr-bps", type=float, default=None,
                    help="ATR tipico em bps; se omitido, o custo NAO e' aplicado")
    ap.add_argument("--alvo-sr", type=float, default=2.0)
    args = ap.parse_args()

    print("=== VERIFICACAO contra o exemplo do livro ===")
    print("  livro: n=260, pi-=-0.01, pi+=0.005, SR=2 -> p=0.72")
    print("  aqui ....................................... p=%.4f\n"
          % precisao_exigida(-0.01, 0.005, 260, 2.0))

    R = args.ptlim / args.stop_atr
    print("=== DESENHO DAS SPECS V2/V3 ===")
    print("  alvo ............. %.2f x ATR   (decisao 24)" % args.ptlim)
    print("  stop (medido) .... %.2f x ATR   (Medicao 3)" % args.stop_atr)
    print("  R = alvo/stop .... %.2f\n" % R)

    for rot, custo_bps in (("SEM custo", None),) + \
            ((("COM custo (%s)" % k, v) for k, v in SPREAD_BPS.items())
             if args.atr_bps else ()):
        if custo_bps is not None and args.atr_bps:
            # custo em fracao do ATR: ida e volta ~ 1 spread completo
            c = custo_bps / args.atr_bps
            pi_mais = (args.ptlim - c) / args.stop_atr
            pi_menos = -(1.0 + c / args.stop_atr)
        else:
            pi_mais, pi_menos = R, -1.0
        print("--- %s ---" % rot)
        print("  %-26s %10s" % ("frequencia (sinais/ano)", "p exigido"))
        for n, nome in ((12, "mensal"), (52, "semanal"), (104, "2x/semana"),
                        (260, "diaria"), (520, "2x/dia"), (1040, "4x/dia")):
            p = precisao_exigida(pi_menos, pi_mais, n, args.alvo_sr)
            marca = "  <- irrealista" if p > 0.70 else ""
            print("  %-26s %9.3f%s" % ("%s (%d)" % (nome, n), p, marca))
        print()

    print(">> Leitura: 'p exigido' e' a taxa de acerto MINIMA para o Sharpe")
    print("   alvo de %.1f. Acima de ~0,70 em estrategia direcional de forex" % args.alvo_sr)
    print("   e' tratado aqui como irrealista -- ⚠️ esse corte e' MEU, nao da fonte.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
