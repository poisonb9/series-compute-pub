#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mede o spread REAL por par e por hora de sessao nos ticks Dukascopy (.bi5).

Por que existe: a decisao 29 do projeto usa `spread_bps + 2*slippage_bps = 4 bps`
com a ressalva declarada de que os 2/1 bps sao "default de codigo, nao modelo
calibrado por par". O `.bi5` traz bid E ask -- o spread e' OBSERVAVEL, nao
precisa de estimador (Roll 1984 existe para quem nao tem o que ja' temos).

Formato .bi5 (Dukascopy): LZMA ALONE, 20 bytes por tick, big-endian:
    uint32  ms desde o inicio da hora do arquivo
    uint32  ask em pontos
    uint32  bid em pontos
    float32 ask "volume"   <- tamanho COTADO, nao execucao (medido pelo projeto 22/09)
    float32 bid "volume"

⛔ AMOSTRAGEM, nao censo: 57.397 arquivos. Amostra estratificada por hora.
"""
import glob, lzma, os, statistics, struct, sys
from collections import defaultdict

for _f in (sys.stdout, sys.stderr):
    try: _f.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

#: raiz do dado. Relativa por padrao; aponte com SC_DATA se estiver noutro lugar.
F = os.path.join(os.environ.get("SC_DATA", os.path.join("data", "external")),
                 "forex_ticks")
# divisor de preco por par: 5 digitos -> 1e5 ; 3 digitos (JPY) -> 1e3
DIV = {"EURUSD": 1e5, "GBPUSD": 1e5, "USDJPY": 1e3}
POR_HORA = 12          # arquivos amostrados por hora do dia, por par


def ticks(caminho):
    """Devolve lista de (ask, bid) em pontos. [] se o arquivo estiver vazio."""
    raw = open(caminho, "rb").read()
    if not raw:
        return []
    try:
        d = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
    except Exception:
        return []
    n = len(d) // 20
    return [struct.unpack_from(">IIIff", d, i * 20)[1:3] for i in range(n)]


def main():
    print("%-8s %6s %8s %9s %9s %9s %9s" %
          ("par", "arqs", "ticks", "spr_med", "spr_p50", "spr_p90", "bps_med"))
    resumo = {}
    for par, div in DIV.items():
        base = os.path.join(F, par)
        if not os.path.isdir(base):
            print("%-8s SEM PASTA" % par); continue
        # agrupa arquivos por hora do dia (o nome e' "HHh_ticks.bi5")
        por_hora = defaultdict(list)
        for c in glob.glob(os.path.join(base, "*", "*", "*", "*.bi5")):
            h = os.path.basename(c)[:2]
            por_hora[h].append(c)
        alvos = []
        for h in sorted(por_hora):
            fs = sorted(por_hora[h])
            if not fs: continue
            passo = max(1, len(fs) // POR_HORA)     # espalha ao longo do periodo
            alvos += fs[::passo][:POR_HORA]

        spreads_pt, por_h = [], defaultdict(list)
        bps = []
        n_arq = 0
        for c in alvos:
            ts = ticks(c)
            if not ts: continue
            n_arq += 1
            h = os.path.basename(c)[:2]
            for ask, bid in ts:
                if ask <= 0 or bid <= 0 or ask < bid:
                    continue
                s = ask - bid
                spreads_pt.append(s)
                por_h[h].append(s)
                mid = (ask + bid) / 2.0
                bps.append(1e4 * s / mid)
        if not spreads_pt:
            print("%-8s SEM TICKS" % par); continue
        spreads_pt.sort()
        pips = [s / 10.0 for s in spreads_pt]       # 1 pip = 10 pontos nos dois casos
        p50 = pips[len(pips) // 2]
        p90 = pips[int(len(pips) * 0.90)]
        print("%-8s %6d %8d %9.2f %9.2f %9.2f %9.2f" %
              (par, n_arq, len(pips), statistics.mean(pips), p50, p90,
               statistics.mean(bps)))
        resumo[par] = (por_h, statistics.mean(bps))

    print()
    print("=== SPREAD MEDIANO (pips) POR HORA UTC ===")
    print("%-5s %10s %10s %10s" % ("hora", "EURUSD", "GBPUSD", "USDJPY"))
    for h in ["%02d" % i for i in range(24)]:
        linha = []
        for par in ("EURUSD", "GBPUSD", "USDJPY"):
            v = resumo.get(par, ({}, 0))[0].get(h)
            if v:
                v = sorted(v); linha.append("%10.2f" % (v[len(v)//2] / 10.0))
            else:
                linha.append("%10s" % "-")
        print("%-5s %s" % (h, "".join(linha)))

    print()
    print("=== CONTRA A DECISAO 29 ===")
    print("decisao 29 usa: spread_bps + 2*slippage_bps = 4 bps (ida e volta)")
    for par, (_, b) in resumo.items():
        print("  %-8s spread medido = %.2f bps  (so' o spread, sem slippage)" % (par, b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
