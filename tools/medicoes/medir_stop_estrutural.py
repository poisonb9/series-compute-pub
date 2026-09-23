#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mede a distancia do STOP ESTRUTURAL em multiplos de ATR — e daí o R das specs.

Por que existe: a peneira `binHR` (LdP cap. 15) exige o par (alvo, stop) na
MESMA unidade. As seis specs V2/V3 declaram:

    alvo = ptlim * ATR_14,  ptlim = 4.0        <- conhecido ex-ante
    stop = "acima do swing high" / "abaixo do  <- ESTRUTURAL, distancia
            fundo anterior"                        NAO conhecida ex-ante

Sem a distancia tipica do swing em ATR, o R = alvo/stop nao existe, e a
peneira nao pode rodar. Este script mede essa distancia nos dados do disco.

Barras horarias: cada arquivo .bi5 E' uma hora -> OHLC dos mids daquele arquivo.
ATR de Wilder, 14 periodos, conforme as convencoes §6 do projeto.
Swing high/low: extremo local com `k` barras de cada lado (pivo confirmado).
"""
import glob
import lzma
import os
import statistics
import struct
import sys

for _f in (sys.stdout, sys.stderr):
    try: _f.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

#: raiz do dado. Relativa por padrao; aponte com SC_DATA se estiver noutro lugar.
FX = os.path.join(os.environ.get("SC_DATA", os.path.join("data", "external")),
                  "forex_ticks")
DIV = {"EURUSD": 1e5, "GBPUSD": 1e5, "USDJPY": 1e3}
K = 3              # barras de cada lado para confirmar um pivo
PERIODO = 14       # ATR de Wilder


def barra(caminho, div):
    """OHLC dos mids de um arquivo .bi5 (uma hora). None se vazio."""
    try:
        raw = open(caminho, "rb").read()
        if not raw:
            return None
        d = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
    except Exception:
        return None
    n = len(d) // 20
    if n == 0:
        return None
    mids = []
    for i in range(n):
        _, ask, bid, _, _ = struct.unpack_from(">IIIff", d, i * 20)
        if ask > 0 and bid > 0:
            mids.append((ask + bid) / 2.0 / div)
    if not mids:
        return None
    return (mids[0], max(mids), min(mids), mids[-1])   # O H L C


def atr_wilder(barras, periodo=PERIODO):
    """ATR de Wilder. Devolve lista alinhada a `barras` (None no warm-up)."""
    trs = []
    for i, (o, h, l, c) in enumerate(barras):
        if i == 0:
            trs.append(h - l)
        else:
            pc = barras[i - 1][3]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    out = [None] * len(barras)
    if len(trs) <= periodo:
        return out
    media = sum(trs[:periodo]) / periodo
    out[periodo - 1] = media
    for i in range(periodo, len(trs)):
        media = (media * (periodo - 1) + trs[i]) / periodo
        out[i] = media
    return out


def main():
    print("barras HORARIAS (1 arquivo .bi5 = 1 hora) | ATR_14 de Wilder | pivo k=%d\n" % K)
    print("%-8s %7s %8s %9s %9s %9s %9s" %
          ("par", "barras", "pivos", "med(ATR)", "p50", "p90", "R_implic"))
    for par, div in DIV.items():
        cs = sorted(glob.glob(os.path.join(FX, par, "2024", "*", "*", "*.bi5")))
        barras, idx = [], []
        for c in cs:
            b = barra(c, div)
            if b:
                barras.append(b); idx.append(c)
        if len(barras) < 200:
            print("%-8s poucos dados (%d)" % (par, len(barras))); continue
        atrs = atr_wilder(barras)

        dists = []
        for i in range(K, len(barras) - K):
            if atrs[i] is None or atrs[i] <= 0:
                continue
            h = [b[1] for b in barras[i - K:i + K + 1]]
            l = [b[2] for b in barras[i - K:i + K + 1]]
            c = barras[i][3]
            # pivo de alta confirmado: a barra i tem a maior maxima da janela
            if barras[i][1] == max(h):
                dists.append(abs(barras[i][1] - c) / atrs[i])
            # pivo de baixa confirmado
            if barras[i][2] == min(l):
                dists.append(abs(c - barras[i][2]) / atrs[i])
        if not dists:
            print("%-8s sem pivos" % par); continue
        dists.sort()
        p50 = dists[len(dists) // 2]
        p90 = dists[int(len(dists) * 0.90)]
        media = statistics.mean(dists)
        r_imp = 4.0 / p50 if p50 > 0 else float("inf")
        print("%-8s %7d %8d %9.2f %9.2f %9.2f %9.2f" %
              (par, len(barras), len(dists), media, p50, p90, r_imp))
    print()
    print("med/p50/p90 = distancia do stop estrutural, em MULTIPLOS DE ATR")
    print("R_implic    = alvo (4 ATR) / stop mediano   <- o R que falta a' peneira")
    return 0


if __name__ == "__main__":
    sys.exit(main())
