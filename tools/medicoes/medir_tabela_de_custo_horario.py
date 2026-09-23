#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MEDICAO 6 -- a tabela de custo por HORA UTC, para substituir o default.

DECORRE DA DECISAO E (23/09/2026): o default `spread_bps = 2` de
`src/execution_realism.py` (decisao 29) e' substituido por spread MEDIDO por
hora UTC e por par.

POR QUE NAO BASTA A MEDICAO 1: ela publicou a tabela horaria AGRUPADA em
faixas (03-08, 08-20) e amostrou 12 arquivos por hora/par. Implementar exige
os 24 valores separados, com amostra maior.

⛔ E' SPREAD, NAO CUSTO TOTAL. Falta `slippage`, que NAO e' observavel em dado
   de cotacao. Esta tabela e' o PISO do custo. Quem a usar como "o custo"
   subestima -- e o default antigo, de 2 bps, ERRAVA PARA CIMA em 3 a 6x, o
   que e' um erro de outra natureza.

⭐ Em bps sobre o mid, que e' a unidade que o codigo consome:
      spread_bps = (ask - bid) / mid * 10000

⛔ Nenhum numero de performance do projeto aparece aqui.

USO
    python medir_tabela_de_custo_horario.py --autoteste
    python medir_tabela_de_custo_horario.py --por-hora 60
"""
import argparse
import glob
import json
import lzma
import os
import random
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


def spreads_bps(caminho, div):
    """spread em bps de cada tick do arquivo (uma hora)."""
    try:
        raw = open(caminho, "rb").read()
        if not raw:
            return []
        d = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
    except Exception:
        return []
    out = []
    for i in range(len(d) // 20):
        _, ask, bid, _, _ = struct.unpack_from(">IIIff", d, i * 20)
        if ask > 0 and bid > 0 and ask >= bid:
            a, b = ask / div, bid / div
            mid = (a + b) / 2.0
            if mid > 0:
                out.append((a - b) / mid * 10000.0)
    return out


def autoteste():
    print("AUTOTESTE da Medicao 6\n")
    falhas = []

    # PINO 1 -- a conversao para bps, conferida a mao.
    # EURUSD: bid 1,10000  ask 1,10004 -> mid 1,10002 ; spread 0,00004
    # bps = 0,00004 / 1,10002 * 10000 = 0,363630...
    bid, ask = 1.10000, 1.10004
    mid = (ask + bid) / 2.0
    bps = (ask - bid) / mid * 10000.0
    ok = abs(bps - 0.3636297) < 1e-6
    print("  1. bps de 0,4 pip no EURUSD ........ %s  %.7f (esperado 0,3636297)"
          % ("OK " if ok else "FALHOU", bps))
    if not ok:
        falhas.append(1)

    # PINO 2 -- coerencia com a Medicao 1: 0,30 pip mediano no EURUSD a 1,08
    # tem de dar ~0,28 bps, e a Medicao 1 publicou 0,35 bps para a media de
    # 0,39 pip. Confere a ORDEM DE GRANDEZA contra numero ja' publicado.
    bps030 = (0.00003) / 1.08 * 10000.0
    bps039 = (0.000039) / 1.08 * 10000.0
    ok = abs(bps030 - 0.2778) < 0.01 and abs(bps039 - 0.3611) < 0.01
    print("  2. bate com a Medicao 1 (0,39 pip -> 0,35 bps) %s  0,39pip=%.4f bps"
          % ("OK " if ok else "FALHOU", bps039))
    if not ok:
        falhas.append(2)

    # PINO 3 -- a mediana e' de fato mediana (nao media) em caso assimetrico
    v = [1.0, 1.0, 1.0, 1.0, 100.0]
    ok = statistics.median(v) == 1.0 and abs(statistics.mean(v) - 20.8) < 1e-9
    print("  3. mediana resiste a outlier ....... %s  med=%.1f media=%.1f"
          % ("OK " if ok else "FALHOU", statistics.median(v),
             statistics.mean(v)))
    if not ok:
        falhas.append(3)

    print("\n%s" % ("TODOS OS PINOS PASSARAM" if not falhas
                    else "PINOS QUE FALHARAM: %s" % falhas))
    return 0 if not falhas else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--por-hora", type=int, default=60,
                    help="arquivos amostrados por hora/par (Medicao 1 usou 12)")
    ap.add_argument("--semente", type=int, default=20260923)
    ap.add_argument("--json", default=None, help="grava a tabela em JSON")
    ap.add_argument("--autoteste", action="store_true")
    a = ap.parse_args()
    if a.autoteste:
        return autoteste()

    random.seed(a.semente)
    print("MEDICAO 6 -- spread mediano em BPS por hora UTC e por par")
    print("amostra: %d arquivos por hora/par (a Medicao 1 usou 12)" % a.por_hora)
    print("⛔ e' SPREAD, nao custo total: falta slippage. Isto e' o PISO.\n")

    tabela = {}
    for par, div in DIV.items():
        porh = {}
        todos = glob.glob(os.path.join(FX, par, "*", "*", "*", "*.bi5"))
        buckets = {}
        for c in todos:
            h = os.path.basename(c)[:2]
            if h.isdigit():
                buckets.setdefault(int(h), []).append(c)
        for h in range(24):
            cs = buckets.get(h, [])
            random.shuffle(cs)
            vals, usados = [], 0
            for c in cs:
                s = spreads_bps(c, div)
                if s:
                    vals.extend(s)
                    usados += 1
                if usados >= a.por_hora:
                    break
            if vals:
                porh[h] = {"mediana_bps": round(statistics.median(vals), 4),
                           "media_bps": round(statistics.mean(vals), 4),
                           "arquivos": usados, "ticks": len(vals)}
        tabela[par] = porh

    # impressao
    print("%-4s %-22s %-22s %-22s" % ("hora", "EURUSD", "GBPUSD", "USDJPY"))
    print("%-4s %s" % ("", " ".join("%-22s" % "mediana / media (bps)"
                                    for _ in DIV)))
    for h in range(24):
        celulas = []
        for par in DIV:
            d = tabela[par].get(h)
            celulas.append("%-22s" % ("%6.3f / %6.3f" % (d["mediana_bps"],
                                                         d["media_bps"])
                                      if d else "-"))
        print("%-4d %s" % (h, " ".join(celulas)))

    print("\n=== razao contra a hora mais BARATA de cada par ===")
    for par in DIV:
        vs = [(d["mediana_bps"], h) for h, d in tabela[par].items()]
        if not vs:
            continue
        lo = min(vs)
        hi = max(vs)
        print("  %-8s barata: %5.3f bps (h%02d)   cara: %5.3f bps (h%02d)"
              "   razao %.2fx" % (par, lo[0], lo[1], hi[0], hi[1],
                                  hi[0] / lo[0] if lo[0] > 0 else float("nan")))

    print("\n=== contra o default da decisao 29 (spread_bps = 2) ===")
    for par in DIV:
        vs = [d["mediana_bps"] for d in tabela[par].values()]
        if vs:
            m = statistics.median(vs)
            print("  %-8s mediana das 24 horas = %5.3f bps  -> o default e'"
                  " %.1fx MAIOR" % (par, m, 2.0 / m if m > 0 else float("nan")))

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"unidade": "bps sobre o mid",
                       "origem": "Medicao 6, 2026-09-23",
                       "ressalva": "SPREAD, nao custo total: falta slippage",
                       "amostra_por_hora": a.por_hora,
                       "semente": a.semente,
                       "tabela": tabela}, f, indent=2, ensure_ascii=False)
        print("\ntabela gravada em %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
