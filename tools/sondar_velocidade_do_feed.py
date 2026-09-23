#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sonda a velocidade do feed ANTES de comprometer a cota de Actions.

⛔⛔ POR QUE EXISTE. Medido nesta casa em 23/09/2026: UM arquivo de 25 KB do
Dukascopy levou **15,1 segundos**. Nesse ritmo, um ano de 3 pares (26.352
arquivos) levaria **110 horas** -- 3,3x os 2.000 minutos mensais inteiros, e
muito acima do teto de 6 h por job do GitHub Actions.

⇒ Se o runner tiver a MESMA velocidade, baixar da fonte nao e' viavel e o dado
  precisa subir pronto. Se for rapido, o download cabe. **Isso nao se adivinha:
  mede-se com 10 arquivos, que custam segundos, antes de gastar horas.**

Esta sonda e' barata de proposito e NAO grava nada: ela so' responde "quanto
tempo levaria o download inteiro, a partir DESTE runner".
"""
import argparse
import sys
import time
import urllib.error
import urllib.request

for _f in (sys.stdout, sys.stderr):
    try: _f.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

FEED = "https://datafeed.dukascopy.com/datafeed"
# horas de um dia util conhecido (2024-01-02, terca), mes 00 = JANEIRO
ALVOS = [("EURUSD", 2024, 0, 2, h) for h in range(24)]
ARQUIVOS_DE_UM_ANO_3_PARES = 26352


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()

    tempos, bytes_totais, falhas = [], 0, 0
    for par, ano, mes0, dia, hora in ALVOS[:a.n]:
        url = "%s/%s/%04d/%02d/%02d/%02dh_ticks.bi5" % (FEED, par, ano, mes0, dia, hora)
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "series-compute/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r:
                d = r.read()
            dt = time.time() - t0
            tempos.append(dt)
            bytes_totais += len(d)
            print("  %02dh  %7d bytes  %6.2f s" % (hora, len(d), dt))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print("  %02dh  sem dado (404)" % hora)
            else:
                falhas += 1
                print("  %02dh  HTTP %s" % (hora, e.code))
        except Exception as e:                        # noqa: BLE001
            falhas += 1
            print("  %02dh  %s" % (hora, type(e).__name__))
        sys.stdout.flush()

    if not tempos:
        print("\n⛔ nenhum arquivo baixado -- o feed esta' recusando deste runner.")
        return 2

    media = sum(tempos) / len(tempos)
    kbs = (bytes_totais / 1024.0) / sum(tempos)
    horas = ARQUIVOS_DE_UM_ANO_3_PARES * media / 3600.0
    print("\n%d arquivos | media %.2f s | %.1f KB/s | falhas %d"
          % (len(tempos), media, kbs, falhas))
    print("PROJECAO para 1 ano x 3 pares (%d arquivos): %.1f HORAS (%.0f min)"
          % (ARQUIVOS_DE_UM_ANO_3_PARES, horas, horas * 60))
    print("referencia medida nesta casa em 23/09/2026: 15,1 s por arquivo")

    if horas > 5.5:
        print("\n⛔ NAO CABE: passa do teto de 6 h por job. O dado tem de subir")
        print("   pronto (release asset ou cache pre-carregado), nao ser baixado.")
        return 1
    if horas * 60 > 400:
        print("\n⚠️ CABE NO JOB, mas consome %.0f dos 2.000 minutos do mes."
              % (horas * 60))
        return 0
    print("\n⭐ VIAVEL: o download cabe no job e na cota.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
