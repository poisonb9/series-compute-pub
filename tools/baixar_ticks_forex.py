#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baixa ticks .bi5 do Dukascopy -- EDUCADO de proposito, e o motivo e' medido.

⛔⛔ O LIMITE E' DO FEED, NAO DA REDE. Medido pelo projeto em 13/09/2026: ao
baixar depressa, o Dukascopy passou a devolver HTTP 429 e cada arquivo de
~60 KB levava 17 a 24 SEGUNDOS (~4 KB/s), com 2 de 5 horas caindo por
URLError. No MESMO instante, GitHub respondia em 0,09 s e Binance em 0,30 s --
a rede estava boa. **Baixar depressa foi o que criou a lentidao.**

⭐ "Um baixador EDUCADO termina antes de um apressado que e' bloqueado."

⛔ POR ISSO ESTE SCRIPT NAO PARALELIZA, e nao deve ser posto em matrix. Vinte
   runners baixando ao mesmo tempo disparam o mesmo bloqueio, so' que juntos.
   O paralelismo do workflow fica na COMPUTACAO.

⛔⛔ 404 SIGNIFICA UMA COISA SO': nao ha' dado naquela hora (fim de semana,
   feriado). Qualquer outra falha esgotada LEVANTA -- nunca vira arquivo vazio.
   A versao antiga do baixador do projeto devolvia "nada" nos dois casos, e o
   resultado era um mes inteiro de arquivos vazios SEM UMA LINHA DE AVISO,
   com a retomada pulando esse mes PARA SEMPRE. Buraco por bloqueio ficava
   indistinguivel de "o mercado estava fechado".

⚠️ ARMADILHA DA URL: o mes do Dukascopy e' ZERO-INDEXADO (janeiro = 00), mas a
   pasta local e' 1-indexada (janeiro = 01). Errar isto baixa o mes errado sem
   nenhum erro visivel.

USO
    python baixar_ticks_forex.py --ano 2024 --pares EURUSD GBPUSD USDJPY
"""
import argparse
import calendar
import os
import sys
import time
import urllib.error
import urllib.request

for _f in (sys.stdout, sys.stderr):
    try: _f.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

FEED = "https://datafeed.dukascopy.com/datafeed"
RAIZ = os.path.join("data", "external", "forex_ticks")
PAUSA = 0.25          # medido: 0,05 nao bastou e gerou 429


class FeedBloqueado(RuntimeError):
    """O feed recusou de forma que NAO e' 'nao ha' dado'."""


def buscar(url, tentativas=5):
    """bytes, ou None SE E SOMENTE SE o feed disser 404."""
    espera = 1.0
    ultimo = None
    for _ in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "series-compute/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None                      # nao ha' dado nesta hora
            ultimo = "HTTP %s" % e.code
            if e.code == 429:
                espera = max(espera, 5.0)        # o feed pediu calma
        except Exception as e:                   # noqa: BLE001
            ultimo = type(e).__name__
        time.sleep(espera)
        espera = min(espera * 2, 60.0)
    raise FeedBloqueado("esgotou %d tentativas em %s (ultimo: %s)"
                        % (tentativas, url, ultimo))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano", type=int, required=True)
    ap.add_argument("--pares", nargs="+", required=True)
    ap.add_argument("--raiz", default=RAIZ)
    a = ap.parse_args()

    baixados = pulados = vazios = 0
    t0 = time.time()
    for par in a.pares:
        for mes in range(1, 13):
            dias = calendar.monthrange(a.ano, mes)[1]
            for dia in range(1, dias + 1):
                pasta = os.path.join(a.raiz, par, "%04d" % a.ano,
                                     "%02d" % mes, "%02d" % dia)
                os.makedirs(pasta, exist_ok=True)
                for hora in range(24):
                    alvo = os.path.join(pasta, "%02dh_ticks.bi5" % hora)
                    if os.path.exists(alvo):
                        pulados += 1
                        continue
                    # ⚠️ mes-1 na URL, mes na pasta
                    url = "%s/%s/%04d/%02d/%02d/%02dh_ticks.bi5" % (
                        FEED, par, a.ano, mes - 1, dia, hora)
                    dados = buscar(url)
                    time.sleep(PAUSA)
                    if dados is None:
                        vazios += 1
                        continue
                    with open(alvo, "wb") as f:
                        f.write(dados)
                    baixados += 1
            # ⭐ ECOAR o valor medido a cada passo. Laco que nao ecoa parece
            #    vigilancia sem ser -- a licao do vigia que rodou 3 horas
            #    falhando em silencio.
            print("  %s %04d-%02d | baixados %d | ja' tinha %d | sem dado %d "
                  "| %.1f min" % (par, a.ano, mes, baixados, pulados, vazios,
                                  (time.time() - t0) / 60.0))
            sys.stdout.flush()

    print("\nTOTAL baixados %d | ja' presentes %d | horas sem dado %d | %.1f min"
          % (baixados, pulados, vazios, (time.time() - t0) / 60.0))
    if baixados == 0 and pulados == 0:
        print("⛔ nada foi obtido -- trate como falha, nao como 'mercado fechado'")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FeedBloqueado as e:
        print("⛔⛔ FEED BLOQUEADO: %s" % e)
        print("   Isto NAO e' 'nao ha' dado'. Nao trate o resultado parcial")
        print("   como completo, e nao grave cache a partir dele.")
        sys.exit(2)
