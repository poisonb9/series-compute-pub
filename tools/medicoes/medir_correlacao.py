#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mede a correlacao entre os instrumentos do disco e o N EFETIVO.

Responde tres perguntas de uma vez, todas ja' abertas no projeto:
  (a) empilhar instrumentos amplia a amostra EFETIVA? (LdP cap. 8.5)
  (b) qual o N efetivo?                                (decisao 12)
  (c) a matriz de covariancia e' estimavel?            (LdP cap. 16)

Metrica de distancia (LdP, apendice 16.A.1), que e' METRICA VERDADEIRA:
    d[x,y] = raiz( 0.5 * (1 - rho[x,y]) )

N efetivo pela variancia da media (mesma forma do CPCV, LdP cap. 12.5):
    var[media] = N^-1 * sigma^2 * (1 + (N-1)*rho_barra)
    => N_efetivo = N / (1 + (N-1)*rho_barra)

⛔ FOREX: para nao ler 57.397 arquivos, usa-se UM arquivo por dia (a hora 20h
   UTC, a ultima liquida antes do rollover) e o ULTIMO tick dele como
   fechamento do dia. E' aproximacao declarada, nao o fechamento oficial.
   A janela 21-22h e' evitada de proposito: a Medicao 1 mostrou spread 3x ali.
"""
import csv
import glob
import io
import lzma
import math
import os
import struct
import sys
import zipfile
from collections import defaultdict

for _f in (sys.stdout, sys.stderr):
    try: _f.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

#: raiz do dado. Relativa por padrao; aponte com SC_DATA se estiver noutro lugar.
RAIZ = os.environ.get("SC_DATA", os.path.join("data", "external"))
FX = os.path.join(RAIZ, "forex_ticks")
CR = os.path.join(RAIZ, "binance_klines_raw")
DIV = {"EURUSD": 1e5, "GBPUSD": 1e5, "USDJPY": 1e3}
HORA_FX = "20"                       # ultima hora liquida; 21-22h e' rollover
INI, FIM = "2023-07-01", "2024-12-31"   # janela comum aos 14 instrumentos


def fecha_forex(par):
    """{data: mid} usando o ultimo tick da hora 20h UTC de cada dia."""
    div = DIV[par]
    out = {}
    padrao = os.path.join(FX, par, "*", "*", "*", HORA_FX + "h_ticks.bi5")
    for c in glob.glob(padrao):
        partes = c.replace("\\", "/").split("/")
        data = "%s-%s-%s" % (partes[-4], partes[-3], partes[-2])
        if not (INI <= data <= FIM):
            continue
        try:
            raw = open(c, "rb").read()
            if not raw:
                continue
            d = lzma.decompress(raw, format=lzma.FORMAT_ALONE)
        except Exception:
            continue
        n = len(d) // 20
        if n == 0:
            continue
        _, ask, bid, _, _ = struct.unpack_from(">IIIff", d, (n - 1) * 20)
        if ask > 0 and bid > 0:
            out[data] = (ask + bid) / 2.0 / div
    return out


def fecha_cripto(ativo):
    """{data: close} a partir dos klines diarios em .zip."""
    out = {}
    for z in sorted(glob.glob(os.path.join(CR, ativo, "1d", "*.zip"))):
        try:
            with zipfile.ZipFile(z) as zf:
                nome = zf.namelist()[0]
                txt = zf.read(nome).decode("utf-8", "replace")
        except Exception:
            continue
        for linha in csv.reader(io.StringIO(txt)):
            if not linha or not linha[0] or not linha[0][0].isdigit():
                continue           # pula cabecalho quando houver
            try:
                ms = int(linha[0])
                # binance passou a usar microsegundos em arquivos recentes
                if ms > 1e13:
                    ms //= 1000
                import datetime as _dt
                data = _dt.datetime.utcfromtimestamp(ms / 1000.0).strftime("%Y-%m-%d")
                if INI <= data <= FIM:
                    out[data] = float(linha[4])
            except Exception:
                continue
    return out


def retornos(serie):
    datas = sorted(serie)
    r = {}
    for i in range(1, len(datas)):
        a, b = serie[datas[i - 1]], serie[datas[i]]
        if a > 0 and b > 0:
            r[datas[i]] = math.log(b / a)
    return r


def correl(x, y):
    comuns = sorted(set(x) & set(y))
    if len(comuns) < 30:
        return None, len(comuns)
    a = [x[d] for d in comuns]
    b = [y[d] for d in comuns]
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((v - ma) ** 2 for v in a)
    vb = sum((v - mb) ** 2 for v in b)
    if va <= 0 or vb <= 0:
        return None, len(comuns)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(len(a)))
    return cov / math.sqrt(va * vb), len(comuns)


def main():
    print("janela comum: %s a %s" % (INI, FIM))
    print("forex: 1 arquivo/dia (hora %sh UTC), ultimo tick\n" % HORA_FX)

    series = {}
    for par in DIV:
        s = fecha_forex(par)
        if s:
            series[par] = retornos(s)
            print("  %-10s %4d dias" % (par, len(s)))
    for ativo in sorted(os.listdir(CR)):
        s = fecha_cripto(ativo)
        if s:
            series[ativo] = retornos(s)
            print("  %-10s %4d dias" % (ativo, len(s)))

    nomes = sorted(series)
    N = len(nomes)
    print("\ninstrumentos com serie: %d" % N)

    print("\n=== MATRIZ DE CORRELACAO (retornos log diarios) ===")
    print("%-10s %s" % ("", " ".join("%7s" % n[:7] for n in nomes)))
    M = {}
    for a in nomes:
        linha = []
        for b in nomes:
            if a == b:
                M[(a, b)] = 1.0; linha.append("%7.2f" % 1.0); continue
            if (b, a) in M:
                M[(a, b)] = M[(b, a)]
            else:
                r, _ = correl(series[a], series[b])
                M[(a, b)] = r
            linha.append("%7s" % ("-" if M[(a, b)] is None else "%.2f" % M[(a, b)]))
        print("%-10s %s" % (a[:10], " ".join(linha)))

    fora = [M[(a, b)] for i, a in enumerate(nomes) for b in nomes[i + 1:]
            if M[(a, b)] is not None]
    if not fora:
        print("\nsem pares suficientes"); return 1
    rho = sum(fora) / len(fora)
    n_eff = N / (1 + (N - 1) * rho) if (1 + (N - 1) * rho) > 0 else float("inf")

    print("\n=== N EFETIVO ===")
    print("  instrumentos (N) ............ %d" % N)
    print("  correlacao media (fora diag)  %.4f" % rho)
    print("  N_efetivo = N/(1+(N-1)*rho) . %.2f" % n_eff)
    print("  ganho de amostra ao empilhar  %.2fx (nao %dx)" % (n_eff, N))

    print("\n=== DISTANCIA d = raiz(0.5*(1-rho)) — pares mais PROXIMOS ===")
    pares = sorted(((math.sqrt(0.5 * (1 - M[(a, b)])), a, b)
                    for i, a in enumerate(nomes) for b in nomes[i + 1:]
                    if M[(a, b)] is not None))
    for d, a, b in pares[:6]:
        print("  %.3f  %s ~ %s  (rho=%.2f)" % (d, a[:10], b[:10], M[(a, b)]))
    print("  ... mais DISTANTES:")
    for d, a, b in pares[-4:]:
        print("  %.3f  %s ~ %s  (rho=%.2f)" % (d, a[:10], b[:10], M[(a, b)]))

    print("\n=== COVARIANCIA ESTIMAVEL? (LdP cap. 16) ===")
    obs = min(len(series[n]) for n in nomes)
    print("  menor serie ......... %d observacoes" % obs)
    print("  parametros da matriz  N(N+1)/2 = %d" % (N * (N + 1) // 2))
    print("  razao obs/parametros  %.2f" % (obs / (N * (N + 1) / 2.0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
