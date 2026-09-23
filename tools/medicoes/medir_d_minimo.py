#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MEDICAO 4 (passo 0e) -- o d MINIMO que torna a serie estacionaria.

PERGUNTA: uma regressao ajustada sobre `close[t]` em NIVEL roda sobre serie
com d = 0. Regressao sobre serie nao-estacionaria e' o caso classico de
correlacao espuria. Qual o d minimo que torna a serie estacionaria?

METODO PRIMARIO -- Lopez de Prado, "Advances in Financial Machine Learning",
cap. 5.5.2 e 5.6:

    pesos:  w[0] = 1 ;  w[k] = -w[k-1] * (d - k + 1) / k
            corta quando |w[k]| < tau           (o livro usa tau = 1e-5)
    FFD:    Xtil[t] = soma_k w[k] * X[t-k]      (janela de largura FIXA)
    d*   :  o menor d cuja serie FFD rejeita raiz unitaria no ADF

⛔ O livro aplica isto a LOG-PRECOS, nao a precos. Seguimos o livro.
⛔ O `0,35` do E-mini e' DAQUELA serie. Copiar o valor seria repetir o erro
   das decisoes 25 e 26. Aqui o d e' MEDIDO por serie.

VALOR CRITICO do ADF: -2,8623 (95%, regressao com constante) -- o mesmo que o
livro imprime na secao 5.6. Ver `--autoteste`, que o reobtem por Monte Carlo.

METODO DE DESEMPATE -- Tsay, "Analysis of Financial Time Series" (3a ed.),
secao 2.11 (linhas 5248-5360), para o processo puro (1-B)^d x[t] = a[t]:

    rho[1]   = d / (1 - d)          =>  d = rho[1] / (1 + rho[1])
    phi[k,k] = d / (k - d)          =>  d = k * phi[k,k] / (1 + phi[k,k])

⛔⛔ O plano falava em "estimador de Tsay" como se fosse o GPH
(Geweke-Porter-Hudak). MEDIDO: "Geweke" e "Porter-Hudak" tem ZERO ocorrencias
no texto integral do Tsay 3a ed. (indice de 2026-09-22). O que Tsay da' e' (a)
as formulas fechadas acima e (b) UMA FRASE sem formula -- "one can estimate d
using either a maximum likelihood method or a regression method with logged
periodogram at the lower frequencies". O GPH esta' nessa frase, sem nome e sem
equacao. Implementamos as formulas fechadas, que sao o que o livro de fato da'.

⚠️ LIMITE DECLARADO do desempate: as formulas de Tsay valem para o processo
PURO fracionario, SEM componente ARMA, e na faixa -0,5 < d < 0,5. Log-preco
nao e' isso. Por isso elas sao aplicadas a' PRIMEIRA DIFERENCA (os log-
retornos), estimando um d residual, e o total e' 1 + d_res. Se a serie tiver
estrutura ARMA -- e retorno de FX tem -- o estimador fica VIESADO. Ele entra
como desempate de ORDEM DE GRANDEZA, nunca como o numero de producao.

⛔ Nenhum numero de performance do projeto aparece aqui. Isto e' estatistica
   descritiva da SERIE, nao avaliacao de estrategia.

USO:
    python medir_d_minimo.py --autoteste      # prova por valor, nao le disco
    python medir_d_minimo.py                  # diario (rapido)
    python medir_d_minimo.py --horario        # + barras horarias 2024 (lento)
"""
import csv
import datetime as dt
import glob
import io
import lzma
import math
import os
import random
import struct
import sys
import zipfile

for _f in (sys.stdout, sys.stderr):
    try: _f.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

#: raiz do dado. Relativa por padrao; aponte com SC_DATA se estiver noutro lugar.
RAIZ = os.environ.get("SC_DATA", os.path.join("data", "external"))
FX = os.path.join(RAIZ, "forex_ticks")
CR = os.path.join(RAIZ, "binance_klines_raw")
DIV = {"EURUSD": 1e5, "GBPUSD": 1e5, "USDJPY": 1e3}
HORA_FX = "20"                          # ultima hora liquida (21-22h e' rollover)
INI, FIM = "2023-07-01", "2024-12-31"   # mesma janela comum da Medicao 2

TAU = 1e-5            # limiar de peso do livro (5.5.2)
CRITICO_95 = -2.8623  # ADF com constante, 95% -- valor impresso na secao 5.6
LAGS_ADF = 1          # ordem do termo de diferencas defasadas


# ---------------------------------------------------------------- algebra

def resolver(A, b):
    """Gauss com pivotamento parcial. Devolve (x, inversa) ou (None, None)."""
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] + [b[i]]
         for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-14:
            return None, None
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [v / d for v in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [M[r][c] - f * M[col][c] for c in range(2 * n + 1)]
    x = [M[i][2 * n] for i in range(n)]
    inv = [[M[i][n + j] for j in range(n)] for i in range(n)]
    return x, inv


def ols(y, X):
    """Minimos quadrados. Devolve (coeficientes, erros-padrao)."""
    n, k = len(y), len(X[0])
    if n <= k:
        return None, None
    XtX = [[sum(X[t][i] * X[t][j] for t in range(n)) for j in range(k)]
           for i in range(k)]
    Xty = [sum(X[t][i] * y[t] for t in range(n)) for i in range(k)]
    beta, inv = resolver(XtX, Xty)
    if beta is None:
        return None, None
    sse = sum((y[t] - sum(beta[i] * X[t][i] for i in range(k))) ** 2
              for t in range(n))
    s2 = sse / (n - k)
    se = [math.sqrt(max(s2 * inv[i][i], 0.0)) for i in range(k)]
    return beta, se


# ------------------------------------------------------------------- ADF

def adf(y, lags=LAGS_ADF):
    """t-estatistica de gamma em  dy[t] = a + gamma*y[t-1] + somatorio b_i*dy[t-i].

    Regressao COM constante e SEM tendencia -- o caso cujo critico e' -2,8623.
    """
    dy = [y[t] - y[t - 1] for t in range(1, len(y))]
    alvo, lin = [], []
    for t in range(lags, len(dy)):
        linha = [1.0, y[t]]                      # y[t] e' o nivel em t-1 de dy[t]
        for i in range(1, lags + 1):
            linha.append(dy[t - i])
        alvo.append(dy[t]); lin.append(linha)
    if len(alvo) < 20:
        return None
    beta, se = ols(alvo, lin)
    if beta is None or se[1] <= 0:
        return None
    return beta[1] / se[1]


# ------------------------------------------------- diferenciacao fracionaria

def pesos_ffd(d, tau=TAU):
    """Snippet 5.3 do livro: w[0]=1, w[k] = -w[k-1]*(d-k+1)/k, corte em |w|<tau."""
    w, k = [1.0], 1
    while True:
        novo = -w[-1] / k * (d - k + 1)
        if abs(novo) < tau:
            break
        w.append(novo); k += 1
        if k > 20000:                  # salvaguarda: d proximo de 0 nao converge
            break
    return w


def fracdiff_ffd(x, d, tau=TAU):
    """Janela de largura FIXA. Devolve (serie, largura descartada)."""
    w = pesos_ffd(d, tau)
    L = len(w) - 1
    if L >= len(x):
        return [], L
    out = [sum(w[k] * x[t - k] for k in range(L + 1)) for t in range(L, len(x))]
    return out, L


def correl(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((v - ma) ** 2 for v in a)
    vb = sum((v - mb) ** 2 for v in b)
    if va <= 0 or vb <= 0:
        return float("nan")
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / math.sqrt(va * vb)


def d_minimo(x, passo=0.05, critico=CRITICO_95):
    """Menor d em [0,1] cuja serie FFD rejeita raiz unitaria. Refina por bisseccao.

    Devolve dict com d, o ADF em d=0 e em d=1, e a correlacao preservada.
    """
    grade = []
    d = 0.0
    while d <= 1.0 + 1e-9:
        s, L = fracdiff_ffd(x, d)
        t = adf(s) if len(s) > 60 else None
        grade.append((round(d, 4), t, len(s), L))
        d += passo
    achou = None
    for i, (dd, t, _n, _L) in enumerate(grade):
        if t is not None and t < critico:
            achou = i
            break
    if achou is None:
        return {"d": None, "grade": grade}
    if achou == 0:
        d_est = 0.0
    else:                                   # bisseccao entre o ultimo nao e o 1o sim
        lo = grade[achou - 1][0]
        hi = grade[achou][0]
        for _ in range(6):                  # ate ~0,008 de resolucao
            meio = (lo + hi) / 2.0
            s, _L = fracdiff_ffd(x, meio)
            t = adf(s) if len(s) > 60 else None
            if t is not None and t < critico:
                hi = meio
            else:
                lo = meio
        d_est = hi
    s, L = fracdiff_ffd(x, d_est)
    orig = x[L:]
    return {"d": d_est, "grade": grade, "adf": adf(s),
            "corr": correl(orig, s), "n": len(s), "largura": L}


# ------------------------------------------------------ desempate de Tsay

def acf1(x):
    n = len(x)
    m = sum(x) / n
    den = sum((v - m) ** 2 for v in x)
    if den <= 0:
        return float("nan")
    return sum((x[t] - m) * (x[t - 1] - m) for t in range(1, n)) / den


def pacf(x, kmax=5):
    """PACF por Durbin-Levinson a partir da ACF amostral."""
    n = len(x)
    m = sum(x) / n
    den = sum((v - m) ** 2 for v in x)
    if den <= 0:
        return []
    r = [1.0] + [sum((x[t] - m) * (x[t - k] - m) for t in range(k, n)) / den
                 for k in range(1, kmax + 1)]
    phi, saida = {}, []
    for k in range(1, kmax + 1):
        if k == 1:
            phi[(1, 1)] = r[1]
        else:
            num = r[k] - sum(phi[(k - 1, j)] * r[k - j] for j in range(1, k))
            den2 = 1 - sum(phi[(k - 1, j)] * r[j] for j in range(1, k))
            if abs(den2) < 1e-14:
                return saida
            phi[(k, k)] = num / den2
            for j in range(1, k):
                phi[(k, j)] = phi[(k - 1, j)] - phi[(k, k)] * phi[(k - 1, k - j)]
        saida.append(phi[(k, k)])
    return saida


def d_tsay(retornos, kmax=5):
    """d residual dos RETORNOS pelas formulas fechadas de Tsay 2.11.

    d = rho1/(1+rho1)  e  d = k*phi_kk/(1+phi_kk). O d total da serie de nivel
    e' 1 + d_residual. ⚠️ Premissa: processo puro, sem ARMA (ver docstring).
    """
    r1 = acf1(retornos)
    por_rho = r1 / (1 + r1) if r1 > -1 else float("nan")
    por_phi = []
    for k, p in enumerate(pacf(retornos, kmax), start=1):
        if p > -1:
            por_phi.append(k * p / (1 + p))
    return por_rho, por_phi


# ------------------------------------------------------------------ dados

def fecha_forex_diario(par):
    div = DIV[par]
    out = {}
    for c in glob.glob(os.path.join(FX, par, "*", "*", "*",
                                    HORA_FX + "h_ticks.bi5")):
        p = c.replace("\\", "/").split("/")
        data = "%s-%s-%s" % (p[-4], p[-3], p[-2])
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
    return [out[k] for k in sorted(out)]


def fecha_cripto_diario(ativo):
    out = {}
    for z in sorted(glob.glob(os.path.join(CR, ativo, "1d", "*.zip"))):
        try:
            with zipfile.ZipFile(z) as zf:
                txt = zf.read(zf.namelist()[0]).decode("utf-8", "replace")
        except Exception:
            continue
        for linha in csv.reader(io.StringIO(txt)):
            if not linha or not linha[0] or not linha[0][0].isdigit():
                continue
            try:
                ms = int(linha[0])
                if ms > 1e13:
                    ms //= 1000
                data = dt.datetime.utcfromtimestamp(ms / 1000.0).strftime("%Y-%m-%d")
                if INI <= data <= FIM:
                    out[data] = float(linha[4])
            except Exception:
                continue
    return [out[k] for k in sorted(out)]


def fecha_forex_horario(par, ano="2024"):
    """Fechamento de cada barra horaria de um ano. LENTO: ~8.700 arquivos."""
    div = DIV[par]
    out = []
    for c in sorted(glob.glob(os.path.join(FX, par, ano, "*", "*", "*.bi5"))):
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
            out.append((ask + bid) / 2.0 / div)
    return out


# --------------------------------------------------------------- autoteste

def autoteste():
    """PROVA POR VALOR -- nao le o disco. Sai 1 se qualquer pino falhar."""
    print("AUTOTESTE (prova por valor; nao toca no disco)\n")
    falhas = []

    # PINO 1 -- pesos de d=1 sao exatamente [1, -1] (diferenca inteira)
    w = pesos_ffd(1.0)
    ok = len(w) == 2 and abs(w[0] - 1.0) < 1e-12 and abs(w[1] + 1.0) < 1e-12
    print("  1. pesos(d=1) == [1, -1] ............ %s  %s"
          % ("OK " if ok else "FALHOU", [round(v, 6) for v in w]))
    if not ok:
        falhas.append(1)

    # PINO 2 -- pesos batem com o binomio (-1)^k * C(d,k), calculado a' parte
    d = 0.4
    w = pesos_ffd(d, tau=1e-4)
    esperado = []
    for k in range(len(w)):
        c = 1.0
        for j in range(k):
            c *= (d - j) / (j + 1)
        esperado.append(((-1) ** k) * c)
    erro = max(abs(w[k] - esperado[k]) for k in range(len(w)))
    ok = erro < 1e-12
    print("  2. pesos(d=0,4) == (-1)^k*C(d,k) .... %s  erro max %.2e"
          % ("OK " if ok else "FALHOU", erro))
    if not ok:
        falhas.append(2)

    # PINO 3 -- FFD com d=1 reproduz a primeira diferenca, valor a valor
    random.seed(7)
    x = [100.0]
    for _ in range(300):
        x.append(x[-1] + random.gauss(0, 1))
    s, L = fracdiff_ffd(x, 1.0)
    dif = [x[t] - x[t - 1] for t in range(1, len(x))]
    erro = max(abs(s[i] - dif[i]) for i in range(len(s)))
    ok = L == 1 and len(s) == len(dif) and erro < 1e-9
    print("  3. FFD(d=1) == primeira diferenca ... %s  erro max %.2e"
          % ("OK " if ok else "FALHOU", erro))
    if not ok:
        falhas.append(3)

    # PINO 4 -- o valor critico do ADF, reobtido por Monte Carlo.
    # Sob H0 (passeio aleatorio) o percentil 5% da t deve cair perto de -2,86,
    # que e' o numero que o livro imprime. Isto prova a regressao ADF por VALOR
    # contra tabela publicada, nao contra a propria implementacao.
    random.seed(20260923)
    ts = []
    for _ in range(2000):
        y, v = [0.0], 0.0
        for _t in range(400):
            v += random.gauss(0, 1); y.append(v)
        t = adf(y, lags=0)
        if t is not None:
            ts.append(t)
    ts.sort()
    p5 = ts[int(0.05 * len(ts))]
    ok = abs(p5 - CRITICO_95) < 0.15
    print("  4. Monte Carlo do critico 5%% ........ %s  medido %.4f  vs %.4f "
          "do livro (2000 sims, n=400)"
          % ("OK " if ok else "FALHOU", p5, CRITICO_95))
    if not ok:
        falhas.append(4)

    # PINO 5 -- sanidade de direcao: ruido branco tem de ser rejeitado com folga
    random.seed(11)
    ruido = [random.gauss(0, 1) for _ in range(500)]
    t_ruido = adf(ruido)
    passeio = [0.0]
    for _ in range(500):
        passeio.append(passeio[-1] + random.gauss(0, 1))
    t_passeio = adf(passeio)
    ok = t_ruido < CRITICO_95 and t_passeio > CRITICO_95
    print("  5. ruido rejeita / passeio nao ...... %s  ruido %.2f | passeio %.2f"
          % ("OK " if ok else "FALHOU", t_ruido, t_passeio))
    if not ok:
        falhas.append(5)

    # PINO 6 -- Tsay: serie fracionaria sintetica de d conhecido volta em d
    d_ver = 0.30
    random.seed(3)
    a = [random.gauss(0, 1) for _ in range(6000)]
    psi = [1.0]
    for k in range(1, 400):
        psi.append(psi[-1] * (k - 1 + d_ver) / k)
    x = [sum(psi[k] * a[t - k] for k in range(min(t + 1, len(psi))))
         for t in range(len(psi), len(a))]
    por_rho, por_phi = d_tsay(x)
    ok = abs(por_rho - d_ver) < 0.06
    print("  6. Tsay rho1 recupera d=0,30 ....... %s  rho1->%.4f | phi_kk->%s"
          % ("OK " if ok else "FALHOU", por_rho,
             " ".join("%.3f" % v for v in por_phi)))
    if not ok:
        falhas.append(6)

    print("\n%s" % ("TODOS OS 6 PINOS PASSARAM"
                    if not falhas else "PINOS QUE FALHARAM: %s" % falhas))
    return 0 if not falhas else 1


# -------------------------------------------------------------------- saida

def analisar(nome, precos, unidade):
    if len(precos) < 120:
        print("%-12s %-8s poucos dados (%d)" % (nome, unidade, len(precos)))
        return
    x = [math.log(p) for p in precos if p > 0]
    r = [x[t] - x[t - 1] for t in range(1, len(x))]
    res = d_minimo(x)
    t0 = res["grade"][0][1]
    t1 = [g for g in res["grade"] if abs(g[0] - 1.0) < 1e-9]
    t1 = t1[0][1] if t1 else None
    por_rho, por_phi = d_tsay(r)
    d_tsay_total = 1.0 + por_rho
    if res["d"] is None:
        print("%-12s %-8s n=%-6d  d* NAO ACHADO em [0,1]  ADF(d=0)=%s"
              % (nome, unidade, len(x), "%.2f" % t0 if t0 else "?"))
        return
    print("%-12s %-8s n=%-6d  d*=%.3f  ADF(d*)=%7.3f  corr=%.4f  "
          "| ADF(d=0)=%8.3f  ADF(d=1)=%8.3f  | Tsay 1+rho1=%.3f"
          % (nome, unidade, len(x), res["d"], res["adf"], res["corr"],
             t0 if t0 is not None else float("nan"),
             t1 if t1 is not None else float("nan"), d_tsay_total))


def main():
    if "--autoteste" in sys.argv:
        return autoteste()

    print("MEDICAO 4 -- d minimo (passo 0e)")
    print("metodo: FFD + ADF (LdP cap. 5.5.2/5.6, tau=%g, critico %.4f)" % (TAU, CRITICO_95))
    print("desempate: formulas fechadas de Tsay 2.11 sobre os retornos")
    print("serie analisada: LOG-PRECO, como no livro\n")
    print("janela diaria: %s a %s (a mesma da Medicao 2)\n" % (INI, FIM))

    print("== DIARIO ==")
    for par in DIV:
        analisar(par, fecha_forex_diario(par), "diario")
    if os.path.isdir(CR):
        for ativo in sorted(os.listdir(CR)):
            s = fecha_cripto_diario(ativo)
            if s:
                analisar(ativo, s, "diario")

    if "--horario" in sys.argv:
        print("\n== HORARIO (2024) ==")
        for par in DIV:
            analisar(par, fecha_forex_horario(par), "horario")
    else:
        print("\n(sem --horario: barras horarias nao foram medidas)")

    print("\n⚠️ d* depende da RESOLUCAO e da JANELA. As specs nao declaram")
    print("   timeframe -- a mesma lacuna que a Medicao 3 apontou para k.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
