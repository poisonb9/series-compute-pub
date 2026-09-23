#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MEDICAO 5 (passo 0d) -- barras de TICK contra barras de TEMPO.

PERGUNTA (laudo 12, §4): as barras de tick tem propriedades estatisticas
melhores que as barras de tempo NO DADO DESTE DISCO? O livro afirma que sim;
ninguem mediu aqui. E' a unica recomendacao da serie que SUBSTITUI trabalho
feito -- trocar a unidade de observacao reescreve todas as specs -- entao a
medicao vem ANTES de reescrever qualquer coisa.

FONTE -- Lopez de Prado, "Advances in Financial Machine Learning", cap. 2:

    "time-sampled series often exhibit poor statistical properties, like
     serial correlation, heteroscedasticity, and non-normality of returns"

    Mandelbrot & Taylor (1967): "Price changes over a FIXED NUMBER OF
    TRANSACTIONS may have a Gaussian distribution. Price changes over a FIXED
    TIME PERIOD may follow a stable Paretian distribution, whose variance is
    INFINITE."

⛔⛔ RESSALVA QUE O PROJETO JA' MEDIU (22/09) E QUE LIMITA ESTA MEDICAO:
   o "tick" da Dukascopy e' ATUALIZACAO DE COTACAO, nao transacao. Mandelbrot
   & Taylor falam de "number of transactions". Barra de volume e de dolar
   estao FECHADAS -- o dado nao tem volume executado. O que se mede aqui e'
   proxy de um proxy, e um resultado favoravel NAO prova a tese do livro:
   prova apenas que amostrar por chegada de COTACAO ajuda neste dado.

AS TRES METRICAS, uma por defeito citado pelo livro:
   nao-normalidade ... Jarque-Bera = n/6 * (S^2 + (K-3)^2/4), critico 5,991
   correlacao serial. ACF(1) dos retornos, contra a banda +-1,96/raiz(n)
   heterocedastic. .. ACF(1) dos retornos AO QUADRADO (e' o teste ARCH)

⭐ COMPARACAO JUSTA: o tamanho da barra de tick e' calibrado para produzir o
   MESMO numero de barras que as de tempo. Sem isso, a diferenca de n mudaria
   a JB (que e' proporcional a n) e nada seria comparavel. A licao do d
   minimo (Medicao 4) foi exatamente essa.

⛔ Nenhum numero de performance do projeto aparece aqui.

USO
    python medir_barras_tick_x_tempo.py --autoteste
    python medir_barras_tick_x_tempo.py --par EURUSD
    python medir_barras_tick_x_tempo.py              # os tres pares (lento)
"""
import argparse
import array
import glob
import lzma
import math
import os
import random
import struct
import sys

for _f in (sys.stdout, sys.stderr):
    try: _f.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

#: raiz do dado. Relativa por padrao; aponte com SC_DATA se estiver noutro lugar.
FX = os.path.join(os.environ.get("SC_DATA", os.path.join("data", "external")),
                  "forex_ticks")
DIV = {"EURUSD": 1e5, "GBPUSD": 1e5, "USDJPY": 1e3}
ANO = "2024"
JB_CRITICO = 5.991        # qui-quadrado, 2 g.l., 95%


# ------------------------------------------------------------- estatistica

def momentos(x):
    n = len(x)
    m = sum(x) / n
    m2 = sum((v - m) ** 2 for v in x) / n
    if m2 <= 0:
        return m, 0.0, 0.0, 0.0
    m3 = sum((v - m) ** 3 for v in x) / n
    m4 = sum((v - m) ** 4 for v in x) / n
    s = m3 / m2 ** 1.5
    k = m4 / m2 ** 2
    return m, math.sqrt(m2), s, k


def jarque_bera(x):
    """JB = n/6 * (S^2 + (K-3)^2/4). Devolve (JB, assimetria, curtose)."""
    n = len(x)
    _m, _sd, s, k = momentos(x)
    return n / 6.0 * (s * s + (k - 3.0) ** 2 / 4.0), s, k


def acf(x, lag=1):
    n = len(x)
    m = sum(x) / n
    den = sum((v - m) ** 2 for v in x)
    if den <= 0:
        return 0.0
    return sum((x[t] - m) * (x[t - lag] - m) for t in range(lag, n)) / den


# ------------------------------------------------------------------ barras

def barras_de_tick(mids, por_barra):
    """Fecha uma barra a cada `por_barra` ticks. Devolve os fechamentos."""
    return [mids[i + por_barra - 1]
            for i in range(0, len(mids) - por_barra + 1, por_barra)]


def retornos_log(fechamentos):
    r = []
    for i in range(1, len(fechamentos)):
        a, b = fechamentos[i - 1], fechamentos[i]
        if a > 0 and b > 0:
            r.append(math.log(b / a))
    return r


def ler_par(par, ano=ANO):
    """Uma passada pelo disco. Devolve (mids de TODOS os ticks, fechos horarios).

    ⚠️ RAM: os mids vao num array('d') -- 4 milhoes de ticks sao ~32 MB. Esta
    maquina tem ~1,6 GB livres, entao guardar tuplas com carimbo de tempo
    estouraria. O carimbo nao e' preciso: 1 arquivo .bi5 = 1 HORA, entao a
    barra de tempo e' o proprio arquivo.
    """
    mids = array.array("d")
    horarias = []
    horas = []            # hora UTC de cada barra horaria, para o teste de diluicao
    div = DIV[par]
    caminhos = sorted(glob.glob(os.path.join(FX, par, ano, "*", "*", "*.bi5")))
    for c in caminhos:
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
        ultimo = None
        for i in range(n):
            _, ask, bid, _, _ = struct.unpack_from(">IIIff", d, i * 20)
            if ask > 0 and bid > 0:
                v = (ask + bid) / 2.0 / div
                mids.append(v)
                ultimo = v
        if ultimo is not None:
            horarias.append(ultimo)
            nome = os.path.basename(c)
            horas.append(int(nome[:2]) if nome[:2].isdigit() else -1)
    return mids, horarias, horas, len(caminhos)


# --------------------------------------------------------------- autoteste

def autoteste():
    print("AUTOTESTE da Medicao 5 (nao le o disco)\n")
    falhas = []

    # PINO 1 -- JB calculado a mao num caso minusculo.
    # x = [1,2,3,4] -> media 2,5 ; m2=1,25 ; m3=0 ; m4=2,5625
    # S = 0 ; K = 2,5625/1,5625 = 1,64 ; JB = 4/6*(0 + (1,64-3)^2/4) = 0,30827
    jb, s, k = jarque_bera([1.0, 2.0, 3.0, 4.0])
    ok = abs(jb - 0.3082666667) < 1e-9 and abs(s) < 1e-12 and abs(k - 1.64) < 1e-12
    print("  1. JB de [1,2,3,4] == 0,3082667 .... %s  JB=%.7f S=%.3f K=%.3f"
          % ("OK " if ok else "FALHOU", jb, s, k))
    if not ok:
        falhas.append(1)

    # PINO 2 -- gaussiana: curtose ~3, assimetria ~0, JB NAO rejeita
    random.seed(42)
    g = [random.gauss(0, 1) for _ in range(20000)]
    jb, s, k = jarque_bera(g)
    ok = jb < JB_CRITICO and abs(k - 3.0) < 0.15 and abs(s) < 0.06
    print("  2. gaussiana nao rejeita ........... %s  JB=%.2f (critico %.3f)"
          " K=%.3f" % ("OK " if ok else "FALHOU", jb, JB_CRITICO, k))
    if not ok:
        falhas.append(2)

    # PINO 3 -- cauda pesada (t de Student ~ razao de gaussianas) REJEITA com
    # folga. Sem este pino, um JB que so' devolve numero baixo passaria.
    random.seed(7)
    pesada = [random.gauss(0, 1) / max(abs(random.gauss(0, 1)), 0.25)
              for _ in range(20000)]
    jb2, _s2, k2 = jarque_bera(pesada)
    ok = jb2 > JB_CRITICO * 100 and k2 > 5
    print("  3. cauda pesada rejeita ............ %s  JB=%.0f K=%.1f"
          % ("OK " if ok else "FALHOU", jb2, k2))
    if not ok:
        falhas.append(3)

    # PINO 4 -- ACF(1) recupera o phi de um AR(1) simulado
    random.seed(3)
    phi, x, v = 0.6, [], 0.0
    for _ in range(50000):
        v = phi * v + random.gauss(0, 1)
        x.append(v)
    a = acf(x, 1)
    ok = abs(a - phi) < 0.02
    print("  4. ACF(1) recupera phi=0,6 ......... %s  medido %.4f"
          % ("OK " if ok else "FALHOU", a))
    if not ok:
        falhas.append(4)

    # PINO 5 -- a barra de tick fecha onde deve, valor a valor
    mids = [float(i) for i in range(1, 101)]      # 1..100
    b = barras_de_tick(mids, 10)
    ok = b == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    print("  5. barra de 10 ticks fecha certo ... %s  %s"
          % ("OK " if ok else "FALHOU", [int(v) for v in b[:4]] + ["..."]))
    if not ok:
        falhas.append(5)

    # PINO 6 -- ACF(1) dos quadrados PEGA heterocedasticidade que a ACF(1)
    # dos niveis NAO pega. E' a distincao que a medicao inteira depende.
    # ARCH(1) cannonico: sig2[t] = w + alpha*x[t-1]^2 ; x[t] = sig[t]*eps[t].
    # (a 1a versao deste pino atualizava sig DUAS vezes por passo e vazava
    #  correlacao para os NIVEIS -- defeito do gerador, nao do estimador.)
    # ⛔ alpha=0,5 e nao 0,85: com alpha > 1/raiz(3) o ARCH(1) NAO tem quarto
    #    momento finito, a ACF amostral nao converge, e o pino media o ruido
    #    do estimador. Isso vale como aviso para o dado real, que tem cauda
    #    pesada: ACF sobre cauda pesada e' estimativa instavel.
    random.seed(11)
    x, ant2 = [], 1.0
    for _ in range(50000):
        sig2 = 0.1 + 0.50 * ant2
        v = math.sqrt(sig2) * random.gauss(0, 1)
        x.append(v)
        ant2 = v * v
    a_niv = abs(acf(x, 1))
    a_qua = acf([v * v for v in x], 1)
    ok = a_niv < 0.03 and a_qua > 0.15
    print("  6. ARCH: quadrados acusam, niveis nao %s  |ACF niveis|=%.3f "
          "ACF quadrados=%.3f" % ("OK " if ok else "FALHOU", a_niv, a_qua))
    if not ok:
        falhas.append(6)

    print("\n%s" % ("TODOS OS 6 PINOS PASSARAM" if not falhas
                    else "PINOS QUE FALHARAM: %s" % falhas))
    return 0 if not falhas else 1


# ------------------------------------------------------------------- saida

def linha(rotulo, r):
    n = len(r)
    jb, s, k = jarque_bera(r)
    a1 = acf(r, 1)
    aq = acf([v * v for v in r], 1)
    banda = 1.96 / math.sqrt(n)
    print("  %-16s n=%-6d  JB=%12.0f  S=%+6.2f K=%8.2f  ACF1=%+.4f%s  "
          "ACF1(r2)=%+.4f%s"
          % (rotulo, n, jb, s, k,
             a1, "*" if abs(a1) > banda else " ",
             aq, "*" if abs(aq) > banda else " "))
    return {"n": n, "jb": jb, "s": s, "k": k, "a1": a1, "aq": aq,
            "banda": banda}


def medir_par(par):
    print("\n===== %s (%s) =====" % (par, ANO))
    mids, horarias, horas, n_arq = ler_par(par)
    if len(mids) < 1000 or len(horarias) < 100:
        print("  poucos dados (%d ticks, %d horas)" % (len(mids), len(horarias)))
        return None
    r_tempo = retornos_log(horarias)
    # calibra a barra de tick para dar o MESMO numero de barras
    por_barra = max(1, len(mids) // len(horarias))
    tick_fech = barras_de_tick(mids, por_barra)
    r_tick = retornos_log(tick_fech)
    print("  %d arquivos | %d ticks | %d barras horarias | %d ticks/barra "
          "-> %d barras de tick"
          % (n_arq, len(mids), len(horarias), por_barra, len(tick_fech)))
    print("  (* = fora da banda +-1,96/raiz(n), isto e', significativo)\n")
    a = linha("barra de TEMPO", r_tempo)
    b = linha("barra de TICK", r_tick)

    # TESTE DE DILUICAO: se a heterocedasticidade menor das barras de TEMPO for
    # artefato das horas mortas (retorno ~0 na madrugada quebra a persistencia
    # da variancia), entao restringir as barras de tempo as horas liquidas deve
    # AUMENTAR a ACF1(r2) delas. Se aumentar, a vantagem aparente do TEMPO nessa
    # metrica e' diluicao, nao virtude.
    liq = [horarias[i] for i in range(len(horarias)) if 8 <= horas[i] <= 16]
    if len(liq) > 200:
        c_ = linha("TEMPO 8-16 UTC", retornos_log(liq))
        print("    ^ so' horas liquidas; retornos cruzando a noite entram como"
              " gap (limite declarado)")
    else:
        c_ = None
    print("\n  veredito por metrica (menor e' melhor nas tres):")
    for nome, ka in (("nao-normalidade (JB)", "jb"),
                     ("correlacao serial |ACF1|", "a1"),
                     ("heterocedastic. ACF1(r2)", "aq")):
        va, vb = abs(a[ka]), abs(b[ka])
        if vb < va:
            veredito = "TICK melhor  (%.1f%% menor)" % (100.0 * (va - vb) / va) \
                if va > 0 else "TICK melhor"
        elif va < vb:
            veredito = "TEMPO melhor (%.1f%% menor)" % (100.0 * (vb - va) / vb) \
                if vb > 0 else "TEMPO melhor"
        else:
            veredito = "empate"
        print("    %-26s %s" % (nome, veredito))
    if c_ is not None:
        print("")
        print("  teste de diluicao: ACF1(r2) do TEMPO passa de %+.4f (24h)"
              " para %+.4f (8-16 UTC)" % (a["aq"], c_["aq"]))
        if abs(c_["aq"]) > abs(a["aq"]):
            print("    ⇒ AUMENTOU: a vantagem do TEMPO nesta metrica e'"
                  " em parte DILUICAO pelas horas mortas.")
        else:
            print("    ⇒ nao aumentou: a diluicao NAO explica a vantagem.")
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", default=None)
    ap.add_argument("--autoteste", action="store_true")
    a = ap.parse_args()
    if a.autoteste:
        return autoteste()

    print("MEDICAO 5 -- barras de TICK x barras de TEMPO (passo 0d)")
    print("⛔ o 'tick' da Dukascopy e' atualizacao de COTACAO, nao transacao.")
    print("   Barra de volume e de dolar estao FECHADAS: falta volume executado.")
    print("   Resultado favoravel aqui NAO prova a tese do livro.")
    pares = [a.par] if a.par else list(DIV)
    for p in pares:
        if p not in DIV:
            print("par desconhecido: %s" % p)
            return 2
        medir_par(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
