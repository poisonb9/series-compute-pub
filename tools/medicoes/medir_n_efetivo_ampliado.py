#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Medicao 2 REFEITA com a janela ampliada -- o N efetivo depois da coleta.

MOTIVO: a Medicao 2 (23/09) mediu N_efetivo = 2,49 numa janela comum de apenas
**18 meses**, e declarou por que: BTCUSDT, ETHUSDT e SOLUSDT paravam em
2024-12 enquanto os outros oito criptos iam ate' 2026-06. Os tres que faltavam
eram os MAIS LIQUIDOS.

⭐ Os meses que faltavam foram baixados em 23/09/2026 (`tools/baixar_klines_
faltantes.py`), da fonte SPOT -- confirmada por hash contra o disco -- e cada
arquivo conferido contra o `.CHECKSUM` publicado pela Binance.

⛔ O STAGING NAO FOI PROMOVIDO para `data/external/`. Este script le' as duas
   fontes LADO A LADO e nao escreve nada. Promover o dado e' ato humano: o
   CSV de origem declara `fresh_data_quarantined_unassigned`.

⚠️ O forex vai ate' 2026-07, entao ele NUNCA foi o limite da janela comum --
   eram os tres criptos. A janela deve passar de ~18 para ~36 meses.

USO
    python medir_n_efetivo_ampliado.py --staging <pasta>
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medir_correlacao as mc                                      # noqa: E402

for _f in (sys.stdout, sys.stderr):
    try: _f.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass


def cripto_unido(ativo, staging):
    """Une os meses de data/external com os do staging. Sem sobrescrever."""
    base = dict(mc.fecha_cripto(ativo))
    antes = len(base)
    orig = mc.CR
    try:
        mc.CR = staging
        novo = mc.fecha_cripto(ativo)
    finally:
        mc.CR = orig
    so_novos = 0
    for d, v in novo.items():
        if d not in base:
            base[d] = v
            so_novos += 1
    return base, antes, so_novos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True)
    ap.add_argument("--ini", default="2023-07-01")
    ap.add_argument("--fim", default="2026-06-30")
    a = ap.parse_args()

    mc.INI, mc.FIM = a.ini, a.fim
    print("MEDICAO 2 AMPLIADA -- janela %s a %s" % (mc.INI, mc.FIM))
    print("cripto: data/external UNIDO ao staging verificado (nao promovido)\n")

    series, brutas = {}, {}
    for par in mc.DIV:
        s = mc.fecha_forex(par)
        if s:
            series[par] = mc.retornos(s)
            brutas[par] = s
            print("  %-10s %4d dias  (forex)" % (par, len(s)))
    for ativo in sorted(os.listdir(mc.CR)):
        if not os.path.isdir(os.path.join(mc.CR, ativo)):
            continue
        s, antes, novos = cripto_unido(ativo, a.staging)
        if s:
            series[ativo] = mc.retornos(s)
            brutas[ativo] = s
            marca = ("  <- +%d do staging" % novos) if novos else ""
            print("  %-10s %4d dias  (%d ja' tinha)%s"
                  % (ativo, len(s), antes, marca))

    nomes = sorted(series)
    N = len(nomes)
    # janela REALMENTE comum: datas presentes em TODOS
    comuns = None
    for n in nomes:
        s = set(brutas[n])
        comuns = s if comuns is None else (comuns & s)
    print("\ninstrumentos: %d | dias comuns a TODOS: %d" % (N, len(comuns)))
    if comuns:
        print("janela comum efetiva: %s a %s" % (min(comuns), max(comuns)))

    M = {}
    for i, x in enumerate(nomes):
        for y in nomes[i + 1:]:
            r, _n = mc.correl(series[x], series[y])
            M[(x, y)] = r
    fora = [v for v in M.values() if v is not None]
    if not fora:
        print("sem pares suficientes")
        return 1
    rho = sum(fora) / len(fora)
    n_eff = N / (1 + (N - 1) * rho) if (1 + (N - 1) * rho) > 0 else float("inf")

    def bloco(sel, rot):
        vs = [M[(x, y)] for i, x in enumerate(sel) for y in sel[i + 1:]
              if M.get((x, y)) is not None]
        if not vs:
            return
        rb = sum(vs) / len(vs)
        k = len(sel)
        ne = k / (1 + (k - 1) * rb) if (1 + (k - 1) * rb) > 0 else float("inf")
        print("  %-18s rho=%+.3f   N_ef = %.2f de %d" % (rot, rb, ne, k))

    print("\n=== N EFETIVO ===")
    print("  instrumentos (N) ............ %d" % N)
    print("  correlacao media (fora diag)  %.4f" % rho)
    print("  N_efetivo ................... %.2f" % n_eff)
    print("\n  por bloco:")
    fx = [n for n in nomes if n in mc.DIV]
    cr = [n for n in nomes if n not in mc.DIV]
    bloco(fx, "dentro do forex")
    bloco(cr, "dentro da cripto")

    anos = len(comuns) / 252.0
    print("\n=== O DEFICIT DE AMOSTRA (contra MinBTL = 4,4 anos) ===")
    print("  janela comum ................ %.2f anos (%d dias)"
          % (anos, len(comuns)))
    print("  anos-equivalentes = janela x N_ef ... %.2f" % (anos * n_eff))
    falta = 4.4 - anos * n_eff
    if falta > 0:
        print("  ⛔ AINDA FALTAM %.2f anos-equivalentes" % falta)
    else:
        print("  ⭐ o deficit FECHOU (sobra %.2f)" % (-falta))
    print("\n⚠️ o N_ef assume equicorrelacao, que a propria medicao mostra ser")
    print("   falsa. E a ortogonalidade do USDJPY e' em parte artefato de")
    print("   convencao de cotacao. Ressalvas da Medicao 2, ainda de pe'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
