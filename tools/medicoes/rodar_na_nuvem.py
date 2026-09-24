#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Despachante das cargas pesadas no runner -- e o CRONOMETRO delas.

⭐ POR QUE ISTO EXISTE, e nao e' so' um atalho de linha de comando: a duvida
que trouxe o projeto para a nuvem nao era "o CPCV funciona?" -- era **"ele
cabe?"**. A decisao de metodo de validacao foi adiada por custo de maquina, e
custo se MEDE. Este script roda a carga e imprime quanto ela levou, para que a
resposta deixe de ser palpite.

⛔ RODA SOBRE DADO SINTETICO, e isso e' deliberado. Baixar tick de forex no
   runner projeta **178 horas** (medido no proprio runner em 24/09/2026:
   24,34 s por arquivo, 8 falhas em 10). O dado real sobe pronto, por outro
   caminho. Aqui o que se exercita e' a ESTEIRA e o CUSTO, que nao dependem de
   o numero ser real.

⛔⛔ E POR ISSO NENHUM NUMERO DAQUI E' RESULTADO DE ESTRATEGIA. Sao
   caracteristicas do particionamento (quantos caminhos, quantos folds) e
   tempo de parede. Ler qualquer coisa daqui como desempenho seria ler ruido
   sintetico como sinal.

USO
    python rodar_na_nuvem.py --alvo cpcv
    python rodar_na_nuvem.py --alvo cpcv --n 5000 --grupos 8 --teste 2
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

for _f in (sys.stdout, sys.stderr):
    try: _f.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass


def carga_cpcv(n, grupos, k_teste, horizonte, embargo):
    from src.purged_cpcv import cpcv_splits, n_paths

    esperado = n_paths(grupos, k_teste)
    print("CPCV -- particionamento purgado e combinatorio")
    print("  observacoes ... %d" % n)
    print("  grupos ........ %d" % grupos)
    print("  grupos/teste .. %d" % k_teste)
    print("  horizonte ..... %d   embargo ... %d" % (horizonte, embargo))
    print("  caminhos previstos (formula) ... %d" % esperado)
    sys.stdout.flush()

    t0 = time.time()
    splits = list(cpcv_splits(n=n, n_groups=grupos, k_test=k_teste,
                              horizon=horizonte, embargo=embargo))
    dt = time.time() - t0

    treinos = [len(s.train) for s in splits]
    testes = [len(s.test) for s in splits]
    print("\n  combinacoes geradas ............ %d" % len(splits))
    print("  tempo .......................... %.3f s" % dt)
    if treinos:
        print("  treino: min %d | mediana %d | max %d"
              % (min(treinos), sorted(treinos)[len(treinos) // 2], max(treinos)))
        print("  teste : min %d | mediana %d | max %d"
              % (min(testes), sorted(testes)[len(testes) // 2], max(testes)))

    # ⭐ PROVA POR VALOR, e no runner: treino e teste NAO podem se tocar.
    # Se a purga falhar, o vazamento e' silencioso e o resultado parece bom.
    vazou = 0
    for s in splits:
        if set(s.train) & set(s.test):
            vazou += 1
    print("\n  splits com sobreposicao treino/teste: %d" % vazou)
    if vazou:
        print("  ⛔⛔ VAZAMENTO: a purga nao esta' isolando. Nao use este resultado.")
        return 1
    print("  ⭐ zero sobreposicao -- a purga isolou em todos os %d splits."
          % len(splits))

    # a formula tem de bater com o que foi de fato gerado
    if len(splits) != esperado:
        print("  ⛔ contagem diverge da formula (%d != %d)" % (len(splits), esperado))
        return 1
    print("  ⭐ contagem confere com n_paths(): %d" % esperado)

    print("\n  CUSTO EXTRAPOLADO (o que trouxe isto para a nuvem):")
    for mult in (2, 4, 10):
        print("    %5dx mais observacoes -> ~%.1f s (se linear)" % (mult, dt * mult))
    print("  ⚠️ 'se linear' e' hipotese: o particionamento e' combinatorio no")
    print("     numero de GRUPOS, nao no de observacoes. Dobrar grupos custa")
    print("     muito mais que dobrar linhas.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alvo", default="cpcv")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--grupos", type=int, default=6)
    ap.add_argument("--teste", type=int, default=2)
    ap.add_argument("--horizonte", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=5)
    a = ap.parse_args()

    print("=" * 70)
    print("CARGA NA NUVEM -- dado SINTETICO, so' esteira e custo")
    print("⛔ nenhum numero daqui e' desempenho de estrategia")
    print("=" * 70)

    if a.alvo == "cpcv":
        return carga_cpcv(a.n, a.grupos, a.teste, a.horizonte, a.embargo)

    print("⛔ alvo desconhecido: %r" % a.alvo)
    print("   'sadf', 'cusum' e 'mcpm' NAO EXISTEM no codigo -- zero")
    print("   ocorrencias. O bloqueio deles nunca foi computacional:")
    print("   ninguem os escreveu ainda. A nuvem nao os destrava.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
