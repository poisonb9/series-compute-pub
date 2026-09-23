# series-compute-pub

Motor de medicao estatistica sobre series de preco. **Metodo publicado** --
ADF e diferenciacao fracionaria (Tsay; Lopez de Prado cap. 5), Jarque-Bera,
ACF, CPCV e PBO (Lopez de Prado; Bailey), bootstrap, walk-forward diagnostics.

Nao ha estrategia aqui: nenhum limiar escolhido, nenhuma regra de entrada ou
saida, nenhum score proprietario. A fronteira foi verificada pelo grafo de
imports com fecho transitivo, ate zero violacoes.

    pytest -q            # 183 testes, sem dado externo
    python tools/medicoes/medir_d_minimo.py --autoteste
    python tools/sondar_velocidade_do_feed.py --n 10

## Antes de qualquer corrida que gaste cota

Rode a sonda. Medido em 23/09/2026: um arquivo .bi5 de 25 KB leva ~15 s, o que
projeta 99 HORAS para um ano de 3 pares -- acima do teto de 6 h por job. O
limite e do feed, nao da rede. O dado precisa subir pronto, nao ser baixado.

Documentacao completa da separacao e das armadilhas medidas:
`docs/DICIONARIO_DA_SEPARACAO_PUBLICO_PRIVADO_20260923.md` no repo principal.
