# Superfícies 3D de volatilidade — Ibovespa + 20 maiores pesos

Pipeline em Python que constrói superfícies de volatilidade implícita para o **IBOV** e
as **20 ações de maior peso** da carteira teórica, a partir do arquivo diário oficial de
cotações da B3 (COTAHIST), e gera gráficos 3D interativos (Plotly).

```
PETR4 · VALE3 · ITUB4 · PETR3 · BBDC4 · SBSP3 · ELET3 · B3SA3 · ITSA4 · BPAC11
WEGE3 · EMBR3 · BBAS3 · ABEV3 · EQTL3 · SUZB3 · RDOR3 · PRIO3 · RENT3 · UGPA3  (+ IBOV)
```
> A carteira teórica é rebalanceada em jan/mai/set — revise `TOP20` em `volsurface/universe.py`.

## Uso

```bash
pip install -r requirements.txt

# Dados reais (baixa COTAHIST_DddmmYYYY.ZIP da B3 para ./data)
python -m volsurface --source cotahist --date 2026-09-24 --rate 0.1425

# Arquivo já baixado, subconjunto de ativos, filtro de liquidez
python -m volsurface --source cotahist --file data/COTAHIST_D24092026.ZIP --tickers IBOV PETR4 VALE3 --min-trades 3

# Demonstração sem internet (cadeias SINTÉTICAS, marcadas como tal nos gráficos)
python -m volsurface --source synthetic --plotlyjs inline
```

Saída em `output/`:
- `index.html` — dashboard: superfície 3D com seletor de ativo + tabela (ATM 1M/3M/6M, skew 90–110 3M, inclinação 3M−1M, RMSE do fit);
- `surfaces/<TICKER>.html` — superfície 3D + smiles por vencimento (SVI vs. pontos de mercado);
- `surfaces/<TICKER>_grid.csv` — grade de vol (DU × K/F);
- `summary.csv`.

## Metodologia

| Etapa | Escolha | Motivo |
|---|---|---|
| Prazo | DU/252, calendário B3 (feriados nacionais + Páscoa móvel) | convenção local |
| Desconto | `(1 + r)^(-DU/252)`, `r` = taxa pré (proxy DI) | idem |
| Preço | mid de melhor oferta compra/venda no fechamento; fallback último negócio | último negócio pode ser stale |
| Forward | paridade put-call `F = K + (C − P)/DF` (mediana dos 5 strikes mais ATM); fallback `S·(1+r)^T` | incorpora dividendos esperados e ajuste de strike por proventos sem precisar modelá-los |
| IV | Black-76 no forward, Brent | |
| Pontos usados | só OTM (puts K<F, calls K≥F), K/F ∈ [0.55, 1.60], ≥ 7 DU | liquidez; evita prêmio de exercício antecipado das calls americanas ITM |
| Smile | SVI raw por vencimento, erro em vol ponderado por vega, multi-start, `w_min ≥ 0` | parametrização padrão de mesa, suave e extrapolável |
| Term structure | interpolação linear em variância total a *k* fixo; `w` forçado não-decrescente em T | ausência de arbitragem de calendário (condição necessária) |

### Limitações conhecidas
- **Opções de IBOV**: identificadas pela raiz `IBOV` no COTAHIST; se não houver liquidez na data, use `--tickers BOVA11` como proxy. O spot do índice não está no COTAHIST — o forward vem só da paridade.
- ESPECI (ON/PN/UNT) desambigua ativos que compartilham raiz (PETR3 × PETR4). Opções de séries com ESPECI atípico podem precisar de ajuste em `universe.py`.
- Taxa flat (não usa a curva DI por vértice). Para prazos longos, trocar por interpolação da curva pré (DI1/BMF).
- Nenhuma checagem de arbitragem de borboleta (densidade ≥ 0) além da forma SVI; para SSVI/e-SVI com garantias, ver Gatheral & Jacquier (2014).
- Liquidez nas asas de ações de menor peso é baixa; use `--min-trades` e confira o RMSE por fatia.

## Testes

```bash
python -m pytest -q
```
