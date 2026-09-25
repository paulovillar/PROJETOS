"""CLI: gera superfícies 3D do IBOV e das 20 maiores ações.

Exemplos::

    python -m volsurface --source synthetic
    python -m volsurface --source cotahist --date 2026-09-24 --rate 0.1425
    python -m volsurface --source cotahist --file data/COTAHIST_D24092026.ZIP --tickers IBOV PETR4 VALE3
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data_sources import CotahistSource, DataSource, SyntheticSource, parse_date
from .plotting import asset_figure, write_dashboard
from .surface import CleaningConfig, build_surface, summary_metrics
from .universe import get_assets

log = logging.getLogger("volsurface")


def _previous_business_day(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["cotahist", "synthetic"], default="cotahist")
    p.add_argument("--date", type=parse_date, default=None, help="Data-base AAAA-MM-DD (padrão: D-1 útil).")
    p.add_argument("--file", type=Path, default=None, help="COTAHIST local (.ZIP ou .TXT); senão baixa da B3.")
    p.add_argument("--rate", type=float, default=0.1425, help="Taxa pré anual base 252 (proxy DI). Padrão 14,25%%.")
    p.add_argument("--tickers", nargs="*", default=None, help="Subconjunto do universo (padrão: IBOV + top 20).")
    p.add_argument("--min-trades", type=int, default=0, help="Descarta opções com menos negócios no dia.")
    p.add_argument("--plotlyjs", choices=["cdn", "inline"], default="cdn", help="inline = HTML autocontido (offline, ~4 MB/arquivo).")
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    plotlyjs: str | bool = True if args.plotlyjs == "inline" else "cdn"
    ref_date = args.date or _previous_business_day(date.today())
    assets = get_assets(args.tickers)

    source: DataSource = SyntheticSource() if args.source == "synthetic" else CotahistSource(args.file)
    snapshot = source.load(ref_date, assets, args.rate)
    if args.min_trades:
        snapshot.quotes = snapshot.quotes[snapshot.quotes["trades"] >= args.min_trades]
    log.info("Data-base %s · %d cotações de opções · fonte %s", snapshot.ref_date, len(snapshot.quotes), snapshot.meta.get("source"))

    (args.out / "surfaces").mkdir(parents=True, exist_ok=True)
    surfaces, rows = [], []
    cfg = CleaningConfig()
    for asset in assets:
        try:
            surf = build_surface(snapshot, asset.ticker, cfg)
        except ValueError as exc:
            log.warning("%s", exc)
            continue
        surfaces.append(surf)
        rows.append(summary_metrics(surf))
        asset_figure(surf, snapshot.synthetic).write_html(args.out / "surfaces" / f"{asset.ticker}.html", include_plotlyjs=plotlyjs)
        grid = pd.DataFrame(surf.vol_grid, index=surf.du_grid.round(1), columns=(100 * np.exp(surf.k_grid)).round(2))
        grid.index.name = "DU \\ K/F(%)"
        grid.to_csv(args.out / "surfaces" / f"{asset.ticker}_grid.csv")
        log.info("%-7s %2d vencimentos · ATM 3M %.2f%%", asset.ticker, len(surf.fitted), 100 * rows[-1]["atm_3m"])

    if not surfaces:
        log.error("Nenhuma superfície construída.")
        return 1
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "summary.csv", index=False)
    path = write_dashboard(surfaces, summary, args.out / "index.html", snapshot.synthetic, snapshot.meta.get("source", ""), plotlyjs)
    log.info("Dashboard: %s", path)
    return 0
