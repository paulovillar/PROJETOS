from datetime import date

import numpy as np
import pandas as pd
import pytest

from volsurface.black_scholes import black_price, implied_vol
from volsurface.calendar import business_days, easter, holidays
from volsurface.data_sources import SyntheticSource, parse_cotahist
from volsurface.surface import SviParams, build_surface, fit_svi, implied_forward, summary_metrics
from volsurface.universe import get_assets


def test_easter_and_holidays() -> None:
    assert easter(2026) == date(2026, 4, 5)
    h = holidays(2026)
    assert date(2026, 2, 16) in h and date(2026, 2, 17) in h  # Carnaval
    assert date(2026, 4, 3) in h  # Sexta-feira Santa
    assert date(2026, 6, 4) in h  # Corpus Christi
    assert date(2026, 11, 20) in h


def test_business_days_skip_weekend_and_holiday() -> None:
    # 2026-11-19 (qui) -> 2026-11-23 (seg): 20/nov é feriado ⇒ apenas 1 DU
    assert business_days(date(2026, 11, 19), date(2026, 11, 23)) == 1


@pytest.mark.parametrize("is_call", [True, False])
@pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
def test_implied_vol_roundtrip(is_call: bool, K: float) -> None:
    F, T, df, vol = 100.0, 0.5, 0.93, 0.32
    px = float(black_price(F, K, T, vol, df, is_call))
    assert implied_vol(px, F, K, T, df, is_call) == pytest.approx(vol, abs=1e-6)


def test_implied_vol_rejects_below_intrinsic() -> None:
    assert np.isnan(implied_vol(5.0, 100.0, 80.0, 0.5, 0.95, True))


def test_svi_fit_recovers_smile() -> None:
    true = SviParams(a=0.01, b=0.08, rho=-0.5, m=0.02, sigma=0.15)
    T = 0.4
    k = np.linspace(-0.4, 0.3, 25)
    iv = np.sqrt(true.total_variance(k) / T)
    params, rmse = fit_svi(k, iv, T)
    assert rmse < 1e-4
    assert np.allclose(params.total_variance(k), true.total_variance(k), atol=1e-5)


def test_implied_forward_from_parity() -> None:
    F, T, df, vol = 105.0, 0.3, 0.96, 0.25
    K = np.arange(90.0, 121.0, 2.5)
    rows = [
        {"strike": k, "type": t, "price": float(black_price(F, k, T, vol, df, t == "C"))} for k in K for t in ("C", "P")
    ]
    assert implied_forward(pd.DataFrame(rows), df, spot_guess=F * df) == pytest.approx(F, rel=1e-9)


def _cotahist_line(**f: str) -> str:
    buf = [" "] * 245
    layout = {"tipreg": (1, 2), "data": (3, 10), "codneg": (13, 24), "tpmerc": (25, 27), "especi": (40, 49),
              "preult": (109, 121), "preofc": (122, 134), "preofv": (135, 147), "totneg": (148, 152),
              "voltot": (171, 188), "preexe": (189, 201), "datven": (203, 210), "fatcot": (211, 217)}
    for name, val in f.items():
        a, b = layout[name]
        width = b - a + 1
        buf[a - 1 : b] = list(val.rjust(width, "0") if val.isdigit() else val.ljust(width))
    return "".join(buf)


def test_parse_cotahist_option_line() -> None:
    line = _cotahist_line(tipreg="01", data="20260924", codneg="PETRJ320", tpmerc="070", especi="PN",
                          preult="123", preofc="120", preofv="125", totneg="42", voltot="100000",
                          preexe="3200", datven="20261016", fatcot="1")
    df = parse_cotahist(["00HEADER", line, "99TRAILER"])
    assert len(df) == 1
    r = df.iloc[0]
    assert r["codneg"] == "PETRJ320" and r["tpmerc"] == "070" and r["especi"] == "PN"
    assert r["preult"] == pytest.approx(1.23) and r["preexe"] == pytest.approx(32.0)
    assert r["preofc"] == pytest.approx(1.20) and r["preofv"] == pytest.approx(1.25)
    assert r["totneg"] == 42 and r["datven"] == date(2026, 10, 16)


def test_pipeline_on_synthetic_data() -> None:
    assets = get_assets(["IBOV", "PETR4"])
    snap = SyntheticSource(seed=1).load(date(2026, 9, 24), assets, 0.14)
    for a in assets:
        surf = build_surface(snap, a.ticker)
        m = summary_metrics(surf)
        assert len(surf.fitted) >= 8
        assert np.all(np.isfinite(surf.vol_grid))
        assert m["rmse_vol_bp"] < 50
        assert m["skew_90_110_3m"] > 0  # skew de put típico de equity
        # w não-decrescente em T (sem arbitragem de calendário na grade)
        w = surf.vol_grid**2 * (surf.du_grid[:, None] / 252)
        assert np.all(np.diff(w, axis=0) >= -1e-12)
