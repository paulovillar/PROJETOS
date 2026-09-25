"""Construção da superfície: limpeza, forward implícito, IV, fit SVI por vencimento
e interpolação em variância total.

Convenções:
    k = ln(K/F)            log-moneyness forward
    w(k, T) = σ²(k, T)·T   variância total implícita
    T = DU/252

Cada vencimento é ajustado por um SVI "raw" (Gatheral, 2004):
    w(k) = a + b·[ρ(k−m) + sqrt((k−m)² + σ²)]
Entre vencimentos, interpola-se linearmente em w a k fixo, e força-se w não
decrescente em T (condição necessária de ausência de arbitragem de calendário).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .black_scholes import black_vega, implied_vol
from .calendar import business_days
from .data_sources import MarketSnapshot


@dataclass(frozen=True)
class CleaningConfig:
    """Filtros de qualidade dos dados."""

    min_du: int = 7
    min_price: float = 0.02
    max_rel_spread: float = 0.60
    moneyness_range: tuple[float, float] = (0.55, 1.60)
    use_last_if_no_quote: bool = True
    min_points_per_slice: int = 5


@dataclass
class SviParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        return self.a + self.b * (self.rho * (k - self.m) + np.sqrt((k - self.m) ** 2 + self.sigma**2))


@dataclass
class Slice:
    """Um vencimento com IVs de mercado e o SVI ajustado."""

    expiry: date
    du: int
    T: float
    forward: float
    forward_method: str
    points: pd.DataFrame
    svi: SviParams | None = None
    rmse_vol: float = float("nan")


@dataclass
class VolSurface:
    """Superfície de um ativo: fatias + grade interpolada."""

    ticker: str
    ref_date: date
    spot: float
    slices: list[Slice]
    k_grid: np.ndarray = field(default_factory=lambda: np.empty(0))
    du_grid: np.ndarray = field(default_factory=lambda: np.empty(0))
    vol_grid: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    @property
    def fitted(self) -> list[Slice]:
        return [s for s in self.slices if s.svi is not None]

    def vol(self, moneyness: float | np.ndarray, du: float) -> np.ndarray:
        """Vol implícita interpolada para moneyness forward K/F e prazo em DU."""
        k = np.log(np.atleast_1d(np.asarray(moneyness, dtype=float)))
        w = _interp_total_variance(self.fitted, k, np.array([du / 252.0]))[0]
        return np.sqrt(w / (du / 252.0))


# --------------------------------------------------------------------------- preparação


def _option_price(q: pd.DataFrame, cfg: CleaningConfig) -> pd.Series:
    """Mid bid/ask quando o book é válido; senão último negócio (se houve negócio)."""
    valid_book = (q["bid"] > 0) & (q["ask"] >= q["bid"])
    mid = (q["bid"] + q["ask"]) / 2
    rel_spread = (q["ask"] - q["bid"]) / mid.where(mid > 0)
    price = mid.where(valid_book & (rel_spread <= cfg.max_rel_spread))
    if cfg.use_last_if_no_quote:
        price = price.fillna(q["last"].where((q["trades"] > 0) & (q["last"] > 0)))
    return price


def implied_forward(chain: pd.DataFrame, df: float, spot_guess: float) -> float:
    """Forward por paridade put-call: F = K + (C − P)/DF.

    Usa os 5 strikes com menor |C − P| (mais próximos do ATM, onde a paridade é
    mais bem precificada) e toma a mediana — robusto a quotes ruins isolados.
    """
    piv = chain.pivot_table(index="strike", columns="type", values="price", aggfunc="mean").dropna()
    if "C" not in piv or "P" not in piv or piv.empty:
        return float("nan")
    diff = piv["C"] - piv["P"]
    near = diff.abs().nsmallest(5).index
    f = (near.to_numpy() + diff.loc[near].to_numpy() / df)
    fwd = float(np.median(f))
    # Sanidade: forward implausível vs. spot descapitalizado ⇒ descarta.
    if np.isfinite(spot_guess) and not (0.7 < fwd / (spot_guess / df) < 1.3):
        return float("nan")
    return fwd


def build_slices(snapshot: MarketSnapshot, ticker: str, cfg: CleaningConfig = CleaningConfig()) -> list[Slice]:
    """Calcula as IVs OTM por vencimento para um ativo."""
    q = snapshot.quotes[snapshot.quotes["underlying"] == ticker].copy()
    if q.empty:
        return []
    q["price"] = _option_price(q, cfg)
    q = q[q["price"] >= cfg.min_price]
    spot = snapshot.spots.get(ticker, float("nan"))

    slices: list[Slice] = []
    for expiry, chain in q.groupby("expiry"):
        du = business_days(snapshot.ref_date, expiry)
        if du < cfg.min_du:
            continue
        T = du / 252.0
        df = (1 + snapshot.rate) ** (-T)
        fwd, method = implied_forward(chain, df, spot), "paridade"
        if not np.isfinite(fwd):
            if not np.isfinite(spot):
                continue
            fwd, method = spot / df, "spot·(1+r)^T"

        # Apenas OTM: puts abaixo do forward, calls acima (mais líquidas e sem
        # prêmio de exercício antecipado relevante nas calls americanas ITM).
        otm = chain[((chain["type"] == "P") & (chain["strike"] < fwd)) | ((chain["type"] == "C") & (chain["strike"] >= fwd))]
        lo, hi = cfg.moneyness_range
        otm = otm[(otm["strike"] / fwd).between(lo, hi)]
        rows = []
        for r in otm.itertuples(index=False):
            iv = implied_vol(r.price, fwd, r.strike, T, df, r.type == "C")
            if np.isfinite(iv):
                rows.append(
                    {
                        "strike": r.strike,
                        "type": r.type,
                        "price": r.price,
                        "iv": iv,
                        "k": np.log(r.strike / fwd),
                        "vega": black_vega(fwd, r.strike, T, iv, df),
                        "option_ticker": r.option_ticker,
                    }
                )
        pts = pd.DataFrame(rows)
        if len(pts) < cfg.min_points_per_slice:
            continue
        pts = pts.groupby("strike", as_index=False).first().sort_values("k")
        slices.append(Slice(expiry, du, T, fwd, method, pts))
    return sorted(slices, key=lambda s: s.T)


# --------------------------------------------------------------------------- SVI


def fit_svi(k: np.ndarray, iv: np.ndarray, T: float, weights: np.ndarray | None = None) -> tuple[SviParams, float]:
    """Ajusta SVI raw minimizando o erro em vol (ponderado por vega).

    Multi-start para robustez; impõe b ≥ 0, |ρ| < 1, σ > 0 e w_min ≥ 0 via penalidade.

    Returns:
        (parâmetros, RMSE em pontos de vol).
    """
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    w_mkt = iv**2 * T
    wts = np.ones_like(k) if weights is None else np.sqrt(np.asarray(weights, float) / np.max(weights))

    def resid(x: np.ndarray) -> np.ndarray:
        p = SviParams(*x)
        w = p.total_variance(k)
        model_vol = np.sqrt(np.maximum(w, 1e-10) / T)
        w_min = p.a + p.b * p.sigma * np.sqrt(1 - p.rho**2)
        penalty = 10.0 * max(0.0, -w_min) / T
        return np.append(wts * (model_vol - iv), penalty)

    w_atm = float(np.interp(0.0, k, w_mkt))
    lb = [-np.max(w_mkt), 1e-6, -0.999, -1.0, 1e-3]
    ub = [np.max(w_mkt), 5.0, 0.999, 1.0, 2.0]
    best = None
    for rho0 in (-0.6, -0.2, 0.2):
        for sig0 in (0.05, 0.2):
            x0 = np.clip([0.5 * w_atm, 0.1, rho0, 0.0, sig0], np.array(lb) + 1e-9, np.array(ub) - 1e-9)
            res = least_squares(resid, x0, bounds=(lb, ub), method="trf", max_nfev=4000)
            if best is None or res.cost < best.cost:
                best = res
    params = SviParams(*best.x)
    model_vol = np.sqrt(np.maximum(params.total_variance(k), 1e-10) / T)
    rmse = float(np.sqrt(np.mean((model_vol - iv) ** 2)))
    return params, rmse


def _interp_total_variance(slices: list[Slice], k: np.ndarray, T: np.ndarray) -> np.ndarray:
    """w(k, T) linear em T entre fatias; extrapola com vol constante nas pontas; monótono em T."""
    Ts = np.array([s.T for s in slices])
    W = np.vstack([s.svi.total_variance(k) for s in slices])  # (n_slices, n_k)
    W = np.maximum.accumulate(np.maximum(W, 1e-8), axis=0)  # calendário: w não-decrescente
    out = np.empty((T.size, k.size))
    for i, t in enumerate(T):
        if t <= Ts[0]:
            out[i] = W[0] * t / Ts[0]
        elif t >= Ts[-1]:
            out[i] = W[-1] * t / Ts[-1]
        else:
            j = np.searchsorted(Ts, t)
            a = (t - Ts[j - 1]) / (Ts[j] - Ts[j - 1])
            out[i] = (1 - a) * W[j - 1] + a * W[j]
    return out


def build_surface(
    snapshot: MarketSnapshot,
    ticker: str,
    cfg: CleaningConfig = CleaningConfig(),
    n_k: int = 41,
    n_t: int = 40,
    moneyness_bounds: tuple[float, float] | None = None,
) -> VolSurface:
    """Pipeline completo para um ativo. Levanta ``ValueError`` se não houver fatias ajustáveis."""
    slices = build_slices(snapshot, ticker, cfg)
    for s in slices:
        s.svi, s.rmse_vol = fit_svi(s.points["k"].to_numpy(), s.points["iv"].to_numpy(), s.T, s.points["vega"].to_numpy())
    surf = VolSurface(ticker, snapshot.ref_date, snapshot.spots.get(ticker, float("nan")), slices)
    if not surf.fitted:
        raise ValueError(f"{ticker}: nenhum vencimento com pontos suficientes para ajuste.")

    if moneyness_bounds is None:
        all_k = np.concatenate([s.points["k"].to_numpy() for s in surf.fitted])
        k_lo, k_hi = np.quantile(all_k, [0.10, 0.90])
        k_lo, k_hi = max(k_lo, np.log(0.70)), min(k_hi, np.log(1.30))
    else:
        k_lo, k_hi = np.log(moneyness_bounds[0]), np.log(moneyness_bounds[1])
    surf.k_grid = np.linspace(k_lo, k_hi, n_k)
    du_min, du_max = surf.fitted[0].du, surf.fitted[-1].du
    surf.du_grid = np.linspace(du_min, du_max, n_t)
    w = _interp_total_variance(surf.fitted, surf.k_grid, surf.du_grid / 252.0)
    surf.vol_grid = np.sqrt(w / (surf.du_grid[:, None] / 252.0))
    return surf


def summary_metrics(surf: VolSurface) -> dict[str, float | str]:
    """Métricas de mesa: ATM 1M/3M/6M, skew 90–110 3M, inclinação da estrutura a termo."""

    def v(m: float, du: float) -> float:
        return float(surf.vol(m, du)[0])

    atm1, atm3, atm6 = v(1.0, 21), v(1.0, 63), v(1.0, 126)
    return {
        "ticker": surf.ticker,
        "spot": surf.spot,
        "atm_1m": atm1,
        "atm_3m": atm3,
        "atm_6m": atm6,
        "skew_90_110_3m": v(0.9, 63) - v(1.1, 63),
        "term_3m_1m": atm3 - atm1,
        "n_expiries": len(surf.fitted),
        "n_points": int(sum(len(s.points) for s in surf.fitted)),
        "rmse_vol_bp": float(np.mean([s.rmse_vol for s in surf.fitted]) * 1e4),
    }
