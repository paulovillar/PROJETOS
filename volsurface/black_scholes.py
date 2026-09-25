"""Black-76 sobre o forward e inversão de volatilidade implícita.

Trabalhar no forward ``F`` (em vez de spot + dividend yield) torna a precificação
agnóstica a dividendos discretos — relevante na B3, onde o strike das opções de
ações é ajustado por proventos e o forward é melhor inferido por paridade.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import norm

VOL_MIN, VOL_MAX = 1e-4, 5.0


def black_price(F: ArrayLike, K: ArrayLike, T: ArrayLike, vol: ArrayLike, df: ArrayLike, is_call: ArrayLike) -> NDArray:
    """Preço Black-76 descontado.

    Args:
        F: forward do ativo-objeto para o vencimento.
        K: strike.
        T: prazo em anos (DU/252).
        vol: volatilidade anualizada.
        df: fator de desconto até o vencimento.
        is_call: ``True`` para call, ``False`` para put.
    """
    F, K, T, vol, df = map(np.asarray, (F, K, T, vol, df))
    sig_t = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sig_t**2) / sig_t
    d2 = d1 - sig_t
    call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


def black_vega(F: float, K: float, T: float, vol: float, df: float) -> float:
    """Vega (dPreço/dVol) Black-76."""
    sig_t = vol * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sig_t**2) / sig_t
    return float(df * F * norm.pdf(d1) * np.sqrt(T))


def implied_vol(price: float, F: float, K: float, T: float, df: float, is_call: bool) -> float:
    """Volatilidade implícita via Brent. Retorna ``nan`` se o preço violar os limites de não-arbitragem."""
    if not (price > 0 and F > 0 and K > 0 and T > 0):
        return float("nan")
    intrinsic = df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    upper = df * F if is_call else df * K
    if not (intrinsic < price < upper):
        return float("nan")

    def f(v: float) -> float:
        return float(black_price(F, K, T, v, df, is_call)) - price

    try:
        return brentq(f, VOL_MIN, VOL_MAX, xtol=1e-8, maxiter=200)
    except ValueError:
        return float("nan")
