"""Calendário de dias úteis B3 (base 252) e convenções de taxa.

Feriados nacionais (ANBIMA): fixos + móveis dependentes da Páscoa
(Carnaval seg/ter, Sexta-feira Santa, Corpus Christi). Consciência Negra
(20/nov) é feriado nacional desde 2024.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import numpy as np

_FIXED = ((1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 25))


def easter(year: int) -> date:
    """Domingo de Páscoa (algoritmo de Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


@lru_cache(maxsize=None)
def holidays(year: int) -> tuple[date, ...]:
    """Feriados nacionais que fecham a B3 em ``year``."""
    e = easter(year)
    days = {date(year, m, d) for m, d in _FIXED}
    days |= {e - timedelta(days=48), e - timedelta(days=47), e - timedelta(days=2), e + timedelta(days=60)}
    if year >= 2024:
        days.add(date(year, 11, 20))
    return tuple(sorted(days))


def business_days(start: date, end: date) -> int:
    """Dias úteis em [start, end) — convenção DU da B3."""
    hol = [h for y in range(start.year, end.year + 1) for h in holidays(y)]
    return int(np.busday_count(start, end, holidays=hol))


def year_fraction(start: date, end: date) -> float:
    """Prazo em anos na base DU/252."""
    return business_days(start, end) / 252.0


def continuous_rate(annual_rate_252: float) -> float:
    """Converte taxa exponencial 252 (convenção DI) em taxa contínua."""
    return float(np.log1p(annual_rate_252))
