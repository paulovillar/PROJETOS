"""Fontes de dados de opções, normalizadas em um ``MarketSnapshot``.

- ``CotahistSource``: arquivo diário oficial da B3 (COTAHIST_DddmmYYYY.ZIP,
  layout fixo de 245 posições). Fonte gratuita e reprodutível; traz último
  preço e melhor oferta de compra/venda no fechamento.
- ``SyntheticSource``: cadeia sintética gerada a partir de um SVI por ativo.
  Serve para testes e demonstração do pipeline — **não é dado de mercado**.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .black_scholes import black_price
from .calendar import business_days, year_fraction
from .universe import Asset

QUOTE_COLUMNS = ["underlying", "option_ticker", "type", "strike", "expiry", "bid", "ask", "last", "trades", "volume"]

COTAHIST_URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_D{:%d%m%Y}.ZIP"

# (nome, início 1-indexado, fim inclusivo) — layout oficial B3 do COTAHIST.
_COTAHIST_FIELDS: tuple[tuple[str, int, int], ...] = (
    ("tipreg", 1, 2),
    ("data", 3, 10),
    ("codbdi", 11, 12),
    ("codneg", 13, 24),
    ("tpmerc", 25, 27),
    ("nomres", 28, 39),
    ("especi", 40, 49),
    ("preult", 109, 121),
    ("preofc", 122, 134),
    ("preofv", 135, 147),
    ("totneg", 148, 152),
    ("voltot", 171, 188),
    ("preexe", 189, 201),
    ("datven", 203, 210),
    ("fatcot", 211, 217),
)
_PRICE_FIELDS = ("preult", "preofc", "preofv", "voltot", "preexe")
TPMERC_SPOT, TPMERC_CALL, TPMERC_PUT = "010", "070", "080"


@dataclass
class MarketSnapshot:
    """Fotografia de mercado em uma data de referência.

    Attributes:
        ref_date: data de referência (pregão).
        spots: preço à vista por ticker (``nan`` para o índice se indisponível).
        quotes: DataFrame com as colunas ``QUOTE_COLUMNS``.
        rate: taxa pré anual exponencial base 252 (proxy do DI).
        synthetic: ``True`` se os dados não são de mercado.
    """

    ref_date: date
    spots: dict[str, float]
    quotes: pd.DataFrame
    rate: float
    synthetic: bool = False
    meta: dict[str, str] = field(default_factory=dict)


class DataSource(Protocol):
    def load(self, ref_date: date, assets: list[Asset], rate: float) -> MarketSnapshot: ...


# --------------------------------------------------------------------------- COTAHIST


def parse_cotahist(lines: list[str] | io.TextIOBase) -> pd.DataFrame:
    """Faz o parse dos registros tipo 01 do COTAHIST em um DataFrame tipado."""
    rows = []
    for line in lines:
        if not line.startswith("01"):
            continue
        rows.append({name: line[a - 1 : b].strip() for name, a, b in _COTAHIST_FIELDS})
    df = pd.DataFrame(rows, columns=[f[0] for f in _COTAHIST_FIELDS])
    for col in _PRICE_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0
    fatcot = pd.to_numeric(df["fatcot"], errors="coerce").replace(0, 1).fillna(1)
    for col in ("preult", "preofc", "preofv"):
        df[col] = df[col] / fatcot  # preços cotados por lote de FATCOT unidades
    df["totneg"] = pd.to_numeric(df["totneg"], errors="coerce").fillna(0).astype(int)
    df["data"] = pd.to_datetime(df["data"], format="%Y%m%d", errors="coerce").dt.date
    df["datven"] = pd.to_datetime(df["datven"], format="%Y%m%d", errors="coerce").dt.date
    return df


class CotahistSource:
    """Lê o COTAHIST diário da B3 (local ou via download)."""

    def __init__(self, path: str | Path | None = None, cache_dir: str | Path = "data") -> None:
        self.path = Path(path) if path else None
        self.cache_dir = Path(cache_dir)

    def _read_text(self, ref_date: date) -> list[str]:
        path = self.path
        if path is None:
            path = self.cache_dir / f"COTAHIST_D{ref_date:%d%m%Y}.ZIP"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                url = COTAHIST_URL.format(ref_date)
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (URL fixa da B3)
                    path.write_bytes(resp.read())
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                raw = zf.read(zf.namelist()[0])
        else:
            raw = path.read_bytes()
        return raw.decode("latin-1").splitlines()

    def load(self, ref_date: date, assets: list[Asset], rate: float) -> MarketSnapshot:
        df = parse_cotahist(self._read_text(ref_date))
        if df.empty:
            raise ValueError("COTAHIST sem registros de negociação.")
        file_date = df["data"].iloc[0]

        spot_rows = df[df["tpmerc"] == TPMERC_SPOT]
        spots: dict[str, float] = {}
        frames = []
        opts = df[df["tpmerc"].isin([TPMERC_CALL, TPMERC_PUT])]
        for asset in assets:
            row = spot_rows[spot_rows["codneg"] == asset.ticker]
            spots[asset.ticker] = float(row["preult"].iloc[0]) if len(row) else float("nan")
            mask = opts["codneg"].str.startswith(asset.option_root)
            if asset.especi:
                mask &= opts["especi"].str.startswith(asset.especi)
            sub = opts[mask]
            if sub.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "underlying": asset.ticker,
                        "option_ticker": sub["codneg"].values,
                        "type": np.where(sub["tpmerc"] == TPMERC_CALL, "C", "P"),
                        "strike": sub["preexe"].values,
                        "expiry": sub["datven"].values,
                        "bid": sub["preofc"].values,
                        "ask": sub["preofv"].values,
                        "last": sub["preult"].values,
                        "trades": sub["totneg"].values,
                        "volume": sub["voltot"].values,
                    }
                )
            )
        quotes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=QUOTE_COLUMNS)
        return MarketSnapshot(file_date, spots, quotes, rate, meta={"source": "B3 COTAHIST"})


# --------------------------------------------------------------------------- Sintético

# Parâmetros ilustrativos (ATM vol 3M, skew, convexidade). NÃO são dados de mercado.
_SYNTH_PARAMS: dict[str, tuple[float, float, float, float]] = {
    # ticker: (spot, atm_vol, rho, curvatura)
    "IBOV": (140_000.0, 0.17, -0.70, 0.25),
    "BOVA11": (135.0, 0.17, -0.70, 0.25),
    "VALE3": (55.0, 0.28, -0.35, 0.35),
    "ITUB4": (38.0, 0.22, -0.55, 0.30),
    "PETR4": (32.0, 0.30, -0.45, 0.40),
    "PETR3": (35.0, 0.29, -0.45, 0.40),
    "BBDC4": (16.0, 0.27, -0.50, 0.35),
    "SBSP3": (110.0, 0.26, -0.40, 0.30),
    "ELET3": (45.0, 0.30, -0.40, 0.35),
    "B3SA3": (13.0, 0.30, -0.45, 0.35),
    "ITSA4": (11.0, 0.21, -0.55, 0.30),
    "BPAC11": (45.0, 0.30, -0.45, 0.35),
    "WEGE3": (42.0, 0.28, -0.30, 0.30),
    "EMBR3": (75.0, 0.38, -0.20, 0.40),
    "BBAS3": (22.0, 0.30, -0.45, 0.35),
    "ABEV3": (13.0, 0.23, -0.35, 0.30),
    "EQTL3": (35.0, 0.25, -0.40, 0.30),
    "SUZB3": (52.0, 0.30, -0.10, 0.35),
    "RDOR3": (40.0, 0.33, -0.40, 0.35),
    "PRIO3": (42.0, 0.38, -0.30, 0.45),
    "RENT3": (45.0, 0.40, -0.40, 0.40),
    "UGPA3": (20.0, 0.32, -0.40, 0.35),
}


def third_friday_like(year: int, month: int) -> date:
    """Vencimento de opções de ações B3: 3ª sexta-feira do mês."""
    d = date(year, month, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


class SyntheticSource:
    """Gera cadeias sintéticas com smile SVI, term structure e ruído bid/ask.

    Útil para validar o pipeline sem acesso a dados. Todo output gerado a partir
    desta fonte é marcado como sintético.
    """

    def __init__(self, n_expiries: int = 8, seed: int = 7) -> None:
        self.n_expiries = n_expiries
        self.rng = np.random.default_rng(seed)

    def load(self, ref_date: date, assets: list[Asset], rate: float) -> MarketSnapshot:
        expiries: list[date] = []
        y, m = ref_date.year, ref_date.month
        while len(expiries) < self.n_expiries:
            exp = third_friday_like(y, m)
            if business_days(ref_date, exp) >= 5:
                expiries.append(exp)
            m += 1
            if m > 12:
                y, m = y + 1, 1
        # Vencimentos trimestrais longos (liquidez típica em mar/jun/set/dez).
        for add in (12, 18):
            yy, mm = divmod(ref_date.month - 1 + add, 12)
            expiries.append(third_friday_like(ref_date.year + yy, mm + 1))

        spots: dict[str, float] = {}
        frames = []
        for asset in assets:
            spot, atm, rho, curv = _SYNTH_PARAMS.get(asset.ticker, (50.0, 0.30, -0.40, 0.35))
            spots[asset.ticker] = spot
            for exp in expiries:
                T = year_fraction(ref_date, exp)
                df_ = (1 + rate) ** (-T)
                F = spot / df_
                # Term structure levemente positiva no ATM e skew que achata com o prazo.
                atm_T = atm * (0.92 + 0.08 * np.tanh(3 * T))
                width = float(np.clip(2.2 * atm * np.sqrt(T), 0.08, 0.60))
                strikes = F * np.exp(np.linspace(-1.3 * width, width, 25))
                strikes = np.unique(np.round(strikes, -3) if asset.is_index else np.round(strikes, 2))
                k = np.log(strikes / F)
                b = curv * atm_T / np.sqrt(max(T, 0.15)) * 0.35
                sig = 0.25
                w_atm = atm_T**2 * T
                raw = rho * k + np.sqrt(k**2 + sig**2) - sig
                w = w_atm + b * np.sqrt(T) * raw
                vol = np.sqrt(np.maximum(w, 1e-6) / T) * (1 + self.rng.normal(0, 0.006, k.size))
                for is_call in (True, False):
                    px = black_price(F, strikes, T, vol, df_, is_call)
                    spread = np.maximum(5.0 if asset.is_index else 0.01, 0.04 * px)
                    frames.append(
                        pd.DataFrame(
                            {
                                "underlying": asset.ticker,
                                "option_ticker": [f"{asset.option_root}{'C' if is_call else 'P'}{i}" for i in range(k.size)],
                                "type": "C" if is_call else "P",
                                "strike": strikes,
                                "expiry": exp,
                                "bid": np.maximum(px - spread / 2, 0.0),
                                "ask": px + spread / 2,
                                "last": px,
                                "trades": self.rng.integers(1, 200, k.size),
                                "volume": px * 1000,
                            }
                        )
                    )
        quotes = pd.concat(frames, ignore_index=True)
        return MarketSnapshot(ref_date, spots, quotes, rate, synthetic=True, meta={"source": "SINTÉTICO"})


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()
