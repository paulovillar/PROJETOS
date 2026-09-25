"""Universo de ativos: Ibovespa + 20 maiores pesos da carteira teórica.

A carteira teórica do IBOV é rebalanceada quadrimestralmente (jan/mai/set);
revise ``TOP20`` a cada rebalanceamento. Para opções na B3, o ticker da opção
começa com a *raiz* de 4 letras do emissor (PETR, VALE...). Como ON e PN
compartilham a raiz, o campo ``especi`` (ESPECI no COTAHIST: ON, PN, UNT, CI...)
desambigua o ativo-objeto.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Asset:
    """Ativo-objeto de uma superfície de volatilidade.

    Attributes:
        ticker: Código de negociação do ativo-objeto (ex.: ``PETR4``).
        name: Nome curto para exibição.
        option_root: Raiz de 4 letras dos tickers de opção.
        especi: Prefixo da especificação (ESPECI) que identifica o objeto.
        is_index: ``True`` para o índice (spot não negociado no mercado à vista).
    """

    ticker: str
    name: str
    option_root: str
    especi: str
    is_index: bool = False


IBOV = Asset("IBOV", "Ibovespa", "IBOV", "", is_index=True)
# Proxy líquido do índice caso as opções de IBOV estejam ilíquidas na data.
BOVA11 = Asset("BOVA11", "iShares Ibovespa (proxy IBOV)", "BOVA", "CI")

TOP20: tuple[Asset, ...] = (
    Asset("VALE3", "Vale", "VALE", "ON"),
    Asset("ITUB4", "Itaú Unibanco", "ITUB", "PN"),
    Asset("PETR4", "Petrobras PN", "PETR", "PN"),
    Asset("PETR3", "Petrobras ON", "PETR", "ON"),
    Asset("BBDC4", "Bradesco", "BBDC", "PN"),
    Asset("SBSP3", "Sabesp", "SBSP", "ON"),
    Asset("ELET3", "Eletrobras", "ELET", "ON"),
    Asset("B3SA3", "B3", "B3SA", "ON"),
    Asset("ITSA4", "Itaúsa", "ITSA", "PN"),
    Asset("BPAC11", "BTG Pactual", "BPAC", "UNT"),
    Asset("WEGE3", "WEG", "WEGE", "ON"),
    Asset("EMBR3", "Embraer", "EMBR", "ON"),
    Asset("BBAS3", "Banco do Brasil", "BBAS", "ON"),
    Asset("ABEV3", "Ambev", "ABEV", "ON"),
    Asset("EQTL3", "Equatorial", "EQTL", "ON"),
    Asset("SUZB3", "Suzano", "SUZB", "ON"),
    Asset("RDOR3", "Rede D'Or", "RDOR", "ON"),
    Asset("PRIO3", "PRIO", "PRIO", "ON"),
    Asset("RENT3", "Localiza", "RENT", "ON"),
    Asset("UGPA3", "Ultrapar", "UGPA", "ON"),
)

UNIVERSE: tuple[Asset, ...] = (IBOV, *TOP20)
EXTRA: tuple[Asset, ...] = (BOVA11,)


def get_assets(tickers: list[str] | None = None) -> list[Asset]:
    """Retorna os ativos pedidos (ou o universo padrão: IBOV + top 20).

    Raises:
        KeyError: se algum ticker não estiver cadastrado.
    """
    if not tickers:
        return list(UNIVERSE)
    lookup = {a.ticker: a for a in (*UNIVERSE, *EXTRA)}
    missing = [t for t in tickers if t.upper() not in lookup]
    if missing:
        raise KeyError(f"Tickers não cadastrados em universe.py: {missing}")
    return [lookup[t.upper()] for t in tickers]
