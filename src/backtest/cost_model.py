"""
Modelagem de Custos Reais (B3) — TradeSystem5000.

Este módulo implementa modelos de custos operacionais específicos para o
mercado brasileiro (B3) e estimativas de slippage.

Componentes:
- **BrazilianCostModel**: Corretagem, emolumentos B3, taxa de liquidação e ISS.
- **SlippageModel**: Estimativa de impacto de mercado baseada no spread e volume.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 15.
Tabela de Tarifas B3 (Ações e Derivativos).
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from config.settings import cost_config


# ---------------------------------------------------------------------------
# Modelo de Custos B3
# ---------------------------------------------------------------------------
class BrazilianCostModel:
    """
    Modela os custos operacionais do mercado brasileiro (B3).

    Parameters
    ----------
    brokerage : float
        Corretagem por contrato/operação.
    emoluments_pct : float
        Taxa de emolumentos B3 (% sobre volume).
    settlement_pct : float
        Taxa de liquidação (% sobre volume).
    iss_pct : float
        ISS sobre a corretagem.
    """

    def __init__(
        self,
        brokerage: float | None = None,
        emoluments_pct: float | None = None,
        settlement_pct: float | None = None,
        iss_pct: float | None = None,
    ) -> None:
        self.brokerage = brokerage if brokerage is not None else cost_config.brokerage_per_contract
        self.emoluments_pct = emoluments_pct if emoluments_pct is not None else cost_config.emoluments_pct
        self.settlement_pct = settlement_pct if settlement_pct is not None else cost_config.settlement_pct
        self.iss_pct = iss_pct if iss_pct is not None else cost_config.iss_pct

    def trade_cost(
        self,
        price: float,
        quantity: int,
        n_operations: int = 2,
    ) -> float:
        """
        Calcula o custo total de um round-trip (entrada + saída).

        Parameters
        ----------
        price : float
            Preço médio da operação.
        quantity : int
            Quantidade de contratos/ações.
        n_operations : int
            Número de operações (2 = round trip).

        Returns
        -------
        float
            Custo total em unidades monetárias.
        """
        if quantity == 0:
            return 0.0

        volume = price * quantity

        # Corretagem
        brokerage_total = self.brokerage * n_operations

        # Emolumentos (sobre volume)
        emoluments = volume * self.emoluments_pct * n_operations

        # Liquidação (sobre volume)
        settlement = volume * self.settlement_pct * n_operations

        # ISS sobre corretagem
        iss = brokerage_total * self.iss_pct

        total = brokerage_total + emoluments + settlement + iss
        return total

    def cost_series(
        self,
        prices: pd.Series,
        quantities: pd.Series,
    ) -> pd.Series:
        """
        Calcula custos para uma série de trades.

        Parameters
        ----------
        prices : pd.Series
            Preços de entrada.
        quantities : pd.Series
            Quantidades.

        Returns
        -------
        pd.Series
            Custo por trade.
        """
        costs = pd.Series(
            [self.trade_cost(p, q) for p, q in zip(prices, quantities)],
            index=prices.index,
            name="cost",
        )
        logger.debug("Custos totais: {:.2f} | Média: {:.4f}", costs.sum(), costs.mean())
        return costs


# ---------------------------------------------------------------------------
# Modelo de Slippage
# ---------------------------------------------------------------------------
class SlippageModel:
    """
    Estima o slippage (impacto de mercado) baseado no spread e volume.

    Parameters
    ----------
    base_slippage_bps : float
        Slippage base em basis points.
    """

    def __init__(self, base_slippage_bps: float | None = None) -> None:
        self.base_slippage_bps = (
            base_slippage_bps if base_slippage_bps is not None
            else cost_config.slippage_bps
        )

    def estimate(
        self,
        price: float,
        quantity: int,
        avg_volume: float = 1_000_000,
    ) -> float:
        """
        Estima o slippage para uma operação.

        O slippage aumenta linearmente com a participação no volume médio.

        Parameters
        ----------
        price : float
            Preço do ativo.
        quantity : int
            Quantidade negociada.
        avg_volume : float
            Volume médio diário do ativo.

        Returns
        -------
        float
            Slippage estimado em unidades monetárias.
        """
        # Participação no volume (quanto maior, mais slippage)
        participation = (quantity * price) / avg_volume if avg_volume > 0 else 0

        # Slippage em BPS, escalado pela participação
        slippage_bps = self.base_slippage_bps * (1 + participation)
        slippage_monetary = price * quantity * (slippage_bps / 10_000)

        return slippage_monetary

    def slippage_series(
        self,
        prices: pd.Series,
        quantities: pd.Series,
        avg_volume: float = 1_000_000,
    ) -> pd.Series:
        """Calcula slippage para uma série de trades."""
        slippages = pd.Series(
            [self.estimate(p, q, avg_volume) for p, q in zip(prices, quantities)],
            index=prices.index,
            name="slippage",
        )
        return slippages


# ---------------------------------------------------------------------------
# Pipeline de custos completo
# ---------------------------------------------------------------------------
def total_cost(
    prices: pd.Series,
    quantities: pd.Series,
    avg_volume: float = 1_000_000,
) -> pd.DataFrame:
    """
    Calcula custo total (corretagem + taxas + slippage) para uma série de trades.

    Returns
    -------
    pd.DataFrame
        Colunas: ``transaction_cost``, ``slippage``, ``total_cost``.
    """
    cost_model = BrazilianCostModel()
    slip_model = SlippageModel()

    tc = cost_model.cost_series(prices, quantities)
    sl = slip_model.slippage_series(prices, quantities, avg_volume)

    result = pd.DataFrame({"transaction_cost": tc, "slippage": sl})
    result["total_cost"] = result.sum(axis=1)

    logger.info(
        "Custos totais: corretagem={:.2f}, slippage={:.2f}, total={:.2f}",
        tc.sum(),
        sl.sum(),
        result["total_cost"].sum(),
    )
    return result
