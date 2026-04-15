"""Modelagem de Custos Reais (B3) — TradeSystem5000.

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
    """Modela os custos operacionais do mercado brasileiro (B3).

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
    symbol : str
        Ativo negociado. Define a regra de custos (ações vs futuros).

    """

    def __init__(
        self,
        brokerage: float | None = None,
        emoluments_pct: float | None = None,
        settlement_pct: float | None = None,
        iss_pct: float | None = None,
        symbol: str = "WIN",
    ) -> None:
        """Inicializa BaseCostModel."""
        self.brokerage = brokerage if brokerage is not None else cost_config.brokerage_per_contract
        self.emoluments_pct = (
            emoluments_pct if emoluments_pct is not None else cost_config.emoluments_pct
        )
        self.settlement_pct = (
            settlement_pct if settlement_pct is not None else cost_config.settlement_pct
        )
        self.iss_pct = iss_pct if iss_pct is not None else cost_config.iss_pct
        self.symbol = symbol

        self.is_future = any(
            self.symbol.startswith(prefix) for prefix in cost_config.asset_multipliers.keys()
        )
        self.multiplier = 1.0
        for prefix, mult in cost_config.asset_multipliers.items():
            if self.symbol.startswith(prefix):
                self.multiplier = mult
                break

    def trade_cost(
        self,
        price: float,
        quantity: int,
        n_operations: int = 2,
    ) -> float:
        """Calcula o custo total de um round-trip (entrada + saída)."""
        if quantity == 0:
            return 0.0

        if self.is_future:
            brokerage_total = self.brokerage * quantity * n_operations
            emoluments = cost_config.emoluments_fixed * quantity * n_operations
            settlement = 0.0  # B3 não cobra taxa de liquidação por contrato em derivativos
        else:
            brokerage_total = self.brokerage * n_operations
            volume = price * quantity
            emoluments = volume * self.emoluments_pct * n_operations
            settlement = volume * self.settlement_pct * n_operations

        iss = brokerage_total * self.iss_pct

        total = brokerage_total + emoluments + settlement + iss
        return total

    def cost_series(
        self,
        prices: pd.Series,
        quantities: pd.Series,
    ) -> pd.Series:
        """Calcula custos para uma série de trades."""
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
    """Estima o slippage (impacto de mercado) baseado no spread e volume."""

    def __init__(self, base_slippage_bps: float | None = None, symbol: str = "WIN") -> None:
        """Inicializa o SlippageModel."""
        self.symbol = symbol
        self.base_slippage_bps = (
            base_slippage_bps if base_slippage_bps is not None else cost_config.slippage_bps
        )

        self.is_future = any(
            self.symbol.startswith(prefix) for prefix in cost_config.asset_multipliers.keys()
        )
        self.multiplier = 1.0
        self.tick_size = 0.01
        matched = False
        for prefix, mult in cost_config.asset_multipliers.items():
            if self.symbol.startswith(prefix):
                self.multiplier = mult
                self.tick_size = cost_config.tick_sizes.get(prefix, 1.0)
                matched = True
                break
        if not matched and self.is_future:
            logger.warning(
                "SlippageModel: símbolo '{}' detectado como futuro mas sem tick_size configurado. "
                "Usando fallback tick_size=0.01 — verifique asset_multipliers e tick_sizes em CostConfig.",
                self.symbol,
            )

    def estimate(
        self,
        price: float,
        quantity: int,
        avg_volume: float = 1_000_000,
    ) -> float:
        """Estima o slippage para uma operação."""
        if quantity == 0:
            return 0.0

        if self.is_future:
            slippage_ticks = cost_config.slippage_ticks
            # Para futuros, avg_volume é em CONTRATOS (não em BRL).
            # O default de 1_000_000 é conservador — WIN gira ~50k contratos/dia.
            # Passe o volume diário real em contratos para ativar o scale de impacto de mercado.
            participation = quantity / avg_volume if avg_volume > 0 else 0
            adjusted_ticks = slippage_ticks * (1 + participation * 10)

            slippage_points = adjusted_ticks * self.tick_size
            slippage_monetary = slippage_points * self.multiplier * quantity
            return slippage_monetary
        else:
            participation = (quantity * price) / avg_volume if avg_volume > 0 else 0
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
    symbol: str = "WIN",
) -> pd.DataFrame:
    """Calcula custo total (corretagem + taxas + slippage) para uma série de trades."""
    cost_model = BrazilianCostModel(symbol=symbol)
    slip_model = SlippageModel(symbol=symbol)

    tc = cost_model.cost_series(prices, quantities)
    sl = slip_model.slippage_series(prices, quantities, avg_volume)

    result = pd.DataFrame({"transaction_cost": tc, "slippage": sl})
    result["total_cost"] = result.sum(axis=1)

    logger.info(
        "Custos totais ({}): corretagem={:.2f}, slippage={:.2f}, total={:.2f}",
        symbol,
        tc.sum(),
        sl.sum(),
        result["total_cost"].sum(),
    )
    return result
