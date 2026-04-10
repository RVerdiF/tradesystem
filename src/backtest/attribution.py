"""
Análise de Atribuição (Alpha vs. Meta-Labeling) — TradeSystem5000.

Este módulo decompõe a performance total do sistema em contribuições de cada
componente da arquitetura AFML:
- **Alpha Model**: Contribuição do sinal direcional primário (aposta original).
- **Meta-Labeling**: Valor agregado pela filtragem/sizing do modelo secundário.
- **Custos**: Impacto dos custos operacionais e slippage no retorno líquido.

A atribuição permite identificar se o meta-modelo está efetivamente mitigando
falsos positivos do alpha e melhorando o Sharpe Ratio final (Sharpe Lift).

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulos 3 e 15.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.metrics import sharpe_ratio


# ---------------------------------------------------------------------------
# Decomposição Alpha vs. Meta-Label
# ---------------------------------------------------------------------------
def attribution_analysis(
    returns_full: pd.Series,
    returns_alpha_only: pd.Series,
    returns_before_costs: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict:
    """
    Decompõe a performance em contribuições de cada componente.

    Parameters
    ----------
    returns_full : pd.Series
        Retornos finais do sistema completo (Alpha + Meta-Label + Custos).
    returns_alpha_only : pd.Series
        Retornos se apenas o Alpha fosse usado (sem filtragem nem sizing).
    returns_before_costs : pd.Series, optional
        Retornos com filtragem mas antes de descontar custos.
    periods_per_year : int
        Períodos por ano.

    Returns
    -------
    dict
        Dicionário com métricas de atribuição.
    """
    sr_full = sharpe_ratio(returns_full, periods_per_year=periods_per_year)
    sr_alpha = sharpe_ratio(returns_alpha_only, periods_per_year=periods_per_year)

    result = {
        "sharpe_full_system": sr_full,
        "sharpe_alpha_only": sr_alpha,
        "sharpe_lift_meta": sr_full - sr_alpha,
        "annualized_return_full": float(np.mean(returns_full) * periods_per_year),
        "annualized_return_alpha": float(np.mean(returns_alpha_only) * periods_per_year),
        "return_lift_meta": float(
            (np.mean(returns_full) - np.mean(returns_alpha_only)) * periods_per_year
        ),
        "vol_full": float(np.std(returns_full, ddof=1) * np.sqrt(periods_per_year)),
        "vol_alpha": float(np.std(returns_alpha_only, ddof=1) * np.sqrt(periods_per_year)),
    }

    if returns_before_costs is not None:
        sr_before_costs = sharpe_ratio(returns_before_costs, periods_per_year=periods_per_year)
        result["sharpe_before_costs"] = sr_before_costs
        result["cost_drag"] = sr_before_costs - sr_full
        result["annualized_cost_impact"] = float(
            (np.mean(returns_before_costs) - np.mean(returns_full)) * periods_per_year
        )

    logger.info(
        "Atribuição: SR(full)={:.2f}, SR(alpha)={:.2f}, Lift={:.2f}",
        result["sharpe_full_system"],
        result["sharpe_alpha_only"],
        result["sharpe_lift_meta"],
    )

    return result


# ---------------------------------------------------------------------------
# Relatório de atribuição por trade
# ---------------------------------------------------------------------------
def trade_level_attribution(
    trades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Análise de atribuição no nível de cada trade individual.

    Parameters
    ----------
    trades : pd.DataFrame
        DataFrame com colunas:
        - ``ret``: retorno do trade
        - ``side``: direção do Alpha (+1/-1)
        - ``meta_label``: predição do meta-model (0/1)
        - ``bet_size``: tamanho da posição (Kelly)
        - ``cost``: custo da operação (opcional)

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas adicionais de atribuição por trade.
    """
    result = trades.copy()

    # Retorno bruto do Alpha (sem sizing)
    if "side" in result.columns and "ret" in result.columns:
        result["alpha_contribution"] = result["ret"] * result["side"]

    # Contribuição do sizing
    if "bet_size" in result.columns:
        result["sized_return"] = result.get("alpha_contribution", result["ret"]) * result["bet_size"]
    else:
        result["sized_return"] = result.get("alpha_contribution", result["ret"])

    # Impacto do filtro
    if "meta_label" in result.columns:
        result["filtered"] = result["meta_label"] == 0
        result["filter_impact"] = np.where(
            result["filtered"],
            -result.get("alpha_contribution", result["ret"]),  # trade evitado
            0,
        )

    # Custo
    if "cost" in result.columns:
        # Aplicar custo apenas se houver execução (meta_label != 0 ou bet_size > 0)
        mask = pd.Series(True, index=result.index)
        if "meta_label" in result.columns:
            mask = mask & (result["meta_label"] != 0)
        if "bet_size" in result.columns:
            mask = mask & (result["bet_size"] > 0)

        result["net_return"] = np.where(
            mask,
            result["sized_return"] - result["cost"],
            result["sized_return"]
        )
    else:
        result["net_return"] = result["sized_return"]

    logger.info(
        "Trade attribution: {} trades | Média bruta={:.4f} | Média líquida={:.4f}",
        len(result),
        result.get("alpha_contribution", result["ret"]).mean(),
        result["net_return"].mean(),
    )

    return result


# ---------------------------------------------------------------------------
# Sumário de atribuição
# ---------------------------------------------------------------------------
def attribution_summary(
    trade_attribution: pd.DataFrame,
) -> dict:
    """
    Resume a atribuição a nível agregado.

    Parameters
    ----------
    trade_attribution : pd.DataFrame
        Output de ``trade_level_attribution``.

    Returns
    -------
    dict
        Métricas agregadas.
    """
    summary = {
        "total_trades": len(trade_attribution),
    }

    if "alpha_contribution" in trade_attribution.columns:
        alpha = trade_attribution["alpha_contribution"]
        summary["alpha_total_return"] = float(alpha.sum())
        summary["alpha_win_rate"] = float((alpha > 0).mean())
        summary["alpha_avg_win"] = float(alpha[alpha > 0].mean()) if (alpha > 0).any() else 0.0
        summary["alpha_avg_loss"] = float(alpha[alpha < 0].mean()) if (alpha < 0).any() else 0.0

    if "net_return" in trade_attribution.columns:
        net = trade_attribution["net_return"]
        summary["net_total_return"] = float(net.sum())
        summary["net_win_rate"] = float((net > 0).mean())
        summary["profit_factor"] = (
            float(net[net > 0].sum() / abs(net[net < 0].sum()))
            if (net < 0).any() and net[net > 0].sum() > 0
            else 0.0
        )

    if "filtered" in trade_attribution.columns:
        summary["trades_filtered_out"] = int(trade_attribution["filtered"].sum())
        summary["filter_rate"] = float(trade_attribution["filtered"].mean())

    logger.info("Attribution summary: {}", summary)
    return summary
