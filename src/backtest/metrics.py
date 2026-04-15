"""Métricas de Performance para Backtesting — TradeSystem5000.

Este módulo implementa métricas estatísticas e de risco para avaliação
rigorosa de estratégias de trading, seguindo os padrões AFML.

Métricas incluídas:
- **Sharpe Ratio**: Anualizado e ajustado.
- **Deflated Sharpe Ratio (DSR)**: Ajuste para múltiplos testes.
- **Maximum Drawdown (MDD)**: Maior queda do pico ao vale.
- **Calmar Ratio**: Retorno ajustado pelo drawdown máximo.
- **Probabilidade de Ruína**: Risco de atingir um nível crítico de perda.

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 14.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Sharpe Ratio
# ---------------------------------------------------------------------------
def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calcula o Sharpe Ratio anualizado.

    Parameters
    ----------
    returns : pd.Series ou np.ndarray
        Série de retornos (simples ou log).
    risk_free : float
        Taxa livre de risco por período.
    periods_per_year : int
        Períodos por ano para anualização (252 para diário, ~12 para mensal).

    Returns
    -------
    float
        Sharpe Ratio anualizado.

    """
    excess = np.asarray(returns) - risk_free
    std = np.std(excess, ddof=1)
    if len(excess) < 2 or std < 1e-10:
        return 0.0

    sr = np.mean(excess) / std
    return float(sr * np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Sharpe Deflacionado
# ---------------------------------------------------------------------------
def deflated_sharpe(
    observed_sr: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calcula o Sharpe Deflacionado (DSR).

    Ajusta o Sharpe Ratio observado pela quantidade de testes (trials)
    realizados, controlando o viés de seleção (data snooping).

    Parameters
    ----------
    observed_sr : float
        Sharpe Ratio observado (anualizado).
    n_trials : int
        Número total de estratégias/modelos testados.
    n_obs : int
        Número de observações (retornos) usado no cálculo do SR.
    skew : float
        Assimetria dos retornos.
    kurtosis : float
        Curtose dos retornos (3.0 = normal).
    risk_free : float
        Taxa livre de risco por período.
    periods_per_year : int
        Períodos por ano.

    Returns
    -------
    float
        p-value do DSR. Valores < 0.05 significam que o SR observado
        é estatisticamente significativo mesmo após ajuste por múltiplos testes.

    """
    if n_trials < 1 or n_obs < 2:
        return 1.0

    # SR esperado sob H0 (melhor SR de n_trials tentativas aleatórias)
    # Usando a aproximação de Euler-Mascheroni para o máximo esperado
    euler_mascheroni = 0.5772156649
    sr_expected = (
        np.sqrt(2 * np.log(n_trials))
        - ((np.log(np.pi) + euler_mascheroni) / (2 * np.sqrt(2 * np.log(n_trials))))
        if n_trials > 1
        else 0.0
    )

    # Desvio padrão do SR estimado (ajustado por skew e kurtosis)
    sr_std = np.sqrt((1 - skew * observed_sr + (kurtosis - 1) / 4 * observed_sr**2) / (n_obs - 1))

    if sr_std == 0:
        return 1.0

    # Estatística Z
    z = (observed_sr - sr_expected) / sr_std

    # p-value (bicaudal à direita)
    p_value = 1 - scipy_stats.norm.cdf(z)

    logger.debug(
        "DSR: SR_obs={:.3f}, SR_exp={:.3f}, z={:.3f}, p={:.4f}",
        observed_sr,
        sr_expected,
        z,
        p_value,
    )

    return float(p_value)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------
def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """Calcula o Maximum Drawdown (MDD).

    Parameters
    ----------
    returns : pd.Series ou np.ndarray
        Série de retornos.

    Returns
    -------
    float
        Maximum Drawdown (valor negativo, ex: -0.15 = 15% de queda).

    """
    cumulative = (1 + pd.Series(returns)).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return float(drawdown.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Retorna a série completa de drawdown."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    dd.name = "drawdown"
    return dd


# ---------------------------------------------------------------------------
# Calmar Ratio
# ---------------------------------------------------------------------------
def calmar_ratio(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """Calcula o Calmar Ratio (retorno anualizado / |MDD|).

    Parameters
    ----------
    returns : pd.Series ou np.ndarray
        Série de retornos.
    periods_per_year : int
        Períodos por ano.

    Returns
    -------
    float
        Calmar Ratio. Quanto maior, melhor o retorno por unidade de drawdown.

    """
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0

    annualized_return = np.mean(returns) * periods_per_year
    return float(annualized_return / abs(mdd))


# ---------------------------------------------------------------------------
# Probabilidade de Ruína
# ---------------------------------------------------------------------------
def probability_of_ruin(
    returns: pd.Series | np.ndarray,
    threshold: float = -0.50,
) -> float:
    """Estima a probabilidade de ruína (drawdown atingir threshold).

    Usa simulação de Monte Carlo simples sobre os retornos observados.

    Parameters
    ----------
    returns : pd.Series ou np.ndarray
        Série de retornos históricos.
    threshold : float
        Nível de drawdown considerado "ruína" (ex: -0.50 = 50%).

    Returns
    -------
    float
        Probabilidade estimada de ruína [0, 1].

    """
    returns = np.asarray(returns)
    n_simulations = 1000
    n_periods = len(returns)
    ruin_count = 0

    rng = np.random.default_rng(seed=42)

    for _ in range(n_simulations):
        sampled = rng.choice(returns, size=n_periods, replace=True)
        cumulative = np.cumprod(1 + sampled)
        peak = np.maximum.accumulate(cumulative)
        dd = (cumulative - peak) / peak

        if dd.min() <= threshold:
            ruin_count += 1

    prob = ruin_count / n_simulations
    logger.debug("P(Ruína < {:.0%}): {:.2%}", threshold, prob)
    return prob


# ---------------------------------------------------------------------------
# Relatório completo
# ---------------------------------------------------------------------------
def performance_report(
    returns: pd.Series,
    n_trials: int = 1,
    risk_free: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    """Gera um relatório completo de performance.

    Parameters
    ----------
    returns : pd.Series
        Série de retornos.
    n_trials : int
        Número de estratégias testadas (para DSR).
    risk_free : float
        Taxa livre de risco.
    periods_per_year : int
        Períodos por ano.

    Returns
    -------
    dict
        Dicionário com todas as métricas.

    """
    sr = sharpe_ratio(returns, risk_free, periods_per_year)
    mdd = max_drawdown(returns)

    report = {
        "n_obs": len(returns),
        "annualized_return": float(np.mean(returns) * periods_per_year),
        "annualized_vol": float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)),
        "sharpe_ratio": sr,
        "max_drawdown": mdd,
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "skewness": float(scipy_stats.skew(returns)),
        "kurtosis": float(scipy_stats.kurtosis(returns, fisher=False)),
    }

    # Sharpe Deflacionado
    if n_trials > 1:
        dsr_pvalue = deflated_sharpe(
            observed_sr=sr,
            n_trials=n_trials,
            n_obs=len(returns),
            skew=report["skewness"],
            kurtosis=report["kurtosis"],
        )
        report["deflated_sharpe_pvalue"] = dsr_pvalue
        report["dsr_significant"] = dsr_pvalue < 0.05

    # Probabilidade de ruína
    report["prob_ruin_50pct"] = probability_of_ruin(returns, -0.50)

    logger.info(
        "Performance: SR={:.2f} | MDD={:.2%} | Calmar={:.2f}",
        report["sharpe_ratio"],
        report["max_drawdown"],
        report["calmar_ratio"],
    )

    return report
