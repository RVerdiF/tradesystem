"""
5.2 — Deflated Sharpe Ratio (DSR).

O DSR corrige o viés de seleção de múltiplos testes (multi-testing),
calculando a probabilidade de que o Sharpe Ratio observado seja real,
após descontar o número de tentativas e a variância dos resultados.

Referência: López de Prado, *Advances in Financial Machine Learning*, Cap. 11/14.
"""

import numpy as np
import scipy.stats as ss
from loguru import logger

def deflated_sharpe_ratio(
    observed_sr: float,
    sr_values: list[float] | np.ndarray,
    n_trials: int,
    n_days: int,
) -> float:
    """
    Calcula o Deflated Sharpe Ratio (DSR) — probabilidade [0, 1].
    
    Parameters
    ----------
    observed_sr : float
        Sharpe Ratio anualizado da melhor configuração.
    sr_values : list ou array
        Lista de todos os Sharpes obtidos durante a otimização.
    n_trials : int
        Número total de trials independentes realizados.
    n_days : int
        Número de observações (dias/barras) no dataset.
        
    Returns
    -------
    float
        DSR (valor p ou probabilidade de significância). 
        Valores > 0.95 indicam significância estatística.
    """
    if len(sr_values) < 2:
        return 1.0
    
    # Converte SR anualizado para SR diário (assumindo 252 dias)
    # López de Prado geralmente usa a versão não anualizada nos cálculos internos
    sr_daily = observed_sr / np.sqrt(252)
    
    # Variância dos Sharpes (anualizados)
    sigma_sr = np.std(sr_values)
    
    # Constante de Euler-Mascheroni
    gamma = 0.5772156649
    
    # Estimativa do Sharpe esperado máximo sob a hipótese nula (H0)
    # Baseado na distribuição do valor máximo de N variáveis normais
    expected_max_sr = sigma_sr * (
        (1 - gamma) * ss.norm.ppf(1 - 1/n_trials) +
        gamma * ss.norm.ppf(1 - 1/(n_trials * np.e))
    )
    
    # Parâmetros para o PSR (Probabilistic Sharpe Ratio)
    # skewness e kurtosis simplificadas (assumindo normalidade dos retornos)
    skew = 0
    kurt = 3
    
    # Ajuste de significância (Z-stat)
    z_stat = (
        (sr_daily - expected_max_sr / np.sqrt(252)) * np.sqrt(n_days - 1) /
        np.sqrt(1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily**2)
    )
    
    # Probabilidade acumulada (DSR)
    dsr = ss.norm.cdf(z_stat)
    
    logger.info(
        "DSR Analysis: Obs_SR={:.2f}, Max_H0_SR={:.2f}, N_Trials={}, DSR={:.4f}",
        observed_sr, expected_max_sr, n_trials, dsr
    )
    
    return float(dsr)
