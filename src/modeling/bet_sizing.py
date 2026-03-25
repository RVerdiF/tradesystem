"""
4.4 — Bet Sizing (Kelly Fracionário).

Usa as probabilidades de sucesso (saída do Meta-Model) e as odds estimadas
para calcular o tamanho ótimo da aposta, limitando a alavancagem com o 
critério de Kelly Fracionário para segurança.

Referência: López de Prado, *Advances in Financial Machine Learning*, Cap. 10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import ml_config, risk_config


def compute_kelly_fraction(
    prob_win: pd.Series | np.ndarray | float,
    odds: float | pd.Series | np.ndarray = 1.0,
    fraction: float | None = None,
) -> pd.Series | float:
    """
    Calcula a fração da banca a ser arriscada segundo o critério de Kelly.

    A fórmula base de Kelly é: f* = p - (q / odds)
    onde:
    - p = probabilidade de vitória
    - q = probabilidade de perda (1 - p)
    - odds = razão entre o lucro em caso de vitória e a perda em caso de derrota
             (Take Profit / Stop Loss)

    Uma aposta só é feita (f > 0) se p * odds > q (Edge positivo).

    Parameters
    ----------
    prob_win : float, np.ndarray ou pd.Series
        Probabilidade de sucesso prevista pelo modelo secundário.
        Valores entre 0 e 1.
    odds : float, np.ndarray ou pd.Series, default 1.0
        Relação recompensa/risco.
        Por padrão = 1.0 se as barreiras de TP/SL forem simétricas (1:1).
    fraction : float, optional
        Fator de redução para o "Fractional Kelly" (ex: 0.5 para half-Kelly).
        Vem da configuração de risco se não for passado.

    Returns
    -------
    f_kelly : float, np.ndarray ou pd.Series
        Fração sugerida a ser apostada, limitada no intervalo [0, 1].
    """
    if fraction is None:
        fraction = risk_config.kelly_fraction

    # lida com inputs diversos e uniformiza as operações
    p = np.clip(prob_win, 0.0, 1.0)
    q = 1.0 - p

    # O edge deve ser positivo p - q/odds > 0
    f_optimal = p - (q / odds)
    
    # Kelly fracionário para suavizar retornos e mitigar estimações ruins da prob
    f_frac = f_optimal * fraction

    # Nunca aposta se o edge for negativo (clip inferior em 0)
    # E limita a 100% da banca permitida pelo modelo (clip superior em 1)
    f_final = np.clip(f_frac, 0.0, 1.0)

    if isinstance(f_final, pd.Series):
        f_final.name = "bet_size"
    
    return f_final


def discretize_bet(
    kelly_fraction: pd.Series,
    max_position: int | None = None,
    step_size: int = 1,
) -> pd.Series:
    """
    Converte a fração contínua de Kelly em tamanho discreto de posição (lotes).

    Na prática de trading, não podemos negociar 0.32 contratos futuros. 
    Este utilitário discretiza a saída contínua para um número inteiro 
    de contratos, respeitando o limite máximo definido na configuração.

    Parameters
    ----------
    kelly_fraction : pd.Series
        Fração sugerida pela função `compute_kelly_fraction` ([0, 1]).
    max_position : int, optional
        Número máximo absoluto de contratos que o sistema permite abrir.
        Se não fornecido, busca no config_ml.
    step_size : int, default 1
        Tamanho do lote mínimo (ex: Win/Winalot = 1).

    Returns
    -------
    pd.Series
        Tamanho da posição em lotes (sempre inteiro e  >= 0).
    """
    if max_position is None:
        max_position = ml_config.max_leverage

    if max_position <= 0:
        return pd.Series(0, index=kelly_fraction.index, dtype=int)

    # Multiplica a fração pelo tamanho máximo (regra de três simples para escalar o Kelly)
    continuous_pos = kelly_fraction * max_position
    
    # Arredonda para o step_size mais próximo e casta pra inteiro
    discrete_pos = (np.round(continuous_pos / step_size) * step_size).astype(int)

    # Clip final just in case
    discrete_pos = discrete_pos.clip(lower=0, upper=max_position)
    
    # Debug info global se for uma série maior, pra não spammar os logs
    if len(discrete_pos) > 1:
        avg_pos = discrete_pos.mean()
        max_pos_reached = discrete_pos.max()
        logger.info(
            "Discretized Pos: Max sugerido={}, Média={:.2f}, Bets ignorados={}", 
            max_pos_reached, 
            avg_pos, 
            (discrete_pos == 0).sum()
        )
        
    return discrete_pos
