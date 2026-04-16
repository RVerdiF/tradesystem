"""Dimensionamento de Apostas (Bet Sizing) — TradeSystem5000.

Este módulo implementa estratégias de dimensionamento de posição baseadas na
probabilidade de sucesso (saída do Meta-Modelo) e no critério de Kelly.

O objetivo é otimizar o crescimento da banca a longo prazo, limitando a
alavancagem através do Kelly Fracionário para garantir a sobrevivência e
mitigar erros de estimação.

Funcionalidades:
- **compute_kelly_fraction**: Cálculo da fração ótima de Kelly (Fractional Kelly).
- **discretize_bet**: Conversão de frações contínuas em lotes operacionais (discretos).
- **apply_conviction_threshold**: Zera probabilidades abaixo do limiar de convicção (pré-Kelly).

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulo 10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import risk_config


def compute_kelly_fraction(
    prob_win: pd.Series | np.ndarray | float,
    odds: float | pd.Series | np.ndarray = 1.0,
    fraction: float | None = None,
) -> pd.Series | float:
    """Calcula a fração da banca a ser arriscada segundo o critério de Kelly.

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
