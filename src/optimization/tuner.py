"""
2.1 — Motor de Otimização Bayesiana (Optuna).

Implementa a busca de hiperparâmetros com penalização de overfitting
e validação cruzada purificada.

Referência: López de Prado, *Advances in Financial Machine Learning*, Cap. 12/13.
"""

import optuna
import pandas as pd
import numpy as np
from loguru import logger
from src.main_backtest import run_pipeline, generate_synthetic_data, fetch_yfinance_data
from config.settings import optimization_config
from src.backtest.dsr import deflated_sharpe_ratio

def objective(trial, df, interval):
    """
    Função objetivo para o Optuna.
    Implementa as fases 1 e 3 do plano de implementação.
    """
    # Fase 1: Espaço de busca restrito
    params = {
        "cusum_threshold": trial.suggest_float("cusum_threshold", *optimization_config.cusum_range),
        "alpha_fast": trial.suggest_int("alpha_fast", *optimization_config.fast_span_range),
        "alpha_slow": trial.suggest_int("alpha_slow", *optimization_config.slow_span_range),
        "pt_sl": (
            trial.suggest_float("pt_mult", *optimization_config.pt_sl_range),
            trial.suggest_float("sl_mult", *optimization_config.pt_sl_range)
        ),
        "xgb_max_depth": trial.suggest_int("xgb_max_depth", *optimization_config.max_depth_range)
    }
    
    # Garantir que slow > fast
    if params["alpha_slow"] <= params["alpha_fast"]:
        return -1.0 # Penaliza configurações inválidas
    
    # Executa o pipeline completo (com CPCV)
    try:
        results = run_pipeline(df, interval=interval, params=params)
    except Exception as e:
        logger.error(f"Erro no pipeline durante trial {trial.number}: {e}")
        return -1.0
    
    if results is None:
        return -1.0

    # Fase 3: Função Objetivo e Penalização de Overfitting
    
    # 3.2 Filtro de Frequência: < 30 trades = Sharpe 0
    if results["n_trades"] < optimization_config.min_trades:
        logger.warning(f"Trial {trial.number} rejeitado: poucos trades ({results['n_trades']})")
        return 0.0
    
    fitness = results["sharpe"]
    
    # 3.3 Sharpe Lift: Se Sharpe_Meta <= Sharpe_Alpha, penalizamos drasticamente
    if results["sharpe_lift"] <= 0:
        logger.debug(f"Trial {trial.number}: Sem Sharpe Lift (Lift={results['sharpe_lift']:.2f})")
        fitness *= 0.1 # Penalidade de 90% se o Meta-Model não agregar valor
        
    # 3.4 Generalization Gap: Penalizar se SR_Train >> SR_Test
    sr_train = results["sharpe_train"]
    sr_test = results["sharpe"]
    
    gap = sr_train - sr_test
    if gap > 1.0: # Exemplo de limite: gap de 1.0 no Sharpe é alto
        logger.debug(f"Trial {trial.number}: Alto Generalization Gap (Gap={gap:.2f})")
        fitness /= (1.0 + gap)
    
    return fitness

def run_optimization(df, interval):
    """Configura e executa o estudo do Optuna."""
    logger.info("Iniciando Otimização Bayesiana ({} trials)...", optimization_config.n_trials)
    
    study = optuna.create_study(
        direction='maximize', 
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="TradeSystem5000_Optimization"
    )
    
    study.optimize(
        lambda trial: objective(trial, df, interval), 
        n_trials=optimization_config.n_trials,
        timeout=optimization_config.timeout,
        show_progress_bar=True
    )
    
    logger.success("Otimização concluída!")
    logger.info("Melhores parâmetros: {}", study.best_params)
    logger.info("Melhor Valor Fitness: {:.2f}", study.best_value)
    
    # Fase 4: Avaliação do DSR
    sr_values = [t.value for t in study.trials if t.value is not None]
    dsr_score = deflated_sharpe_ratio(
        observed_sr=study.best_value,
        sr_values=sr_values,
        n_trials=len(study.trials),
        n_days=len(df)
    )
    
    if dsr_score >= 0.95:
        logger.success("SIGNIFICÂNCIA CONFIRMADA: O melhor Sharpe é real (DSR = {:.2f})", dsr_score)
    else:
        logger.warning("ALERTA DE OVERFITTING: O melhor Sharpe pode ser sorte (DSR = {:.2f})", dsr_score)
    
    return study

if __name__ == "__main__":
    # Script para teste rápido da otimização
    logger.info("Rodando teste de otimização com dados sintéticos...")
    df_test = generate_synthetic_data(n_days=1500)
    run_optimization(df_test, interval="1d")
