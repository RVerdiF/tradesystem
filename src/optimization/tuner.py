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
from src.main_backtest import run_pipeline, fetch_mt5_data
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
    
    # Prepara dicionário de parâmetros normalizado (separar pt_sl, etc)
    # A estrutura deve bater com a forma como é injetada em run_pipeline / train_model
    best_params = dict(study.best_params)
    
    # Recria o dicionário na estrutura final desejada (em train_model / main_backtest)
    params = {
        "cusum_threshold": best_params.get("cusum_threshold"),
        "alpha_fast": best_params.get("alpha_fast"),
        "alpha_slow": best_params.get("alpha_slow"),
        "pt_sl": (best_params.get("pt_mult"), best_params.get("sl_mult")),
        "xgb_max_depth": best_params.get("xgb_max_depth")
    }
    
    return {
        "study": study,
        "params": params,
        "metadata": {
            "dsr_score": dsr_score,
            "best_sharpe": study.best_value,
            "n_trials": len(study.trials)
        }
    }

if __name__ == "__main__":
    # Script para teste rápido da otimização
    logger.info("Rodando teste de otimização com dados reais do MT5...")
    df_test = fetch_mt5_data(symbol="PETR4", n_bars=1500, interval="1d")
    results = run_optimization(df_test, interval="1d")
    print(results)
