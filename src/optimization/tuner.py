"""
Motor de Otimização Bayesiana (Optuna) — TradeSystem5000.

Este módulo implementa a lógica de busca de hiperparâmetros utilizando o
framework Optuna, com foco em robustez estatística e mitigação de overfitting.

Funcionalidades:
- Otimização em Duas Fases: Alpha/Labeling primeiro, seguido pelo Meta-Model.
- **objective**: Função objetivo com penalizações graduadas (sem cliffs) por baixa
  frequência de trades, cherry-picking excessivo e ausência de Sharpe Lift.
- **run_optimization**: Estudo bayesiano com suporte a timeout e DSR.
- Integração do Deflated Sharpe Ratio (DSR) para validar a significância.
- Penalização por Generalization Gap (SR_train vs SR_test).

Referências
-----------
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
Capítulos 11, 12 e 13.
"""

import math

import optuna
from loguru import logger

from config.settings import optimization_config
from src.backtest.dsr import deflated_sharpe_ratio
from src.main_backtest import fetch_mt5_data, run_pipeline


# ---------------------------------------------------------------------------
# Constantes de penalização (centralizadas para facilitar ajuste futuro)
# ---------------------------------------------------------------------------
_LIFT_DECAY_COEFFICIENT = 2.0  # used as: exp(coef * lift) with lift<0 ≡ exp(-coef*|lift|)
_FILTER_RATE_SOFT_START = 0.70  # acima disso, penalidade quadrática inicia
_FILTER_RATE_HARD_REJECT = 0.92  # acima disso, trial é descartado
_FILTER_RATE_PENALTY_RANGE = _FILTER_RATE_HARD_REJECT - _FILTER_RATE_SOFT_START  # 0.22
_FILTER_RATE_MIN_MULTIPLIER = 0.05  # piso da penalidade quadrática


def _sharpe_lift_multiplier(lift: float) -> float:
    """
    Retorna o multiplicador de fitness baseado no Sharpe Lift.
    """
    if lift >= 0.0:
        return 1.0
    return math.exp(_LIFT_DECAY_COEFFICIENT * lift)  # lift < 0, então exp(negativo)


def _filter_rate_multiplier(filter_rate: float) -> float | None:
    """
    Retorna o multiplicador de fitness baseado na filter_rate.
    """
    if filter_rate <= _FILTER_RATE_SOFT_START:
        return 1.0
    if filter_rate > _FILTER_RATE_HARD_REJECT:
        return None  # sinaliza hard reject
    excess = (filter_rate - _FILTER_RATE_SOFT_START) / _FILTER_RATE_PENALTY_RANGE
    multiplier = 1.0 - excess**2
    return max(_FILTER_RATE_MIN_MULTIPLIER, multiplier)


def apply_penalties(results, trial, fitness):
    """Aplica as penalizações de fitness aos resultados do trial."""
    # Filtro de Frequência Contínuo
    min_trades = optimization_config.min_trades
    if results["n_trades"] < min_trades:
        penalty = results["n_trades"] / min_trades
        fitness *= penalty
        logger.debug(
            f"Trial {trial.number}: Penalidade de frequência "
            f"(trades={results['n_trades']}, penalty={penalty:.2f})"
        )

    # Sharpe Lift
    lift = results.get("sharpe_lift", 0.0)
    lift_mult = _sharpe_lift_multiplier(lift)
    if lift_mult < 1.0:
        logger.debug(
            f"Trial {trial.number}: Sharpe Lift penalty "
            f"(lift={lift:.3f}, multiplier={lift_mult:.3f})"
        )
    fitness *= lift_mult

    # Filter Rate
    filter_rate = results.get("filter_rate", 0.0)
    fr_mult = _filter_rate_multiplier(filter_rate)
    if fr_mult is None:
        logger.warning(
            f"Trial {trial.number}: rejeitado por filter_rate={filter_rate:.3f} "
            f"(>{_FILTER_RATE_HARD_REJECT})"
        )
        return -1.0
    if fr_mult < 1.0:
        logger.debug(
            f"Trial {trial.number}: Filter rate penalty "
            f"(filter_rate={filter_rate:.3f}, multiplier={fr_mult:.3f})"
        )
    fitness *= fr_mult

    # Generalization Gap
    sr_train = results.get(
        "sharpe_train", results["sharpe"]
    )  # fallback: assume train == test (no gap)
    sr_test = results["sharpe"]
    gap = sr_train - sr_test

    if gap > 3.0:
        logger.warning(
            f"Trial {trial.number}: rejeitado por generalization gap alto "
            f"(sr_train={sr_train:.2f}, sr_test={sr_test:.2f}, gap={gap:.2f})"
        )
        return -1.0

    if gap > 1.0:
        logger.debug(f"Trial {trial.number}: Alto Generalization Gap (Gap={gap:.2f})")
        fitness /= 1.0 + gap

    return fitness


def objective_phase1(trial, df, interval):
    """
    Fase 1: Otimiza Alpha (Labels, CUSUM, Features).
    Parâmetros do Meta-Model são fixados para evitar overfitting cruzado e focar na qualidade do sinal.
    """
    params = {
        "cusum_threshold": trial.suggest_float("cusum_threshold", *optimization_config.cusum_range),
        "alpha_fast": trial.suggest_int("alpha_fast", *optimization_config.fast_span_range),
        "alpha_slow": trial.suggest_int("alpha_slow", *optimization_config.slow_span_range),
        "pt_sl": (
            trial.suggest_float("pt_mult", *optimization_config.pt_sl_range),
            trial.suggest_float(
                "sl_mult",
                max(1.5, float(optimization_config.pt_sl_range[0]) if hasattr(optimization_config.pt_sl_range[0], "__float__") else 1.5),
                optimization_config.pt_sl_range[1],
            ),
        ),
        "ma_dist_fast_period": trial.suggest_int(
            "ma_dist_fast_period", *optimization_config.ma_dist_fast_range
        ),
        "ma_dist_slow_period": trial.suggest_int(
            "ma_dist_slow_period", *optimization_config.ma_dist_slow_range
        ),
        "moments_window": trial.suggest_int(
            "moments_window", *optimization_config.moments_window_range
        ),
        "be_trigger": trial.suggest_float("be_trigger", *optimization_config.be_trigger_range),
        "ffd_d": trial.suggest_float("ffd_d", *optimization_config.ffd_d_range),
        "atr_period": trial.suggest_int("atr_period", *optimization_config.atr_period_range),
        # FIXADOS PARA FASE 1: Meta-Model rápido e raso
        "xgb_max_depth": 1,
        "xgb_gamma": 0.0,
        "xgb_lambda": 1.0,
        "xgb_alpha": 0.0,
        "meta_threshold": 0.50,
    }

    if params["alpha_slow"] <= params["alpha_fast"]:
        return -1.0

    try:
        results = run_pipeline(df, interval=interval, params=params)
    except Exception as e:
        logger.error(f"Erro no pipeline Fase 1 durante trial {trial.number}: {e}")
        return -1.0

    if results is None:
        return -1.0

    # Usamos o Calmar ou Sharpe base para avaliar a qualidade pura do alpha e das labels.
    fitness = results.get("calmar_ratio", results["sharpe"])
    return apply_penalties(results, trial, fitness)


def objective_phase2(trial, df, interval, base_params):
    """
    Fase 2: Otimiza Hyperparâmetros do ML (Meta-Model).
    Recebe os parâmetros do Alpha otimizados e fixos.
    """
    params = dict(base_params)  # Copia os melhores parâmetros da Fase 1

    # Atualiza com as variações específicas de ML
    params.update(
        {
            "meta_threshold": trial.suggest_float(
                "meta_threshold", optimization_config.meta_threshold_range[0], optimization_config.meta_threshold_range[1]
            ),
            "xgb_max_depth": trial.suggest_int(
                "xgb_max_depth", optimization_config.max_depth_range[0], optimization_config.max_depth_range[1]
            ),
            "xgb_gamma": trial.suggest_float("xgb_gamma", optimization_config.xgb_gamma_range[0], optimization_config.xgb_gamma_range[1]),
            "xgb_lambda": trial.suggest_float("xgb_lambda", optimization_config.xgb_lambda_range[0], optimization_config.xgb_lambda_range[1]),
            "xgb_alpha": trial.suggest_float("xgb_alpha", optimization_config.xgb_alpha_range[0], optimization_config.xgb_alpha_range[1]),
            "xgb_min_child_weight": trial.suggest_float(
                "xgb_min_child_weight", optimization_config.xgb_min_child_weight_range[0], optimization_config.xgb_min_child_weight_range[1]
            ),
            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", optimization_config.scale_pos_weight_range[0], optimization_config.scale_pos_weight_range[1]
            ),
        }
    )

    try:
        results = run_pipeline(df, interval=interval, params=params)
    except Exception as e:
        logger.error(f"Erro no pipeline Fase 2 durante trial {trial.number}: {e}")
        return -1.0

    if results is None:
        return -1.0

    fitness = results.get("calmar_ratio", results["sharpe"])
    return apply_penalties(results, trial, fitness)


def run_optimization(
    df, interval, n_trials=None, n_trials_phase1=None, n_trials_phase2=None, symbol="default"
):
    """
    Configura e executa o estudo do Optuna em duas fases.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com dados históricos.
    interval : str
        Intervalo de tempo das barras.
    n_trials : int, optional
        Número total de trials (legado). Se fornecido sem n_trials_phase1/phase2,
        Phase 1 recebe n_trials trials e Phase 2 recebe max(10, n_trials // 3) —
        total MAIOR que n_trials. Prefira usar n_trials_phase1/phase2 explicitamente.
    n_trials_phase1 : int, optional
        Trials para Phase 1 (Alpha/Labels). Default: optimization_config.n_trials_phase1.
    n_trials_phase2 : int, optional
        Trials para Phase 2 (Meta-Model). Default: optimization_config.n_trials_phase2.
    symbol : str
        Nome do ativo. Usado no study_name para evitar colisão em armazenamento persistente.

    Notes
    -----
    Data Leakage entre fases: Phase 2 re-executa run_pipeline com os params fixos da Phase 1,
    usando o mesmo df e os mesmos splits temporais (determinísticos pelo índice de tempo).
    Não há vazamento pois Phase 2 não vê os labels de Phase 1 diretamente — eles são
    re-gerados deterministicamente dentro de run_pipeline.
    """
    if n_trials_phase1 is None:
        n_trials_phase1 = optimization_config.n_trials_phase1 if n_trials is None else n_trials

    if n_trials_phase2 is None:
        n_trials_phase2 = (
            optimization_config.n_trials_phase2 if n_trials is None else max(10, n_trials // 3)
        )

    logger.info(f"--- INICIANDO FASE 1: OTIMIZAÇÃO ALPHA/LABELING ({n_trials_phase1} trials) ---")
    study_phase1 = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"TradeSystem5000_{symbol}_Phase1",  # symbol-qualified: evita colisão em storage persistente
    )

    study_phase1.optimize(
        lambda trial: objective_phase1(trial, df, interval),
        n_trials=n_trials_phase1,
        timeout=optimization_config.timeout,
        show_progress_bar=True,
    )

    logger.success("FASE 1 CONCLUÍDA!")
    best_phase1_params = study_phase1.best_params

    # Prepara PT/SL como tupla para o Phase 2
    best_phase1_params["pt_sl"] = (
        best_phase1_params.get("pt_mult", 2.0),
        best_phase1_params.get("sl_mult", 2.0),
    )

    logger.info("Melhores Parâmetros Fase 1: {}", best_phase1_params)

    logger.info(f"--- INICIANDO FASE 2: OTIMIZAÇÃO META-MODEL ({n_trials_phase2} trials) ---")
    study_phase2 = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"TradeSystem5000_{symbol}_Phase2",  # symbol-qualified: evita colisão em storage persistente
    )

    study_phase2.optimize(
        lambda trial: objective_phase2(trial, df, interval, best_phase1_params),
        n_trials=n_trials_phase2,
        timeout=optimization_config.timeout,
        show_progress_bar=True,
    )

    logger.success("FASE 2 CONCLUÍDA!")

    # Merge final dos parâmetros
    final_best_params = dict(best_phase1_params)
    final_best_params.update(study_phase2.best_params)

    # Limpeza para serialização (remover tuplas compostas)
    params_output = dict(final_best_params)
    if "pt_mult" in params_output:
        params_output.pop("pt_mult")
    if "sl_mult" in params_output:
        params_output.pop("sl_mult")

    # Fase 4: Avaliação do DSR no resultado final (Fase 2)
    sr_values = [t.value for t in study_phase2.trials if t.value is not None and t.value != -1.0]
    dsr_score = deflated_sharpe_ratio(
        observed_sr=study_phase2.best_value,
        sr_values=sr_values,
        n_trials=len(study_phase2.trials),
        n_days=len(df),
    )

    if dsr_score >= 0.95:
        logger.success("SIGNIFICÂNCIA CONFIRMADA: O melhor Sharpe é real (DSR = {:.4f})", dsr_score)
    else:
        logger.warning(
            "ALERTA DE OVERFITTING: O melhor Sharpe pode ser sorte (DSR = {:.4f})", dsr_score
        )

    return {
        "study": study_phase2,
        "params": params_output,
        "metadata": {
            "dsr_score": dsr_score,
            "best_sharpe": study_phase2.best_value,
            "n_trials_phase1": len(study_phase1.trials),
            "n_trials_phase2": len(study_phase2.trials),
        },
    }


if __name__ == "__main__":
    # Script para teste rápido da otimização
    logger.info("Rodando teste de otimização com dados reais do MT5...")
    df_test = fetch_mt5_data(symbol="PETR4", n_bars=1500, interval="1d")
    results = run_optimization(df_test, interval="1d", n_trials=5)
    print(results)
