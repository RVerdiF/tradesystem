"""
Motor de Otimização Bayesiana (Optuna) — TradeSystem5000.

Este módulo implementa a lógica de busca de hiperparâmetros utilizando o
framework Optuna, com foco em robustez estatística e mitigação de overfitting.

Funcionalidades:
- **objective**: Função objetivo com penalizações graduadas (sem cliffs) por baixa
  frequência de trades, cherry-picking excessivo e ausência de Sharpe Lift.
- **run_optimization**: Estudo bayesiano com suporte a timeout e DSR.
- Integração do Deflated Sharpe Ratio (DSR) para validar a significância.
- Penalização por Generalization Gap (SR_train vs SR_test).

Mudanças v2 (2026-04-13):
- Substituído cliff `fitness *= 0.1` para Sharpe Lift negativo por curva exponencial suave.
- Substituído hard-reject binário em filter_rate > 0.90 por penalidade quadrática
  progressiva (0.70–0.92) + hard-reject apenas em > 0.92.
- Mantido hard-reject por generalization gap > 3.0.

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
_LIFT_DECAY_COEFFICIENT = 2.0      # used as: exp(coef * lift) with lift<0 ≡ exp(-coef*|lift|)
_FILTER_RATE_SOFT_START = 0.70     # acima disso, penalidade quadrática inicia
_FILTER_RATE_HARD_REJECT = 0.92    # acima disso, trial é descartado
_FILTER_RATE_PENALTY_RANGE = _FILTER_RATE_HARD_REJECT - _FILTER_RATE_SOFT_START  # 0.22
_FILTER_RATE_MIN_MULTIPLIER = 0.05 # piso da penalidade quadrática


def _sharpe_lift_multiplier(lift: float) -> float:
    """
    Retorna o multiplicador de fitness baseado no Sharpe Lift.

    Curva exponencial suave — sem cliff:
      - lift >= 0  → multiplier = 1.0 (sem penalidade)
      - lift < 0   → multiplier = exp(-2.0 * |lift|)
        Ex: lift=-0.1 → 0.819, lift=-1.0 → 0.135, lift=-3.0 → 0.002

    Razão: o TPE sampler precisa de gradiente suave para construir uma
    densidade de probabilidade útil. Um cliff em lift=0 faz o sampler
    tratar todos os negativos como igualmente inúteis.
    """
    if lift >= 0.0:
        return 1.0
    return math.exp(_LIFT_DECAY_COEFFICIENT * lift)  # lift < 0, então exp(negativo)


def _filter_rate_multiplier(filter_rate: float) -> float | None:
    """
    Retorna o multiplicador de fitness baseado na filter_rate.

    - filter_rate <= 0.70       → 1.0 (sem penalidade)
    - 0.70 < rate <= 0.92       → penalidade quadrática: max(0.05, 1 - ((rate-0.70)/0.22)^2)
                                  (em rate=0.92: excess=1.0 → multiplier=max(0.05,0.0)=0.05)
    - filter_rate > 0.92        → None (hard reject — retornar -1.0 no caller)

    Razão: filter_rate=0.85 (trail de cherry-picking moderado) antes passava sem
    penalidade. Agora recebe multiplier≈0.54. O hard-reject é ligeiramente
    relaxado de 0.90 → 0.92 para evitar rejeitar configurações de alta precisão
    que caem marginalmente acima do threshold anterior.
    """
    if filter_rate <= _FILTER_RATE_SOFT_START:
        return 1.0
    if filter_rate > _FILTER_RATE_HARD_REJECT:
        return None  # sinaliza hard reject
    excess = (filter_rate - _FILTER_RATE_SOFT_START) / _FILTER_RATE_PENALTY_RANGE
    multiplier = 1.0 - excess ** 2
    return max(_FILTER_RATE_MIN_MULTIPLIER, multiplier)


def objective(trial, df, interval):
    """
    Função objetivo para o Optuna.
    Implementa as fases 1 e 3 do plano de implementação.
    """
    # Fase 1: Espaço de busca restrito (Top 10 - Faxina Real)
    params = {
        "cusum_threshold": trial.suggest_float(
            "cusum_threshold", *optimization_config.cusum_range
        ),
        "alpha_fast": trial.suggest_int("alpha_fast", *optimization_config.fast_span_range),
        "alpha_slow": trial.suggest_int("alpha_slow", *optimization_config.slow_span_range),
        "pt_sl": (
            trial.suggest_float("pt_mult", *optimization_config.pt_sl_range),
            trial.suggest_float("sl_mult", *optimization_config.pt_sl_range),
        ),
        "meta_threshold": trial.suggest_float(
            "meta_threshold", *optimization_config.meta_threshold_range
        ),
        "xgb_max_depth": trial.suggest_int(
            "xgb_max_depth", *optimization_config.max_depth_range
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
        "be_trigger": trial.suggest_float(
            "be_trigger", *optimization_config.be_trigger_range
        ),
        "ffd_d": trial.suggest_float(
            "ffd_d", *optimization_config.ffd_d_range
        ),
        "atr_period": trial.suggest_int(
            "atr_period", *optimization_config.atr_period_range
        ),
        "xgb_gamma": trial.suggest_float(
            "xgb_gamma", *optimization_config.xgb_gamma_range
        ),
        "xgb_lambda": trial.suggest_float(
            "xgb_lambda", *optimization_config.xgb_lambda_range
        ),
        "xgb_alpha": trial.suggest_float(
            "xgb_alpha", *optimization_config.xgb_alpha_range
        ),
    }

    # Garantir que slow > fast para o Alpha Model
    if params["alpha_slow"] <= params["alpha_fast"]:
        return -1.0

    # Executa o pipeline completo (com CPCV)
    try:
        results = run_pipeline(df, interval=interval, params=params)
    except Exception as e:
        logger.error(f"Erro no pipeline durante trial {trial.number}: {e}")
        return -1.0

    if results is None:
        return -1.0

    # Fase 3: Função Objetivo e Penalização de Overfitting

    # 3.1 Fitness base: Calmar quando disponível, Sharpe como fallback
    fitness = results.get("calmar_ratio", results["sharpe"])

    # 3.2 Filtro de Frequência Contínuo (Evita Cliffs)
    min_trades = optimization_config.min_trades
    if results["n_trades"] < min_trades:
        penalty = results["n_trades"] / min_trades
        fitness *= penalty
        logger.debug(
            f"Trial {trial.number}: Penalidade de frequência "
            f"(trades={results['n_trades']}, penalty={penalty:.2f})"
        )

    # 3.3 Sharpe Lift: curva exponencial suave (sem cliff em lift=0).
    # Razão: cliff anterior (fitness *= 0.1) distorcia o density model do TPE —
    # todos os trials negativos eram igualmente inúteis do ponto de vista do sampler.
    lift = results.get("sharpe_lift", 0.0)
    lift_mult = _sharpe_lift_multiplier(lift)
    if lift_mult < 1.0:
        logger.debug(
            f"Trial {trial.number}: Sharpe Lift penalty "
            f"(lift={lift:.3f}, multiplier={lift_mult:.3f})"
        )
    fitness *= lift_mult

    # 3.4 Filter Rate: penalidade quadrática progressiva acima de 70%.
    # Razão: 0.70 < filter_rate <= 0.92 era zona cega — passava sem penalidade.
    filter_rate = results.get("filter_rate", 0.0)
    fr_mult = _filter_rate_multiplier(filter_rate)
    if fr_mult is None:
        # Hard reject: filter_rate > 0.92
        logger.warning(
            f"Trial {trial.number}: rejeitado por filter_rate={filter_rate:.3f} "
            f"(>{_FILTER_RATE_HARD_REJECT} indica stump decorando assimetria do treino)"
        )
        return -1.0
    if fr_mult < 1.0:
        logger.debug(
            f"Trial {trial.number}: Filter rate penalty "
            f"(filter_rate={filter_rate:.3f}, multiplier={fr_mult:.3f})"
        )
    fitness *= fr_mult

    # 3.5 Generalization Gap: Penalizar se SR_Train >> SR_Test
    sr_train = results["sharpe_train"]
    sr_test = results["sharpe"]
    gap = sr_train - sr_test

    # Guardrail: generalization gap — trial-level stop para overfitting severo.
    if gap > 3.0:
        logger.warning(
            f"Trial {trial.number}: rejeitado por generalization gap alto "
            f"(sr_train={sr_train:.2f}, sr_test={sr_test:.2f}, gap={gap:.2f})"
        )
        return -1.0

    if gap > 1.0:
        logger.debug(f"Trial {trial.number}: Alto Generalization Gap (Gap={gap:.2f})")
        fitness /= (1.0 + gap)

    return fitness


def run_optimization(df, interval, n_trials=None):
    """Configura e executa o estudo do Optuna."""
    if n_trials is None:
        n_trials = optimization_config.n_trials

    logger.info("Iniciando Otimização Bayesiana ({} trials)...", n_trials)

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="TradeSystem5000_Optimization"
    )

    study.optimize(
        lambda trial: objective(trial, df, interval),
        n_trials=n_trials,
        timeout=optimization_config.timeout,
        show_progress_bar=True
    )

    logger.success("Otimização concluída!")
    logger.info("Melhores parâmetros: {}", study.best_params)
    logger.info("Melhor Valor Fitness: {:.2f}", study.best_value)

    # Fase 4: Avaliação do DSR
    # DSR é um critério de aceitação, não um parâmetro a ser "consertado".
    # Um DSR >= 0.95 indica que o melhor Sharpe é estatisticamente significativo
    # dado o número de trials testados. Com um pipeline corrigido (entrada T+1,
    # slippage calibrado, penalidades suaves), espera-se DSR > 0.
    sr_values = [t.value for t in study.trials if t.value is not None]
    dsr_score = deflated_sharpe_ratio(
        observed_sr=study.best_value,
        sr_values=sr_values,
        n_trials=len(study.trials),
        n_days=len(df)
    )

    if dsr_score >= 0.95:
        logger.success(
            "SIGNIFICÂNCIA CONFIRMADA: O melhor Sharpe é real (DSR = {:.4f})", dsr_score
        )
    else:
        logger.warning(
            "ALERTA DE OVERFITTING: O melhor Sharpe pode ser sorte (DSR = {:.4f})", dsr_score
        )

    # Prepara dicionário de parâmetros normalizado
    best_params = dict(study.best_params)
    params = dict(best_params)
    params["pt_sl"] = (best_params.get("pt_mult"), best_params.get("sl_mult"))
    params.pop("pt_mult", None)
    params.pop("sl_mult", None)

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