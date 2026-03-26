# Plano de Implementação: Otimização Bayesiana Global e Prevenção de Overfitting

Este plano detalha a implementação da otimização de hiperparâmetros utilizando o Optuna, integrando as métricas de penalização e validação cruzada exigidas pela literatura quantitativa (López de Prado, Ernest Chan).

## Fase 1: Restrição do Espaço de Busca
Para evitar o viés de *data-snooping*, o espaço de busca será restrito a um máximo de 5 a 6 parâmetros, todos com justificativa econômica.

* [ ] **1.1. Arquivo:** `config/settings.py` (ou no próprio script do Optuna)
* [ ] **1.2. Ação:** Definir os limites estatísticos (ranges) dos parâmetros:
    * `cusum_threshold`: Volatilidade baseada no mercado (ex: 0.01 a 0.05).
    * `alpha_fast` / `alpha_slow`: Períodos clássicos de momentum (ex: 5-20 / 20-60).
    * `pt_sl` (Profit Taking / Stop Loss): Múltiplos da volatilidade diária (ex: 1.0 a 3.0).
    * `xgb_max_depth`: Limitar severamente para evitar árvores complexas (ex: 2 a 4).

## Fase 2: Criação do Motor de Otimização (Optuna)
Implementação do otimizador utilizando o estimador TPE (Tree-structured Parzen Estimator) com pruning.

* [ ] **2.1. Arquivo:** Criar `src/optimization/tuner.py`
* [ ] **2.2. Ação:** Desenvolver a função `objective(trial)` que inicializa o pipeline completo (`run_pipeline`).
* [ ] **2.3. Ação:** Configurar o `optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())`.

## Fase 3: Função Objetivo e Penalização de Overfitting
A métrica de *fitness* deve descartar configurações não generalizáveis.

* [ ] **3.1. Arquivo:** `src/optimization/tuner.py`
* [ ] **3.2. Ação - Filtro de Frequência:** Retornar Sharpe Ratio = 0.0 se o número total de trades na validação cruzada (CPCV) for inferior a um limite estatisticamente significativo (ex: < 30 trades).
* [ ] **3.3. Ação - Sharpe Lift:** Aplicar penalidade ou rejeitar o trial se o desempenho do Meta-Model for inferior ao do Alpha puro ($Sharpe_{Meta} \le Sharpe_{Alpha}$).
* [ ] **3.4. Ação - Generalization Gap:** Penalizar o Sharpe de Validação se a diferença entre o Sharpe de Treino e Validação for superior a um limite aceitável (indicador claro de under/overfitting).

## Fase 4: Avaliação do Deflated Sharpe Ratio (DSR)
Após a conclusão da otimização, é necessário descontar o efeito das múltiplas tentativas.

* [ ] **4.1. Arquivo:** Criar `src/backtest/dsr.py` (ou integrar em `metrics.py`)
* [ ] **4.2. Ação:** Ao final dos *N* trials do Optuna, extrair a variância dos resultados (Sharpes testados) e o número total de trials executados.
* [ ] **4.3. Ação:** Calcular o **Deflated Sharpe Ratio (DSR)** sobre o melhor modelo encontrado, utilizando a fórmula de López de Prado para confirmar se o Sharpe reportado ainda é estatisticamente significativo a 95% de confiança após descontar o número de testes realizados.