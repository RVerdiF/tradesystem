# Plano de Implementação: Melhoria de Identificação de Oportunidades e Robustez do Modelo

Este documento detalha as etapas para aumentar o volume de sinais gerados pelo Alpha Model que chegam ao Meta-Model e para tornar o treinamento preditivo mais robusto.

## Fase 1: Aumento do Fluxo de Dados e Sinais

O objetivo é reduzir a restrição do funil de eventos para alimentar o modelo com mais dados.

* [ ] **1.1 Ajustar Threshold do CUSUM Filter**
    * **Arquivo:** `config/settings.py`
    * **Ação:** Reduzir a variável `cusum_threshold_pct` para `0.5` ou um valor estatisticamente menor que a volatilidade média diária.
* [ ] **1.2 Alterar Amostragem de Barras**
    * **Arquivo:** `src/main_backtest.py`
    * **Ação:** Substituir a amostragem baseada em tempo (Time Bars) por `volume_bars` ou `dollar_bars` (disponíveis em `src/data/bar_sampler.py`).

## Fase 2: Aprimoramento e Robustez do Treinamento

O objetivo é evitar o underfitting e garantir que a validação reflita o desempenho real.

* [ ] **2.1 Atualizar o Algoritmo do Meta-Classificador**
    * **Arquivo:** `src/modeling/classifier.py`
    * **Ação:** Aumentar o `max_depth` da Random Forest atual ou substituí-la pelo XGBoost (`XGBClassifier`).
* [ ] **2.2 Implementar Pesos por Retorno Absoluto (Sample Weighting)**
    * **Arquivo:** `src/main_backtest.py`
    * **Ação:** Passar o valor absoluto dos retornos no parâmetro `sample_weight` ao chamar o método `.fit()` do classificador.
* [ ] **2.3 Configurar Validação Cruzada Purificada**
    * **Arquivo:** `src/main_backtest.py`
    * **Ação:** Substituir o K-Fold padrão pelo `PurgedKFold` ou `CPCV` (`src/backtest/cpcv.py` ou `src/modeling/purge_embargo.py`), utilizando pelo menos 10 splits e embargo configurado.

## Fase 3: Diagnóstico de Features e Atribuição

Garantir que os sinais filtrados possuem valor preditivo de fato.

* [ ] **3.1 Otimizar Diferenciação Fracionária**
    * **Arquivo:** `src/features/frac_diff.py`
    * **Ação:** Executar `find_min_d()` nas séries de preços para encontrar o menor *d* que passe no teste ADF e atualizar a geração de features com este valor.
* [ ] **3.2 Analisar o Sharpe Lift**
    * **Arquivo:** `src/main_backtest.py`
    * **Ação:** Rodar `attribution_analysis` de `src/backtest/attribution.py`. O Sharpe Ratio do portfólio filtrado pelo Meta-Model deve ser superior ao da estratégia Alpha original.