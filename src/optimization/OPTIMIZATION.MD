# Módulo de Otimização (src/optimization)

Este módulo é responsável pela busca automatizada dos melhores hiperparâmetros para os modelos e estratégias. Ele utiliza técnicas de otimização bayesiana e validação estatística rigorosa para mitigar o viés de seleção (*selection bias*) e o sobreajuste ao *backtest*.

A implementação segue os princípios de **López de Prado** (*Advances in Financial Machine Learning*, Cap. 12 e 13).

---

## 1. Componentes e Responsabilidades

O subsistema de otimização é composto por três módulos principais:

### 1.1 Sintonizador de Hiperparâmetros (`tuner.py`)
*   **Otimização com Optuna**: Utiliza o framework **Optuna** para realizar buscas eficientes no espaço de parâmetros (ex: learning rate, depth, thresholds de volatilidade).
*   **Early Stopping**: Implementa mecanismos para interromper tentativas de baixo desempenho, economizando recursos computacionais.
*   **Multi-Objetivo**: Suporta a otimização de múltiplas métricas simultaneamente (ex: Maximizar Sharpe Ratio e Minimizar Drawdown).

### 1.2 Prevenção de Selection Bias (`tuner.py` / `metrics.py`)
*   **Deflated Sharpe Ratio (DSR)**: Implementa o DSR para ajustar o Sharpe Ratio observado pelo número de tentativas realizadas. Isso permite verificar se o resultado "ótimo" encontrado é estatisticamente significativo ou apenas fruto do acaso (*data snooping*) (Cap. 12-13 AFML).

### 1.3 Armazenamento de Parâmetros (`params_store.py`)
*   **Persistência em SQLite**: Salva os melhores parâmetros encontrados por símbolo, timeframe e data de execução em um banco de dados local (`data/tradesystem.db`).
*   **Versionamento**: Permite que o sistema de execução carregue a versão mais recente dos parâmetros otimizados para cada ativo de forma dinâmica.

### 1.4 Executor CLI (`run_opt.py`)
*   **Interface de Comando**: Permite disparar rotinas de otimização em lote via terminal, facilitando a automação de re-treinamento periódico dos modelos.

---

## 2. Fluxo de Otimização

O processo de busca de parâmetros segue esta sequência:

1.  **Definição do Espaço**: O desenvolvedor define o `search_space` (intervalos de valores).
2.  **Execução do Estudo**: O Optuna dispara N tentativas (*trials*).
3.  **Backtest Interno**: Cada tentativa executa um backtest rápido com Purged-CV.
4.  **Cálculo de DSR**: O resultado final é validado pelo Deflated Sharpe Ratio.
5.  **Persistência**: Se aprovado nos critérios de significância, os parâmetros são salvos no `ParamsStore`.

---

## 3. Exemplo de Uso

```python
from src.optimization.tuner import run_study
from src.optimization.params_store import store_params

# 1. Executar estudo de otimização
best_params, dsr_val = run_study(symbol="PETR4", n_trials=100)

# 2. Verificar significância (DSR > threshold)
if dsr_val > 0.95:
    # 3. Persistir parâmetros aprovados
    store_params("PETR4", "meta_model", best_params)
    print("Parâmetros salvos com sucesso!")
else:
    print("Otimização rejeitada por falta de significância estatística (DSR baixo).")
```

---

## 4. Referências Técnicas
*   *Advances in Financial Machine Learning* (Cap. 12: Selection Bias e Cap. 13: Backtest Overfitting), Marcos López de Prado.
*   *Optuna: A Next-generation Hyperparameter Optimization Framework*, Akiba et al.
