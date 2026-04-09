# TradeSystem5000 🚀

Sistema de trading algorítmico de alta performance baseado em **Financial Machine Learning (AFML)**, seguindo rigorosamente as metodologias de **Marcos López de Prado**.

O TradeSystem5000 não é apenas um bot de execução, mas uma plataforma de pesquisa quantitativa que utiliza **Meta-Labeling** para filtrar sinais e gerenciar o risco de forma estatística, mitigando o sobreajuste (overfitting) e a não-estacionaridade dos dados financeiros.

---

## 🗺️ Mapa do Projeto (File Tree)

```text
tradesystem5000/
├── 📂 .agent/                  # Configurações do Framework Superpowers
│   ├── 📂 rules/               # Regras de operação (Plan gate, TDD, Review)
│   ├── 📂 skills/              # Habilidades especializadas (TDD, Python Automation)
│   └── 📂 workflows/           # Fluxos de trabalho padronizados
├── 📂 src/                     # Core Business Logic (Pipeline AFML)
│   ├── 📂 data/                # Ingestão (MT5), Limpeza e Amostragem (Informative Bars)
│   ├── 📂 features/            # FracDiff, Filtro CUSUM, Microestrutura (OFI, VPIN)
│   ├── 📂 labeling/            # Triple Barrier Method, Meta-Labeling, Volatilidade EWMA
│   ├── 📂 modeling/            # Meta-Model (XGBoost), Purged CV, Kelly Bet Sizing
│   ├── 📂 backtest/            # Validação CPCV, Deflated Sharpe Ratio (DSR), Custos B3
│   ├── 📂 optimization/        # Otimização Bayesiana (Optuna), DSR Validation
│   └── 📂 execution/           # Async Engine, Order Manager, Risk Manager
├── 📂 data/                    # Persistência e Amostras
│   ├── 📄 tradesystem.db       # SQLite (Fonte de Verdade: Parâmetros, Sinais, Auditoria)
│   ├── 📂 raw/                 # Dados brutos em Parquet (Ticks/Bars)
│   └── 📂 processed/           # Dados processados para treinamento
├── 📂 config/                  # Configurações Globais
│   └── 📄 settings.py          # Limites de risco, credenciais e parâmetros
├── 📂 tests/                   # Camada de Verificação e QA
│   ├── 📂 test_data/           # Testes de integridade de dados e conectores
│   ├── 📂 test_execution/      # Testes de fluxo de ordens e travas de risco
│   └── ...                     # Testes específicos por módulo
└── 📂 artifacts/               # Rastro de Auditoria e Planejamento
    └── 📂 superpowers/         # Planos, reviews e resultados persistidos
```

---

## 🧠 Funcionamento Detalhado (O Pipeline AFML)

O sistema opera em um fluxo cíclico que transforma ticks brutos em decisões de execução com alta confiança estatística.

### 1. Ingestão e Amostragem de Informação (`src/data`)
Diferente do trading tradicional baseado em barras de tempo (que são heterocedásticas), o TradeSystem5000 utiliza **Bars de Informação**:
*   **Dollar Bars**: Amostradas quando um valor financeiro fixo é trocado. Isso recupera a normalidade estatística dos retornos.
*   **Volume Bars**: Amostradas por quantidade de ativos.
*   **Tick Bars**: Amostradas por número de transações.

### 2. Estacionaridade e Microestrutura (`src/features`)
Modelos de ML falham em dados não-estacionários (preços). Resolvemos isso com:
*   **FracDiff (Fixed-Width Window)**: Remove a tendência mas mantém a "memória" histórica, essencial para modelos preditivos.
*   **Microestrutura**: Cálculo de **VPIN** (Volume-Synchronized Probability of Informed Trading) e **OFI** (Order Flow Imbalance) para detectar desequilíbrios no fluxo de ordens.
*   **Filtro CUSUM**: Detecta mudanças estruturais nos preços para disparar a amostragem de eventos.

### 3. Rotulagem Avançada (`src/labeling`)
Utilizamos o **Triple Barrier Method (TBM)**:
1.  **Take Profit (Horizontal)**: Alvo de lucro dinâmico baseado na volatilidade.
2.  **Stop Loss (Horizontal)**: Limite de perda dinâmico.
3.  **Time Barrier (Vertical)**: Saída compulsória após N barras para evitar capital preso.
*   **Meta-Labeling**: O modelo não prevê a direção (Alpha), mas sim se o sinal Alpha atingirá o TP antes do SL (Binary 0/1).

### 4. Validação Cruzada Purged & Embargoed (`src/modeling`)
Para evitar o *Data Leakage* (vazamento de dados), o sistema implementa:
*   **Purging**: Remoção de observações de treino que se sobrepõem temporalmente ao teste.
*   **Embargo**: Intervalo de segurança após o teste para garantir que a autocorrelação serial não contamine o modelo.
*   **Bet Sizing**: Dimensionamento via **Kelly Criterion**, onde a exposição é proporcional à probabilidade de acerto do meta-modelo.

### 5. Backtesting Rigoroso (`src/backtest`)
*   **Combinatorial Purged CV (CPCV)**: Gera múltiplos caminhos *Out-of-Sample* para testar a estratégia em diversos regimes.
*   **Deflated Sharpe Ratio (DSR)**: Corrige o Sharpe Ratio pelo viés de seleção, informando se o resultado é real ou apenas "sorte" de testar muitas variações.

---

## 📊 Banco de Dados (SQLite)

O sistema utiliza o `data/tradesystem.db` como o oráculo central para persistência e auditoria.

| Tabela | Função |
|---|---|
| `optimized_params` | Melhores hiperparâmetros (Optuna) validados por DSR. |
| `audit_signals` | Todos os sinais gerados, probabilidades e frações de Kelly. |
| `audit_orders` | Logs detalhados de execução (Paper e Live) com Tickets MT5. |
| `audit_errors` | Registro de exceções críticas e interrupções de risco. |

### Consultas de Auditoria Rápidas

```bash
# Sinais com maior probabilidade de acerto recente
sqlite3 data/tradesystem.db "SELECT symbol, prob, timestamp FROM audit_signals WHERE meta_label=1 ORDER BY prob DESC LIMIT 5;"

# Verificar se houve interrupção por Risk Manager (Circuit Breaker)
sqlite3 data/tradesystem.db "SELECT * FROM audit_errors WHERE critical=1 ORDER BY timestamp DESC;"
```

---

## 🚀 Como Operar

1.  **Configuração**: Edite `config/settings.py` para definir seus limites de risco (Max Daily Loss, Max Drawdown).
2.  **Ingestão**: `python src/main_backtest.py --mode mt5 --symbol WINZ25 --n-bars 10000`
3.  **Otimização**: `python src/optimization/run_opt.py --symbol WINZ25`
4.  **Execução**: `python src/main_execution.py --mode paper` (Simulado) ou `--mode live` (Real).

---

## 📚 Referências Técnicas
*   *Advances in Financial Machine Learning*, Marcos López de Prado (2018).
*   *Machine Learning for Asset Managers*, Marcos López de Prado (2020).
