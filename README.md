# TradeSystem5000 🚀

O **TradeSystem5000** é uma plataforma avançada de pesquisa quantitativa e sistema de trading algorítmico (HFT/Mid-Frequency) de alta performance. O software difere dos robôs de varejo tradicionais por implementar metodologias rigorosas baseadas no *Advances in Financial Machine Learning (AFML)* de Marcos López de Prado.

Seu principal foco não é encontrar "O Indicador Mágico", mas fornecer um pipeline à prova de balas focado em: **Mitigação de Overfitting**, **Avaliação de Sinal (Meta-Labeling)** e **Gestão Adaptativa de Risco (Circuit Breakers e Sizing Fracionário)**.

---

## 🗺️ Arquitetura do Projeto

Abaixo segue a divisão detalhada da infraestrutura do software:

```text
tradesystem5000/
├── 📂 src/                     # Core Business Logic (Pipeline AFML)
│   ├── 📂 data/                # Ingestão (MT5), Limpeza (Z-Score) e Amostragem Adaptativa.
│   ├── 📂 features/            # Microestrutura de Mercado (VPIN, OFI) e Diferenciação Fracionária.
│   ├── 📂 labeling/            # Geração do Ground Truth (Tripla Barreira e Meta-Labeling).
│   ├── 📂 modeling/            # Predição Binária, Saneamento de Dados (Purging) e Bet-Sizing (Kelly).
│   ├── 📂 optimization/        # Sintonia Bayesiana (TPE via Optuna) com proteção contra Snooping (DSR).
│   ├── 📂 backtest/            # Simulador de Custos B3, Combinatorial Cross-Validation (CPCV).
│   └── 📂 execution/           # Motor Assíncrono para Live/Paper Trading com rastreabilidade total (Audit).
├── 📂 data/                    # Persistência e Amostras Locais
│   ├── 📄 tradesystem.db       # SQLite (Oráculo central: Parâmetros Otimizados, Logs de Risco e Sinais).
│   └── 📂 raw/ & processed/    # Base de dados em Apache Parquet para carregamento veloz via pyarrow.
├── 📂 config/                  # Configurações Globais (`settings.py`: Limiares e credenciais).
└── 📂 tests/                   # Bateria de Verificação usando Pytest (Data, Execution, Mocking de MT5).
```

---

## 🧠 Funcionamento Detalhado por Módulo

O sistema processa o mercado operando através de um ciclo ordenado (Pipeline), garantindo que dados sujos e informações futuras não corrompam as predições de trading ao vivo.

### 1. Ingestão e Estruturação (`src/data`)
Diferente do varejo, que plota o tempo como um cronômetro rígido (Time Bars), o módulo de Dados retira ruídos anômalos e usa Numba para criar agrupamentos estatisticamente maduros.
*   **Barras Alternativas**: A amostragem baseia-se em *Volume Accumulation* ou *Dollar Values* em vez de tempo puro. Isso acalma distorções do pregão (ex: leilão hiperativo versus hora do almoço estagnada) devolvendo *Machine Learning-friendly features*.

### 2. Engenharia de Features (`src/features`)
Transforma informações transacionais cruas em features enriquecidas matematicamente.
*   **FracDiff (Diferenciação Fracionária)**: Encontra a memória ideal sem matar a estacionariedade (Teste ADF).
*   **Event-Based CUSUM**: Detecta apenas mudanças que rompem as paredes estruturais do preço, não operando no ruído branco.
*   **Microestrutura Oculta**: Calcula o OFI (Order Flow Imbalance) e VPIN (Pressão Informada) para identificar fluxo direcional instantes antes da deflexão do preço.

### 3. Rotulagem Realista (`src/labeling`)
Prever direções (Sobe ou Desce) resulta em modelos perigosos e cegos para a volatilidade. O sistema aplica o conceito de Meta-Labeling:
*   **Triple Barrier Method (TBM)**: Simula o labirinto real de uma ordem. Um trade possui limite dinâmico de Lucro (TP), limite de Perda (SL) atrelados à volatilidade EWM e limite de tempo.
*   **Meta-Labeling**: O XGBoost não tenta adivinhar o mercado, mas atua como supervisor, prevendo "0 ou 1" sobre a chance de sucesso das intenções rudimentares de um modelo Alpha subjacente.

### 4. Validação Rigorosa e Bet Sizing (`src/modeling` & `src/backtest`)
Onde 99% das estratégias quebram. O sistema roda algoritmos robustos anti-leakage:
*   **Purging & Embargoing**: Exclui rigorosamente *K-Folds* temporais cuja sobreposição causal contamine os dados de treino com o "futuro".
*   **Kelly Sizing**: Quando as predições do Meta-Model terminam, ele filtra *probabilidades baixas* (ex: 55%). Já para 95% de chance, ele extrai a **Kelly Fraction**, mandando a boleta mais ou menos cheia consoante as estatísticas.
*   **CPCV e DSR**: Um backtest clássico traça 1 curva histórica. O Combinatorial (CPCV) traça centenas. A penalidade Bayesiana do *Deflated Sharpe Ratio (DSR)* garante que a otimização não foi mera sorte de *Data Snooping*.

### 5. Otimização Bayesiana (`src/optimization`)
*   Usa TPE (via `Optuna`) em duas fases imutáveis para tunar multiplicadores geométricos de barreiras (Fase 1) e o classificador (Fase 2). Salva inteligentemente no SQLite os melhores parâmetros usando RegEx para adaptação a contratos futuros rolados da B3 (ex: de WING26 para WIN$).

### 6. Motor Assíncrono Live (`src/execution`)
*   O cérebro live. Emprega `asyncio` para escrutinar múltiplos símbolos (ex: WIN, WDO e PETR4 simultâneos).
*   **Gestão Extrema de Risco (Circuit Breakers)**: Calcula instantaneamente PnL em tempo real; trava a conta (`HALTED`) limitando saques máximos (Max Drawdown Diário).
*   **Reconciliação Corretora**: Monitora fantasmas e latências, detectando assincronicamente quando o Broker limpou a mão via TP e reiniciando a máquina num *cool-down*.

---

## 🚀 Como Operar

O ecossistema é preparado via linha de comando ou invocação direta:

1.  **Configuração Inicial**: Valide as conexões e os limiares de Max Loss e Corretagem em `config/settings.py`.
2.  **Backtest Rápido**:
    ```bash
    python src/main_backtest.py --mode mt5 --symbol WIN$ --n-bars 10000
    ```
3.  **Tuning Extremo**:
    ```bash
    python src/optimization/run_opt.py --symbol WDO$
    ```
4.  **Bot ao Vivo / Paper Trading**:
    ```bash
    python src/main_execution.py --mode live
    ```

---

## 📊 Repositório e Auditoria (SQLite)

O sistema guarda tudo o que pensa e o que faz no arquivo nativo `data/tradesystem.db`. Onde é fácil auditar:

```sql
-- Verificar as previsões mais cegas do Meta-Model que recusaram boletações na última sessão
SELECT symbol, prob, timestamp FROM audit_signals WHERE meta_label = 0 ORDER BY prob DESC;

-- Verificar paradas compulsórias por risco estourado
SELECT * FROM audit_errors WHERE critical = 1;
```

---

## 📚 Referências Técnicas
*   *Advances in Financial Machine Learning*, Marcos López de Prado (2018).
*   *Machine Learning for Asset Managers*, Marcos López de Prado (2020).
*   "Volume-Synchronized Probability of Informed Trading" (VPIN), Easley et al.
