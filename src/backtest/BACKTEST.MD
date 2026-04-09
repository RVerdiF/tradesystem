# Módulo de Backtest - TradeSystem5000

Este módulo é responsável pela validação rigorosa de estratégias de trading, utilizando técnicas avançadas de Machine Learning Financeiro (AFML) para evitar overfitting e garantir estimativas realistas de performance.

## 1. Arquitetura do Sistema

O backtesting no TradeSystem5000 não é apenas uma simulação de preços, mas um pipeline integrado que percorre desde a amostragem de dados até a atribuição de performance.

### Componentes Principais (`src/backtest/`)

- **`metrics.py`**: Implementa métricas estatísticas de performance.
  - **Sharpe Ratio Anualizado**: Ajustado pela frequência dos dados.
  - **Deflated Sharpe Ratio (DSR)**: Corrige o viés de seleção (data snooping) quando múltiplos testes são realizados.
  - **Drawdown & Calmar**: Medidas de risco e retorno sobre o risco máximo.
  - **Probabilidade de Ruína**: Estimativa via Monte Carlo do risco de quebra.

- **`cost_model.py`**: Modelagem realista de custos para o mercado brasileiro (B3).
  - **Custos Fixos/Variáveis**: Corretagem, emolumentos B3, taxa de liquidação e ISS.
  - **Slippage**: Modelo de impacto de mercado baseado na participação do volume médio diário.

- **`cpcv.py`**: *Combinatorial Purged Cross-Validation*.
  - Técnica fundamental para gerar múltiplos caminhos *Out-of-Sample* (OOS) sem vazamento de dados (*leakage*), essencial para meta-labeling.

- **`attribution.py`**: Análise de atribuição de performance.
  - Decompõe o retorno entre o modelo Alpha (sinal base) e o Meta-Model (filtro de confiança), permitindo medir o "Sharpe Lift" agregado pelo Machine Learning.

- **`dsr.py`**: Implementação específica do Sharpe Deflacionado baseado em López de Prado (Cap. 14).

## 2. Fluxo de Execução (Pipeline)

O orquestrador central é o `src/main_backtest.py`. O fluxo segue estas etapas:

1.  **Ingestão de Dados**: Carregamento via MT5, Yahoo Finance ou geração sintética.
2.  **Amostragem**: Opcionalmente converte barras de tempo para barras de volume ou dólar.
3.  **Features & Labeling**: Cálculo de indicadores, diferenciação fracionária (FFD) e aplicação da Tripla Barreira.
4.  **Treinamento & CV**: O Meta-Model é treinado usando CPCV para garantir que cada teste seja verdadeiramente OOS.
5.  **Simulação de Trades**: Aplicação dos tamanhos de aposta (*bet sizing*) e cálculo de retornos líquidos (pós-custos).
6.  **Relatório**: Geração de métricas de performance e análise de atribuição.

## 3. Como Executar

O backtest pode ser iniciado via CLI com diferentes fontes de dados e configurações.

### Exemplos de Comandos:

**1. Backtest com dados do MetaTrader 5 (Padrão):**
```bash
python src/main_backtest.py --mode mt5 --symbol PETR4 --interval 1h --n-bars 5000
```

**2. Backtest com dados do Yahoo Finance (Diário):**
```bash
python src/main_backtest.py --mode yfinance --symbol VALE3.SA --years 5 --interval 1d
```

**3. Teste com Barras de Volume (Volume Bars):**
```bash
python src/main_backtest.py --mode yfinance --symbol WINJ24 --volume-bars --interval 5m
```

**4. Simulação Sintética (Sanity Check):**
```bash
python src/main_backtest.py --mode synthetic
```

## 4. Referências Técnicas

A metodologia deste módulo é baseada em:
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge University Press.
