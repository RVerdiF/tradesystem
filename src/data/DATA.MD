# Módulo de Dados (src/data)

## Objetivo
O módulo de dados (`src/data`) é responsável pela ingestão, limpeza, transformação e persistência dos dados financeiros históricos e em tempo real. Ele conecta o sistema ao MetaTrader 5 (MT5), baixa e organiza as cotações (ticks e barras), aplica filtros de qualidade (limpeza de anomalias) e amostra os dados de forma que sejam apropriados para o treinamento de modelos de Machine Learning (via barras de volume ou valor financeiro, em vez de barras de tempo convencionais).

## Integração com outros módulos
- **Backtest (`src/backtest`) e Otimização (`src/optimization`)**: Fornece os dados limpos e convertidos (barras de volume/dólar) para alimentar os processos de simulação, treinamento e otimização.
- **Configurações Globais (`config/settings.py`)**: Consome as credenciais do MT5, diretórios para salvar dados no disco, e limiares (thresholds) padrão para geração de barras.
- **Features (`src/features`)**: A saída desse módulo (as barras ou ticks amostrados) serve como matéria-prima direta para a extração de métricas de microestrutura, diferenciação fracionária e cálculo de volatilidade.

## Mapa de funções

### Ingestão e Conectividade (`mt5_connector.py` / `extractor.py`)
- `MT5Connector`: Gerencia o ciclo de vida da conexão com o MetaTrader 5, implementando tentativas de reconexão (retry automático) e backoff exponencial.
- `mt5_session`: Atalho de Context Manager para abrir e fechar a conexão de forma segura.
- `extract_ticks`: Baixa o histórico granular de ticks em um intervalo de tempo específico.
- `extract_ohlc`: Baixa o histórico de barras OHLCV clássicas (por timeframes do MT5).
- `extract_ohlc_incremental`: Realiza o download apenas de dados novos posteriores ao último registro salvo.

### Processamento e Limpeza (`cleaner.py`)
- `clean_ticks`: Remove "bad ticks" usando filtros Z-score adaptativos ou absolutos para preço e volume, corrigindo dados espúrios comuns em séries intradiárias.
- `forward_fill_gaps`: Preenche lacunas temporais para evitar saltos artificiais de preço causados pela inatividade do mercado.
- `_filter_outliers_zscore`: Função auxiliar que detecta picos utilizando o desvio em relação à média móvel.

### Amostragem de Barras (`bar_sampler.py`)
- `volume_bars`: Amostra e constrói barras OHLCV a cada $V$ volume transacionado.
- `dollar_bars`: Amostra e constrói barras OHLCV a cada valor financeiro cumulativo atingido ($Preço \times Volume$).
- `tick_bars`: Amostra barras a cada contagem $N$ fixa de ticks.

### Armazenamento (`storage.py`)
- `ParquetStore`: Persiste os dados na estrutura do sistema de arquivos utilizando Apache Parquet comprimido com Snappy. Oferece as funções `save`, `load`, `exists` e `info`.

## Lógica geral
A lógica segue o pipeline de dados clássico mas estendido para finanças quantitativas. A extração conecta e baixa os dados brutos, salvando-os. Em seguida, os dados passam pela limpeza para evitar lixo nas features. Como o tempo no mercado não é uniforme (há mais negociações no começo e fim do pregão), a amostragem foca no volume e dinheiro para normalizar as distribuições estatísticas. A persistência em Parquet visa otimização do uso de I/O em leitura e escrita posterior pelas outras etapas.

## Lógica matemática
O principal processo matemático e estatístico está em `cleaner.py` (filtros) e em `bar_sampler.py` (amostragem adaptativa):

1. **Filtro Z-Score**:
   $$ Z = \frac{x - \mu_{window}}{\sigma_{window}} $$
   Os ticks em que $|Z| > \text{threshold}$ são identificados como falhas de microestrutura e descartados.

2. **Barras Alternativas (Amostragem de López de Prado)**:
   Uma barra é amostrada no momento $T$ tal que a restrição se cumpra:
   - Volume Bars: $\sum_{t=1}^{T} v_t \ge V_{threshold}$
   - Dollar Bars: $\sum_{t=1}^{T} p_t \cdot v_t \ge D_{threshold}$
   Onde $p_t$ e $v_t$ são o preço e o volume no tick $t$. Isso atenua a heterocedasticidade da variância, melhorando as propriedades i.i.d. (independentes e identicamente distribuídas) para o Meta-Labeling. O cálculo disto utiliza funções otimizadas (JIT via numba) para performar agregações iterativas numa matriz massiva.
