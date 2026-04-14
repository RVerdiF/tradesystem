# Módulo de Dados (src/data)

## Objetivo Detalhado
O módulo de dados (`src/data`) atua como a fundação de ingestão do TradeSystem5000. Sua responsabilidade é estabelecer conectividade de rede com a corretora via MetaTrader 5, gerenciar o download contínuo de dados brutos (ticks e barras), aplicar rotinas de saneamento de anomalias (tais como picos de latência) e reamostrar os dados utilizando uma abordagem quantitativa adaptativa baseada em volume.

Para contornar o problema de heterocedasticidade comum a séries temporais particionadas por intervalos de tempo estáticos, a reamostragem baseia-se na teoria das "barras de informação" proposta por Marcos López de Prado. Esse método encapsula transações atreladas ao volume de lotes ou volume financeiro real transacionado, melhorando as propriedades de normalidade da distribuição subjacente.

## Integração com Outros Módulos
- **Backtest (`src/backtest`) e Otimização (`src/optimization`)**: Fornece os dataframes de alta performance no formato Parquet para treinamento e testes retroativos, permitindo simulações de cenários indexados em intervalos contínuos de volume financeiro.
- **Configurações Globais (`config/settings.py`)**: Submete-se aos parâmetros de ambiente que definem diretórios locais (`RAW_DATA_DIR` e `PROCESSED_DATA_DIR`), credenciais de acesso, tempos de limite de sessão e constantes operacionais para agrupamentos.
- **Features (`src/features`)**: A saída final deste módulo serve como base padronizada e sanitizada para os algoritmos transformacionais, sendo o input primário para a computação de filtros direcionais baseados em CUSUM e diferenciações polinomiais de preços.

## Mapa de Funções e Componentes Chave

### 1. Ingestão e Conectividade (`mt5_connector.py` e `extractor.py`)
Responsável pela manutenção da camada de rede. Encapsula as requisições à API oficial do MetaTrader 5, implementando controles de paginação histórica, gerência de limites e restabelecimento automático de conexões.

- `MT5Connector`: Gerencia o ciclo de vida de rede da conexão, incorporando lógicas de *retry* em caso de falhas e progressão de *backoff* exponencial.
- `mt5_session`: Gerenciador de contexto que previne travamentos e vazamentos de recursos em caso de falhas processuais no script principal.
- `extract_ticks` e `extract_ohlc`: Procedimentos de requisição de dados brutos em massa e extrações granulares temporais contínuas.
- `extract_ohlc_incremental`: Realiza o rastreio do último timestamp registrado no armazenamento local antes de solicitar uma extração diferencial (delta) à corretora, garantindo sincronização de alta eficiência.

**Exemplo de Utilização da Conectividade:**
```python
from src.data.mt5_connector import mt5_session
from src.data.extractor import extract_ticks
from datetime import datetime

with mt5_session() as conn:
    if conn.is_connected:
        ticks_raw = extract_ticks(
            symbol="WIN$N",
            date_from=datetime(2024, 1, 1),
            date_to=datetime(2024, 6, 1)
        )
        print(f"Total de ticks transferidos: {len(ticks_raw)}")
```

### 2. Processamento e Limpeza (`cleaner.py`)
O fluxo advindo diretamente de *exchanges* apresenta imperfeições comumente descritas como ruídos microestruturais. Este módulo remove anomalias para evitar a contaminação nos cálculos subsequentes de volatilidade.

- `clean_ticks`: Coordena o processo iterativo de purificação de falhas, aplicando regras que excluem anomalias de transação e injetam valores interpolados nas janelas onde o pregão apresentou ausência anômala de cotações (gap filling).
- `forward_fill_gaps`: Atua na ausência prolongada de formação de preço. Preencher lacunas garante a robustez das operações de diferenciação fracionária (FFD), nas quais distanciamentos sintéticos podem quebrar o decaimento serial contínuo.
- `_filter_outliers_zscore`: Modela uma distribuição local adaptativa em janelas móveis para determinar cortes dinâmicos em vez de valores absolutos constantes. As cotações cujos desvios da média violam uma constante parametrizável de variação são expurgadas.

**Exemplo de Saneamento:**
```python
from src.data.cleaner import clean_ticks

cleaned_data = clean_ticks(ticks_raw, zscore_threshold=5.0)
```

### 3. Reamostragem Alternativa (`bar_sampler.py`)
Implementa as soluções de reamostragem baseadas na micro-estratificação de volume em detrimento das marcações cronológicas periódicas convencionais, restabelecendo propriedades de independência na distribuição de probabilidade.

- `volume_bars`: Agrega iterações unindo as ocorrências de tick até a transação subjacente bater o limiar de quantidade transacional de contratos ($V$).
- `dollar_bars`: Idêntico à agregação por volume, ponderado pelo valor do respectivo fluxo na cotação. Elimina as discrepâncias que contratos futuros podem originar frente a grandes ciclos de deflação/inflação de preços a longo prazo.
- `tick_bars`: Um procedimento de acumulação contínua onde a consolidação da barra ocorre após uma contagem finita de negociações executadas.

**Exemplo de Amostragem Compilada (Numba JIT):**
```python
from src.data.bar_sampler import dollar_bars

bars_df = dollar_bars(
    ticks=cleaned_data,
    threshold=50_000_000,
    price_col="last",
    volume_col="volume"
)
```

### 4. Armazenamento e Persistência (`storage.py`)
O sistema gerencia requisições assíncronas que exigem baixo grau de latência na manipulação de matrizes gigantes. O formato Parquet possibilita o armazenamento otimizado via layout colunar.
- `ParquetStore`: Classe Singleton encarregada da serialização e deserialização nos caminhos `data/raw/` e `data/processed/`. O formato acarreta a compressão (algoritmo *Snappy*), otimizando radicalmente a utilização de hardware em leitura sequencial.

## Paradigma do Processamento Sequencial

O encadeamento do fluxo de informações atua da seguinte forma:
1. Ingestão paralela via camada de *sockets* gerenciada no terminal Broker local.
2. Descarga temporária inalterada em ambiente estático (`data/raw`).
3. Avaliação dinâmica baseada em *Z-Scores* adaptativos descartando lacunas isoladas.
4. Transformação da lista purificada usando operações vetoriais em arrays `numpy` reamostrando sob janelas baseadas puramente no montante transacionado.
5. Emissão do novo `DataFrame` validado (*Processed*), garantindo normalidade probabilística para o repasse à inteligência da arquitetura secundária preditiva.

## Especificação Matemática

### Filtro Adaptativo para Anomalias (Z-Score)
Para evitar que registros isolados oriundos de problemas de processamento de fila na corretora causem impacto na estimativa de Volatilidade Diária contínua, os eventos de mercado são filtrados ativamente.
Assumindo a série dos retornos dos preços observados ($p_t$) contida em uma janela flutuante ($w$) definimos a média local $\mu_{w}$ e o respectivo desvio padrão local $\sigma_{w}$.
O escore transacional $Z$ atrelado no passo unitário $t$ afere a dispersão normalizada:
$$ Z_t = \frac{p_t - \mu_{t-w:t}}{\sigma_{t-w:t}} $$
Nos casos de ocorrência extrema $|Z_t| > C$ (sendo $C$ a definição configurável base para anomalias, geralmente fixado em 4.0 ou superior), estipulamos que o registro carece de integridade estocástica e retira-se o index correspondente antes da reamostragem.

### Critério de Amostragem (Dollar Bars)
Para devolver à base temporal de estudo o caráter estacionário, ignoramos a partição fixa ($T \to [5, 10, 15]$ minutos) adotando marcos de preenchimento flutuantes associados diretamente à densidade negocial contínua da massa de *traders*.
Dada uma negociação em $t$ executada sob uma taxa de cotação de $p_t$ equivalendo à transação parcial $v_t$.
Um ciclo delimitado por sua abertura no tempo de índice contínuo $t_0$ declarará uma nova barra OHLC no fechamento subsequente no exato vetor iterativo $T$ em que satisfizer pela primeira vez a incompletude monetária global de um limiar $D_{threshold}$:
$$ \sum_{t=t_0}^{T} (p_t \cdot v_t) \ge D_{threshold} $$
Nesse limite recém atingido $T$, registra-se um vetor contendo valores empíricos máximos e mínimos operados (*High*, *Low*), retornando os agregados sem a distorção temporal em que volumes ausentes causam perturbações estatísticas para as equações matemáticas subsequentes.