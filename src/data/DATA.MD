# Módulo de Dados (src/data)

## Objetivo Detalhado
O módulo de dados (`src/data`) é o pilar fundamental do TradeSystem5000, servindo como a porta de entrada para a realidade empírica do mercado financeiro. A qualidade de qualquer modelo preditivo é diretamente limitada pela integridade e formatação da sua base de dados de treino (*garbage in, garbage out*). Este módulo assume a responsabilidade de conectar a plataforma de forma confiável à corretora via MetaTrader 5, orquestrar o fluxo massivo de extração de milhões de ticks por segundo, realizar saneamento estatístico para remover distorções criadas por microestrutura e ineficiências de rede da B3 e reamostrar estes dados de uma maneira puramente quantitativa e adaptativa.

Para alcançar estacionariedade sem destruição de sinal e contornar a heterocedasticidade intrínseca do tempo (mercados andam de lado em horários de almoço e disparam na abertura, resultando em dados onde a variância não é constante), o módulo descarta o uso de cronômetros fixos para a construção dos tradicionais *candlesticks*. Em vez de barras temporais de 5 minutos, ele constrói as "barras informativas", propostas por López de Prado, empacotando os trades de acordo com o volume de contratos ou volume financeiro real transitado.

## Integração com Outros Módulos
- **Backtest (`src/backtest`) e Otimização (`src/optimization`)**: Estes módulos rodam simulações sobre um subconjunto de anos passados e requerem os DataFrames Parquet ultra-rápidos que as rotinas de extração do `data` organizam, permitindo o treinamento do classificador com barras já consolidadas em limites customizáveis de volume/dólar.
- **Configurações Globais (`config/settings.py`)**: As definições arquiteturais — como caminhos para os discos (RAW_DATA_DIR e PROCESSED_DATA_DIR), thresholds que decidem o tamanho da barra amostrada, e os credenciais para login e timeout no MT5 — fluem daqui.
- **Features (`src/features`)**: A saída final deste módulo é o sinal de *start* para a computação matemática do modelo. Uma série temporal de preços devidamente expurgada de falsas variações (spikes de rede) é consumida para que o `cusum_filter` possa rastrear e engatilhar micro-mudanças baseadas exclusivamente na pressão verdadeira do preço.

## Mapa de Funções e Componentes Chave

### 1. Ingestão e Conectividade (`mt5_connector.py` e `extractor.py`)
Responsável pela rede. A API do MetaTrader 5 é escrita em C nativo com bindings para Python e sofre por não ser intuitiva; este sub-módulo cria um invólucro (Wrapper) robusto, tratando quedas, desconexões e paginação histórica.

- `MT5Connector`: Gerencia o ciclo de vida e a memória da conexão com o MetaTrader 5, implementando tentativas de reconexão (retry automático) e backoff exponencial.
- `mt5_session`: Um contexto seguro que garante que se algo falhar gravemente no motor do bot, a conexão não fique órfã em background drenando RAM.
- `extract_ticks` e `extract_ohlc`: Trabalhadores rústicos que buscam em blocos grandes o histórico cru para o disco ou requisições `live` de pequenas quantidades.
- `extract_ohlc_incremental`: Verifica no cache do disco onde a série temporal anterior parou, fazendo sync *diff* na corretora e trazendo apenas o delta de tempo necessário, evitando baixar os últimos 10 anos a cada novo start.

**Snippet de Uso da Ingestão:**
```python
from src.data.mt5_connector import mt5_session
from src.data.extractor import extract_ticks
from datetime import datetime

# Uso limpo do contexto para evitar memory leaks
with mt5_session() as conn:
    if conn.is_connected:
        # Puxa dados hipergranulares num intervalo (milhões de pontos)
        ticks_raw = extract_ticks(
            symbol="WIN$N",
            date_from=datetime(2024, 1, 1),
            date_to=datetime(2024, 6, 1)
        )
        print(f"Buscados {len(ticks_raw)} ticks da bolsa.")
```

### 2. Processamento e Limpeza (`cleaner.py`)
Módulo responsável pela sanitização dos dados "raw". Os dados fluindo diretamente do Exchange carregam ruído (*spikes* provocados pela latência do roteamento onde uma ordem a um centavo passa num milissegundo vazio) que destrói as médias de volatilidade.

- `clean_ticks`: Atua como orquestrador dos filtros em série para a tabela bruta, invocando limpeza progressiva de discrepâncias de preço até interpolações de tempo nulo.
- `forward_fill_gaps`: Em janelas inativas (como leilões interrompidos ou ativos ilíquidos), injeta artificialmente os últimos valores sabidos. Em algoritmos de diferenciação fracionária (FFD), saltos de 4 horas sem cotação distorcem enormemente o decaimento polinomial.
- `_filter_outliers_zscore`: Em vez de definir limites fixos imbecis (ex: "remova quedas de 200 pontos no índice"), a função normaliza a variação ao longo do dia em uma distribuição, detectando e exterminando os dados que fogem à anomalia local permitida.

**Snippet de Uso de Limpeza:**
```python
from src.data.cleaner import clean_ticks

# Ticks brutos -> Ticks normalizados sem anomalias de rede
cleaned_data = clean_ticks(ticks_raw, zscore_threshold=5.0)
```

### 3. Amostragem de Barras Alternativas (`bar_sampler.py`)
Aqui repousa o cerne do AFML. Ticks individuais são ruidosos demais e candlesticks temporais (5 minutos) mascaram o comportamento humano; uma barra de 5 minutos da manhã vale exponencialmente mais do que uma do final do dia.

- `volume_bars`: Adiciona tick por tick em um acumulador e cria uma fatia (um "O,H,L,C") no instante $T$ em que a quantidade transacionada bate no limite do parâmetro $V$.
- `dollar_bars`: Idêntico ao volume, mas multiplica pelo preço atual na hora da troca. Especialmente útil para não precisar reajustar o tamanho da amostra depois de fortes correções (1 contrato a 130 mil pontos pesa financeiramente diferente de 1 contrato a 90 mil pontos).
- `tick_bars`: Um acumulador que corta cada vez que exatos 1.000 negócios foram concretizados, não importando os tamanhos individuais do pacote.

**Snippet de Uso da Amostragem (Compilada por JIT Numba):**
```python
from src.data.bar_sampler import dollar_bars

# Gera a versão limpa do dataset de estudo onde cada nova barra OHLC
# representará exatamente R$ 50 milhões trocados entre investidores.
bars_df = dollar_bars(
    ticks=cleaned_data,
    threshold=50_000_000,
    price_col="last",
    volume_col="volume"
)
```

### 4. Armazenamento Eficiente (`storage.py`)
A modelagem consome muita I/O e a memória precisa de performance extrema. CSVs e arquivos SQL nativos não comportam centenas de milhões de linhas facilmente sob Python.
- `ParquetStore`: Classe Singleton que gerencia a árvore diretorial `data/raw/` e `data/processed/`. O formato colunar com compressão *Snappy* reduz em até 80% o uso de disco ao mesmo tempo que carrega arquivos multimegabytes quase instantaneamente.

## A Lógica Geral da Transformação de Ticks

O paradigma de dados funciona como um rio unidirecional:
1. Começa caótico recebendo em rajadas via sockets de `extractor.py` diretamente do Broker.
2. É descarregado temporariamente na área `/data/raw` como cópia de salvaguarda inalterada via `.parquet`.
3. Passa por peneiras e filtros Z-Score adaptativos que retiram grãos podres (ordens corrompidas e gaps vazios).
4. É agregado usando vetores compilados com `numba` onde o relógio não marca as horas, mas marca sim a quantia de moedas gastas pelos players institucionais.
5. Torna-se então a base *Processed*, um `DataFrame` polido de barras que ostenta a valiosa propriedade I.I.D. (Dados Independentes e Identicamente Distribuídos) pronta para os preditores e árvores de decisão que a seguirão nos próximos passos do pipeline.

## A Lógica Matemática Rigorosa

### Filtro Dinâmico Z-Score para Outliers (Spikes de Preço)
Para evitar corromper o cálculo de Volatilidade Diária com preços espúrios (bad-ticks), processamos os sinais através de uma padronização normal móvel:
A série do histórico numérico dos retornos dos preços ($p_t$) numa janela móvel ($w$) assume uma média $\mu_{w}$ e desvio padrão $\sigma_{w}$.
O desvio estandardizado para um tick momentâneo em $t$ é:
$$ Z_t = \frac{p_t - \mu_{t-w:t}}{\sigma_{t-w:t}} $$
Se $|Z_t| > C$ (onde $C$ geralmente é 4 ou 5 desvios padrões limitados pela configuração de outliers), nós entendemos estatisticamente que a probabilidade da transação ter ocorrido de fato e se mantido é infinitesimal (cravando ruído sintético) e dropamos a respectiva linha de índice da matriz do pandas.

### A Amostragem Adaptativa (López de Prado's Dollar Bars)
Para restaurar a normalidade de uma série temporal, as barras não podem sofrer fechamentos indexados por tempo ($T \to [5, 10, 15...]$ min), e sim pelas flutuações e ritmos do mercado.
Seja a transação no tick $t$ portando o preço negociado $p_t$ e o volume de ativos trocados $v_t$.
Uma barra iniciada num tempo anterior $t_0$ será considerada completa quando o índice atual $T$ satisfazer, pela primeira vez, a incompletude monetária global estipulada como limiar $D_{threshold}$:
$$ \sum_{t=t_0}^{T} (p_t \cdot v_t) \ge D_{threshold} $$
Nesse limite recém atingido $T$, é forjado um corte que resultará num vetor representativo do *micro-regime*. Registra-se para o modelo a abertura em $t_0$, e se calcula os tradicionais alvos empíricos *High* e *Low* do subconjunto entre esses pontos estocásticos, fechando a barra no respectivo final do acúmulo no mercado, resultando num formato limpo para os inputs preditivos das dezenas de colunas do `features`.
