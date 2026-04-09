# Módulo de Dados (src/data)

Este módulo é responsável por todo o ciclo de vida dos dados financeiros, desde a ingestão via MetaTrader 5 (MT5) até a amostragem de barras alternativas para modelagem.

A arquitetura segue os princípios de robustez, performance e reprodutibilidade, baseando-se em técnicas avançadas de Financial Machine Learning.

---

## 1. Componentes e Responsabilidades

O módulo é composto pelos seguintes arquivos principais:

### 1.1 Ingestão e Conectividade
*   **`mt5_connector.py`**: Gerencia a conexão e autenticação com o terminal MetaTrader 5. Implementa um `MT5Connector` (context manager) com lógica de **retry automático** e **backoff exponencial**, garantindo resiliência contra falhas de rede.
*   **`extractor.py`**: Realiza o download de dados históricos (ticks e barras OHLC). Suporta **download incremental**, baixando apenas os dados novos posteriores ao último registro local, otimizando o tempo de sincronização.

### 1.2 Armazenamento
*   **`storage.py`**: Implementa a classe `ParquetStore` para persistência de alta performance. Utiliza o formato **Parquet com compressão Snappy**, organizando os dados no disco por símbolo e tipo, permitindo leitura e escrita eficientes de grandes volumes.

### 1.3 Processamento e Limpeza
*   **`cleaner.py`**: Pipeline de sanitização de dados. Inclui:
    *   **Remoção de Spikes**: Filtros estatísticos (Z-score) para remover "bad ticks" e outliers de preço.
    *   **Preenchimento de Lacunas**: Técnicas de *forward fill* e interpolação para tratar períodos sem negociação.
    *   **Validação OHLC**: Garante a integridade lógica das barras (ex: High >= Low, Open/Close dentro do range).

### 1.4 Amostragem de Barras (Bar Sampling)
*   **`bar_sampler.py`**: Transforma ticks em barras informativas, reduzindo a heterocedasticidade dos retornos. Implementa as técnicas descritas por **Marcos López de Prado** (*Advances in Financial Machine Learning*, Cap. 2):
    *   **Tick Bars**: Amostragem por contagem fixa de transações.
    *   **Volume Bars**: Amostragem por volume acumulado (ativos trocados).
    *   **Dollar Bars**: Amostragem por valor financeiro acumulado (preço × volume).
    *   *Nota: O processamento é acelerado via **Numba (JIT)** para máxima performance em grandes volumes de ticks.*

---

## 2. Fluxo de Dados

O fluxo típico de operação no módulo é:

1.  **Conexão**: `mt5_connector` estabelece a sessão.
2.  **Extração**: `extractor` baixa os ticks/barras brutas.
3.  **Armazenamento (Raw)**: `storage` salva os dados brutos na pasta `data/raw`.
4.  **Limpeza**: `cleaner` remove ruídos e valida a integridade.
5.  **Amostragem**: `bar_sampler` gera barras informativas a partir dos ticks limpos.
6.  **Armazenamento (Processed)**: `storage` salva os dados finais na pasta `data/processed`.

---

## 3. Exemplo de Uso

```python
from src.data.mt5_connector import mt5_session
from src.data.extractor import extract_ticks
from src.data.cleaner import clean_ticks
from src.data.bar_sampler import dollar_bars

# 1. Extrair e Limpar
with mt5_session() as conn:
    ticks_raw = extract_ticks("PETR4", date_from=datetime(2023, 1, 1), date_to=datetime.now())
    ticks_clean = clean_ticks(ticks_raw)

# 2. Gerar Barras de Dólar
bars = dollar_bars(ticks_clean, threshold=1_000_000)

# 3. Salvar
from src.data.storage import ParquetStore
store = ParquetStore()
store.save(bars, symbol="PETR4", data_type="dollar_bars")
```

---

## 4. Referências Técnicas
*   *Advances in Financial Machine Learning* (Cap. 2), Marcos López de Prado.
*   Documentação da API MetaTrader 5 para Python.
