# Módulo de Engenharia de Features (src/features)

Este módulo é responsável pela transformação dos dados brutos em séries temporais informativas e estacionárias, preservando ao máximo a memória histórica dos preços.

A arquitetura utiliza técnicas avançadas do **Financial Machine Learning (AFML)** para mitigar o viés de look-ahead e garantir que as features sejam prontas para o treinamento de modelos de alta performance.

---

## 1. Componentes e Responsabilidades

O subsistema de features é organizado em quatro pilares fundamentais:

### 1.1 Filtragem de Eventos (`cusum_filter.py`)
*   **Filtro CUSUM Simétrico**: Implementa o algoritmo de López de Prado (Cap. 2.3.6) para detectar mudanças estruturais nos preços.
*   **Event-Based Sampling**: Em vez de barras temporais fixas, o sistema foca em pontos onde a variação cumulativa (fixa ou adaptativa via EWMA) ultrapassa um limiar (*threshold*).
*   **Performance**: Utiliza **Numba (JIT)** para o kernel de detecção, permitindo processar milhões de ticks em milissegundos.

### 1.2 Diferenciação Fracionária (`frac_diff.py`)
*   **FracDiff FFD (Fixed-Width Window)**: Transforma séries não estacionárias (preços) em estacionárias sem remover toda a memória histórica (ao contrário da diferenciação inteira).
*   **Busca de d Mínimo**: Inclui rotina `find_min_d` que utiliza o teste **ADF (Augmented Dickey-Fuller)** para encontrar automaticamente o menor valor de `d` que atinge a estacionaridade, preservando o máximo de informação (Cap. 5 AFML).

### 1.3 Indicadores Técnicos e Microestrutura (`indicators.py`)
*   **Momentum**: RSI, MACD e ROC calculados de forma *point-in-time*.
*   **Volatilidade**: ATR, Bollinger Width e Volatilidade Realizada.
*   **Microestrutura**: Implementa métricas de fluxo de ordens (Order Flow Imbalance - **OFI**) e o Volume-Synchronized Probability of Informed Trading (**VPIN**), fundamentais para prever movimentos de curto prazo (Cap. 18-19 AFML).

### 1.4 Normalização Temporal (`normalizer.py`)
*   **Z-Score Móvel**: Normaliza as features para escala [0, 1] ou média 0/std 1 usando janelas móveis para evitar vazamento de dados futuros.
*   **Expanding Rank**: Transforma features em ranks percentuais expandidos, garantindo robustez contra outliers.
*   **Validação Anti Look-Ahead**: Inclui o método `validate_no_lookahead` que verifica rigorosamente se o valor normalizado em `t` depende apenas de informações `≤ t`.

---

## 2. Fluxo de Processamento

O pipeline típico de geração de features segue esta sequência:

1.  **Amostragem**: O `cusum_filter` identifica os timestamps de interesse.
2.  **Estacionaridade**: O `frac_diff_ffd` é aplicado aos preços para gerar séries estacionárias.
3.  **Indicadores**: `compute_all_features` gera o conjunto completo de indicadores técnicos e de microestrutura.
4.  **Normalização**: O `normalize_features` ajusta a escala de todas as colunas para o meta-modelo.
5.  **Validação**: O sistema garante a integridade estatística antes de enviar para o treinamento.

---

## 3. Exemplo de Uso

```python
from src.features.indicators import compute_all_features
from src.features.normalizer import normalize_features
from src.features.frac_diff import frac_diff_ffd

# 1. Aplicar FracDiff para preservar memória
df["frac_diff_close"] = frac_diff_ffd(df["close"], d=0.4)

# 2. Gerar lote completo de features
features = compute_all_features(df)

# 3. Normalizar para o modelo secundário
features_norm = normalize_features(features, method="zscore", window=100)
```

---

## 4. Referências Técnicas
*   *Advances in Financial Machine Learning* (Cap. 2, 5, 18 e 19), Marcos López de Prado.
*   "Volume-Synchronized Probability of Informed Trading" (VPIN), Easley et al.
*   Documentação Numba para aceleração de kernels financeiros.
