# Módulo de Engenharia de Features (src/features)

## Objetivo
O módulo de Features converte séries temporais processadas (como as barras de volume ou dólar) em variáveis preditivas (features) otimizadas para modelos de Machine Learning (ML). O objetivo primário é mitigar o dilema entre estacionariedade (exigência estatística de modelos como XGBoost/Regressões) e a perda de memória (consequência da diferenciação inteira tradicional). O módulo aplica técnicas robustas da teoria de microestrutura de mercado e da metodologia de López de Prado para gerar sinais confiáveis, tudo sem introduzir o perigoso *look-ahead bias* (vazamento de informações do futuro para o passado).

## Integração com outros módulos
- **Data (`src/data`)**: Consome as barras alternativas ou ticks formatados que servem de entrada bruta.
- **Labeling (`src/labeling`) e Modeling (`src/modeling`)**: Fornece os *DataFrames* finais repletos de features que serão usados junto aos labels triplos/meta-labels para o treinamento ou inferência do modelo (`classifier`).

## Mapa de funções

### Diferenciação Fracionária (`frac_diff.py`)
- `get_weights`: Gera os pesos baseados na série de Taylor para um dado grau fracionário $d$.
- `frac_diff_ffd`: Aplica a Diferenciação Fracionária de Janela Fixa (Fixed-Width Window Fractional Differentiation) numa série temporal, limitando a queda de peso pelo limiar $tau$.
- `find_min_d`: Algoritmo iterativo que varre valores de $d$ no teste Augmented Dickey-Fuller (ADF) para achar o menor grau fracionário que atinge um p-valor (ex: 5%) aceitável de estacionariedade.

### Microestrutura de Ordem e Volume (`order_flow.py`)
- `order_flow_imbalance`: Calcula o OFI combinando as pontas do livro para derivar o desequilíbrio na agressão direcional.
- `vpin`: Estima o Volume-Synchronized Probability of Informed Trading calculando a pressão de compra/venda intrabarra.
- `volume_imbalance` e `volume_imbalance_zscore`: Normalizam a pressão de fluxo de compra vs venda de forma adaptativa.
- `volume_spread_analysis`: Emite features de VSA combinando geometria das barras (tamanho do spread e do pavio) frente à agressividade do volume.

### Filtro de Eventos CUSUM (`cusum_filter.py`)
- `cusum_filter_events`: Varre a série para encontrar desvios acumulados positivos e negativos, retornando apenas os timestamps onde esses desvios excedem um limiar (threshold).

### Indicadores e Agregação (`indicators.py`)
- Funções clássicas adaptadas: `roc`, `moving_average_distance`, `atr`, `rolling_volatility`, `garman_klass_volatility`, `rolling_moments` (skew/kurtosis), `rolling_hurst_exponent`.
- `compute_all_features`: Função de fachada (facade) que unifica as extrações e retorna o `DataFrame` consolidado com todas features macro e microestruturais.

### Normalização Temporal (`normalizer.py`)
- `rolling_zscore`: Produz um z-score expansivo para padronizar features de forma estritamente dependente do tempo $t$.
- `expanding_rank`: Gera o percentil temporal das features garantindo um intervalo [0,1].
- `normalize_features` e `validate_no_lookahead`: Aplica as escalas e audita o conjunto resultante confirmando que não há vazamento do futuro.

## Lógica geral
Uma vez que os dados puros são difíceis para as redes preverem sem overfitting em regimes temporais específicos, o módulo atua criando dezenas de colunas derivadas focando em estacionariedade e informações de micro-mercado.
A arquitetura assegura desempenho utilizando vetores pandas/numpy e validações `isclose` contra cálculos puramente iterativos (sem lookahead). O cálculo completo chama `compute_all_features`, em seguida se pode aplicar `frac_diff_ffd` e finalmente passa os retornos no `normalize_features`.

## Lógica matemática
1. **Filtro CUSUM Simétrico (Cumulative Sum)**:
   Calcula os componentes positivos $S^+$ e negativos $S^-$ das variações diárias $y_t$.
   $$ S_t^+ = \max(0, S_{t-1}^+ + y_t) $$
   $$ S_t^- = \min(0, S_{t-1}^- + y_t) $$
   Eventos são emitidos quando $S_t^+ > h$ ou $S_t^- < -h$, após o qual reseta-se $S_t$.

2. **Diferenciação Fracionária (FFD)**:
   Baseado na expansão de uma potência $(1-L)^d$, onde $L$ é o operador *lag* e $d \in [0, 1]$.
   $$ w_k = -w_{k-1} \frac{d - k + 1}{k} $$
   Os preços são multiplicados por essa longa série de pesos. Remove a tendência central mantendo a memória estrutural. O `tau` define o limiar para cortes dos pesos na série convergente e melhora a velocidade computacional.

3. **VPIN**:
   A diferença absoluta entre o Volume de Compra ($V^B$) e de Venda ($V^S$) como uma fração do Volume total numa barra de volume.
   $$ VPIN = \frac{|V^B - V^S|}{V_{total}} $$
   (Estimado intrabarra via $ \frac{C - L}{H - L} $).
