# Módulo de Engenharia de Features (src/features)

## Objetivo Detalhado
Modelos de Machine Learning puros, como XGBoost ou Random Forests, são péssimos atuando sobre os preços secos do mercado (não-estacionários e sem referencial estático). O objetivo supremo deste módulo é criar **transformações matemáticas de características** que revelem os padrões ocultos das negociações de forma digerível para os algoritmos. Aqui é resolvido o antigo dilema do Quant Finance: se você fornece o preço puro, o modelo decora tendências passadas e falha em mercados novos (ausência de estacionariedade); se você o ensina apenas retornos puros (diferenciação inteira, $x_t - x_{t-1}$), você joga fora toda a memória histórica do patamar de preços (apaga o impacto estrutural de suportes e resistências antigas).

Este módulo foca na conversão rigorosa das barras limpas da camada `src/data` aplicando Diferenciação Fracionária para reter memória, Filtro CUSUM para amostragem tática em eventos pontuais, e desdobramento focado do desequilíbrio profundo do livro de negociações através de lógicas de VPIN e VSA (Volume Spread Analysis). O output desta operação transforma tabelas finas de 5 colunas em matrizes imensas e extremamente informativas com proteção absoluta contra o enviesamento analítico (look-ahead bias).

## Integração com Outros Módulos
- **Data (`src/data`)**: Consome nativamente as barras financeiras (Dollar Bars) que já suavizaram as disparidades temporais das horas de almoço da B3 e leilões.
- **Labeling (`src/labeling`) e Modeling (`src/modeling`)**: Projeta a mega-tabela $X$ contendo a "física do mercado" no exato momento $t$, cujos *labels* de sucesso/fracasso triplo serão acoplados lado-a-lado para o posterior ensinamento supervisionado no `classifier.py`.

## Mapa de Funções e Componentes Chave

### 1. Diferenciação Fracionária e Retenção de Memória (`frac_diff.py`)
Muitos fundos falham por basear Machine Learning em *log returns* (retornos isolados a cada minuto). Eles tornam a série perfeitamente estacionária (o teste Dickey-Fuller ama), mas perdem 100% da informação relacional de quanto o ativo caiu na semana anterior. Diferenciar "um pouco menos" de 1 preserva a memória.

- `get_weights_ffd`: Função pura auxiliar que constrói a progressão infinita de pesos decrescentes seguindo o teorema de Taylor para a potência racional.
- `find_min_d`: Algoritmo de busca por "força bruta" inteligente. Ao invés do humano chutar a fração $d$, essa função itera a diferenciação testando via `adfuller` até obter o exato menor valor que atinja um limite p-valor (como $p \le 0.05$) garantindo máxima retenção de memória com o selo de estacionariedade mínima exigida pelo ML.
- `frac_diff_ffd`: O coração do cálculo. Diferencia a série e corta a cauda que ficaria muito pesada usando o limite `tau`, barateando os custos de memória de cálculo no pandas.

**Snippet de Estacionariedade Parcial:**
```python
from src.features.frac_diff import find_min_d, frac_diff_ffd

# 1. Encontra a fração perfeita iterando d de [0, 1] que torna
# a série estacionária ao teste ADF sem destruir muita memória
d_optimal = find_min_d(bars_df["close"], p_value_threshold=0.05)

# 2. Transforma o preço para a representação super-poderosa
# de Memória Longa
df["frac_diff"] = frac_diff_ffd(bars_df["close"], d=d_optimal, tau=1e-4)
```

### 2. Filtro de Eventos de Supresa - CUSUM (`cusum_filter.py`)
Sempre que você manda um algoritmo chutar previsões em barras monótonas ou mercados consolidados minúsculos, as taxas de erro explodem. O CUSUM (Cumulative Sum) foi adotado para "acordar" o robô apenas quando os ruídos formarem ondas reais direcionalmente surpreendentes em relação à volatilidade intrínseca da fase atual.
- `cusum_filter_events`: Computa ativamente dois funis contínuos — um acumulando surpresas negativas e outro surpresas positivas. Corta todos os retornos e gera timestamps onde, e apenas onde, o sinal quebrou a barreira estática ou adaptativa baseada em histerese.

### 3. A Arte Oculta do Micro-Mercado (`order_flow.py`)
No mundo intradiário do HFT, médias móveis estão sempre atrasadas; o fluxo de ordens (Book) é o único indicador antecedente confiável.
- `order_flow_imbalance` (OFI): Mede o desbalanceamento não pelo preço, mas contabilizando o deslocamento das pás de agressão de compradores vs. vendedores da fita.
- `vpin`: Computa a Probabilidade de *Informed Trading* em volume simultâneo.

### 4. Orquestrador Holístico de Atributos (`indicators.py`)
O armazém aglomerador que empacota dezenas de indicativos tradicionais repensados para uso quantitativo, fugindo de sinais simplórios (não existe if RSI > 70 = venda).
- Possui agregadores macro (`roc`, `atr`, `garman_klass_volatility`, `rolling_moments` — Skew e Curtose da distribuição real de preços em dadas janelas) e a grande função agregadora (Facade Pattern) `compute_all_features`.

**Snippet de Computação Massiva Encadeada:**
```python
from src.features.indicators import compute_all_features
from src.features.normalizer import normalize_features

# Computa as >15 colunas complexas baseadas nos microdados de OHLCV
raw_features_df = compute_all_features(bars_df, config=my_cfg)

# Processa Z-Scores contínuos (expansão point-in-time e rankeamento temporal)
# garantindo que o DataFrame inteiro esteja escalado [-3 a +3] para as
# árvores XGBoost consumirem.
norm_features_df = normalize_features(raw_features_df, method="zscore", window=100)
```

### 5. Estrita Proibição à Viagem no Tempo (`normalizer.py`)
Se a normalização utilizar o `MinMaxScaler` ou `StandardScaler` clássicos globais em toda a série do backtest, a barra do ano de 2021 saberá qual foi o limite máximo alcançado em 2024, e o Meta-Model trapaceará.
- `rolling_zscore` e `expanding_rank`: O único antídoto contra look-ahead. Usa apenas o $t_0 - 1$ para escalar o momento presente.
- `validate_no_lookahead`: Uma função de autoauditoria que atua como advogado do diabo, recomputando índices de ponta a ponta e assegurando erro em caso de corrupção linear.

## A Física Matemática do Micro-Mercado (OFI & VPIN)

### Diferenciação Fracionária FFD de López de Prado
Dado um grau racional de diferenciação $d \in (0, 1)$, derivamos sua série expansiva de convergência baseada no atraso serial das cotações (Lag operator $L$). O vetor iterativo dos pesos $w$ é definido por:
$$ w_k = -w_{k-1} \frac{d - k + 1}{k} \quad \text{onde } w_0 = 1 $$
Aplicando nos preços para criar as séries modificadas $\tilde{P}_t$:
$$ \tilde{P}_t = \sum_{k=0}^{\infty} w_k P_{t-k} $$
Ao jogar fora a cauda cujo erro dos pesos acumulados em declínio cruzem o sub-limiar minúsculo $\tau$ (e.g. $10^{-5}$), o modelo processa num vetor restrito muito mais performático para séries contendo milhões de registros (FFD: Fixed-Width Window Fractional Differentiation).

### Volume-Synchronized Probability of Informed Trading (VPIN)
A métrica captura assimetrias tóxicas causadas por agressões não-verbais institucionais dentro das próprias velas de volume trocadas. A estimativa intradiária não depende das exaustivas e instáveis *Order Books*, usando ao invés a cinemática relativa intrabarra:
A fração bruta direcional do fluxo de compradores $V_B$ versus vendedores $V_S$ é estimada sob o gradiente:
$$ \Delta_{P} = \frac{\text{Close} - \text{Low}}{\text{High} - \text{Low}} $$
$$ V^B_t \approx \Delta_P \cdot V_{total, t} $$
O VPIN calcula o choque relativo do desbalanço para com a profundidade global fluida:
$$ VPIN = \frac{ \sum |V^B_{\tau} - V^S_{\tau}| }{ \sum V_{total, \tau} } $$
Valores acentuados e saltos repentinos no VPIN demonstram estatisticamente alta probabilidade matemática de que as mãos institucionais ("informados") estão enxugando ativamente liquidez pregressa numa ponta específica, antevendo pânicos ou disparadas iminentes e permitindo ao robô desativar a agressão contramão de liquidez.

### O CUSUM Simétrico (Filtro Direcional Oculto)
Usado para identificar não oscilações de tempo, mas viradas de direção genuínas do fluxo (Surpresas que geram Eventos):
$$ S_t^+ = \max \left( 0, S_{t-1}^+ + y_t - E_{t-1}[y_t] \right) $$
$$ S_t^- = \min \left( 0, S_{t-1}^- + y_t - E_{t-1}[y_t] \right) $$
Onde $y_t$ representa retornos numéricos de alta frequência e os eventos são guardados na memória unicamente no instante exato $t$ que resulta no rompimento:
$$ t_{evento} : S_t^+ > h_t \quad \lor \quad S_t^- < -h_t $$
Após isso, ambas as parcelas re-agregadoras colapsam a 0 e o ciclo de contagem começa de novo sobre os retornos. Esse sistema purifica em 70 a 90% todo o "ruído branco" perigoso das barras convencionais em períodos de acumulação do Índice e Dólar.
