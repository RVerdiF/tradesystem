# Módulo de Engenharia de Features (src/features)

## Objetivo Detalhado
Modelos de aprendizado de máquina necessitam de predições que operem sobre representações matriciais estruturadas (features) em vez de fiarem-se unicamente nas cotações brutas do mercado financeiro. O objetivo primário deste módulo é derivar e escalar essas representações, aplicando transformações que mitigam a falta de referencial estacionário das séries de preços puros e contornam o problema da perda de memória (observado em diferenciações inteiras clássicas, ex: log-returns simples de primeira ordem).

O módulo executa métodos desenvolvidos pela literatura de microestrutura de mercado (como AFML) e utiliza vetores computacionais para extrair indicações prévias do balanço do fluxo de ordens, aplicar Diferenciação Fracionária e indexar eventos transacionais via Filtros CUSUM, tudo isso garantindo a estrita proibição matemática de qualquer vazamento temporal das observações futuras (look-ahead bias) para o período validado.

## Integração com Outros Módulos
- **Data (`src/data`)**: Ingestão primária da matriz convertida em barras de informação (Dollar/Volume Bars) que padroniza o horizonte de formação das distribuições subjacentes.
- **Labeling (`src/labeling`) e Modeling (`src/modeling`)**: Transmite a tabela preditiva independente $X$ (as *features* formatadas), perfeitamente dimensionadas sob os *labels* triplos e dinâmicos para a modelagem supervisionada do classificador em XGBoost.

## Mapa de Funções e Componentes Chave

### 1. Diferenciação Fracionária e Retenção de Memória (`frac_diff.py`)
Muitos sistemas falham devido à adoção de retornos estáticos em que o valor do patamar estrutural de suportes ou resistências antigas ($t - 100$) é descartado perante as cotações de sub-minuto atuais ($t$). A Diferenciação Fracionária adota fatores não inteiros de derivada temporal.

- `get_weights_ffd`: Função utilitária estática para inicialização expansiva do decaimento das razões dos pesos fracionados segundo a série de Taylor.
- `find_min_d`: Rotina de busca estrutural. Itera o multiplicador $d$ sobre o teste Augmented Dickey-Fuller (ADF) computando e verificando iterativamente o exato menor diferencial admissível para atingir a estacionariedade imposta (ex: P-valor de $\le 0.05$), otimizando a restrição.
- `frac_diff_ffd`: Desempenha o cálculo recursivo central de Diferenciação Fracionária de Janela Fixa (Fixed-Width Window). Trunca o polinômio pela precisão sub-decimal `tau`, prevenindo excessos computacionais iterativos na modelagem da biblioteca pandas.

**Exemplo de Diferenciação Estacionária (FFD):**
```python
from src.features.frac_diff import find_min_d, frac_diff_ffd

d_optimal = find_min_d(bars_df["close"], p_value_threshold=0.05)
df["frac_diff"] = frac_diff_ffd(bars_df["close"], d=d_optimal, tau=1e-4)
```

### 2. Filtro Quantitativo CUSUM (`cusum_filter.py`)
Estimativas emitidas sobre todo e qualquer *tick* contêm altos índices de ruído residual (White Noise) que comprometem o rastreamento do classificador. O CUSUM (Cumulative Sum) foi integrado para criar interrupções pontuais, amostrando ocorrências em que surpresas da distribuição de retorno transbordam os limites preestabelecidos da volatilidade condicional.
- `cusum_filter_events`: Computa duas esteiras simultâneas isolando variações percentuais positivas e negativas. Consolida os retornos emitindo sub-carimbos de tempo (timestamps) nos exatos indexadores logarítmicos em que as funções de histerese colapsam as barreiras estáticas predefinidas.

### 3. Análise da Microestrutura do Livro de Ordens (`order_flow.py`)
Incorpora a identificação e rastreamento contínuo das disparidades de pressões subjacentes, utilizando-se das execuções passadas para derivar estimativas relativas sobre o topo de agressões do livro (L2) simulado.
- `order_flow_imbalance` (OFI): Acompanha assimetrias no montante passivo e agressivo por intermédio das variações cruzadas observadas pela liquidez remanescente.
- `vpin`: Métrica preditiva (Volume-Synchronized Probability of Informed Trading) deduzindo o nível inter-cruzado do fluxo não balanceado em transações sincronizadas ao longo das barras de volume.

### 4. Orquestração e Derivação Estrutural (`indicators.py`)
Fornece métodos secundários focando nas oscilações quantitativas em adição a indicadores clássicos.
- Reúne os agregadores transversais da distribuição (`roc`, `atr`, `garman_klass_volatility`, `rolling_moments` — Skew/Curtose temporal) provendo o agrupador generalizado de features atômicas através da centralizadora `compute_all_features`.

**Exemplo de Orquestração Analítica:**
```python
from src.features.indicators import compute_all_features
from src.features.normalizer import normalize_features

raw_features_df = compute_all_features(bars_df, config=my_cfg)
norm_features_df = normalize_features(raw_features_df, method="zscore", window=100)
```

### 5. Restrições à Contaminação e Vazamento Temporal (`normalizer.py`)
Se a normalização escalonada sobre a série temporal for imposta retrospectivamente via `MinMaxScaler` estático de ponta-a-ponta, todo o histórico transacional repassaria viés da média total ou da variação final sobre amostras primárias.
- `rolling_zscore` e `expanding_rank`: Normalizam os parâmetros subjacentes aplicando exclusivamente vetores móveis iterativos avaliando apenas o intervalo condicional até o instante validado $t_0 - 1$.
- `validate_no_lookahead`: Auditoria recursiva executando avaliações unitárias em intervalos escalonados atestando se a tabela final da normalização converge para resultados temporalmente cegos perante a variação futura inalterada do mercado local.

## Especificação Matemática de Extração (OFI, CUSUM e VPIN)

### Diferenciação Fracionária de López de Prado (FFD)
Adotando $d$ sendo um grau derivativo de raiz real e fracionado contido de $d \in (0, 1)$, estipula-se a série expansiva baseada sobre a convergência do operador linear iterativo ($L$). Constrói-se os pesos $w$ tal que:
$$ w_k = -w_{k-1} \frac{d - k + 1}{k} \quad \text{onde } w_0 = 1 $$
Aplicando essa formulação a uma respectiva sequência transacional indexada para derivar os níveis de série estacionária $\tilde{P}_t$:
$$ \tilde{P}_t = \sum_{k=0}^{\infty} w_k P_{t-k} $$
Ao adotar o limitador tolerante de truncagem (`tau`) ex: $\tau = 10^{-5}$, impõe-se a poda logarítmica reduzindo massivamente os laços iterativos no escopo temporal histórico durante a montagem das janelas fixas da biblioteca vetorial subjacente de cálculos Pandas.

### Volume-Synchronized Probability of Informed Trading (VPIN)
Em formulações microestruturais intradiárias, adota-se métricas correlacionais de agressões intra-velas baseando o cálculo cinemático na simetria do preenchimento da barra local:
A proporção estimada do saldo subjacente de ordens agressoras de compradores $V_B$ vs vendedores $V_S$ é expressa perante o gradiente de dispersão direcional sub-barra:
$$ \Delta_{P} = \frac{\text{Close} - \text{Low}}{\text{High} - \text{Low}} $$
$$ V^B_t \approx \Delta_P \cdot V_{total, t} $$
Calculando a probabilidade empírica baseada pelo choque e assimetria estocástica transacional nas janelas consecutivas de volume contíguo:
$$ VPIN = \frac{ \sum |V^B_{\tau} - V^S_{\tau}| }{ \sum V_{total, \tau} } $$
Níveis anômalos no limiar de distribuição VPIN identificam estatisticamente concentrações atípicas institucionais agredindo um espectro passivo desbalanceado do mercado.

### Monitoramento Adaptativo de Amostragem (Filtro CUSUM Simétrico)
Apropriado para criar desvios purificados indexando oscilações estruturais diretas do preenchimento contínuo, a rotina armazena desvios de flutuação positiva $S^+$ e negativa $S^-$ acumuladas com base no patamar da surpresa iterativa baseada na respectiva taxa:
$$ S_t^+ = \max \left( 0, S_{t-1}^+ + y_t - E_{t-1}[y_t] \right) $$
$$ S_t^- = \min \left( 0, S_{t-1}^- + y_t - E_{t-1}[y_t] \right) $$
Assumindo $y_t$ como retornos matemáticos ou logarítmicos e os limiares simétricos contidos sobre uma margem estática $h$, demarca-se vetorialmente o intervalo e a respectiva indexação contendo as variações que superam:
$$ t_{evento} : S_t^+ > h_t \quad \lor \quad S_t^- < -h_t $$
Colapsando os re-agregadores iterativos a $0$ subitamente e iniciando sucessivas e constantes iterações nas varreduras do DataFrame contínuo processado.