# Módulo de Rotulagem (src/labeling)

## Objetivo Detalhado
A abordagem de aprendizado de máquina convencional para finanças comumente define rótulos (*labels*) estáticos, tentando prever o sinal de retornos a intervalos de tempo fixos (e.g., fechamento positivo ou negativo da próxima barra). Esta metodologia negligencia a gestão de risco dinâmica intrínseca à operação algorítmica, ignorando variáveis como limiares de *Stop Loss* e *Take Profit* ajustados à volatilidade. O TradeSystem5000 substitui essa abordagem pela metodologia conjunta do **Método da Tripla Barreira (Triple Barrier Method - TBM)** e **Meta-Labeling**, formalizados na literatura de Machine Learning Financeiro.

O objetivo deste módulo é a geração de um *ground truth* probabilístico e realista. A estrutura avalia cenários contrafactuais operacionais: dado um sinal tático primário em um instante $t_0$, e aplicando restrições dinâmicas de risco baseadas na volatilidade estimada no mesmo período, a trajetória do ativo resultaria em um retorno favorável (sucesso) ou na violação dos limites de risco antes do alcance do objetivo (falha)?

## Integração com Outros Módulos
- **Features (`src/features`)**: A amostragem de eventos estruturais (ex: originados pelo Filtro CUSUM) define o subconjunto de instantes $t$ nos quais a avaliação da trajetória (rotulagem) será desencadeada, reduzindo o ruído inerente à amostragem contínua.
- **Modeling (`src/modeling`)**: O *array* resultante da função `get_meta_labels` constitui o vetor alvo $y$ (composto de $0$s e $1$s). Esse vetor é alinhado simetricamente à matriz de features $X$ para a formulação do treinamento supervisionado via classificadores em árvores de decisão.
- **Configurações (`config/settings.py`)**: Define os parâmetros constantes estritos do sistema: a janela de decaimento para estimação da volatilidade (`volatility_span`), a assimetria paramétrica de risco-retorno para as barreiras (`pt_sl_ratio`), e a restrição máxima de tempo de retenção da posição (`max_holding_bars`).

## Mapa de Funções e Componentes Chave

### 1. Estimativa de Volatilidade Dinâmica (`volatility.py`)
A modelagem de alvos fixos apresenta degradação estatística severa frente à não-estacionariedade da volatilidade intrassíntese. O dimensionamento das barreiras de risco necessita ser elástico.
- `get_daily_vol`: Computa o desvio padrão ponderado exponencialmente (EWMA) dos retornos em uma janela móvel (span). Essa estimativa serve como unidade de medida base para a largura transversal das barreiras de lucro e perda, adaptando as proporções de risco ao regime corrente do mercado.

### 2. Sinais Base (Modelo Alpha - `alpha.py`)
A estratégia de Meta-Labeling requer um modelo primário de recomendação direcional. Este sub-módulo congrega heurísticas táticas tradicionais para originar o "viés" ou intuição da operação.
- `bollinger_alpha` e `macd_alpha`: Operam as derivações métricas indicando intenções dicotômicas de entrada longa ($1$), entrada curta ($-1$) ou abstenção neutralidade ($0$).
- `composite_alpha`: Estrutura o mecanismo de votação em *ensemble* unificando vetores independentes em um sinal direcional unificado.

### 3. Mecanismo da Tripla Barreira (`triple_barrier.py`)
Simula a trajetória do preço intradia submetido às premissas rígidas do gerenciamento de risco paramétrico.
- `create_events`: Dado um evento de interesse em $t_0$, estabelece o horizonte de expiração ($t_1$) determinando a barreira vertical através da limitação de permanência temporal contínua na posição (`max_holding`).
- `apply_triple_barrier` & `_find_dynamic_touch`: Operam as varreduras determinísticas vetoriais otimizadas por compilação *Just-in-Time* (Numba). O algoritmo rastreia o inter-cruzamento das flutuações das máximas (*High*) e mínimas (*Low*) contra as barreiras flutuantes superior (+PT) e inferior (-SL). Para garantir conservadorismo frente ao *Slippage* intradiário, em casos de violação simultânea das barreiras na mesma amostra observada, o algoritmo contabiliza deterministicamente a trajetória como perda (atingimento do Stop Loss).

**Exemplo de Avaliação de Trajetória da Tripla Barreira:**
```python
from src.labeling.triple_barrier import get_labels
from src.labeling.volatility import get_daily_vol

target_vol = get_daily_vol(df["close"], span=100)

truth_labels = get_labels(
    close=df["close"],
    events=events_list,
    pt_sl=(2.0, 1.0),
    open_prices=df["open"],
    high_prices=df["high"],
    low_prices=df["low"],
    min_return=0.001
)
```

### 4. Formulação do Rótulo Secundário (Meta-Labeling) (`meta_labeling.py`)
Isola a performance da trajetória estrita computada pela barreira contrafactual da respectiva intuição direcional originada pelo sinal Alpha.
- `get_meta_labels`: Confronta a direção proposta do vetor Alpha com a resolução determinística computada pelo `triple_barrier.py`. A convergência estatística do êxito na mesma direção do Alpha atribui o rótulo $1$, enquanto qualquer interrupção por Stop Loss ou atingimento direcional incongruente com a expiração consolida o rótulo negativo $0$.

**Exemplo da Extração do Meta-Label:**
```python
from src.labeling.meta_labeling import get_meta_labels

meta_y = get_meta_labels(
    events=truth_labels_df,
    side=df["alpha_side"]
)
```

## Especificação Matemática e Dinâmica da Rotulagem

### Estimação de Volatilidade e Parametrização Dinâmica das Barreiras
A formulação da volatilidade estocástica aplica o decaimento exponencial à variância intradiária:
$$ \sigma_t = \sqrt{ \text{EWM}[ (R_t)^2, \, \text{span}=\Delta] } $$
Sendo $\sigma_{t_0}$ o parâmetro escalar do instante de emissão $t_0$. As fronteiras horizontais de realização de lucros ($PT$) e limitação de perdas ($SL$) operam diretamente sob multiplicadores constantes $m_{pt}$ e $m_{sl}$:
$$ \text{Upper Boundary (PT)} = P_{t_0} \times \left[ 1 + (\sigma_{t_0} \cdot m_{pt}) \right] $$
$$ \text{Lower Boundary (SL)} = P_{t_0} \times \left[ 1 - (\sigma_{t_0} \cdot m_{sl}) \right] $$
A barreira temporal delimita a viabilidade e risco da permanência inercial:
$$ t_1 = t_0 + \text{max\_holding\_bars} $$
Se a trajetória sequencial do array $P_{t_0} \dots P_{t_1}$ romper as condições condicionais predeterminadas nos patamares acima descritos, a função retorna deterministicamente a primeira violação, cessando o processo de checagem.

### Retorno Ajustado por Lado (Side-Adjusted Return)
A métrica analítica de sucesso precisa comportar simetria independentemente da direcionalidade (Long ou Short) do sinal primário.
O cálculo de lucro/perda aferido pela barreira normaliza-se à seguinte função vetorial:
$$ r_{adjusted} = \left( \frac{P_{end}}{P_{entry}} - 1 \right) \cdot Side $$
Garantindo a simetria probabilística: variações declinantes acopladas a intenções de venda ($Side = -1$) convertem a avaliação final à rentabilidade marginal estritamente positiva, balizando e unificando os labels para modelagem secundária.
