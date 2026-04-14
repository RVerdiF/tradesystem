# Módulo de Rotulagem (src/labeling)

## Objetivo Detalhado
No Machine Learning clássico aplicado a reconhecimento de imagens ou NLP, os rótulos (*labels*) são dados por fato (é um cachorro ou é um gato). Em finanças, se modelarmos um classificador para prever se a próxima barra do índice fechará positiva ou negativa, estaremos treinando o computador a fazer algo terrivelmente perdedor: prever micro-passos sem compreender as forças do *Stop Loss* nem o Risco-Retorno dinâmico do ativo. O TradeSystem5000 reverte isso utilizando a arquitetura de **Meta-Labeling** e a **Tripla Barreira**, descritos na literatura avançada de ML Financeiro.

O objetivo deste módulo é fabricar uma verdade fundamental (Ground Truth) de qualidade inquestionável, respondendo retrospectivamente à seguinte pergunta de ouro: *"Se meu robô Alpha rudimentar recomendasse Compra naquele exato minuto X e os meus limites estivessem configurados para a dinâmica atual de volatilidade local, o robô seria engolido por um drawdown mortal (falha) ou extrairia lucro líquido sem dores (sucesso)?"*

## Integração com Outros Módulos
- **Features (`src/features`)**: A saída dos eventos estritamente detectados (ex: Filtro CUSUM, que só dispara quando há real assimetria de fluxo) gera as datas de interesse onde a rotulagem deve engatilhar as "boletas matemáticas" para avaliar o desenrolar das triplas barreiras.
- **Modeling (`src/modeling`)**: O resultado global transacionado do `get_meta_labels` cria uma série de Zeros e Uns que forma o vetor vetor-alvo $y$. Este vetor alimentará lado a lado os DataFrames matriciais formados de indicadores e features ($X$) treinando implacavelmente o classificador em *XGBoost*.
- **Configurações (`config/settings.py`)**: As variáveis rígidas do comportamento elástico (quantos dias abrange o *span* da volatilidade EWM? Qual a razão de Risco para Recompensa default?) ditam o tamanho da caixa da Barreira.

## Mapa de Funções e Componentes Chave

### 1. Volatilidade Elástica Dinâmica (`volatility.py`)
No regime das 9h de leilão do WIN, oscilações de 300 pontos não significam nada (e ser estopado nesse martelete custa dinheiro à toa). Às 15h, sem notícias, oscilar 300 pontos é um terremoto. Operadores humanos utilizam *feelings*; a máquina usa *EWMA*.
- `get_daily_vol`: Calcula a variação bruta temporal com um multiplicador de raiz e decaimento elástico das amostras, traçando e projetando em toda barra exata do pregão quantos *bps* o ativo precisa viajar matematicamente para caracterizar uma perna (alvo).

### 2. Sinais de Base Fracos (Modelo Alpha - `alpha.py`)
O Meta-Labeling requer uma premissa básica a ser testada, e aqui instanciamos lógicas quantitativas puramente "táticas".
- `bollinger_alpha` e `macd_alpha`: Operam as margens cruzadas indicando compras (1), vendas (-1) ou abstenção neutralidade (0).
- `composite_alpha`: Empacota os votos cegos usando lógicas puramente direcionais sem dimensionamento.

### 3. A Máquina da Tripla Barreira (`triple_barrier.py`)
Simula as proteções mecânicas da conta para decidir sem ambiguidades se o trade fictício teve sucesso.
- `create_events`: Dado um sinal detectado no carimbo do tempo $T$, demarca que o limite temporal obrigatório para zeração (Stop por expiração de Tempo) cairá exatas barras à frente do calendário do DataFrame (`max_holding`).
- `apply_triple_barrier` & `_find_dynamic_touch`: Esse é o miolo JIT Compilado ultra-rápido do sistema. Em C#, C++ ou Python iterativo demoraria dias processar 15 anos. Essa função recebe o limite superior (+PT), limite inferior (-SL) calculados através da volatilidade dinâmica, o Stop no Relógio e escrutina linha a linha (no intrabar com *High* e *Low*) descobrindo o vencedor. E mais ainda, implementa lógica de "Viés Conservador": Se na mesma barra o Índice violou os absurdos limites tanto do Gain e Stop (como numa *Payroll* explosiva), o compilador força o retorno assumir a falha em favor da pior margem possível de sobrevivência, estressando duramente a otimização.

**Snippet de Rastreio com a Tripla Barreira Intrabarra:**
```python
from src.labeling.triple_barrier import get_labels
from src.labeling.volatility import get_daily_vol

# Calcula a volatilidade flutuante usando decaimento longo do histórico recente
target_vol = get_daily_vol(df["close"], span=100)

# Processa o Ground Truth absoluto do mercado baseando-se que a caixa de alvo e proteção
# da "Aposta" possui Reward = 2x o target e Risk = 1x o target
truth_labels = get_labels(
    close=df["close"],
    events=events_list,           # Saídas vindas do CUSUM
    pt_sl=(2.0, 1.0),             # Ex: Lucra 400 pts e Estopa 200 pts
    open_prices=df["open"],
    high_prices=df["high"],       # Garantem tracking sem "look-ahead" dentro da barra
    low_prices=df["low"],
    min_return=0.001
)
# `truth_labels` retorna +1 (Gain), -1 (Loss) e 0 (Atingiu o tempo limite vertical)
```

### 4. Extração do Verdadeiro Meta-Label (`meta_labeling.py`)
Confronta o Sinal Seco emitido em `alpha.py` (Ex: "Vende!") com as estatísticas do rastreamento estrito de `triple_barrier.py` (Ex: "A venda atingiu Stop Loss e tomou -1").
- `get_meta_labels`: Cruza e varre, definindo que a falha da união das intenções significa Zero (*Aprenda a não boletar isto futuramente*), e quando a união casa positivamente atríbui 1 (*Aprenda e tente encontrar novamente estas nuances nos seus inputs*).

**Snippet do Destilamento do Label Secundário:**
```python
from src.labeling.meta_labeling import get_meta_labels

# Cruzando a intuição crua do Alpha (Comprar=1 ou Vender=-1) com
# a cruel realidade calculada dos eventos reais batidos no labirinto de Preço.
meta_y = get_meta_labels(
    events=truth_labels_df,
    side=df["alpha_side"]  # Intenção sugerida gerada em alpha.py
)
# `meta_y` é agora o vetor binomial super puro [0, 1, 1, 0, 0, 1] preparado para
# ser a variável Y Alvo consumida impiedosamente pelo Random Forest.
```

## Lógica Matemática e Mecânicas de Volatilidade do AFML

### Estimação Progressiva de Volatilidade e Fronteiras Dinâmicas (Targeting)
Para calcular a oscilação que regerá as caixas de trade baseando-se num span exponencial, definimos primeiro a média EWMA das variações diárias puras baseada no vetor de retornos logarítmicos subjacentes ou absolutos:
$$ \sigma_t = \sqrt{ \text{EWM}[ (R_t)^2, \, \text{span}=\Delta] } $$
Isso nos fornece a régua instantânea no tempo $t_0$. As margens horizontais ($PT$ - Take Profit e $SL$ - Stop Loss) que cercam as probabilidades operacionais são diretamente ponderadas contra os multiplicadores $m$:
$$ \text{Upper Boundary (PT)} = P_{t_0} \times \left[ 1 + (\sigma_{t_0} \cdot m_{pt}) \right] $$
$$ \text{Lower Boundary (SL)} = P_{t_0} \times \left[ 1 - (\sigma_{t_0} \cdot m_{sl}) \right] $$
A Barreira de Tempo restringe os dias nos quais o capital ficará preso exposto sem definição de *Drift* (tendência perambulante no fundo):
$$ t_1 = t_0 + \text{max\_holding\_bars} $$
Se após as avaliações sequenciais o vetor de fechamentos $P_T \dots P_{t1}$ violar qualquer margem inferior a SL, a operação quebra imediatamente garantindo que retornos positivos futuros nunca mais farão o modelo achar que uma aposta de -5% para ganhar +2% foi vitoriosa.

### Semântica Matemática do Retorno Direcional Pós-Processamento (Side-Adjusted Return)
Quando a barreira toca num alvo ou expira na barra $t_{end}$, o retorno puramente percentual não significa muita coisa (Ex: Se a Bolsa caiu e você estava Short, foi Gain).
A função da tripla barreira corrige para que o Machine Learning entenda sucesso puramente como "Lucro Bruto Variacional":
$$ r_{adjusted} = \left( \frac{P_{end}}{P_{entry}} - 1 \right) \cdot Side $$
Garantindo que tanto uma alocação massiva vendida (-1) numa crise que caiu -5% reverta e se mostre aos olhos cruéis do algoritmo como uma vitória majestosa matemática de +5% direcional, homogeneizando e balanceando estatisticamente os labels.
