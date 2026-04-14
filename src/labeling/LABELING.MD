# Módulo de Rotulagem (src/labeling)

## Objetivo
O módulo de Labeling é responsável por definir a variável alvo (*target* ou rotulagem) para o treinamento dos modelos de Machine Learning. Diferente de rotulagens fixas simplistas (como prever o preço daqui a N dias), o módulo implementa o método da Tripla Barreira e Meta-Labeling (baseado nas técnicas de Marcos López de Prado). Isso fornece um *ground truth* muito mais realista do risco-retorno na execução de um trade, considerando volatilidade dinâmica e filtragem rigorosa de alvos ao longo do tempo de duração do sinal.

## Integração com outros módulos
- **Features (`src/features`)**: A saída dos eventos detectados (ex: Filtro CUSUM) são enviados ao pipeline de rotulagem.
- **Modeling (`src/modeling`)**: O resultado deste módulo são os *meta-labels* (0 ou 1) que alimentam o modelo `classifier.py` permitindo-o aprender se deve ou não entrar num sinal alpha secundário.
- **Configurações (`config/settings.py`)**: Define limiares base como span de volatilidade (`volatility_span`), proporção de risco-retorno para barreiras (`pt_sl_ratio`), o limite de tempo em barras (`max_holding_bars`), etc.

## Mapa de funções

### Volatilidade Dinâmica (`volatility.py`)
- `get_daily_vol`: Estima a volatilidade de um ativo com base na Média Móvel Exponencial Ponderada (EWMA) dos retornos absolutos para um determinado intervalo de barras. Isso garante que a largura dos alvos e stops no método de Tripla Barreira seja diretamente proporcional ao humor volátil do mercado daquele regime temporal específico.

### Modelo Alpha Simulado (`alpha.py`)
- `bollinger_alpha`: Gera sinais básicos de reversão ou momento à margem das bandas de Bollinger.
- `macd_alpha`: Sinais táticos puros advindos de cruzamento e divergência de Médias Moveis.
- `composite_alpha`: Reúne múltiplos alphas fracos combinando seus votos.

### Meta-Labeling (`meta_labeling.py`)
- `get_meta_labels`: Cruza o sucesso final retornado pelas Triplas Barreiras com o sinal direcional pretendido pelo Alpha para definir o label binomial que aprova ou rejeita (0 ou 1) o sinal original.

### Tripla Barreira (`triple_barrier.py`)
- `create_events`: Identifica o ciclo de vida do trade com o *start* e o *end* estrito pelo tempo máximo permitido em mercado `max_holding`.
- `apply_triple_barrier`: Dispara um rastreador intrabarra (verificando mínimas e máximas da série) para checar quem é rompido primeiro: O limite vertical (tempo), o alvo de lucro superior horizontal ou o Stop loss inferior horizontal (além de permitir Breakeven progressivo).
- `_find_dynamic_touch`: Função núcleo em numba que resolve o caminho dentro do labirinto dos preços, decidindo com viés conservador as ambiguidades onde o alvo e o stop são batidos simultaneamente (retornando stop primeiro).
- `get_labels`: Função de unificação do processo de Tripla barreira devolvendo a tabela de labels completa para modelagem primária (+1, -1, 0).

## Lógica geral
A premissa baseada no fluxo de *Meta-Labeling* é que tentar prever direções financeiras (sobe ou desce) com Machine Learning usualmente produz falsos positivos custosos. O modelo propõe separar o processo. O sistema usa "modelos" teóricos de `alpha` para gerar a intenção do trade. A `volatilidade` dita o tamanho dos stops. O `triple_barrier` resolve simulando o trajeto do sinal. E o `meta_labeling` pega as vitórias provadas para treinar o classificador secundário (Se Alpha disse Compra e Trade foi Positivo = 1; Se foi Negativo = 0). O foco se volta para dimensionar e prever a qualidade (falsos positivos).

## Lógica matemática
1. **Volatilidade (EWMA)**:
   A volatilidade é baseada num decaimento exponencial, calculando a raiz quadrada da variância diária média:
   $$ \sigma_t = \sqrt{ \text{EWM}[ (R_t)^2, \, \text{span}=T] } $$

2. **Cálculo da Tripla Barreira (Take Profit e Stop Loss Dinâmicos)**:
   Os limites horizontais são funções da volatilidade do ativo no instante $t_0$:
   $$ \text{Upper Target} (PT) = p_{t0} \cdot \sigma_{t0} \cdot m_{pt} $$
   $$ \text{Lower Limit} (SL) = -p_{t0} \cdot \sigma_{t0} \cdot m_{sl} $$
   A barreira vertical ($t1$) é:
   $$ t_1 = t_0 + \text{max\_holding\_bars} $$
   Se a barreira superior ou inferior não forem tocadas, força-se a saída pelo preço em $t_1$.

3. **Retorno Ajustado por Lado (Side-Adjusted Return)**:
   A avaliação de lucro e prejuízo em compras ($Side=1$) e vendas ($Side=-1$) obedece:
   $$ r_{adj} = \left( \frac{p_{t}}{p_{entry}} - 1 \right) \cdot Side $$
   Isso normaliza a lógica para focar estritamente se o viés subjacente e a força previstos estavam corretos, independentemente de ser operação short ou long.
