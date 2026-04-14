# Módulo de Backtest (src/backtest)

## Objetivo Detalhado
O backtest de um software algorítmico profissional não se presta a imprimir um gráfico bonitinho em linhas verdes mostrando "o quão rico você seria se tivesse operado isso ontem". Gráficos simples sofrem de duas moléstias fundamentais: Overfitting histórico e Subestimação de atrito de mercado (Custos falsos de corretagem + Slippage irrealista do livro).

O TradeSystem5000 aborda o Backtest (em todo o subdiretório `src/backtest`) com um rigor tido como excessivamente paranóico. É melhor destruir na simulação uma estratégia falsa do que financiá-la em conta real com dinheiro tangível. Para isso, o módulo implementa o estressante fluxo quantitativo do **Combinatorial Purged Cross-Validation (CPCV)**, varrendo milhares de retornos, a desconstrução percentual exata de Impostos (ISS), Emolumentos e Taxas (TCA) do mercado B3, somado a um modelo elástico estocástico do *Slippage* (derrapagem ao agressar grandes quantidades e raspar as ordens na pedra), e uma classe de Atribuição e Simulação de Monte Carlo para ruína que entrega índices analíticos profissionais e isentos de viés.

## Integração com Outros Módulos
- **Modeling (`src/modeling`)**: Este módulo é parasita do Modeling. Ele abraça e empacota o `PurgedKFold` temporal que foi forjado em ML, extraindo a física das previsões cegas de dobras estritamente separadas e unindo no CPCV todos os caminhos fractais do retorno temporal para análise de métricas financeiras cruas.
- **Optimization (`src/optimization`)**: Age também como sentinela perante a otimização (atuando em conluio com o Sharpe Deflacionado `dsr.py`). Quando a Otimização grita "Eureka, achei os hiperparâmetros de máxima performance", ela submete essa curva a esse módulo; caso o risco de falso positivo e calmar ajustado escorregar, a parametrização não é creditada.
- **Execution (`src/execution`)**: Todas as parametrizações restritivas criadas via modelo de ruínas e cálculos monetários dos custos base B3 simulados são consumidos no sistema restritivo real de risco diário quando rodando via MT5 ao vivo no intraday.

## Mapa de Funções e Componentes Chave

### 1. A Simulação de Atrito Completo B3 (`cost_model.py`)
Não vivemos num mundo ideal sem atrito; ao entrar longo com uma meta de 20 pontos de índice em 5 contratos, o scalping precisa superar o imposto das taxas e o buraco sem liquidez entre o *Bid* e o *Ask*.
- `BrazilianCostModel`: Sub-classe hardcoded baseada na parametrização pública da estrutura de tarifas da B3 e Impostos de São Paulo para Emolumentos + Liquidação + Corretagens em centavos em Dólar Futuro Cheio, Mini Dólar, Ibovespa e Mini-Índice.
- `SlippageModel`: A cada quantidade executada e não engolida instantaneamente no *spread* mínimo do *Book* da cotação, assume uma função estocástica simulada que piora percentualmente a entrada real proporcionalmente a força submetida e a escassez do Volume Médio Diário na ponta.
- `total_cost`: Facade de união orquestradora empacotando numa só chamada a linha do atrito completo.

**Snippet do Modelo Empírico de Custos:**
```python
from src.backtest.cost_model import total_cost
import pandas as pd

# Assume os vetores massivos de ordens fechadas simuladas do XGBoost:
precos_ordem = pd.Series([120000.0, 120500.0, 119800.0]) # Índice a 120 mil pt
vols_ordem = pd.Series([10.0, 50.0, 100.0])              # Mandou agressões de lotes distintos

# Simula o verdadeiro buraco que se afunda a cotação e a conta da B3:
df_custos = total_cost(
    prices=precos_ordem,
    quantities=vols_ordem,
    avg_volume=500000.0, # Liquidez e profundidade média local estipulada do ativo
    symbol="WIN"
)
print("Penalização de Retornos Brutos na boleta devido Atrito e Taxas:")
print(df_custos["total_cost"]) # Array numérico de redução per-trade real em Reais ($).
```

### 2. Rotinas Quantitativas do Deflacionador e Estatística (`metrics.py` e `dsr.py`)
Um caldeirão de analítica estatística institucional para o relatório final contábil de modelagem.
- Funções em `metrics`: Calcula o verdadeiro Sharpe Anual de dados de alta frequência sub-minuto, Calmar Ratio para sensibilidade, Sortino Ratio isolando as sub-distribuições de caudas penais e uma rotina estocástica profunda calculando Probabilidades de Ruínas usando Matrizes Aleatórias Gaussianas em série baseadas no capital em exposição e histórico da série.
- `deflated_sharpe_ratio`: Retorna a confiabilidade matemática sob testes de hipóteses de viés de otimização de múltiplas comparações (o selo de verificação de não-overfitting global).

### 3. A Máquina do Tempo Alternativa (CPCV - `cpcv.py`)
Módulo ultra especial que reflete o poder preditivo sem fatiar cronicamente o tempo sequencial em apenas "1 trajeto" ou sub-ciclos K-Folds simplistas.
- `CombinatorialPurgedCV`: Gera os múltiplos *Backtest Paths* únicos ao criar permutações combinatoriais usando $N$ blocos globais limitados.

**Snippet do Cross-Validation Combinatório (Backtest Múltiplo):**
```python
from src.backtest.cpcv import CombinatorialPurgedCV

# Se queremos simular todos os multiversos possíveis que os blocos de crises passadas
# (Ex: COVID, J&J, Sub-prime, Circuit-Breaker IBOV) e bull-markets reagiriam em série.
cpcv = CombinatorialPurgedCV(
    n_groups=6,          # Fatia o mega-histórico de 10 anos em 6 épocas
    n_test_groups=2,     # Agarra a junção cega de 2 épocas (não adjacentes) para testes
    samples_info=df_trade_timeline,
    pct_embargo=0.01
)

print(f"Iremos gerar, expurgar, embargar e reconstruir {cpcv.n_paths} trajetos paralelos de histórico.")
# Executa as junções matemáticas retornando arrays perfeitos para fitar e simular.
splits_de_cruzamento = cpcv.split(X_features, y_labels)
```

## A Lógica Geral Da Destruição da Sub-Amostragem Base B3

Um Backtest ingênuo em MT5 traça uma curva de fundos onde se comprou Índice e subiu. No CPCV nós traçamos 15 curvas para a *mesma* estratégia ao mesmo tempo.
Se você usou K-Folds clássico, a primeira simulação treina no Bull Market, e testa na Pandemia. A Segunda treina na Pandemia e tenta operar no Pós-crise; isso é fraco, não demonstra versatilidade do modelo porque na realidade os regimes vêm misturados.
No CPCV agrupamos permutações (treinar na pandemia + Bull Market para testar numa era de juros altos isolada), costurando essas predições retalhadas não de volta numa tripa só, mas em dezenas de trilhas ramificadas independentes de "Capital Evolutivo", traçando a teia simulada de todos os destinos lógicos se o modelo tivesse sido plugado em épocas invertidas e observando a média de lucros gerados se o portfólio não quebra em nehum dos sub-horizontes de multiverso gerados. É o atestado máximo universal de não-superestimação (*Over-Estimation*) para gestoras quantitativas de ponta.

## Equações Matemáticas Chave da Validação de Risco e Simulação de Mercado

### Análise Combinatória Subjacente ao CPCV (Combinatorial Paths)
Ao particionar arbitrariamente todo o histórico de dados em exatos $N$ partições (Grupos temporais independentes com tamanhos idênticos) e ao exigirmos formar dobras de Validação selecionando estritamente um pacote com tamanho agrupado em $k$ test groups sem reposição, o número combinatório de todas as rodadas parciais de Backtests únicas e expurgadas extraídas do dataset total (A quantidade de Multiversos da estratégia gerada em `n_paths`) atinge de imediato:
$$ P_{aths} = \binom{N}{N-k} = \frac{N!}{k! \cdot (N-k)!} $$

### Simulação de Slippage Punitivo Contínuo
Mesmo uma estratégia fantástica naufragará na vida real se ela tenta virar 2.000 Minis em uma fração de segundo e engolir todos os níveis de livro rasgando-os sem Liquidez (A *derrapagem* devora 3 a 4 ticks a favor rasgando o lucro de scalping). A mecânica teórica modelada calcula um desvio relativo atrelado à fragilidade ou profundidade subjacente da hora perante o volume médio empírico transacionável do ativo local no dia:
Assumindo Constante Institucional empírica $\lambda$ do sub-modelo da B3:
$$ \text{Spread\_Derrapado}_t = P_t \times \left( \lambda \times \sqrt{\frac{V_{ordem}}{V_{avg\_daily}}} \right) $$

### Relatórios e Atribuição do Alpha Marginal
A atribuição decompõe os log-retornos para revelar a força independente do Machine Learning ($ret_{ML}$) operando perante a pura direção inercial base cega do sistema sem os Zeros impostos para corte ($ret_{Base}$):
Dado que $\text{Sharpe} = \frac{E[R_p]}{\sigma_p}$:
$$ \text{Sharpe\_Lift\_Percentual} = \frac{\text{Sharpe}(ret_{ML}) - \text{Sharpe}(ret_{Base})}{ \max(\text{Sharpe}(ret_{Base}), 0.001) } $$
Informando metricamente se a complexidade extrema inferida do algoritmo na arquitetura AFML compensa o peso que cria em infraestrutura no pipeline e se sua capacidade atritiva filtra com maestria as ruínas da Base sem extirpar sua inércia direcional nativa.
