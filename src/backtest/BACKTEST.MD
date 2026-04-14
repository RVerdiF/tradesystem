# Módulo de Backtest (src/backtest)

## Objetivo
O Módulo de Backtest é um simulador de execução e validador estatístico. Diferente de plataformas comerciais de varejo que meramente plotam curvas de capital que sofrem de overfitting crônico, o TradeSystem5000 foca na robustez. O módulo reproduz a operação do pipeline e calcula métricas institucionais, precifica realisticamente derrapagens e taxas brasileiras (B3), e principalmente: resolve o problema de vazamento de dados gerando validações cruzadas purgadas (CPCV) e testando a verdadeira força da estratégia por meio da Deflação de Sharpe (DSR).

## Integração com outros módulos
- **Modeling (`src/modeling`)**: Interage profundamente para rodar os splits de Cross-Validation com purga (Purged CV), e para testar retornos previstos versus caminhos alternativos da história.
- **Optimization (`src/optimization`)**: Recebe como *callback* o Deflated Sharpe Ratio que julga no final se o afinador Bayesiano encontrou Ouro ou apenas ajustou dados ao ruído ("Sorte").
- **Execution (`src/execution`)**: Serve como o "gêmeo digital" do sistema de risco, estimando cenários via simulações de ruína que depois regem o Risk Manager.

## Mapa de funções

### Validação Cruzada Combinatória (`cpcv.py`)
- `CombinatorialPurgedCV`: Classe que fatia a série temporal em $N$ blocos. Diferente do K-Fold tradicional, ele cria permutações gerando *Backtest Paths* únicos. Ex: Se temos 6 blocos de tempo onde a crise de 2020 é o bloco 3 e o bull-market é o bloco 5, ele treina e testa em várias permutações, garantindo que o Meta-Modelo preveja a crise antes e depois dela ter acontecido no bloco de teste.
- `split`: Executa a purga e embargo e provê as coordenadas (índices).
- `backtest_paths`: Reconstrói visualmente e financeiramente as múltiplas histórias paralelas do que "poderia ter acontecido".

### Sharpe Deflacionado (`dsr.py`)
- `deflated_sharpe_ratio`: Calcula a probabilidade do Sharpe Ratio máximo observado na bateria de testes Bayesiana não ser o viés do limite máximo esperado sob a Hipótese Nula ($H_0$).

### Modelo de Custo (B3) (`cost_model.py`)
- `BrazilianCostModel`: Parametriza Custos de Corretagem, Emolumentos e Taxa de Liquidação B3 para o contrato de mini índice e dólar, além de ISS e impostos.
- `SlippageModel`: Calcula a *derrapagem* do preço baseada na quantidade que o robô empurra contra o livro restrito ao volume diário de negociação (usando constante elástica).
- `total_cost`: Pipeline de fachada para estimar o Spread transacional total em série.

### Atribuição (`attribution.py`) e Métricas (`metrics.py`)
- Funções essenciais para relatório quantitativo. Em `metrics`, calculam-se o Rácio de Calmar, Rácio de Sharpe (Anualizado), Drawdown e Simulações de Monte Carlo de Ruína.
- Em `attribution`, ele decompõe os lucros em fatias, respondendo matematicamente: *Quanto do meu lucro se deve ao Sinal Alpha cego, e quanto do Sharpe "Shift/Lift" é mérito da restrição do Meta-Labeling ML?*

## Lógica geral
Um Backtest não gera um valor absoluto; gera uma distribuição. A série é cortada no Combinatorial CV; os testes são expurgados com Purging e Embargo limitando a capacidade do ML de espiar o Teste. A série de retornos brutos passa pelo `total_cost` retirando da equidade a perda irrecuperável de taxas e ineficiência de Book. A lista dos melhores Sharpe Ratios obtidos passa pelo Deflator (DSR). Se o valor final for maior que `0.95`, a estratégia é aprovada.

## Lógica matemática
1. **Modelagem de Custos (Slippage Estimada)**:
   A derrapagem percentual é modelada baseada na raiz da proporção transacionada:
   $$ \text{Slippage}_t = \text{Const} \times \sqrt{\frac{V_{trade}}{V_{avg\_daily}}} $$
2. **Deflated Sharpe Ratio (DSR)**:
   Baseado na variância dos Sharpes Testados $\sigma^2_{SR}$, calcula o máximo Sharpe esperado caso a estratégia não tivesse valor real ($H_0$):
   $$ \text{E}[max(SR)] = \sigma_{SR} \left[ (1 - \gamma) Z^{-1}(1-\frac{1}{N}) + \gamma Z^{-1}\left(1-\frac{1}{N e}\right) \right] $$
   E o DSR converte isso num valor $Z$ de probabilidade acumulada normal, considerando as assimetrias (Skew e Kurtosis).
3. **Múltiplos Caminhos (CPCV)**:
   O número de trajetos simuláveis é a análise combinatória dos Blocos:
   $$ P = \binom{N}{k} = \frac{N!}{k!(N-k)!} $$
