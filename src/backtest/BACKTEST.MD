# Módulo de Backtest (src/backtest)

## Objetivo Detalhado
O backtesting em ambientes algorítmicos profissionais requer abstrações estritas para refletir as fricções mecânicas da execução real, eliminando heurísticas triviais de precificação exata do mercado. O TradeSystem5000 implementa neste módulo a base metodológica para avaliação probabilística através da métrica do **Combinatorial Purged Cross-Validation (CPCV)**, produzindo distribuições representativas de caminhos evolutivos para os retornos e, de maneira complementar, estressando os resíduos transacionais e as comissões por intermédio de modelos empíricos da Bolsa (B3).

Através da abstração e precificação do *Slippage* (derrapagem ao agredir a liquidez do livro) acoplado a métodos quantitativos de simulação estatística (Simulação de Monte Carlo), o sistema objetiva a obtenção de indicativos analíticos da performance e a validação do Sharpe Ratio Deflacionado (DSR) sem subestimações.

## Integração com Outros Módulos
- **Modeling (`src/modeling`)**: Estrutura a interoperabilidade para extração de predições. Submete as previsões de teste temporal de instâncias *PurgedKFold* originárias de XGBoost, avaliando caminhos alternativos independentes do classificador e calculando a métrica total do conjunto das permutações históricas.
- **Optimization (`src/optimization`)**: Módulo de avaliação recursiva das hipóteses durante o processo bayesiano. Transmite os retornos de múltiplos trials para a função penalizadora (DSR), avaliando o grau de viés estatístico de exploração na nuvem de simulação de resultados.
- **Execution (`src/execution`)**: A base modelada de custos diretos (taxas baseadas em lotes de índice e dólar B3) e restrições estimadas do drawdown retroativo estabelecem os limiares transpostos ativamente sobre os Circuit Breakers diários no ambiente produtivo (Live).

## Mapa de Funções e Componentes Chave

### 1. Modelo Empírico de Custos B3 (`cost_model.py`)
Estima as despesas proporcionais incidentes, atreladas a transações institucionais, que representam perdas permanentes sobre a cotação aferida no log de simulação.
- `BrazilianCostModel`: Classe parametrizada contendo tabelas exatas para a Bolsa de Valores Brasileira. Determina as taxas flutuantes de corretagem (zero para certas operadoras institucionais), Emolumentos e Taxa de Liquidação da B3 e os impostos percentuais incidentes por tipo de contrato (WDO, WIN, DOL, IND).
- `SlippageModel`: Função modeladora de *Impacto de Mercado* (Market Impact). Calcula a assimetria percentual da ordem perante a execução integral estática, derivando estocasticamente a degradação do spread na absorção dependente da proporção lançada (Quantity) frente ao Volume Médio Diário (ADV) transacionado para o subjacente.
- `total_cost`: Estrutura de fachada (Facade) operando o mapeamento seqüencial de ambos os fluxos de penalização atritiva para abater de imediato do vetor original dos resultados do backtest.

**Exemplo do Escalonamento Transacional B3:**
```python
from src.backtest.cost_model import total_cost
import pandas as pd

precos_ordem = pd.Series([120000.0, 120500.0, 119800.0])
vols_ordem = pd.Series([10.0, 50.0, 100.0])

df_custos = total_cost(
    prices=precos_ordem,
    quantities=vols_ordem,
    avg_volume=500000.0,
    symbol="WIN"
)
```

### 2. Análise Estatística e Desempenho Contínuo (`metrics.py` e `dsr.py`)
Fornece métricas padronizadas institucionais de risco de cauda e estabilidade de evolução patrimonial.
- Matriz unificada e extensiva para derivação de Sharpe Anualizado em predições sub-diárias de alta-frequência, Rácio de Calmar (sensibilidade da recuperação temporal), Sortino (isolando os retornos de variações à esquerda) e a respectiva inferência de Ruína via distribuição Gaussiana iterativa baseada em Matrizes de Monte Carlo (MCMC).
- `deflated_sharpe_ratio`: Rotina quantificadora encarregada do Deflated Sharpe Ratio. Reflete a avaliação das otimizações atestando as probabilidades globais do resultado ser isento do viés otimizado multivariado.

### 3. Simulação Multidirecional e Validação de Trajetória (`cpcv.py`)
Inovação fundamental da metodologia de validação de modelos sequenciais de Marcos López de Prado.
- `CombinatorialPurgedCV`: Gera partições combinatórias sobrepostas e as reordena, produzindo inúmeras trajetórias únicas da evolução do capital (Backtest Paths) sem violar as regras da cronologia cruzada ou prever os resultados com sobreposições no modelo.

**Exemplo da Disposição do Combinatorial CV:**
```python
from src.backtest.cpcv import CombinatorialPurgedCV

cpcv = CombinatorialPurgedCV(
    n_groups=6,
    n_test_groups=2,
    samples_info=df_trade_timeline,
    pct_embargo=0.01
)

splits_de_cruzamento = cpcv.split(X_features, y_labels)
```

## Fundamentação Matemática do Simulador Evolutivo

### Distribuição Subjacente da Avaliação Combinatória (CPCV)
O conceito K-Fold CV sequencial falha na estruturação temporal de fenômenos financeiros porque agrupa subconjuntos invariáveis do passado. A proposição do *Combinatorial Purged CV* consiste em segmentar formalmente os registros históricos em $N$ blocos de tamanho simétrico e contíguo. Com o particionamento imposto de forma rígida em conjuntos operacionais simultâneos de teste $k$ grupos, as combinações e a ordenação purificada propiciam as múltiplas ramificações independentes de simulação sem duplicatas do capital $P_{aths}$:
$$ P_{aths} = \binom{N}{N-k} = \frac{N!}{k! \cdot (N-k)!} $$

### Penalização Condicional e Risco Sistêmico (Slippage)
O impacto gerado nas cotações quando a agressão transacional avança níveis da estrutura primária do Livro de Ofertas reflete decaimento em escala de dependência à liquidez e atrito operacional da infraestrutura da bolsa B3. Modelamos matematicamente as despesas contínuas derivando a *Derrapagem* através da volatilidade marginal base e da quantidade estipulada do vetor:
Assumindo a sensibilidade elástica ($\lambda$) e o sub-fator condicional de mercado do Volume:
$$ \text{Spread\_Derrapado}_t = P_t \times \left( \lambda \times \sqrt{\frac{V_{ordem}}{V_{avg\_daily}}} \right) $$

### Mensuração Marginal das Predições do Alpha
A avaliação fracionada das decomposições das metas operacionais evidencia isoladamente o benefício condicional de agregar restrições de filtragem via Meta-Labeling perante execuções irrestritas nativas baseadas no Sinal Base. Expressamos analiticamente esse Lift sobre o Sharpe:
Dado que o Sharpe corresponde ao retorno logarítmico ajustado perante a volatilidade $\text{Sharpe} = \frac{E[R_p]}{\sigma_p}$:
$$ \text{Sharpe\_Lift\_Percentual} = \frac{\text{Sharpe}(ret_{ML}) - \text{Sharpe}(ret_{Base})}{ \max(\text{Sharpe}(ret_{Base}), 0.001) } $$
Estabelecendo a verificação paramétrica rigorosa que atesta se a implementação sistêmica global das métricas purificadoras da modelagem do classificador binário, sob os pesados embargos transacionais do XGBoost em *CPCV*, excede suficientemente a base da inércia primária ($ret_{Base}$).