# Módulo de Otimização (src/optimization)

## Objetivo Detalhado
Sistemas algorítmicos complexos frequentemente incorporam parâmetros contínuos ou discretos para as decisões de negociação (e.g., taxas de aprendizado, multiplicadores de *Stop Loss*, limiares de *CUSUM* e pesos de classe para florestas aleatórias). A Otimização Bayesiana substitui a busca humana intuitiva ou os métodos tradicionais de varredura (*Grid Search* e *Random Search*), explorando o espaço de configuração de forma direcional.

Ao efetuar buscas estocásticas ao longo de um banco de dados restrito, incorre-se sistematicamente no viés da seleção (ou *Data Snooping*), no qual o algoritmo localiza hiperparâmetros ajustados aos ruídos do período em vez do sinal inerente (*Overfitting*). O Módulo de Otimização aplica penalidades quantitativas (Sharpe Deflacionado - DSR) validando, através de testes de hipótese formais, se o melhor resultado descoberto apresenta significância estatística em relação ao número total de tentativas avaliadas.

## Integração com Outros Módulos
- **Data (`src/data`)**: Invoca requisições sob demanda para garantir a amostragem cronologicamente atualizada para as extrações *out-of-sample* contínuas do MetaTrader.
- **Modeling (`src/modeling`) e Labeling (`src/labeling`)**: Atua como controlador hierárquico (Orquestrador) inicializando e estressando exaustivamente os módulos paramétricos em tempo de execução, alimentando e derivando as *folds* através do *PurgedKFold*.
- **Backtest (`src/backtest`)**: Depende integralmente dos submódulos do simulador, incluindo avaliadores estatísticos para DSR, e do processador Combinatorial (CPCV) e custos operacionais de liquidação da estrutura da Bolsa local.
- **Configurações Globais (`config/settings.py`)**: Consome os *namespaces* das variáveis estáticas que impõem fronteiras lógicas à exploração algorítmica do modelo (limites de árvores, volumes absolutos e diretivas do Optuna).

## Mapa de Funções e Componentes Chave

### 1. Otimização Bayesiana Multietapas (`tuner.py`)
Abstrai a pesquisa paramétrica orientada pelo estimador TPE (Tree-structured Parzen Estimator) empregado na biblioteca *Optuna*.
A resolução paramétrica do sistema de negociação foi decomposta em duas avaliações (Phase 1 e Phase 2), visando contornar a explosão combinatória dimensional na rede.
- `objective_phase1`: Segmento que foca nas configurações geométricas restritas da tripla barreira. Avalia empiricamente os limiares de Volatilidade e os filtros dinâmicos de Eventos. A função submete uma métrica punitiva de ajuste (fitness penalty) àqueles vetores estocásticos que resultam em uma margem populacional insustentável de trades (e.g., baixa frequência extrema de execuções perante a janela de estudo).
- `objective_phase2`: Congela os resultados ótimos definidos em Phase 1 e transiciona a otimização aos *hiperparâmetros intrínsecos* dos preditores de árvores de regressão binária XGBoost, testando os balanceamentos da floresta de decisão sem provocar ruídos de causalidade nas margens físicas de negociação já calibradas.
- `run_optimization`: Função orquestradora unificando e persistindo as simulações e, como *output* final, realizando a compilação cruzada na análise Deflacionada (DSR) dos resultados globais contra todos os *trials* rejeitados e aceitos no percurso experimental.

### 2. Gestão Assíncrona dos Parâmetros Locais (`params_store.py`)
Mecanismo concebido para armazenar em formato acessível o último banco otimizado.
- `save_optimized_params`: Serializa os esquemas validados para formatações JSON, efetivando rotinas seguras de `INSERT OR REPLACE` no SQLite, retendo os metadados avaliativos (Sharpe e DSR alcançado) em carimbos temporais de *deployment*.
- `load_optimized_params` e `get_continuous_symbol`: Sistema de contingência transacional. Caso os parâmetros armazenados sejam relativos a derivativos em constante vencimento (`WINV25`), a função incorpora rotinas via expressões regulares (RegEx) que mapeiam os parâmetros herdados de um ciclo de cotação superior consolidado (`WIN$`), assegurando robustez temporal contínua na chamada autônoma.

### 3. Pipeline Otimizador Executivo CLI (`run_opt.py`)
Encapsulamento utilitário projetado para execução contínua ou autônoma via crontab (e.g., no encerramento noturno do pregão). Este submódulo agrupa e coordena a atualização das cotações cruas do MetaTrader seguida pela invocações das rotinas *Optuna*.

**Exemplo de Inicialização do Otimizador:**
```python
import pandas as pd
from src.optimization.run_opt import run_optimization
from src.optimization.params_store import load_optimized_params, params_exist

otimizacao_log = run_optimization(
    df=barras_raw_db,
    interval="5m",
    n_trials_phase1=100,
    n_trials_phase2=100,
    symbol="WDO$"
)

if params_exist("WDO$"):
    best_config = load_optimized_params("WDO$")
    print(f"Base de Parâmetros Operacionais: {best_config['metadata']}")
```

## Especificações Analíticas: Estimação de Distribuição e Testes de Viés

### Tree-structured Parzen Estimator (TPE)
Uma busca parametrizada que evita os desperdícios iterativos do grid estático tradicional. O algoritmo de Optuna (TPE) ordena os últimos testes baseando-se no desempenho retornado e segmenta progressivamente o ranqueamento populacional através da distribuição Gaussiana com resultados empíricos ruins $l(x)$ separada das densidades campeãs com alta assertividade $g(x)$.
Deste modo, aproxima a densidade da Expetativa de Melhoria (*Expected Improvement* - EI), avançando a probabilidade estocástica focado na equação:
$$ EI(x) \propto \frac{l(x)}{g(x)} $$
Isolando os novos testes aos sub-espaços de exploração cujas configurações pareçam promissoras para a minimização das perdas do modelo contínuo.

### Teste de Hipótese para o Viés de Seleção (DSR - Deflated Sharpe Ratio)
O cálculo analítico sobre a otimização refuta os casos de viés subjacentes (Data Snooping) impostos pelas repetições (*Trials*). Com a execução de testes estendidos, a volatilidade do espaço assegura que, sob probabilidade randômica iterativa ($N_{trials}$), resultados de alta magnitude ($Max(SR)$) ocorram arbitrariamente por pura sorte estatística ($H_0$).
Dado a variância agregada de todos os testes iterativos $\sigma^2_{SR}$, define-se o pico Sharpe estimado (Máximo Esperado) perante a constante padronizada Euler-Mascheroni $\gamma$:
$$ \text{E}[max(SR)] = \sigma_{SR} \left[ (1 - \gamma) \cdot \Phi^{-1}\left(1-\frac{1}{N_{trials}}\right) + \gamma \cdot \Phi^{-1}\left(1-\frac{1}{N_{trials} \cdot e}\right) \right] $$
O Índice de Sharpe Deflacionado computa e isola a dispersão convertendo esse limite referencial contínuo para o Valor-P através da função de Distribuição Cumulativa Normal (ajustado ao Skew e Curtose da série), refutando a adoção do algoritmo quando a validação probabilística da robustez exata for inferior a $\sim 0.95$ ($\ge 95\%$ significância analítica).
