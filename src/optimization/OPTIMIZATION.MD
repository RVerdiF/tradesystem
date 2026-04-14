# Módulo de Otimização (src/optimization)

## Objetivo
O Módulo de Otimização visa sintonizar de forma autônoma e científica os hiperparâmetros vitais de todo o ecossistema (desde a sensibilidade de filtros até a profundidade das árvores do XGBoost). Ele se propõe a ser robusto contra o *Backtest Overfitting* (viés de seleção). Utilizando algoritmos Bayesianos (como TPE via biblioteca `Optuna`), o sistema roda duas fases sequenciais para achar a melhor configuração que sobreviva às validações cruzadas expurgadas (CPCV) e à punição implacável pelo Sharpe Ratio Deflacionado (DSR).

## Integração com outros módulos
- **Backtest (`src/backtest`) e Modeling (`src/modeling`)**: Interage intimamente com os ambientes que constroem as árvores e calculam o Calmar/Sharpe SR e DSR para testar hipóteses.
- **Data (`src/data`)**: Consome o `fetch_mt5_data` para obter a matéria prima de barras.
- **Execution (`src/execution`)**: Alimenta a execução "Live", salvando os parâmetros campeões no banco de dados SQLite que o Motor de Execução acessa na hora H.

## Mapa de funções

### Tuning Bayesiano e Testes (`tuner.py`)
- `objective_phase1`: Otimiza a macroestrutura geradora dos sinais primários (Alphas e Tripla Barreira), sintonizando os multiplicadores Take-Profit/Stop-Loss e limiares de *CUSUM* e Volatilidade. É penalizado se produzir poucos trades (buscando significância estatística).
- `objective_phase2`: Otimiza puramente os hiperparâmetros de Machine Learning (XGBoost) como profundidade máxima (`max_depth`), taxa de aprendizado (`learning_rate`) e pesos de classe para corrigir classes de labels desbalanceadas, fixando a geometria já descoberta na Fase 1.
- `run_optimization`: Função core que encapsula a orquestração do Optuna: Instancia os *studies*, roda a Fase 1 e em seguida a Fase 2, unifica os parâmetros e calcula o Deflated Sharpe Ratio (DSR) em cima de todo o experimento.

### Persistência de Parâmetros (`params_store.py`)
- `save_optimized_params`: Serializa em JSON e faz o *upsert* (insere ou substitui) da melhor configuração descoberta no banco SQLite (`tradesystem.db`).
- `load_optimized_params`: Consome os dicionários salvos por ativo, provendo uma utilidade poderosa de regressão para ativos contínuos (se você pede WING26 e ele tem WIN$ guardado, faz o fallback inteligente para o metadado principal do contrato de índice cheio).
- `params_exist`: Fast check booleano para constatar presença no repositório.

### Runner CLI (`run_opt.py`)
- *Ponto de entrada (Entrypoint) para rodar o tuner.* Ele inicializa a coleta de dados massivos do MetaTrader e chuta o loop assíncrono que delega processamento aos núcleos via paralelismo e imprime os retornos finais.

## Lógica geral
A premissa da otimização aqui baseada em *Lopez de Prado* condena abordagens comuns como Grid Search exaustivo sobre a mesma curva. Como testar um milhão de variações sempre encontra um setup lucrativo por puro viés estatístico (sorte de Data Snooping), a lógica foi particionada:
1. **Fase 1**: Define as características do "Campo e do Time" (Sinais de Volatilidade, Multiplicadores).
2. **Fase 2**: Ajusta as táticas do Classificador (Hiperparâmetros do ML), com as regras imutáveis de Fase 1.
3. No final, todas as simulações executadas fornecem sua distribuição de retornos para o avaliador DSR. Se o Sharpe atingido não for estatisticamente forte perante todas as falhas, o pipeline não salva os resultados, recusando a ilusão de um ótimo global falso.

## Lógica matemática
1. **Tree-structured Parzen Estimator (TPE)**:
   Algoritmo no núcleo do Optuna. Avalia iterativamente a distribuição $l(x)$ (piores resultados) e $g(x)$ (melhores resultados) de uma métrica de fitness (como o Ratio de Calmar ou Sharpe):
   $$ EI(x) \propto \frac{l(x)}{g(x)} $$
   O TPE explora as áreas com maior Expectativa de Melhoria ($EI$).

2. **Penalidades por baixa amostragem**:
   Na Fase 1, o sistema induz penalidades sobre os rácios brutos caso o número de negócios seja muito baixo.
   $$ \text{Penalizado} = \text{Fitness} \cdot \max(0.1, \frac{n\_trades}{\text{min\_trades\_esperado}}) $$
   Garantindo que a otimização não pare em uma geometria onde fez 2 trades em 5 anos dando 100% de lucro (quebra estatística).

3. **Fallback Contínuo (RegEx)**:
   Em `params_store.py`, Regex resolve os meses para contratos B3:
   $$ ^(WIN|WDO)[FGHJKMNQUVXZ]\d{2}$ \to WIN\$, WDO\$ $$
