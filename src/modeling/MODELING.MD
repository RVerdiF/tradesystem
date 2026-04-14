# Módulo de Modelagem e Validação (src/modeling)

## Objetivo
O Módulo de Modelagem é o coração preditivo do TradeSystem5000. Ele unifica as features e os rótulos (labels) para treinar um modelo de Machine Learning focado em resolver o problema de classificação binária gerado pelo Meta-Labeling (ver `src/labeling`). Diferente dos pacotes clássicos de ML (como `scikit-learn`), a arquitetura aqui é customizada pesadamente para séries temporais financeiras, abordando o grave problema de sobreajuste (*overfitting*) e vazamento de dados (*data leakage*) inerente aos dados temporalmente sobrepostos. Também é responsável por dimensionar o tamanho financeiro das apostas (Bet Sizing) em tempo real após a inferência.

## Integração com outros módulos
- **Labeling (`src/labeling`) e Features (`src/features`)**: Consome o histórico temporal com inputs ($X$) de features processadas e outputs ($y$) de meta-labels e barreira tripla.
- **Optimization (`src/optimization`)**: O sintonizador bayesiano utiliza as dobras (folds) da validação cruzada exposta aqui para encontrar hiperparâmetros ótimos para o `classifier`.
- **Execution (`src/execution`)**: No ambiente de trading real ou simulação backtest, consome as funções de inferência de `classifier.py` e extrai o dimensionamento de ordens via `bet_sizing.py`.

## Mapa de funções

### Validação Cruzada Especializada (`cv.py`)
- `PurgedKFold`: Implementa uma classe (compatível com a interface do scikit-learn) para dividir o *dataset* em dobras de treino e teste não sucessivas, purificando sobreposições para impedir vazamento (look-ahead) provocado pelas barras dependentes (como a duração não-fixa das triplas barreiras).

### Saneamento de Amostras (`purge_embargo.py`)
- `get_train_times`: Aplica o *Purging*, varrendo o conjunto temporal e excluindo qualquer observação de treino que inicie antes e termine dentro (ou depois) da janela temporal de teste.
- `apply_embargo`: Aplica o *Embargoing*, cortando uma faixa de amostras do treino posicionadas imediatamente após o término da dobra de teste. Previne vazamento pela autocorrelação serial comum nos mercados de capitais.
- `purge_and_embargo`: Orquestra ambas as funções acima fornecendo os limiares rigorosos de subconjuntos de treino limpos.

### Meta-Modelo Classificador (`classifier.py`)
- `train_meta_model`: Rotina global que compila o Pipeline de aprendizado (usualmente um `XGBoost` ou `RandomForest`). Ajusta o `class_weight` dinâmico caso os sinais negativos desbalanceiem o *dataset*.
- `predict_meta_model`: Encapsula a inferência e retorna explicitamente o `predict_proba` (a confiança no modelo primário), um valor contínuo essencial para dimensionar as apostas, ao invés do tradicional output binário bruto.

### Dimensionamento da Posição / Bet Sizing (`bet_sizing.py`)
- `compute_kelly_fraction`: Deriva a agressividade do risco do trade calculando a fração do critério de Kelly, unindo a probabilidade extraída do classificador contra a razão Risco/Recompensa.
- `discretize_bet`: Converte as frações contínuas sugeridas por Kelly em número inteiro de contratos futuros negociáveis (ex: WINFUT), arredondando aos blocos e respeitando limites estritos.
- `apply_conviction_threshold`: Zera o Kelly / tamanho da posição se o Meta-Modelo reportar uma convicção insatisfatória, mitigando execuções de ruído.

### Avaliação de Features (`feature_evaluation.py`)
- *Utilidades para investigar importância preditiva de features, como MDA (Mean Decrease Accuracy) ou MDI, garantindo que colunas irrelevantes não diluam o sinal na modelagem.* (Implementação contextual das métricas financeiras).

## Lógica geral
Séries temporais financeiras não são vetores de dados independentes. Se o Trade A começa na hora 1 e acaba na hora 3, e o Trade B (que cai numa dobra de Teste do K-Fold) começa na hora 2, o algoritmo de ML "vê" o futuro (o fechamento das horas 2 e 3 está embutido nos retornos de A). Portanto, `cv.py` e `purge_embargo.py` fazem fatiamentos matemáticos temporais baseados no índice de tempo do trade ($t_0$) até a sua liquidação ($t_1$).
Só depois desse preparo laborioso que o XGBoost (em `classifier.py`) é efetivamente treinado.
Na fase de *Deploy*/Inferência, o sistema pega as probabilidades reportadas pelo classificador e passa pelo filtro rigoroso do critério de Kelly. Assim, o robô aposta grande em trades de alta clareza (90%+ de acerto) e pequeno em cenários ambíguos.

## Lógica matemática
1. **Purging & Embargoing**:
   Seja $I_{test} = [t_{test\_start}, t_{test\_end}]$ o intervalo de teste.
   Um trade de treino em $I_{train} = [t_0, t_1]$ é descartado (purged) se:
   $$ \text{Overlap} \implies (t_0 \le t_{test\_end}) \land (t_1 \ge t_{test\_start}) $$
   Após o expurgo de sobreposição, o Embargo exclui amostras de treino se:
   $$ \text{Embargo} \implies (t_0 > t_{test\_end}) \land (t_0 \le t_{test\_end} + E) $$
   Onde $E$ é a duração da fração de embargo $h$.

2. **Fração de Kelly (Half/Fractional Kelly)**:
   Baseado na probabilidade contínua gerada por um Sigmoid ou Softmax (XGBoost):
   $$ p = \text{prob\_win} $$
   $$ q = 1 - p $$
   O Kelly fracionário é expresso como a vantagem sobre as *Odds* (razão recompensa-risco do Alpha/Tripla Barreira):
   $$ f\_kelly = \left( p - \frac{q}{\text{Odds}} \right) \cdot \text{fraction\_limit} $$
   Se as Odds são $1:1$ (TP e SL são proporcionais ao risco):
   $$ f = (2p - 1) $$
   Apenas sinais com vantagem real ($p > 0.5$) resultarão num tamanho de aposta $> 0$.
