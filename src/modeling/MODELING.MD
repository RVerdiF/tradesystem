# Módulo de Modelagem e Validação (src/modeling)

Este módulo é responsável pelo treinamento do meta-modelo (modelo secundário), pela validação estatística rigorosa e pelo dimensionamento de posição (*bet sizing*). A arquitetura foca na robustez contra o sobreajuste (*overfitting*) e no controle do vazamento de dados (*data leakage*).

A implementação segue as diretrizes de **López de Prado** (*Advances in Financial Machine Learning*, Cap. 3, 7 e 10).

---

## 1. Componentes e Responsabilidades

O subsistema de modelagem é composto por quatro módulos principais:

### 1.1 Meta-Modelo (`classifier.py`)
*   **Classificador Binário**: Implementa modelos baseados em árvores (XGBoost ou RandomForest) para prever a probabilidade de sucesso de um sinal Alpha.
*   **Output de Probabilidade**: O modelo não retorna apenas 0 ou 1, mas uma probabilidade contínua que é utilizada pelo motor de execução para filtrar operações e dimensionar lotes.

### 1.2 Prevenção de Vazamento de Dados (`purge_embargo.py`)
*   **Purging**: Remove do conjunto de treinamento as observações que se sobrepõem temporalmente com o conjunto de teste.
*   **Embargo**: Adiciona um intervalo extra após o conjunto de teste para garantir que a autocorrelação serial não cause vazamento de informações para o treinamento (Cap. 7 AFML).

### 1.3 Validação Cruzada (`cv.py`)
*   **Purged K-Fold CV**: Uma evolução do K-Fold tradicional adaptada para séries temporais financeiras. Ela integra as técnicas de Purging e Embargo em cada dobra (*fold*) da validação, garantindo que as métricas de performance sejam realistas e não infladas por vazamento de dados.

### 1.4 Dimensionamento de Posição (`bet_sizing.py`)
*   **Fração de Kelly**: Utiliza o critério de Kelly (ou Kelly Fracionário) para definir o tamanho da posição baseando-se na probabilidade retornada pelo meta-modelo e na relação risco/retorno esperada.
*   **Ajuste por Probabilidade**: Permite que o sistema aumente a exposição em sinais de alta confiança e reduza (ou zere) em sinais duvidosos.

---

## 2. Fluxo de Modelagem

O processo de treinamento e validação segue esta sequência:

1.  **Preparação**: As features e labels (Meta-Labels) são carregadas.
2.  **Definição de Dobras**: O `cv.py` define os intervalos de treino/teste com Purge e Embargo.
3.  **Treinamento**: O classificador é treinado em cada dobra.
4.  **Avaliação**: Métricas como Log-Loss, Precision e F1-Score são calculadas de forma agregada.
5.  **Bet Sizing**: A função de Kelly é calibrada com base nos resultados da validação cruzada.

---

## 3. Exemplo de Uso

```python
from src.modeling.classifier import train_meta_model
from src.modeling.cv import PurgedKFold
from src.modeling.bet_sizing import get_kelly_size

# 1. Configurar Validação Cruzada com Purge e Embargo
cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

# 2. Treinar Meta-Modelo
model = train_meta_model(X, y, cv=cv)

# 3. Calcular Tamanho da Posição para novo sinal
prob = model.predict_proba(X_new)[:, 1]
size = get_kelly_size(prob, win_rate=0.55, edge=1.5)
```

---

## 4. Referências Técnicas
*   *Advances in Financial Machine Learning* (Cap. 7: Cross-Validation e Cap. 10: Bet Sizing), Marcos López de Prado.
*   *The Elements of Statistical Learning*, Hastie et al.
