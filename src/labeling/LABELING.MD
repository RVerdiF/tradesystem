# Módulo de Rotulagem (src/labeling)

Este módulo é responsável pela definição da variável alvo (*target*) para o treinamento dos modelos. Ele implementa o **Triple Barrier Method (TBM)** e o **Meta-Labeling**, técnicas fundamentais para evitar o viés de look-ahead e lidar com a natureza não estacionária dos dados financeiros.

A lógica de rotulagem segue os princípios de **López de Prado** (*Advances in Financial Machine Learning*, Cap. 3).

---

## 1. Componentes e Responsabilidades

O subsistema de labeling é composto por quatro módulos principais:

### 1.1 Volatilidade Dinâmica (`volatility.py`)
*   **Estimativa de Volatilidade (EWMA)**: Calcula a volatilidade diária usando uma média móvel exponencial ponderada (EWMA).
*   **Target Sizing**: Define o tamanho dos alvos (Take Profit e Stop Loss) de forma dinâmica, baseando-se na volatilidade atual do ativo. Isso garante que o sistema seja adaptável a diferentes regimes de mercado.

### 1.2 Triple Barrier Method (`triple_barrier.py`)
*   **As Três Barreiras**:
    1.  **Barreira Superior (Horizontal)**: Take Profit (TP), atingida se o preço sobe até o alvo.
    2.  **Barreira Inferior (Horizontal)**: Stop Loss (SL), atingida se o preço cai até o limite.
    3.  **Barreira de Tempo (Vertical)**: Saída por tempo, atingida se nenhuma barreira horizontal for tocada dentro do limite de tempo pré-definido.
*   **Rotulagem Primária**: O rótulo é determinado por qual barreira foi tocada primeiro, permitindo capturar o resultado real de uma operação.

### 1.3 Modelo Primário (Alpha) (`alpha.py`)
*   **Geração de Sinais de Tendência**: Define a direção inicial da operação (Buy/Sell/Neutral). 
*   **Identificação de Side**: O sinal do modelo Alpha é o ponto de partida para o processo de Meta-Labeling.

### 1.4 Meta-Labeling (`meta_labeling.py`)
*   **Rotulagem Secundária**: Em vez de prever a direção (Buy/Sell), o Meta-Labeling prevê se o sinal do modelo Alpha será bem-sucedido ou não (0 ou 1).
*   **Filtragem de Falsos Positivos**: Esta técnica permite que o sistema filtre sinais de baixa probabilidade de lucro, aumentando a precisão e o Sharpe Ratio global da estratégia (Cap. 3.6 AFML).

---

## 2. Fluxo de Rotulagem

O processo completo de rotulagem segue estes passos:

1.  **Cálculo da Volatilidade**: Define os alvos dinâmicos baseados no retorno recente.
2.  **Geração de Sinais Alpha**: O modelo primário identifica oportunidades de entrada.
3.  **Aplicação do Triple Barrier**: Para cada entrada, o sistema rastreia o preço até que uma das três barreiras seja tocada.
4.  **Meta-Labeling**: O resultado do Triple Barrier é comparado com o sinal Alpha para gerar o rótulo binário (Sucesso/Falha) para o modelo secundário.

---

## 3. Exemplo de Uso

```python
from src.labeling.volatility import get_daily_vol
from src.labeling.triple_barrier import apply_tbm
from src.labeling.meta_labeling import get_meta_labels

# 1. Calcular volatilidade dinâmica
vol = get_daily_vol(df["close"], span=100)

# 2. Aplicar Triple Barrier
events = apply_tbm(df["close"], t_events=events_list, pt_sl=[1, 1], target=vol)

# 3. Gerar labels binários para Meta-Model
labels = get_meta_labels(events, side=df["alpha_side"])
```

---

## 4. Referências Técnicas
*   *Advances in Financial Machine Learning* (Cap. 3), Marcos López de Prado.
*   *Machine Learning for Asset Managers*, Marcos López de Prado.
