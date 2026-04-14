# Módulo de Modelagem e Validação (src/modeling)

## Objetivo Detalhado
O escopo principal deste módulo é orquestrar o processo de modelagem preditiva utilizando as variáveis derivadas ($X$) e os alvos estabilizados por barreira ($y$) sob um referencial supervisionado rigoroso (Classificação Binária). Contudo, em virtude das propriedades inerentes de séries financeiras temporais, o fluxo de *fitting* tradicional empregado em bases independentes não se aplica sem gerar superestimação de performance severa (*Data Leakage*).

Desta forma, o módulo engloba ferramentas projetadas especificamente para mitigação dessa fuga de dados pela adoção do modelo de purga temporal (Purging) e inibição da autocorrelação (Embargo). Na extremidade posterior à predição (Inferência), este módulo traduz a avaliação estocástica (probabilidades do modelo) em frações reais de alocação de risco dimensionadas por variantes do Critério de Kelly (*Bet Sizing*).

## Integração com Outros Módulos
- **Labeling (`src/labeling`) e Features (`src/features`)**: Depende integralmente da carga dos vetores correspondentes às features ($X$) e do dicionário lógico processado de barreiras e meta-labels ($y$).
- **Optimization (`src/optimization`)**: Dispõe o iterador modificado de amostragem K-Fold (`PurgedKFold`) e o pipeline preconfigurado do algoritmo XGBoost, oferecendo à otimização Bayesiana o espaço para o teste serial das métricas (Sharpe/Calmar) durante a tunagem de hiperparâmetros.
- **Execution (`src/execution`)**: Na configuração operacional *Live* ou na avaliação de *Backtest*, expõe funções estáticas (`predict_meta_model`) encarregadas da classificação e dimensionamento imediato das ordens pela calibração final do `bet_sizing`.

## Mapa de Funções e Componentes Chave

### 1. Tratamento de Fuga de Dados: Purging e Embargo (`purge_embargo.py`)
A divisão aleatória de partições (tais como *train_test_split* do `scikit-learn`) provoca que dados do horizonte futuro (período de teste) sejam enxertados aleatoriamente nos dados passados de validação. Devido à autocorrelação adjacente ou da contiguidade inerente da tripla barreira, isso causa vazamento (*Data Leakage*).

- `get_train_times`: Aplica a "Purga". Varre os *timestamps* atestando intersecções parciais ou integrais entre os períodos da tripla barreira de operações destinadas ao treino com as janelas demarcadas exclusivamente ao pacote de teste, eliminando amostras ambíguas da base de validação.
- `apply_embargo`: Estabelece a margem condicional de *Embargo*. A correlação direcional persiste após o corte sequencial dos testes. Impõe a exclusão temporária (baseada em *TimeDelta* percentual ou explícito) das primeiras observações adjacentes ao período final da janela do teste.

### 2. Validação Cruzada Otimizada (`cv.py`)
Implementação estruturada para compatibilidade nativa com o ecossistema `scikit-learn`, assegurando que todos os blocos de teste processados pelo classificador sejam completamente desacoplados.
- `PurgedKFold`: Classe iteradora modificada que unifica as rotinas de Purga e Embargo a fim de particionar o DataFrame cronológico em blocos contíguos validados.

**Exemplo do Setup de Validação Purificada e Embargada:**
```python
from src.modeling.cv import PurgedKFold
from src.modeling.classifier import train_meta_model
import xgboost as xgb

cv_generator = PurgedKFold(
    n_splits=5,
    samples_info=df_trade_timeline,
    pct_embargo=0.01
)

modelo_validado = train_meta_model(X_features, y_meta_labels, cv=cv_generator)
```

### 3. Modelagem Supervisionada de Classificação (`classifier.py`)
Encapsula o fluxo do *XGBoost* dimensionando correções para distribuições não uniformes, nas quais a preponderância natural da classe majoritária (Stop Loss - 0) ofusca ou anula as raras convergências de Take Profit (1).
- `train_meta_model`: Efetua o processo completo, atribuindo rebalanceamento implícito (`scale_pos_weight`) caso as classes sejam flagrantemente desbalanceadas.
- `predict_meta_model`: Extrai o escalonamento estrito das probabilidades (`predict_proba`) que sustentam a continuidade analítica exigida pelas frações de alocação subsequentes.

### 4. Dimensionamento e Modulação de Exposição (`bet_sizing.py`)
Transições discretas de predições (*True* / *False*) falham na adaptação probabilística do gerenciamento de carteiras. Este submódulo implementa a premissa quantitativa de que previsões carregam instâncias contínuas de acurácia, necessitando reflexão algorítmica exata na quantidade apostada (Bet-Sizing).
- `compute_kelly_fraction`: Integra a variável escalar do estimador binário contra a razão Risco/Retorno das margens da operação, retornando a proporção mitigada sob limites contínuos predeterminados entre $[0, 1]$.
- `discretize_bet`: Adota funções lineares convertendo as sugestões numéricas ideais para arranjos modulares transacionáveis nos patamares discretos restritivos exigidos pelas corretoras (e.g., Lotes mínimos Múltiplos).

**Exemplo do Dimensionamento (Kelly Sizing):**
```python
from src.modeling.bet_sizing import compute_kelly_fraction, discretize_bet

prob_win = model.predict_proba(current_market_features)[:, 1]

kelly_size_perc = compute_kelly_fraction(
    prob_win=prob_win,
    odds=2.0,
    fraction=0.5
)

lotes_alvo = discretize_bet(kelly_size_perc, max_position=10, step_size=1)
```

## Especificação Matemática: Isolamento e Alocação

### Purga e Embargo: Isolamento Causal
Dada a premissa de um *Fold* em que a janela transacional subjacente à matriz $I_{test}$ é compreendida entre $[t_{test\_start}, t_{test\_end}]$.
Uma trajetória operacional em $I_{train} = [t_0, t_1]$ será eliminada (*Purged*) em virtude do *Data Leakage* sob a circunstância imposta por:
$$ \text{Overlap} \implies (t_0 \le t_{test\_end}) \land (t_1 \ge t_{test\_start}) $$
Procedendo ao descarte adjacente subsequente (*Embargo*) se:
$$ \text{Embargo} \implies (t_0 > t_{test\_end}) \land (t_0 \le t_{test\_end} + E) $$
Onde $E$ equivale ao coeficiente do subconjunto proporcional da margem paramétrica de atraso $h$.

### Modulação Adaptativa do Critério de Kelly (Fractional Kelly)
Na busca da maximização assimptótica dos retornos geométricos sob proteção rigorosa, adaptamos a avaliação das probabilidades $\hat{p}$ preditas na extremidade contínua (*Softmax/Logloss*) do classificador.
Sendo $b$ a razão intrínseca de recompensa pelo risco da arquitetura estipulada pela Tripla Barreira (Take Profit $/$ Stop Loss), o percentual nominal inicial de alocação agressiva derivado de Kelly $\hat{f}$ assume:
$$ \hat{f} = \hat{p} - \frac{1 - \hat{p}}{b} $$
Introduzindo fator atenuante conservador $c$ (*Fractional Kelly* restritivo, e.g., $c = 0.5$ para *Half-Kelly*), temos o peso condicionado final na proporção total aceita para transação em lote $L$:
$$ F_{final} = \max(0, \hat{f} \cdot c) $$
Sinais direcionais onde $\hat{p}$ resulta marginalmente negativo ou $\hat{f} \le 0$ são anulados. Garante-se, portanto, a omissão total de apostas ($F_{final} = 0$) em cenários com expectativa estatística matematicamente desfavorável.