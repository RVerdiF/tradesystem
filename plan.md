Analisando os scripts fornecidos (`run_opt.py`, `tuner.py`, `metrics.py` e `dsr.py`), estruturei os passos exatos para implementar a gestão dinâmica de posições e evoluir o motor do Optuna.

---

### 1. Evolução da Tripla Barreira: Breakeven e Execução Parcial

Atualmente, o seu arquivo `triple_barrier.py` implementa uma lógica de "primeiro toque" (First-Touch). Para permitir Stops Dinâmicos (como mover para a entrada após X% de ganho), você precisará alterar a abstração matemática da Tripla Barreira para um sistema de **Barreiras Dinâmicas Baseadas em Estado**.

**Como implementar na prática (Microestrutura):**
Você precisará alterar a função `@njit _find_first_touch` para rastrear o estado. 

* **Lógica de Breakeven:** Adicione um parâmetro `breakeven_activation` (ex: 0.5, significando que se o preço atingir 50% do alvo do Take Profit, o Stop Loss se move para 0).
* **Exemplo de modificação no kernel (Numba):**
    ```python
    @njit
    def _find_dynamic_touch(close_values, start, end, entry_price, side, upper, lower, be_trigger):
        breakeven_active = False
        for i in range(start + 1, end + 1):
            ret = (close_values[i] / entry_price - 1.0) * side
            
            # 1. Checa ativação de Breakeven
            if not breakeven_active and ret >= (upper * be_trigger):
                lower = 0.0001 # Move o Stop para a entrada (com leve margem para custos)
                breakeven_active = True
                
            # 2. Checa Barreiras
            if ret >= upper:
                return i, ret, 0 # TP atingido
            if ret <= lower:
                return i, ret, 1 # SL (ou Breakeven) atingido
                
        return end, (close_values[end] / entry_price - 1.0) * side, 2 # Tempo esgotado
    ```
* **Impacto no Meta-Labeling:** Isso é altamente benéfico. O Meta-Model passará a aprender não apenas "Lucro vs Perda", mas quais características de mercado (features) permitem que a operação alcance a segurança do *breakeven* sem ser estopada prematuramente pela volatilidade.

---

### 2. Otimização de Parâmetros (Optuna) - Onde buscar Alpha

Avaliando o seu arquivo `tuner.py`, a estrutura do estudo Bayesiano está muito bem montada, mas a **Função Objetivo (`objective`)** possui vulnerabilidades clássicas de otimização heurística.

#### A. O Problema das "Penalidades Hardcoded" (Cliffs)
No seu `tuner.py`, você utiliza:
```python
if results["n_trades"] < optimization_config.min_trades:
    return 0.0
```
O algoritmo do Optuna (TPE - Tree-structured Parzen Estimator) aprende mapeando um espaço contínuo. Se você retorna `0.0` abruptamente para menos de 30 trades, você cria um "penhasco matemático" cego. O Optuna não consegue saber se uma configuração com 29 trades era quase perfeita e deveria ser levemente ajustada.
**Solução:** Substitua por uma penalidade contínua. Exemplo: `fitness = sharpe * (n_trades / min_trades)` caso seja menor que o mínimo.

#### B. Aprimoramento da Função de Fitness (Sharpe vs DSR)
Atualmente, o otimizador busca o maior Sharpe Ratio puramente e, ao final, roda a validação do Deflated Sharpe Ratio (DSR). 
* **Melhoria:** Para ativos de 5-15 minutos, o Sharpe puro mascara riscos de cauda grossa. A função objetivo ideal seria maximizar uma métrica de consistência. Como o DSR exige o histórico de todos os *trials* (impossível avaliar no meio do processo), você deve otimizar o **Calmar Ratio** (Retorno / Maximum Drawdown), que já está disponível no seu script `metrics.py`. Uma estratégia de day-trade que sobrevive sem Drawdowns massivos é estatisticamente mais viável do que uma com Sharpe alto impulsionado por 3 ou 4 *trades* de extrema sorte.

#### C. Variáveis Ocultas a serem Otimizadas
Para extrair verdadeiro Alpha no Optuna, adicione estas variáveis ao `trial.suggest_...`:
1.  **A Barreira Vertical (`max_holding_periods`):** No *medium-frequency*, o custo de oportunidade é altíssimo. O Optuna deve encontrar se é melhor segurar uma operação ruim por 10 barras ou fechá-la compulsóriamente na 4ª barra.
2.  **O Gatilho de Breakeven (`be_trigger`):** Se implementar a lógica citada no passo 1, deixe o Optuna descobrir se é melhor mover o Stop para a entrada com 30%, 50% ou 70% de percurso percorrido em direção ao alvo.
3.  **Threshold do CUSUM (`cusum_threshold`):** Você já o otimiza (linha 22 do `tuner.py`), o que é excelente. Contudo, certifique-se de que a margem de busca não está tão alta a ponto de gerar eventos insuficientes em timeframes de 15m.
4.  **Meta-Model Decision Threshold:** Normalmente assumimos que o Meta-Model emite ordem se a probabilidade `p > 0.5`. Permitir que o Optuna busque esse *threshold* (ex: só executar se a convicção do ML for `p > 0.65`) eleva o Sharpe e o *Sharpe Lift* dramaticamente.

**Próximo Passo Recomendado:**
Faça uma cópia do `tuner.py` e altere a variável `fitness` para retornar o **Calmar Ratio** descontado pela variância entre os *folds* do CPCV. Isso criará modelos ligeiramente menos lucrativos no papel, mas com curvas de capital imensamente mais suaves para o ambiente real (MT5).