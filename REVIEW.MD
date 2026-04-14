# TradeSystem5000: Revisão Profunda do Sistema e Avaliação de Sinais

Esta é uma avaliação detalhada do funcionamento do sistema TradeSystem5000, focando na integração de seus componentes (Alpha, Meta-Modelo, Pipeline) e na fidelidade do cálculo das features matemáticas.

## 1. Problema Estrutural Crítico: Alpha Models e Séries Estacionárias

O problema mais grave identificado na base de código encontra-se na definição e no uso do `TrendFollowingAlpha` (e `MeanReversionAlpha`) em `src/labeling/alpha.py`.

### A Lógica Atual
A documentação em `TrendFollowingAlpha` cita corretamente Marcos López de Prado, afirmando que modelos preditivos sobre séries não-estacionárias sofrem de vazamento e regressões espúrias. Para contornar isso, o pipeline diferencia fracionariamente o preço (`frac_diff_ffd`), transformando-o em uma série estacionária (memória controlada, centrada em torno de uma média constante, usualmente zero).

Entretanto, o código aplica um **cruzamento de Médias Móveis Exponenciais (EMAs)** *diretamente* sobre essa série estacionária:
```python
price_series = df["close_fracdiff"]
ema_fast = price_series.ewm(span=self.fast_span, adjust=False).mean()
ema_slow = price_series.ewm(span=self.slow_span, adjust=False).mean()
```

### O Desalinhamento com a Realidade
O conceito de **Trend Following** (Seguidor de Tendência) pressupõe matematicamente a existência de não-estacionaridade. Um cruzamento de médias sobre o preço real indica que o preço está ganhando tração em uma direção persistente.
Se você aplica médias móveis em uma série estritamente estacionária (que, por definição, não tem tendência e orbita sua média, frequentemente oscilando em torno do zero de forma estocástica), o resultado **não é a captura de uma tendência**, mas sim a detecção de micro-flutuações e ruído. O sinal gerado por este modelo é puramente errático e não reflete o "Momentum" do ativo no mundo real.

A literatura de AFML prega que a **máquina** (Meta-Modelo ou modelo preditivo final) deve operar sobre a série estacionária. Porém, o **sinal Alpha direcional baseado em regras** (heurística do trader) deve refletir o comportamento não-estacionário. A rotulação de eventos (CUSUM, Triple Barrier) atua justamente como ponte.

**Impacto:** Seus sinais de entrada Alpha são aleatórios. Consequentemente, o meta-modelo XGBoost está tentando classificar se o "ruído" vai atingir o take-profit, o que limita severamente a performance e gera falsos underfittings.

## 2. Indicadores e Feature Engineering

Ao revisar a pasta `src/features`, as features parecem sólidas e alinhadas ao plano de refatoração recente, com exceções em sua integração.

### 2.1 Equação de Garman-Klass
O cálculo de volatilidade Intraday de Garman-Klass foi introduzido via plano, e a fórmula implementada encontra-se correta teoricamente:
```python
0.5 * [log(High/Low)]^2 - (2*log(2) - 1) * [log(Close/Open)]^2
```
A verificação no código (`indicators.py`) confirma essa implementação. O fato de usar logaritmos dos spreads mitiga a heteroscedasticidade, criando uma excelente feature de risco para o meta-modelo.

### 2.2 Microestrutura (VSA, OFI, VPIN)
Os cálculos relacionados ao volume e agressão, estimando o Tick Rule com `np.sign(close.diff())`, são precisos. Estão alinhados com a estimativa clássica de VPIN para tempos agregados, sendo normalizados pelo volume para capturar exaustivamente desequilíbrios (*Volume Imbalance*).

### 2.3 Cálculo Fracionário (FracDiff FFD)
O código para a convolução FFD (`frac_diff_ffd` e kernel `numba`) usa uma janela estrita de pesos que não vazam informações futuras (`len(weights)` e truncamento na convolução). Essa é uma implementação canônica impecável de Marcos López de Prado.

## 3. Avaliação da Integração e Pipeline (Backtest vs. Execução)

A integração entre Backtest e Execução `main_backtest.py` vs `main_execution.py` apresenta um forte acoplamento no modo de salvar/carregar modelos, mas estruturalmente o pipeline é obediente ao pipeline AFML.

**No entanto, há uma discrepância perigosa de pipeline de dados:**
Em `src/main_execution.py`, durante a execução do motor (*Live/Paper*):
```python
features["ffd"] = frac_diff_ffd(df["close"], d=self.optimal_d)
...
df["close_fracdiff"] = ffd_series
```
Isso calcula a diferenciação fracionária em tempo real usando `df["close"]` e `optimal_d`. Porém, a série diferenciada só se estabiliza adequadamente e converge à verdadeira variância fracionária se for calculada em uma janela profunda de histórico. Se o snapshot `df` passado em modo *Live* for muito curto (ex: 500 barras, enquanto FFD requer ~1000 barras para perder peso significativo, dependendo do `threshold`), o `ffd` sofrerá distorções de borda severas em produção comparado ao backtest, o que invalidará as previsões do modelo XGBoost.

## 4. O Sistema de Rotulagem (Triple Barrier)

A função `apply_triple_barrier` (Semântica V2) corrige um erro frequente em AFML, que é ignorar as máximas e mínimas intrabar (O que causa look-ahead bias). O sistema corretamente prioriza as Mínimas e Máximas, resolve empates punindo o sinal com a Barreira SL e considera a ativação *Break-Even* antes da avaliação. O cálculo e a resolução de sinais para a criação das labels +1/0/-1 operam exatamente como se espera.

## 5. Recomendações e Próximos Passos (Ação Urgente)

1. **Desacoplar Alpha de FracDiff:**
   No módulo `src/labeling/alpha.py`, modifique `TrendFollowingAlpha` e `MeanReversionAlpha` para operar sobre `df["close"]` (a série real de preço) em vez de `df["close_fracdiff"]`. Modelos primários baseados em regras técnicas foram construídos empiricamente em cima do preço real e dependem da não-estacionaridade para capturar distorções de mercado.
2. **Buffer Histórico na Execução:**
   Certifique-se de que o DataFrame alimentado à classe `LivePipeline` em `src/main_execution.py` no método `__call__` possua um comprimento de barras substancial (muito superior ao truncamento dos pesos da janela FFD) para não induzir *burn-in error* (viés de inicialização de convolução).
3. **Métricas de Overfitting de Meta-Model:**
   Atente-se aos limites de AUC reportados (que ocasionalmente chegam a `> 0.80` nos logs descritos em código). Para os próximos treinamentos, monitore a restrição que já está posta na classe base (fixar `scale_pos_weight=1.0` ajuda muito no viés natural).
