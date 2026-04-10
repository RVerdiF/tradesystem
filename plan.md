Aqui está um plano de execução detalhado, dividido em fases lógicas, para implementar a redução de parâmetros e a nova engenharia de features no seu sistema.

Este plano assume a arquitetura padrão do seu projeto (como `src/features/`, `src/modeling/`, etc.).

---

### **Fase 1: Refatoração da Engenharia de Features (A Grande Limpeza)**
Nesta fase, vamos alterar os scripts responsáveis pela criação dos indicadores técnicos (provavelmente em um arquivo como `src/features/indicators.py` ou `src/data/build_features.py`).

**1.1. Remoção de Features Antigas**
* **Ação:** Comente ou exclua as funções que calculam MACD (e suas linhas de sinal/histograma), RSI e Bandas de Bollinger.
* **Objetivo:** Eliminar imediatamente 6 parâmetros do seu sistema (3 do MACD, 1 do RSI, 2 das Bandas).

**1.2. Implementação: Distância para Médias Móveis**
* **Ação:** Crie uma função para calcular duas médias móveis (ex: Rápida e Lenta, como EMA 9 e EMA 21). Em vez de usar o valor da média, calcule a distância percentual do Preço de Fechamento (Close) para cada média.
* *Fórmula sugerida:* `(Close - MA) / MA`
* **Objetivo:** Substituir a intuição de momentum do MACD e sobrecompra/sobrevenda do RSI por uma métrica normalizada e sem limites fixos.

**1.3. Implementação: Volatilidade de Garman-Klass**
* **Ação:** Implemente a equação de Garman-Klass utilizando as colunas Open, High, Low e Close.
* *Fórmula:* `0.5 * [log(High/Low)]^2 - (2*log(2) - 1) * [log(Close/Open)]^2` (aplicar uma média móvel suave sobre esse resultado, ex: janela de 10 a 20 períodos).
* **Objetivo:** Capturar a volatilidade intradiária de forma altamente eficiente sem adicionar multiplicadores (como nas Bandas de Bollinger).

**1.4. Implementação: Momentos Superiores (Skewness e Kurtosis)**
* **Ação:** Utilize o pandas para calcular a assimetria e curtose em uma janela rolante sobre os retornos do preço.
* *Código prático:* `df['returns'].rolling(window=20).skew()` e `df['returns'].rolling(window=20).kurt()`
* **Objetivo:** Fornecer ao XGBoost uma visão clara sobre "caudas gordas" e mudanças bruscas de regime direcional.

**1.5. Validação da Microestrutura (VSA)**
* **Ação:** Garanta que as features de Volume Spread Analysis (Volume, Spread de Preço, Direção da Barra) continuem intactas no dataset final.

---

### **Fase 2: Implementação do Módulo de Avaliação de Features**
Agora vamos preparar o terreno matemático para provar quais features realmente importam, seguindo a metodologia de López de Prado.

**2.1. Criação do Script de Avaliação**
* **Ação:** Crie um novo arquivo: `src/modeling/feature_evaluation.py`.
* **Conteúdo:** Cole as duas funções fornecidas na resposta anterior (`evaluate_features_shap` e `evaluate_features_mda`).
* **Dependências:** Certifique-se de instalar as bibliotecas necessárias rodando `pip install shap scikit-learn matplotlib`.

**2.2. Integração no Pipeline de Treinamento**
* **Ação:** No seu script principal de treinamento (provavelmente onde o `MetaClassifier` é instanciado e o `.fit()` é chamado), importe o módulo criado.
* **Fluxo:** Logo *após* o treinamento do modelo primário, adicione a chamada das funções de avaliação para que o sistema gere os relatórios automaticamente a cada ciclo.

---

### **Fase 3: Execução e Auditoria Matemática (A "Faxina" Real)**
Com o código pronto, é hora de rodar o sistema e tomar decisões baseadas em dados.

**3.1. Geração do Modelo Baseline**
* **Ação:** Rode um backtest completo de treinamento utilizando seu novo conjunto de dados (Apenas VSA, Distância das Médias, Garman-Klass, Skewness e Kurtosis).

**3.2. Análise por Permutação (MDA)**
* **Ação:** Observe o log gerado pelo `evaluate_features_mda`.
* **Regra de Corte:** Se qualquer feature (incluindo as de VSA) apresentar um `MDA_Mean_Drop` menor que zero ou estatisticamente insignificante (ex: menor que 0.001), **ela deve ser removida permanentemente do código**. Isso significa que o modelo prevê melhor quando essa variável é aleatorizada.

**3.3. Análise Explicativa (SHAP)**
* **Ação:** Abra a imagem gerada (`shap_summary.png`) e os logs do `evaluate_features_shap`.
* **Verificação:** Confirme se as features com alto valor SHAP fazem sentido econômico. Por exemplo, veja se picos altos de Garman-Klass (volatilidade) estão reduzindo a probabilidade de trades (comportamento de aversão ao risco esperado do Meta-Model).

---

### **Fase 4: Restrição da Otimização (Optuna)**
Para garantir que não voltaremos ao ciclo de *overfitting*, precisamos travar o otimizador.

**4.1. Limite de Hiperparâmetros de Busca**
* **Ação:** Vá ao arquivo de configuração do seu otimizador (ex: `src/optimization/tuner.py` ou `config/settings.py`).
* **Restrição:** Conte quantas variáveis o Optuna está tentando ajustar. Remova a busca para os antigos MACD/RSI/Bollinger.
* **Meta Final:** Você deve ter **no máximo 10 parâmetros totais** sendo otimizados pelo algoritmo (ex: 2 janelas de média móvel, 1 janela para momentos, 1 limite de CUSUM, 1 multiplicador de volatilidade para barreiras, e max_depth/n_estimators do XGBoost).

### **Checklist de Conclusão**
- [ ] MACD, RSI e Bollinger removidos.
- [ ] Distância MA, Garman-Klass, Skew e Kurtosis implementados.
- [ ] Script `feature_evaluation.py` criado e integrado.
- [ ] Modelo retreinado e relatório MDA/SHAP analisado.
- [ ] Features com MDA negativo removidas do pipeline.
- [ ] Optuna configurado para buscar no máximo 10 parâmetros.