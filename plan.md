Com base no diagnóstico técnico dos logs, aqui está o plano de ação estruturado para estabilizar o sistema e validar a lógica do pipeline.

---

## 1. Correção Imediata da Engenharia (Custo de Transação)
O objetivo é interromper a "sangria" artificial de capital em operações não executadas.

* **Local do código:** `cost_model.py`, `attribution.py` ou `run_pipeline.py`.
* **Ação:** Implementar uma máscara booleana para garantir que custos de corretagem e *slippage* sejam aplicados apenas quando o tamanho da posição for superior a zero.
* **Lógica a aplicar:**
$$\text{retorno\_líquido} = \text{retorno\_bruto} - (\text{custo} \times \mathbb{I}_{\{\text{tamanho\_posição} \neq 0\}})$$
* **Implementação em Python:**
```python
df['net_return'] = df['gross_return'] - (df['cost'] * (df['position_size'] != 0))
```

## 2. Integridade de Dados e Alinhamento de Features
O objetivo é eliminar os `NaNs` que impedem o treinamento do XGBoost.

* **Ação:** Revisar a função `build_training_dataset`.
* **Procedimento:**
    1.  Verificar o descarte de dados após a aplicação do **Fractional Differentiation (FracDiff)**.
    2.  Garantir que a janela de 202 valores iniciais (instabilidade dos pesos) seja removida via `.dropna()` antes de passar a matriz $X$ para o modelo.
    3.  Inserir um *debug print* preventivo no `classifier.py`:
        ```python
        print(f"Valores nulos no X_train: {X_train.isnull().sum().sum()}")
        ```

## 3. Recalibração de Volatilidade (CUSUM)
O objetivo é adequar a captura de eventos à microestrutura do tempo gráfico (5-15 min).

* **Local do código:** `config/settings.py`.
* **Ação:** Alterar o `cusum_range` para evitar ruído estatístico.
* **Novos parâmetros sugeridos:**
    * Mínimo: $0.002$ ($0.2\%$)
    * Máximo: $0.01$ ($1.0\%$)
* **Justificativa:** Limiares de $0.02\%$ são absorvidos pelo *spread* e ruído de execução, impossibilitando o aprendizado de padrões direcionais.

## 4. Teste de Sanidade do Modelo (Stress Test)
O objetivo é confirmar se o problema é a qualidade do sinal ou a arquitetura do código.

* **Ação:** Forçar um *overfitting* controlado no `classifier.py`.
* **Configuração temporária:**
    * `max_depth`: $8$
    * `gamma`: $0$
    * `min_child_weight`: $1$
* **Validação:**
    * Se o AUC subir para $>0.90$ no treino: O pipeline de dados está saudável; o problema era a parametrização do Optuna ou ruído excessivo no alvo.
    * Se o AUC permanecer em $0.50$: Os dados de entrada (features) não possuem correlação com o alvo ou continuam corrompidos.

---

### Resumo de Prioridades
| Prioridade | Tarefa | Arquivo Alvo |
| :--- | :--- | :--- |
| **Alta** | Aplicar máscara de custo zero para trades filtrados | `attribution.py` |
| **Alta** | Limpeza de NaNs pós-FracDiff | `data_processing.py` |
| **Média** | Ajustar limites do CUSUM no Optuna | `settings.py` |
| **Média** | Executar teste de sanidade com XGBoost liberado | `classifier.py` |