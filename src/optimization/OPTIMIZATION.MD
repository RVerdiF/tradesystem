# Módulo de Otimização (src/optimization)

## Objetivo Detalhado
Sistemas algorítmicos complexos, que dependem de dezenas de constantes numéricas como pesos de classes, profundidade de árvores, multiplicadores de risco, limiares de *CUSUM* e distanciamentos para embargos, esbarram no viés empírico humano. Acreditamos que o homem não deve "tentar adivinhar" que um stop a 2.5x a Volatilidade Diária funciona melhor do que 2.45x no mercado de Dólar Futuro em 2024. A Otimização assume a carga computacional e automatiza a busca exaustiva científica e matemática desses hiperparâmetros (Tuning).

O grande cuidado — e a principal diferença em relação às metodologias simplistas e comerciais de Grid Search exaustivos (Onde testa-se milhares de permutações guardando cegamente a que atingiu o maior lucro) — é o fato inerente de que, testando $N$ chaves lógicas diferentes em um histórico finito, a probabilidade matemática e o viés de Data Snooping garantem a descoberta de estratégias "perfeitas" formadas de ruídos mortos (overfitting). Portanto, o Módulo engloba a *Deflação e Desinflação da Significância da Otimização*, descartando brutalmente os resultados ilusórios.

## Integração com Outros Módulos
- **Data (`src/data`)**: Consome as bibliotecas robustas de download incremental para que, momentos antes de acionar dezenas de instâncias de XGBoost otimizados, puxe o histórico do mês faltante na corretora perfeitamente acoplado em tempo real ao DB.
- **Modeling (`src/modeling`) e Labeling (`src/labeling`)**: Opera estes como "Caixa Preta". Submete instâncias das funções de `triple_barrier` e `classifier.py` enviando em cada Trial novas combinações numéricas em milissegundos tentando encontrar qual geometria estocástica das árvores devolve o melhor Calmar ou DSR da base em questão.
- **Backtest (`src/backtest`)**: Consome nativamente o funil combinatório de cross-validation (CPCV) e os cálculos penalizadores rigorosos na fase primária para mensurar *fitness*.
- **Configurações Globais (`config/settings.py`)**: Acata limites globais de ranges configuráveis do algorítmo Bayesiano (ex: não testar árvores profundas demais limitando `max_depth_range`) que vêm embutidos na carga do sistema.

## Mapa de Funções e Componentes Chave

### 1. Afinador Central Bayesiano (`tuner.py`)
Contém o cerne do funil progressivo que coordena a busca utilizando o algoritmo TPE (Tree-structured Parzen Estimator) importado e orquestrado pela biblioteca de ponta *Optuna*.
A Otimização não acontece tudo de uma vez. Ela se parte em dupla validação (*Phase 1* e *Phase 2*) previnindo explosão dimensional.
- `objective_phase1`: Limpa a geometria física. Mantém os hiperparâmetros do Classificador nulos ou fixos num patamar medíocre padrão, focando toda a rede neural na descoberta exclusiva de *Quão largas ou justas as barreiras devem ser?* (Multiplicador SL/PT) e *Quão responsiva aos sustos devemos ser?* (Filtro CUSUM Threshold). Se um loop gera poucos trades ou se amarra em nichos muito estritos, injeta-se uma pesada métrica punitiva destruindo seu Sharpe artificial para reverter as preferências do TPE.
- `objective_phase2`: Congela e sela em pedra de vez os limitadores geográficos matemáticos recém-descobertos em Phase 1 (já considerados ótimos globais relativos ao problema) para estressar intensamente só o XGBoost, brincando agora infinitamente com pesos imbalançados e decaimentos de aprendizado na inteligência analítica do Classificador Binário.
- `run_optimization`: Cria o escopo para invocar, registrar e salvar o *Study* (Um objeto unificador do Optuna) contendo o ciclo das Fases de Otimização amarrado com `run_pipeline`, colapsando a matriz de melhores features e jogando-a contra a validadora suprema da deflação do DSR do `metrics.py`.

### 2. Gestão de Memória Perene (`params_store.py`)
No mundo intradiário do HFT não abrimos arquivo por arquivo procurando o `json` do modelo ajustado do dia.
- `save_optimized_params`: Serializa os hiperesquemas otimizados da Phase 2, carimba o timestamp cronológico de *deploy*, insere o veredito Deflacionado de sobrevivência, e *Commita* nas tabelas transacionais do SQLite Central do robô via conectores diretos.
- `load_optimized_params` e `get_continuous_symbol`: Sistema engenhoso *Fail-Safe*. Muitas vezes no mercado o robô quer operar o ativo futuro na rolagem do ticker atual (`WINZ25`). O carregador verifica o DB e ao detectar a falência do símbolo específico, aplica regressões lógicas via expressões regulares complexas (RegEx) para decodificar e fazer cache estendido ao histórico do super-ativo consolidado mãe (`WIN$`), resgatando do banco os parâmetros perfeitamente válidos e atualizados e mantendo a operação fluída e robusta.

### 3. Pipeline CLI Master (`run_opt.py`)
O painel de comando do cientista de dados. Exposto para fácil automação *cron job* e via bash, este código inicializa a orquestração do MT5 se conectando e disparando o sub-sistema Optuna com apenas um parâmetro (Ticker), salvando no disco os resultados sem interações complexas de terminal.

**Snippet do Ciclo Exaustivo Automatizado CLI:**
```python
import pandas as pd
from src.optimization.run_opt import run_optimization
from src.optimization.params_store import load_optimized_params, params_exist

# A mágica da Invocação: Um só comando encapsulando 200 Trials exaustivos divididos,
# CPCV Cross-Validation purgado em 5 Folds, Regressor TPE Optuna em cada fold, e o DSR global.
# Ocorre uma simulação com vazamento 0 e salvaguarda estatística 100%.
otimizacao_log = run_optimization(
    df=barras_raw_db,
    interval="5m",
    n_trials_phase1=100,
    n_trials_phase2=100,
    symbol="WDO$"
)

# Após os resultados serem persistidos na tabela transacional de TradeSystem5000:
if params_exist("WDO$"):
    best_config = load_optimized_params("WDO$")
    print(f"Buscado do Banco SQLite -> WinRate Esperado Baseado no Modelo Opt: {best_config['metadata']}")
```

## Entendimento Crítico: A Mecânica Bayesiana vs Penalidade Estatística de Snooping

### Algoritmo Tree-structured Parzen Estimator (TPE)
Uma Otimização Grid convencional em 3 dimensões (Profundidade: 1 a 10; Multiplicador de Risco: 0.5 a 4.0; Learning Rate: 0.01 a 0.3) forçaria processar $10 \times 35 \times 30 = 10.500$ simulações completas massacrantes.
O algoritmo Bayesiano de Optuna (TPE) não atira para todos os lados. Ele agrupa os últimos testes ranqueando os que performaram horrivelmente (um subgrupo de distribuição populacional ruim $l(x)$) e os que extraíram Sharpe bom do mercado (uma densidade campeã paralela $g(x)$).
Sua aproximação tenta localizar o máximo de Qualidade Esperada da Exploração da Busca (*Expected Improvement* - EI), saltando os hiperparâmetros de tentativa focado na razão marginal:
$$ EI(x) \propto \frac{l(x)}{g(x)} $$
Isso orienta a busca e a poda do TPE massivamente focando só em sub-espaços que pareçam promissores na nuvem de simulação de $XGBoost$.

### Punição do Índice DSR (O Anti-Fraude)
Se a Otimização acima fizesse incansáveis mil trials ($N = 1000$), pela própria lei universal dos grandes números aleatórios atrelada a variâncias na série, algumas dessas configurações conseguiriam, mesmo sendo medíocres de berço, acertar os picos da volatilidade ao acaso gerando um Sharpe Ratio anual (SR) estrondosamente absurdo.
Mas e se esse for de fato o pico natural do sistema? É real ou fraude?
A função `deflated_sharpe_ratio` é a condenação da fase. Calcula o pico estocástico que o modelo atingiria pelo mero capricho de ter testado infinitas vezes ($E[max(SR)]$) a uma variância geral computada de toda a distribuição esférica dos Sharpes fracassados da sessão ($\sigma^2_{SR}$), via limite Gumbel, provando pela Constante de Euler-Mascheroni ($\gamma$):
$$ \text{E}[max(SR)] = \sigma_{SR} \left[ (1 - \gamma) \cdot \Phi^{-1}\left(1-\frac{1}{N_{trials}}\right) + \gamma \cdot \Phi^{-1}\left(1-\frac{1}{N_{trials} \cdot e}\right) \right] $$
O DSR devolve a conversão contínua em Probabilidade P-Value para julgar esse prêmio suposto de Sharpe perante o índice z-stat corrigido de assimetrias ($Skew$) e caudas longas ($Kurtosis$) em séries temporais complexas limitando e chancelando os descobrimentos de Optuna apenas quando a confiança probabilística que os retornos operacionais são inquestionavelmente verídicos excede o intervalo base universal de $0.95$ ($> 95\%$ de certeza).
