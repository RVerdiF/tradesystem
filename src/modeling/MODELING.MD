# Módulo de Modelagem e Validação (src/modeling)

## Objetivo Detalhado
Este Módulo encapsula o motor cognitivo e gerencial do TradeSystem5000. No escopo algorítmico tradicional, o treinamento (fit) do Machine Learning é meramente estatístico: injeta-se todo o passado ($X, y$) e cospe-se pesos treinados sem levar o viés temporal a sério. Aqui, essa prática ingênua é categoricamente estirpada com as técnicas agressivas da Teoria Financeira Quantitativa do *Purging* e do *Embargo*.

Este diretório atua unindo o preparo imunológico contra a fuga/vazamento de dados (Data Leakage) antes de fornecer as matrizes embutidas para as árvores do *XGBoost* avaliarem se um sinal secundário compensa ou não (Classificação Binária). Para além do aprendizado, ele toma a frente na Matemática Financeira Aplicada, transmutando probabilidades gélidas em recomendações ativas de apostas proporcionais via Teoria de Kelly.

## Integração com Outros Módulos
- **Labeling (`src/labeling`) e Features (`src/features`)**: Carrega a tabela complexa e final do ciclo (Features $X$ geradas da Diferenciação e CUSUM vs Meta-Labels Binomiais $y$).
- **Optimization (`src/optimization`)**: Empresta a lógica modular `PurgedKFold` e a injeção do modelo de Classificador para que o Optuna os mutile milhões de vezes validando scores imunes e achando os hiperparâmetros campeões do ML.
- **Execution (`src/execution`)**: Alimenta a execução "Live", servindo como uma biblioteca *plug-and-play* fornecendo as funções pesadas já preparadas `predict_meta_model` para despachar probabilidades puras que, na mesma tacada, acionam as calculadoras de lotes do `bet_sizing`.

## Mapa de Funções e Componentes Chave

### 1. Purga e Embargo - A Imunização de Data Leakage (`purge_embargo.py`)
No ML clássico (como `scikit-learn`), a função `train_test_split` randomiza e mistura aleatoriamente os dados. Em séries financeiras (se os Trades de Treino abrangerem eventos onde o Fechamento da barra cai temporalmente nos dias reservados para Teste), o modelo espia o futuro.
- `get_train_times`: Aplica a "Purga". Escrutina todas as datas abertas contra os splits designados para testes ocultos, decapando e exterminando impiedosamente qualquer operação do treino que nasça ou persista durante e após o relógio do teste ter iniciado.
- `apply_embargo`: Após testarmos modelos sobre o período de Dezembro/2023, não podemos simplesmente ligar o Treino no primeiro de Janeiro/2024. A autocorrelação temporal (Momentum de longo prazo) derramará o aprendizado de Dezembro artificialmente. Cria-se o "Embargo", um abismo sintético invisível de tempo de duração estrita logo após a dobra de teste expirar para isolar regimes temporais sucessivos.

### 2. K-Fold Purificado Compatível (`cv.py`)
Módulo de *facade* arquitetural. Como toda a comunidade trabalha em cima da padronização de compatibilidade de bibliotecas, foi escrito do zero esse módulo para "enganar" as métricas do Optuna/Scikit, fornecendo arrays limpos no padrão compatível com `cross_val_score`.
- `PurgedKFold`: Injeta a Purga e Embargo nos iteradores *yield* nativos para serem consumidos como dobras de dados intocados.

**Snippet do Setup Validação Purificada e Embargada:**
```python
from src.modeling.cv import PurgedKFold
from src.modeling.classifier import train_meta_model
import xgboost as xgb

# Gera um split de validação anti-vazamento separando a história
# em 5 dobras onde 1% de todos os tempos são queimados como "Embargo"
# após cada subconjunto de teste.
cv_generator = PurgedKFold(
    n_splits=5,
    samples_info=df_trade_timeline, # Série pandas de T0(início) e T1(fim) do Trade
    pct_embargo=0.01
)

# Aciona o XGBoost com a certeza que ele não decorará o gabarito
modelo_seguro = train_meta_model(X_features, y_meta_labels, cv=cv_generator)
```

### 3. Modelo Treinador de Probabilidade (`classifier.py`)
Sub-núcleo algorítmico do XGBoost. Diferente da classificação ordinária, ele tem pesos manipulados (Scale_Pos_Weight) contra a disparidade de classes severas nas bases de dados (geralmente sobram 90% falhas para 10% sucessos no trade).
- `predict_meta_model`: A engrenagem de resposta contínua. Nós exigimos da inferência o flutuante escalar probabilístico (ex: $P=0.68$) do estimador ao invés da mera saída bruta arredondada $[0, 1]$, fator inegociável para calibragem matemática da posição futura com Kelly.

### 4. Gestão Quantitativa da Banca e Contratos (`bet_sizing.py`)
Enquanto Analistas Físicos operam *stop/take* fixo ("1 lote sempre e para todo trade"), Sistemas AFML não correm esse nível de risco burro. Um trade com Probabilidade $P = 55\%$ deve despender muito menos capital exposto ao leão mercadológico que um trade que cruzou o classificador reportando $P = 98\%$.
- `compute_kelly_fraction`: Utiliza a Probabilidade Contínua do classificador injetada diretamente na Fórmula Fracionária de Kelly, modulada pelo limiar e *odds*. Retorna números não lineares cravados em $[0, 1]$ de exposição sugerida.
- `discretize_bet`: Lida com a chata infraestrutura das corretoras; afinal, você não pode negociar "0,42 contratos do WING26", logo precisa traduzir 42% fracionários num envelope unitário absoluto com teto pré-aprovado (`max_position` permitida pelo Robô) re-enviando os pacotes finais pro MT5.

**Snippet do Consumo Direto para Betting Size (Kelly Sizing):**
```python
from src.modeling.bet_sizing import compute_kelly_fraction, discretize_bet

# 1. Pega as probabilidades do modelo que rodou os dados de hoje
prob_win = model.predict_proba(current_market_features)[:, 1] # Retorna ex: 0.82 (82% de convicção)

# 2. Insere a convicção para extrair o quanto da banca vamos expor
kelly_size_perc = compute_kelly_fraction(
    prob_win=prob_win,
    odds=2.0,      # Aquele setup histórico possui uma taxa de Win/Loss media de $2 p/ $1
    fraction=0.5   # Half-Kelly (Corta o Kelly agressivo puro pela metade por segurança)
)

# 3. Transforma o output da matemática contínua num volume palpável e real em Lotes pra B3
lotes_alvo = discretize_bet(kelly_size_perc, max_position=10, step_size=1)
print(f"Comando Final de Máquina: Emitir agressão a mercado no tamanho de {lotes_alvo} lotes.")
```

## A Lógica Geral Sobre O Dilema Crônico do Vazamento (Leakage)

A armadilha na predição de ações consiste em pensar no mercado como planilhas médicas ou perfis de cliente independentes (um diagnóstico do paciente A não altera nem predispõe ou prevê o paciente B).
Num histórico intradiário do Ibovespa, uma tendência se propaga serialmente e temporalmente criando um fantasma de eco durável pelas próximas centenas de ticks (a autocorrelação temporal).
Se numa dobra o seu CV divide a barra temporal estrita $t=499$ para teste e a amostra de $t=500$ é sorteada magicamente para integrar a base de treino subjacente que forjará os pesos lógicos do estimador daquele CV, ocorre o Vazamento do Futuro, estourando as avaliações com uma precisão mágica fraudulenta (Overfitting Extremo). Por isso nós purgamos rigidamente dobras adjacentes.

## Fisiologia Matemática Aplicada e Otimizada

### A Matemática de Kelly Adaptada Para Retornos e Perdas Empíricas
O Critério de Kelly não fracionário é originário de Bell Labs pra telefonia e modificado por Edward Thorp (Afastando as falácias dos cassinos). Se pautado diretamente para trading, tem a característica violenta de quebrar os especuladores se as probabilidades sofrerem ínfima mutação frente à calibração (overbetting contínuo gerando falência matemática assintótica).
Considerando as probabilidades extraídas da predição $\hat{p}$:
Seja $b$ o Odds (lucro do trade perdedor em razão à perda) ditado pela sua meta na tripla barreira. A fração agressiva inicial da carteira otimizada $\hat{f}$ que maximiza matematicamente o Log-Retorno histórico do capital seria:
$$ \hat{f} = \hat{p} - \frac{1 - \hat{p}}{b} $$
Nós introduzimos o redutor mitigatório (*Fractional Kelly*) designado pela variável empírica de conservação $c$ (Ex: $0.5$ que dita Half-Kelly):
$$ F_{final} = \hat{f} \cdot c $$
Sinais que cruzem o limiar do limbo de vantagem de casa $\hat{f} \le 0$ são clipados em zero. Não se envia uma ordem para predição incerta ou desvantajosa estatisticamente. A decisão booleana "Devo apostar ou ficar de fora das taxas B3?" se aninha perfeitamente num único output escalar suave gerado entre $[0, c]$.
