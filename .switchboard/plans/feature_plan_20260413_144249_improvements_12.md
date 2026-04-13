# Improvements 12.

Há um erro estrutural crítico na forma como as **saídas (exits)** estão sendo calculadas no seu módulo de Tripla Barreira (`triple_barrier.py`). Essa falha contamina a simulação e é a causa mais provável para que a otimização esteja retornando *Sharpe Ratios* sistematicamente negativos.

### O Problema: Avaliação Limitada ao Preço de Fechamento (`close_values`)
Analisando as funções do kernel otimizado que detectam o toque nas barreiras (`_find_first_touch` e `_find_dynamic_touch`), nota-se que **o sistema está utilizando apenas o preço de fechamento (`close_values`) para verificar as saídas**.

Isso cria três distorções severas que destroem o *Sharpe* da estratégia no backtest:

1. ***Take Profits* (Ganhos) Ignorados:** O código verifica se `ret >= upper` usando o fechamento da barra. Se o mercado atingir o seu *Take Profit* no meio da sessão (máxima da barra), mas recuar e fechar abaixo do alvo, o sistema não registra a vitória.
2. ***Stop Losses* com *Slippage* Catastrófico:** Quando uma barra fecha abaixo do seu *Stop Loss*, o backtest retorna o retorno exato do fechamento daquela barra (`ret = (close_values[i] / entry_price - 1.0) * side`) e não o limite cravado do Stop. Ou seja, se o seu limite era -1% mas a barra derreteu e fechou em -4%, o backtest engole a perda de -4%.
3. ***Stops* Ignorados por Recuperação Intrabarra:** Se o preço violar seu *Stop Loss* durante a barra, mas o fechamento for acima dele, a condição `ret <= lower` será falsa. O trade continuará aberto indevidamente na simulação, acumulando distorções para o futuro.

O mesmo erro lógico se aplica ao **Breakeven**: a ativação da proteção em `_find_dynamic_touch` só ocorre se o fechamento da barra ultrapassar o `be_trigger`. Ele não protege posições que atingiram o alvo e devolveram tudo na mesma barra.

### A Entrada (Entry) Está Correta?
**Sim.** Ao contrário das saídas, a sua mecânica de entrada dos trades está robusta e previne viés de antecipação (*lookahead bias*). O código força explicitamente a passagem dos preços de abertura (`open_prices`) e executa a entrada na abertura da barra seguinte ao sinal da estratégia (`entry_price = open_prices.values[start_loc + 1]`). O problema está restrito unicamente à validação intrabarra dos *exits*.

### Como o Motor de Otimização Reage a Isso
No seu script de otimização (`tuner.py`), existem salvaguardas rigorosas contra *overfitting* e penalizações severas. Se o Meta-Modelo não consegue agregar valor (*Sharpe Lift* <= 0), o *fitness* é dizimado (`fitness *= 0.1`). Como a estratégia base (Alpha) está acumulando perdas irreais pela falha da Tripla Barreira, o Meta-Modelo falha em achar um ganho consistente, levando o Optuna a devolver Sharpes finais negativos ou até mesmo abortar os *trials*.

### Como Corrigir
Para corrigir o comportamento das saídas e alinhar o backtest com a realidade do mercado:

1. Modifique a assinatura de `apply_triple_barrier` e dos métodos `@njit` para receberem também os arrays de **máximas (`high_values`)** e **mínimas (`low_values`)**.
2. Dentro do loop `for i in range(start + 1, end + 1)`, substitua o cálculo único do fechamento. Para posições *Long* (`side == 1`), verifique se o retorno gerado pela Máxima atinge o `upper` (Take Profit) e se a Mínima atinge o `lower` (Stop Loss). Faça o inverso para as posições *Short* (`side == -1`).
3. Em vez de retornar `ret` calculado sobre o preço da barra, **retorne o valor estrito da barreira (`upper` ou `lower`)** sempre que houver um cruzamento. O retorno do fechamento (`final_ret`) só deve ser utilizado quando ocorre o esgotamento do tempo (Barreira Vertical).