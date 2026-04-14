# Módulo de Execução e Risco (src/execution)

## Objetivo
O Módulo de Execução é a linha de frente do TradeSystem5000. Ele unifica o fluxo assíncrono de sondagem do mercado (event loop) com restrições rigorosas de gerenciamento de risco e execução de ordens via corretora (MetaTrader 5). Diferente de robôs ingênuos baseados em if-else de indicadores, a engine consome diretamente os pipelines do Meta-Modelo e do Bet-Sizing e aplica roteamento dinâmico sem bloquear threads, mantendo rastreabilidade total (Auditoria) dos lucros, lotes e circuit breakers diários.

## Integração com outros módulos
- **Configurações Globais (`config/settings.py`)**: Consome limites de risco de capital (Max Drawdown, Max Loss), modo de paper-trading ou live e tamanho de contratos.
- **Modeling (`src/modeling`)**: Interage com o `model_pipeline`, recebendo um dicionário contendo o Sinal Direcional (Alpha), Probabilidade de Vitória (Meta) e a Fração de Kelly sugerida.
- **Data (`src/data`)**: Consome internamente a API `mt5.copy_rates_from_pos` (através de stubs/MT5 real) para calcular predições de tempo real.

## Mapa de funções

### Motor Assíncrono (`engine.py`)
- `AsyncTradingEngine`: Orquestrador principal baseado na biblioteca nativa `asyncio`. Instancia os Risk e Order Managers e gerencia a lógica de negócio por trás das negociações baseadas em ticks. Mantém controle do tamanho anterior da posição para detectar se o *Broker* fechou a ordem (Take profit ou StopLoss nativo).
- `_process_symbol`: Função núcleo avaliada por *symbol*. Checa regras de horário, baixa barras de 5 minutos exclusivas, despacha o processamento do pipeline do modelo, decide se corta as fatias de Lotes pelo limite de Kelly e finalmente invoca posições inversas ou de Stop-and-Reverse (Fechando curtas para entrar longas e vice-versa).
- `run_forever`: O Event-loop. Faz o pooling simultâneo de inúmeros ativos utilizando `asyncio.gather`, garantindo um *sleep interval* para não sobrecarregar as APIs nem a CPU local e disparando Heartbeats ocasionais.

### Gerenciador de Ordens (`order_manager.py`)
- `OrderManager`: Encapsula a API C-like arcaica do MetaTrader 5 transformando-a em abstrações de alto nível. Despacha transações em rede, gerencia devoluções de erros e atua como ponte se o sistema estiver apenas em Paper Trading (simulado via logger em vez de roteamento MQL5).
- Funções chave: `send_market_order` (Agressão na ponta), `close_positions` (Zerar Custódia no MagicNumber) e `get_net_position` (Somatório direcional para evitar ordens sobrepostas desnecessárias).

### Gestão de Risco (`risk.py`)
- `RiskManager`: Vigia do Capital. É checado *antes* de cada trade do sistema. Mantém estado da equidade.
- `update_equity` e `can_trade`: Confere perdas acumuladas (Circuit Breaker se a Perda Diária Máxima é violada ou o Drawdown Máximo é perfurado). Se estourar a conta, joga o motor em modo `HALTED`. Se for Swing Trade ele barra novas posições; se for Day-Trade, aciona pânico geral cortando todas as posições imediatamente e travando até a meia noite.
- `validate_order`: Trava limites rígidos do modelo (Nunca entrar num sinal fraco e nunca apostar um volume maior que o `max_position`).

### Auditoria e Logs (`audit.py`)
- `AuditLogger`: Persiste rastreabilidade no SQLite (`tradesystem.db`). Salva a linha do tempo exata das tomadas de decisão.
- Tabelas primárias: `audit_signals` (Rastreia porque um sinal não abriu ou abriu, salvando a prob do modelo), `audit_orders` (Recebimento de ordens aceitas ou negadas do banco) e `audit_errors` (Exceptions globais ou erros severos de execução/MT5).

## Lógica geral
O script de execução não usa threads múltiplas difíceis de depurar, mas a biblioteca `asyncio` do Python. Em tempo real, o robô consome as fatias abertas no mercado, prevê probabilidades usando o pipeline treinado ($Meta\_prob$) e aplica o limiar de Kelly ($Kelly\_f$). Se $Meta\_prob$ não ultrapassar a linha de convicção mínima, o volume apostado é $0$. Se passar, aprova-se no Risk Manager e lança ordens, atualizando imediatamente o buffer local e auditando o traçado das operações. O sistema conta com um cooldown/reconhecimento: se o MT5 reportar posição nula mas a *engine* estiver esperando lotes abertos, conclui-se que o limitador nativo (TP/SL) foi tocado, então aciona-se um cooldown para evitar reentradas caóticas.

## Lógica matemática
1. **Volumetria (Sizing)**:
   A decisão de quantos lotes entrar deriva do pipeline (que gera a Kelly fraction limitada em $[0, 1]$):
   $$ Target\_Volume = round(Kelly\_F \times Max\_Position) $$
2. **Circuit Breaker (Drawdown/Daily Loss)**:
   A cada atualização de PnL:
   $$ Daily\_Loss\_Pct = \frac{Current\_Balance - Start\_Balance}{Start\_Balance} $$
   $$ Drawdown\_Pct = \frac{Peak\_Equity - Current\_Equity}{Peak\_Equity} $$
   Se $Daily\_Loss \le -MaxDailyLoss$ ou $Drawdown\_Pct \ge MaxDrawdown$, entra-se em estado Halt (Zera tudo no caso Day Trade).
