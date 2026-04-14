# Módulo de Execução e Risco (src/execution)

## Objetivo Detalhado
Uma estratégia testada imaculadamente por meses não passa de teoria acadêmica até ser engatada num conector financeiro. O objetivo do módulo de Execução é instanciar o "Corpo Seco" do robô. Trata-se da carcaça *front-end* operante, o software em nível de rede que escuta a corretora (no caso B3 via MetaTrader 5 Terminal), injeta o pulso arterial dos preços *Live* recém colhidos em milissegundos dentro do complexo cérebro do Pipeline AFML (Machine Learning, Frações Kelly), avalia os relatórios devolvidos por esses algoritmos em tempo O(1), estressa com as chaves gerais e duras das alavancas de limitação de Risco de Falência, e assina e roteia (dispara) os pacotes MQL5 das ordens reais de lote de contrato pra Mesa (HFT Proxy).

Tudo isso rodando de forma **não-bloqueante** (Assíncrona) mantendo auditoria restrita temporal in-memory e descarregada transacionalmente no disco, resolvendo perfeitamente falhas vitais em bots comuns como: "Desconexão sem aviso", "Travamento da Thread durante predição derrubando o bot e perdendo o Stop", e principalmente a falta da reconciliação de estado entre a "Memória ilusória do Robô" versus o "Book Real do Broker" que fecham alvos de TPs na ponta por latência divergente sem avisar o software, enlouquecendo o loop.

## Integração com Outros Módulos
- **Configurações Globais (`config/settings.py`)**: Sub-absorve ativamente limites estritos intransponíveis. Puxa do `risk_config` os multiplicadores, capital nocional base, percentual máximo em % de queda dolorosa antes da eutanásia diária do bot (Circuit Breaker), o limite cravado duro absoluto do `max_position` permitido nas pontas.
- **Modeling (`src/modeling`)**: Ele importa como função *Callback* a pipeline pre-fabricada do classificador (que engloba o extrator de Alpha direcional fraco + Features massivas e Pesos de XGBoost + o crivador e quantizador numérico matemático da Kelly). O *Callback* preve em `dict` e este módulo apenas acata e traduz os floats para pacotes de rede MT5.
- **Data (`src/data`) e MetaTrader**: Dialoga abertamente exigindo, entre suas sub-rotinas de micro-loop iterativas, as pequenas janelas temporais fixas ultra recentes de M1 ou M5.

## Mapa de Funções e Componentes Chave

### 1. Loop Coração: O Engine Assíncrono (`engine.py`)
Não lida com loops arcaicos com paradas de `time.sleep` travando a resposta de CPU. A classe principal se enraíza nativamente no núcleo Event-Loop do interpretador moderno `asyncio`.
- `AsyncTradingEngine`: Sub-inicializa as peças de Gestão na carga e carrega dicionários nativos para *track records*.
- `_process_symbol`: Função coroutine atomizada isoladamente para "1 ativo". Essa função realiza as dezenas de checagens obrigatórias contra as rotinas subjacentes (`RiskManager`) garantindo obediência civil de calendário, baixando snapshot micro-temporais (só do *last tick* limpo das velas) do MT5, e ativando o cálculo. Aqui as predições cospem do Modelo o sub-retorno: `side = 1, prob = 0.99, kelly = 0.5`. E o roteamento de S&R (Stop-and-Reverse) é engatilhado.
- `run_forever`: Ponto de amarração universal dos Coroutines. Invoca com superposição paralela a análise e monitoramento de 5, 10 ou centenas de símbolos da B3 ao exato mesmo tempo disparando o limitador `gather`, controlando a fritura dos clocks da CPU via sleeps relativos dinâmicos.

**Snippet do Motor Assíncrono Multi-Ativo em Disparo:**
```python
import asyncio
from src.execution.engine import AsyncTradingEngine

# Cria um Dummy Pipeline de Inferência HFT
def my_ml_pipeline(df_recent_bars):
    # Finge processar as features pesadas e rodar o XGBoost
    return {"side": -1, "meta_prob": 0.82, "kelly_fraction": 0.3}

# Instancia o cérebro que conectará na rede do terminal
engine = AsyncTradingEngine(
    model_pipeline=my_ml_pipeline,
    symbols=["WIN$", "WDO$", "PETR4"], # Escaneia e opera em paralelo os três ao mesmo tempo!
    max_position=5,                    # Absoluto inegociável
    trade_type="day_trade"             # Força zeragem 100% no horário de limite de fechamento
)

# Lança a Main-Thread no vácuo contínuo monitorando o mercado passivamente pra sempre
asyncio.run(engine.run_forever())
```

### 2. Guardião Estrito: O Gestor de Risco (`risk.py`)
O sentinela do nocional da carteira. Em grandes fundos de HFT se as métricas explodirem as restrições da sub-área, sistemas fecham as chaves de rede fisicamente paralisando perdas caóticas derivadas de anomalias algorítmicas imprevistas (*Flash-crashes* e bugs).
- `RiskManager`: Mantém vivo no buffer RAM o estado patrimonial corrente instantâneo `equity`.
- `update_equity` e `can_trade`: Vigília incessante das duas quedas mortais: 1) Circuit Breaker Geral na Perda Corrente Diária base ($> X\%$ do capital base perdido numa sessão isolada, paralisia imediata) e 2) Acionamento Tailing de Extrema Rejeição do *Max Drawdown* a partir da maior montanha tocada e aferida ($> Y\%$ do *HWM* histórico rompido num crash brusco intradiário).
- `validate_order`: Trava local para cada boleta atômica unitária, rechaçando agressões alucinantes maiores que o lote preestabelecido da gestão, bloqueando falsas quebras intencionais por falha pontual de *sizing* de pipeline do ML.

### 3. A Central de Rede e Envio Rápido (`order_manager.py`)
Tradução C++ API Bridge. Pega os desejos do motor quantitativo local em ditar ordens, insere num dict brutal com *MQL5 Enums* (tipo de trade, filling policies, mágicas numéricas para não misturar ordens feitas à mão com algorítmicas, derrapagem aceita e stop nativos) roteando via `OrderSend`.
- `send_market_order` e `close_positions` (Agressão Limpa).
- `get_net_position`: Conta de inventário, buscando no Broker de volta um JSON contendo exatos os lotes abertos para reconciliação do Engine (impedindo loops caóticos se o TP/SL nativo do broker for atingido e o nosso script em Python não perceber fechando a posição da cabeça).

**Snippet do Vigia de Risco do Capital (Circuit Breaker Engine):**
```python
from src.execution.risk import RiskManager

risk = RiskManager(trade_type="day_trade") # Objeto Stateful da sessão em RAM

# Ao longo do loop diário, atualizamos a cada fração de milissegundo as contabilidades da Corretora
risk.update_equity(
    current_balance=100_000.0,
    current_equity=97_000.0    # Patrimônio total líquido em tempo real sofrendo perdas
)

# Verificação implacável que impede o Engine_Core de lançar novas ordens suicidas
if not risk.can_trade("WIN$"):
    print(f"Risco Ativado! O Sistema abortou e barrou o Bot pelo motivo severo de: {risk.halt_reason}")
    # Resultado Ex: O Sistema abortou [...] HALTED: DAILY LOSS LIMIT (Perdeu + de 3% no dia estourando a restrição).
```

### 4. Gravador Contábil Ilegível e Incorruptível (`audit.py`)
Mantém os oráculos da lei em log. Se após três dias o bot perde dinheiro e fecha o saldo diário em colapso negativo, você não se questiona com dúvidas ou adivinhações do porque isso ocorreu. O Log exato e absoluto diz qual era o volume real ditado, qual modelo ativou o sinal, qual era a porcentagem da confiança $P$ estressada com o Kelly gerado para esse exato desvio, bem como o id cravado original do servidor e da corretora (Ticket MQL5) rastreáveis eternamente no DB de `tradesystem.db` com o carimbo do tempo.

## Dinâmica da Reconciliação do Descompasso Front-End (Cool-Down Mecânico)

Uma inovação extrema arquitetural para robôs HFT sobre MT5 ocorre no trecho de *Polling* no motor `AsyncTradingEngine`.
As corretoras fecham por inércia os alvos (Taked Profit ou Stop Loss enfileirados estáticos no servidor remoto). Elas não pingam um websocket de volta ativamente avisando a Thread Python isolada: "Ei, acabei de sair do seu lote de dólar".
O motor mantêm sua tabela na RAM de sua posição interior base e, a cada giro de relógio, puxa a contagem de estoques oficiais via API em $t_{atual}$:
Se $\text{Memory}_{Lotes} = 10 \quad E \quad \text{Broker}_{Lotes} = 0$:
O sistema infere e diagnostica passivamente em runtime: Um limitador de choque ocorreu na nuvem da B3 esvaziando minha ordem sem eu saber de imediato. A engine assume uma interrupção provisória de "Surpresa e Descanso" engatando um cronômetro na função isolada `notify_trade_closed` que proíbe por exatos e duros *N Segundos* (Cool-down configurado local) qualquer reentrada agressiva em loop descontrolado na nova mínima ou máxima do choque recém criado, preservando a tranquilidade mecânica do modelo e prevenindo re-agressões algorítmicas num vácuo de Book estressado pós liquidação.

## Matemática da Trava Máxima Global do Risco Contínuo

### Drawdown Rastejante de Alta Frequência Dinâmica (High-Water Mark)
A avaliação clássica afere o Drawdown usando vetores mortos do último dia ($E\_Final$). Em HFT monitora-se a quebra brusca a todo instante. Na inicialização do módulo, traça-se e mantém a equidade contínua guardando em memória variável $HWM$:
$$ \text{Peak\_Equity\_HWM} = \max(\text{Peak\_Equity}_{t-1}, \text{Current\_Equity}_t) $$
Para todo instante temporal micro-secundário em que houver pulsação do mercado flutuante:
$$ \text{DD\_Corrente \%}_t = \left( \frac{\text{Peak\_Equity\_HWM} - \text{Current\_Equity}_t}{\text{Peak\_Equity\_HWM}} \right) $$
Se essa variável $\text{DD \%}$ esbarrar cruzar na tolerância do limite (ex: 2.5%): O sinal booleano $\text{Halted}$ é disparado cortando e isolando as predições.
O mesmo cálculo simétrico purificado é ativado contra a variável do balanço estático congelado das 09h00 da manhã do caixa da empresa na conta para barrar o dia caso não haja crescimento do $HWM$, travando perdas base.
