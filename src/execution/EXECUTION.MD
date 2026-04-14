# Módulo de Execução e Risco (src/execution)

## Objetivo Detalhado
O escopo principal do módulo de Execução é a integração do modelo algorítmico treinado com a infraestrutura de roteamento de mercado (neste caso, MetaTrader 5 via B3). O módulo atua como a interface final do sistema, orquestrando a captura em tempo real de variações de cotação (*ticks* ou barras recentes), o processamento imediato pelo pipeline de aprendizado de máquina (Feature Engineering e Meta-Model) e a submissão das ordens dimensionadas pelo Critério de Kelly.

A arquitetura adota um padrão de execução estritamente não-bloqueante (*asynchronous event loop*). Essa decisão técnica visa mitigar problemas clássicos de latência e simultaneidade, garantindo que o processamento preditivo de um ativo não interrompa o monitoramento de limites de risco sistêmicos ou a liquidação de posições em outros ativos gerenciados paralelamente pelo mesmo processo. A auditoria contínua e as travas de proteção de capital (Circuit Breakers) operam interconectadas neste módulo para assegurar o encerramento determinístico das atividades sob condições anômalas.

## Integração com Outros Módulos
- **Configurações Globais (`config/settings.py`)**: Interage primariamente com o dicionário `risk_config`, absorvendo as constantes críticas que definem os limites de Drawdown, teto máximo de perda diária, tamanhos de nocional máximo permitido (`max_position`) e tolerâncias para latência do ciclo.
- **Modeling (`src/modeling`)**: Requer e invoca o pipeline consolidado de inferência. Ele injeta as cotações em tempo real e recebe o dicionário preditivo contendo a sinalização direcional (Alpha), a probabilidade computada do classificador e a correspondente fração recomendada (Kelly Fraction).
- **Data (`src/data`) e Corretora**: Comunica-se ativamente por intermédio das bibliotecas de conexão para obter resumos ultrarrecontes da série de preços e submeter pacotes MQL5 estruturados à fila de execução do *broker*.

## Mapa de Funções e Componentes Chave

### 1. Motor de Roteamento Assíncrono (`engine.py`)
Implementa o *core* de monitoramento de fluxo com base no pacote nativo `asyncio`, eliminando tempos ociosos da CPU atrelados a esperas de rede (I/O bound).
- `AsyncTradingEngine`: Orquestrador hierárquico que encapsula as instâncias de Gerenciamento de Risco e Gerenciamento de Ordens. Ele rastreia ativamente em memória local os estados das posições emitidas, detectando divergências causadas por liquidações autônomas do servidor da corretora (e.g., atingimento nativo de Take Profit ou Stop Loss).
- `_process_symbol`: Corrotina unitária avaliada individualmente para cada ativo monitorado. Contempla a sequência: validação dos limites de risco de horário, aquisição da última amostra de preço consolidado, acionamento da predição do modelo de ML, aplicação do filtro paramétrico da probabilidade vs fração de Kelly, e efetivação das operações transacionais de encerramento, entrada direcional ou *Stop-and-Reverse*.
- `run_forever`: O loop contínuo principal. Coordena a análise superposta através de `asyncio.gather`, processando os ativos do portfólio simultaneamente enquanto gerencia intervalos dinâmicos de repouso (`sleep`) e registra pulsações sistêmicas (Heartbeats) para log operacional.

**Exemplo do Motor Assíncrono Multi-Ativo:**
```python
import asyncio
from src.execution.engine import AsyncTradingEngine

def ml_pipeline_inference(df_recent_bars):
    return {"side": -1, "meta_prob": 0.82, "kelly_fraction": 0.3}

engine = AsyncTradingEngine(
    model_pipeline=ml_pipeline_inference,
    symbols=["WIN$", "WDO$", "PETR4"],
    max_position=5,
    trade_type="day_trade"
)

asyncio.run(engine.run_forever())
```

### 2. Controle do Capital e Limites de Exposição (`risk.py`)
Submódulo encarregado da avaliação de viabilidade financeira antecedente a toda emissão de lote. Mantém os estados contábeis em RAM para resposta imediata (*O(1)*).
- `RiskManager`: Classe de estado (Stateful) encarregada da vigilância patrimonial.
- `update_equity` e `can_trade`: Executa o registro da evolução da equidade total. Implementa as duas avaliações de Circuit Breaker: Perda Diária Corrente Absoluta e Drawdown Máximo relativo ao pico de capital (High-Water Mark). Caso os limiares excedam a parametrização tolerada, o método induz a suspensão operacional global (estado `HALTED`) daquele ativo ou da carteira integral.
- `validate_order`: Verifica o enquadramento do volume gerado pelo dimensionador fracionário frente aos limites de custódia globais preestabelecidos, reprovando execuções de ordens matematicamente inconsistentes antes do envio.

### 3. Integração com API da Corretora (`order_manager.py`)
Atua como camada de tradução (Bridge) convertendo os resultados e decisões estocásticas do modelo local em pacotes legíveis e conformes aos dicionários e enums estritos do MetaTrader 5 (MQL5).
- `send_market_order` e `close_positions`: Compilam as matrizes submetidas para ordens do tipo mercado, alocando os parâmetros intrínsecos como *Slippage* admitida, Número Mágico (rastreamento do robô) e *Fill Policies*.
- `get_net_position`: Estabelece a auditoria da posição real, interrogando a custódia efetiva mantida na corretora sob aquele número identificador e assegurando que as ordens submetidas não dupliquem exposições (reconciliação contábil).

**Exemplo da Validação do Risco do Capital (Circuit Breaker):**
```python
from src.execution.risk import RiskManager

risk = RiskManager(trade_type="day_trade")

risk.update_equity(
    current_balance=100_000.0,
    current_equity=97_000.0
)

if not risk.can_trade("WIN$"):
    print(f"Sistema interrompido pelo controle de risco: {risk.halt_reason}")
```

### 4. Persistência de Auditoria e Rastreador Transacional (`audit.py`)
Um banco de registros estruturados e persistentes da infraestrutura. A gravação baseia-se em armazenamento SQLite leve (`tradesystem.db`) desenhada para mitigar incertezas em avaliações pós-mercado. Mantém o mapeamento tabular exato das predições com convicção ($P$), do valor recomendado para lote, da aceitação real da corretora e registro absoluto das exceções ocorridas por meio de classes de log transacionais (`audit_signals`, `audit_orders`, `audit_errors`).

## Dinâmica de Reconciliação do Estado e Tolerância do Loop

O motor central aborda a ocorrência não reportada do fechamento de posições ativas. Uma vez que as execuções restritivas nativas (e.g., Take Profit e Stop Loss despachados para armazenamento no lado do servidor da B3) são acionadas, a API pode não emitir respostas assíncronas ativas notificando o interpretador local.
Para resolver este descompasso analítico na amostragem subjacente temporal, a corrotina central averigua o inventário oficial de custódia e, caso o balanço estático ($Memory\_Lotes = X \neq Broker\_Lotes = 0$) revele a interrupção imposta pela corretora, aciona-se um mecanismo paramétrico temporário (`notify_trade_closed`) impondo um bloqueio local (*Cool-down*) para novas iterações. Tal método neutraliza potenciais re-entradas do modelo durante a volatilidade momentânea do fechamento abrupto, resguardando a simetria preditiva da infraestrutura do AFML.

## Especificação Matemática das Fronteiras Globais de Risco

### Aferição Contínua do High-Water Mark (Max Drawdown HFT)
A apuração do recuo patrimonial é conduzida com base em variações correntes imediatas e não estritamente atrelada aos saldos diários de encerramento do pregão. Definindo e rastreando ativamente a maior montanha atingida:
$$ \text{Peak\_Equity\_HWM}_t = \max(\text{Peak\_Equity}_{t-1}, \text{Current\_Equity}_t) $$
Para todo instante temporal $t$ submetido ao processamento iterativo, deriva-se a margem regressiva suportada pela carteira transacionada:
$$ \text{DD\_Corrente \%}_t = \frac{\text{Peak\_Equity\_HWM}_t - \text{Current\_Equity}_t}{\text{Peak\_Equity\_HWM}_t} $$
Quando a razão computada $\text{DD\_Corrente \%}_t$ ultrapassa a tolerância predefinida (e.g., $\ge MaxDrawdown\%$), a sinalização transacional de estado restrito ($\text{Halted}$) é acionada, suprimindo o fluxo preditivo primário. Esta checagem é paralela e indissociável da aferição da queda simétrica diária, originada da comparação estrita entre as grandezas do capital e a custódia aferida no instante inaugural das negociações do dia corrente.