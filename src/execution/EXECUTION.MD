# Módulo de Execução e Risco (src/execution)

Este módulo é o "cérebro operacional" do TradeSystem5000. Ele gerencia o loop de eventos assíncrono, a conectividade de ordens com o MetaTrader 5, as travas de segurança (risk management) e a auditoria de todas as decisões tomadas.

A arquitetura prioriza a **segurança da conta** e a **rastreabilidade total**, garantindo que nenhum sinal seja executado sem passar por múltiplas camadas de validação.

---

## 1. Componentes Principais

O subsistema de execução é dividido em quatro pilares:

### 1.1 Motor Assíncrono (`engine.py`)
*   **`AsyncTradingEngine`**: O motor principal que coordena o fluxo. Ele utiliza um loop de polling assíncrono (`asyncio`) para monitorar múltiplos ativos em paralelo sem bloquear a execução.
*   **Ciclo de Decisão**: A cada iteração, o motor:
    1.  Verifica os *Circuit Breakers* globais.
    2.  Consome dados em tempo real do MT5.
    3.  Chama o pipeline de modelos (Alpha + Meta).
    4.  Valida o sinal contra as regras de risco locais.
    5.  Envia comandos para o `OrderManager`.

### 1.2 Gerenciador de Ordens (`order_manager.py`)
*   **`OrderManager`**: Responsável pela comunicação de baixo nível com a API `MetaTrader5`.
*   **Modo Paper vs Live**: Identifica automaticamente o modo de execução configurado. No modo `live`, envia ordens reais; no modo `paper`, simula a execução e apenas registra no log de auditoria.
*   **Identificação por Magic Number**: Utiliza o `magic_number` para rastrear apenas as posições abertas pelo sistema, evitando interferência com operações manuais na mesma conta.

### 1.3 Gestão de Risco (`risk.py`)
*   **`RiskManager`**: Atua como uma camada de proteção "Hard-Stop".
*   **Circuit Breakers Diários**: Monitora Drawdown máximo e Perda Diária Máxima. Se atingidos, o sistema trava (`is_halted = True`) e para de enviar ordens até o próximo dia de trading.
*   **Validação de Exposição**: Garante que o volume de uma nova ordem não exceda o limite de exposição máxima definido para o ativo ou para a conta.

### 1.4 Auditoria (`audit.py`)
*   **`AuditLogger`**: Persiste todas as ações operacionais em um banco de dados **SQLite** (`data/tradesystem.db`).
*   **Tabelas de Auditoria**:
    *   `audit_signals`: Todos os sinais gerados, incluindo probabilidade do meta-modelo e fração de Kelly.
    *   `audit_orders`: Logs detalhados de ordens enviadas, executadas ou rejeitadas (com Ticket e Comentário do MT5).
    *   `audit_errors`: Registro de exceções críticas e avisos operacionais.

---

## 2. Ciclo de Vida de um Sinal

O fluxo de processamento de um sinal segue rigorosamente esta ordem:

1.  **Sinal Gerado**: O modelo Alpha propõe uma direção.
2.  **Filtro Meta-Modelo**: O sinal só prossegue se a probabilidade de acerto for >= 50% (`meta_label=1`).
3.  **Dimensionamento (Sizing)**: A fração de Kelly define o volume proporcional ao capital disponível.
4.  **Check de Risco**: O `RiskManager` verifica se o Drawdown/Perda Diária permite a operação e se o volume respeita a exposição máxima.
5.  **Execução**: Se aprovado, o `OrderManager` envia a ordem a mercado para o MT5.
6.  **Persistência**: Todo o processo é auditado no SQLite via `AuditLogger`.

---

## 3. Configuração de Segurança

As travas de segurança são configuradas via `config/settings.py`:
*   `max_daily_loss_pct`: Limite de perda diária (ex: 1.5%).
*   `max_drawdown_pct`: Limite de queda desde o topo da equidade (ex: 3.0%).
*   `max_position`: Volume máximo permitido por ativo.

---

## 4. Auditoria e Reconciliação

Para consultar o histórico de auditoria diretamente via Python:

```python
from src.execution.audit import audit

# Consultar ordens do dia
orders = audit.query_orders(symbol="WINZ25")
for o in orders:
    print(f"[{o['timestamp']}] {o['action']} {o['symbol']} Vol: {o['volume']}")

# Consultar erros críticos
errors = audit.query_errors(critical_only=True)
```
