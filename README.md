# TradeSystem5000

Sistema de trading algorítmico baseado em ML (metodologia AFML — López de Prado).

## Banco de Dados (SQLite)

O sistema usa um banco SQLite centralizado em `data/tradesystem.db` para persistência de:

| Tabela | Conteúdo |
|---|---|
| `optimized_params` | Hiperparâmetros otimizados por ativo (Optuna) |
| `audit_signals` | Sinais gerados pelo meta-modelo em tempo real |
| `audit_orders` | Ordens enviadas/simuladas (paper e live) |
| `audit_errors` | Erros operacionais do sistema |

### Inspecionando o banco

```bash
# Listar tabelas
sqlite3 data/tradesystem.db ".tables"

# Ver schema completo
sqlite3 data/tradesystem.db ".schema"

# Últimos 10 sinais
sqlite3 data/tradesystem.db "SELECT * FROM audit_signals ORDER BY timestamp DESC LIMIT 10;"

# Parâmetros persistidos
sqlite3 data/tradesystem.db "SELECT symbol, updated_at FROM optimized_params;"

# Erros críticos
sqlite3 data/tradesystem.db "SELECT * FROM audit_errors WHERE critical=1 ORDER BY timestamp DESC;"
```

### Notas de migração

Os arquivos legados `logs/audit/trades_*.jsonl`, `logs/audit/signals_*.jsonl` e `logs/audit/critical_errors.log` podem ser removidos manualmente após validação em produção. Os parâmetros JSON em `models/params_*.json` também podem ser removidos — a fonte de verdade agora é o SQLite.