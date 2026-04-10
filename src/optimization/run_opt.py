"""
Script de Otimização TradeSystem5000

Como utilizar via linha de comando (CMD/PowerShell):

Uso básico (recomenda-se o uso do módulo com -m):
    uv run python -m src.optimization.run_opt --symbol PETR4

Uso com parâmetros customizados:
    uv run python -m src.optimization.run_opt --symbol VALE3 --n-bars 50000 --interval 1h --n-trials 100

Argumentos:
    --symbol:   Ativo para otimizar (obrigatório). Ex: PETR4
    --n-bars:   Quantidade de candles para o histórico (padrão: 10000)
    --interval: Tempo gráfico (padrão: 1h). Opções: 1m, 5m, 15m, 30m, 1h, 1d
    --n-trials: Número de tentativas de otimização (opcional, padrão: definido em config)
"""
import argparse
from loguru import logger
from src.main_backtest import fetch_mt5_data, run_pipeline
from src.optimization.tuner import run_optimization
from src.optimization.params_store import save_optimized_params

def main():
    parser = argparse.ArgumentParser(description="Script Genérico de Otimização TradeSystem5000")
    parser.add_argument("--symbol", type=str, required=True, help="Ativo para otimizar (ex: PETR4)")
    parser.add_argument("--n-bars", type=int, default=10000, help="Número de barras para extrair (padrão: 10000)")
    parser.add_argument("--interval", type=str, default="1h", choices=["1m", "5m", "15m", "30m", "1h", "1d"], help="Intervalo (padrão: 1h)")
    parser.add_argument("--n-trials", type=int, default=None, help="Número de tentativas de otimização (opcional)")
    
    args = parser.parse_args()
    
    # Remove o sufixo '.SA' se o usuário esquecer
    symbol = args.symbol.replace(".SA", "")
    
    logger.info(f"--- Iniciando Otimização {symbol} ({args.n_bars} barras, {args.interval}) ---")
    
    try:
        # Busca os dados no MT5
        df = fetch_mt5_data(symbol=symbol, n_bars=args.n_bars, interval=args.interval)
        
        # Executa a otimização (run_optimization já retorna os metadados necessários)
        opt_results = run_optimization(df, interval=args.interval, n_trials=args.n_trials)
        best_params = opt_results["params"]
        metadata = opt_results["metadata"]
        
        # Opcionalmente podemos salvar os parâmetros utilizando nossa store
        save_optimized_params(
            symbol=symbol,
            params=best_params,
            metadata=metadata
        )
        
        # Fase Final: Re-executa o pipeline para obter métricas de desempenho completas (OOS)
        logger.info("Gerando relatório de desempenho final...")
        oos_results = run_pipeline(df, interval=args.interval, params=best_params)
        
        if oos_results:
            print("\n" + "="*50)
            print("        RESULTADOS DA MELHOR CONFIGURAÇÃO")
            print("="*50)
            print(f"Ativo:              {symbol}")
            print(f"Sharpe (Sistema):    {oos_results['sharpe']:.4f}")
            print(f"Sharpe (Alpha):     {oos_results['sharpe_alpha']:.4f}")
            print(f"Sharpe Lift:        {oos_results['sharpe_lift']:.4f}")
            print(f"Calmar Ratio:       {oos_results['calmar_ratio']:.4f}")
            print(f"Total de Trades:     {oos_results['n_trades']}")
            print(f"Score DSR:          {metadata['dsr_score']:.4f}")
            
            sig_status = "CONFIRMADO" if metadata['dsr_score'] >= 0.95 else "FALHOU (Possível Overfit)"
            print(f"Significância:      {sig_status}")
            
            print("\nParâmetros Otimizados:")
            for key, value in best_params.items():
                print(f"  - {key}: {value}")
            print("="*50 + "\n")
        else:
            logger.error("Não foi possível gerar métricas de desempenho final.")

        
    except Exception as e:
        logger.error(f"Erro durante a execução da otimização: {e}")

if __name__ == "__main__":
    main()
