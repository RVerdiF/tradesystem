from src.main_backtest import fetch_yfinance_data
from src.optimization.tuner import run_optimization
import pandas as pd

if __name__ == "__main__":
    symbol = "PETR4.SA"
    years = 1.9
    interval = "1h"
    
    print(f"--- Iniciando Otimização PETR4.SA ({years} anos, {interval}) ---")
    
    try:
        # Busca os dados
        df = fetch_yfinance_data(symbol=symbol, years=years, interval=interval)
        
        # Executa a otimização
        study = run_optimization(df, interval=interval)
        
        print("\n" + "="*50)
        print("RESULTADOS DA OTIMIZAÇÃO")
        print("="*50)
        print(f"Melhor Sharpe (Fitness): {study.best_value:.4f}")
        print(f"Parâmetros Otimizados:")
        for key, value in study.best_params.items():
            print(f"  - {key}: {value}")
        print("="*50)
        
    except Exception as e:
        print(f"Erro durante a execução: {e}")
