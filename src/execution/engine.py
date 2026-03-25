"""
6.1 — Motor Assíncrono Principal (Event Loop).

- Conecta-se à stream de dados (via Polling ou Callbacks, aqui polling leve).
- A cada tick (ou N barras), processa features, alpha model e ML.
- Roteia sinais aceitos para o OrderManager e RiskManager.
"""

from __future__ import annotations

import time
import asyncio
import pandas as pd
from loguru import logger

from config.settings import execution_config
from src.execution.audit import audit
from src.execution.risk import RiskManager
from src.execution.order_manager import OrderManager


class AsyncTradingEngine:
    """
    Motor de execução principal do TradeSystem5000.
    """

    def __init__(
        self, 
        model_pipeline, 
        symbols: list[str],
        max_position: float = 1.0
    ) -> None:
        """
        Parameters
        ----------
        model_pipeline : object
            Objeto/Função que recebe um DataFrame contínuo de preços ohlcv e 
            retorna um dict {"side": int, "meta_prob": float, "kelly_fraction": float}.
        symbols : list[str]
            Lista de ativos monitorados.
        max_position : float
            Máximo em lotes.
        """
        self.model_pipeline = model_pipeline
        self.symbols = symbols
        self.max_position = max_position
        
        self.risk = RiskManager()
        self.om = OrderManager()
        
        self.is_running = False
        
        logger.info("Engine inicializado. Modo: {}", execution_config.mode.upper())

    async def _process_symbol(self, symbol: str) -> None:
        """Lógica executada a cada iteração do polling para um ativo."""
        
        # 1. Checa circuit breakers (Se estourou PnL)
        if not self.risk.can_trade():
            # Se bateu circuit breaker, garante que estamos zerados e pausa
            self.om.close_positions(symbol)
            return

        try:
            # 2. Requisita novos dados (simulado ou do feed de streaming real)
            # Na Fase 6 completa isso viria direto do copy_rates / copy_ticks
            df_snapshot = pd.DataFrame() # self.data_feed.get_snapshot(symbol)
            
            # Se não há barra nova, ignora
            if df_snapshot.empty:
                return

            # 3. Predição Completa (Alpha -> MT5 Bar -> FeatureGen -> ML Meta)
            signal_data = self.model_pipeline(df_snapshot)
            
            # Formato esperado: {"side": 1/-1/0, "kelly_fraction": 0.5, "price": 100.50}
            alpha_side = signal_data.get("side", 0)
            meta_label = 1 if signal_data.get("meta_prob", 0.0) >= 0.5 else 0
            kelly_f = signal_data.get("kelly_fraction", 0.0)
            price = signal_data.get("price", 0.0)

            # Rastreia
            if alpha_side != 0:
                audit.log_signal(symbol, alpha_side, meta_label, kelly_f, price)

            # 4. Avalia Ação Direcional
            net_position = self.om.get_net_position(symbol)
            
            # Exemplo simples de Stop-and-Reverse contínuo:
            if alpha_side == 1 and net_position <= 0 and meta_label == 1 and kelly_f > 0:
                # Comprar! Fecha posicao vendida antes
                self.om.close_positions(symbol)
                
                # Sizing final (converte Kelly em contratos)
                target_volume = kelly_f * self.max_position
                
                if self.risk.validate_order(0, target_volume, self.max_position):
                    self.om.send_market_order(symbol, "buy", target_volume)
                    
            elif alpha_side == -1 and net_position >= 0 and meta_label == 1 and kelly_f > 0:
                # Vender!
                self.om.close_positions(symbol)
                
                target_volume = kelly_f * self.max_position
                
                if self.risk.validate_order(0, target_volume, self.max_position):
                    self.om.send_market_order(symbol, "sell", target_volume)

        except Exception as e:
            audit.log_error("EngineLoop", f"Erro crítico processando {symbol}: {e}", critical=True)

    async def run_forever(self) -> None:
        """Loop infinito de sondagem (Polling loop asíncrono)."""
        self.is_running = True
        logger.success("Iniciando Event Loop Assíncrono do TradeSystem5000...")

        while self.is_running:
            start_time = time.monotonic()
            
            # Atualiza saldo global no Risk Manager
            # O ideal é risk.update_equity(mt5.account_info().balance, mt5.account_info().equity)
            self.risk.update_equity(100000.0, 100000.0)  # stub param paper
            
            # Executa todos os processamentos em paralelo usando asyncio.gather
            tasks = [self._process_symbol(sym) for sym in self.symbols]
            await asyncio.gather(*tasks)

            # Controla a frequência de polling para não fritar CPU / API MT5
            elapsed = time.monotonic() - start_time
            sleep_time = max(0.0, execution_config.poll_interval - elapsed)
            await asyncio.sleep(sleep_time)

    def stop(self) -> None:
        """Interrompe o loop principal."""
        self.is_running = False
        logger.warning("Sinal de parada recebido. Motor sendo deligado...")
