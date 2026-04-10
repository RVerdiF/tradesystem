"""
Motor de Execução Assíncrono (Event Loop) — TradeSystem5000.

Este módulo implementa o coração da execução em tempo real, gerenciando o
loop de eventos que processa dados, gera sinais e roteia ordens para o
mercado de forma não-bloqueante.

Componentes:
- **AsyncTradingEngine**: Orquestrador central do ciclo de vida da execução.
- Integração com RiskManager e OrderManager para segurança operacional.
- Suporte a múltiplos ativos em paralelo através de concorrência assíncrona.

Referências
-----------
Documentação Oficial do Python asyncio (Event Loop).
López de Prado, M. (2018). Advances in Financial Machine Learning. John Wiley & Sons.
"""

from __future__ import annotations

import asyncio
import time

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

from config.settings import execution_config
from src.execution.audit import audit
from src.execution.order_manager import OrderManager
from src.execution.risk import RiskManager


class AsyncTradingEngine:
    """
    Motor de execução principal do TradeSystem5000.
    """

    def __init__(
        self,
        model_pipeline,
        symbols: list[str],
        max_position: float = 1.0,
        trade_type: str = "day_trade"
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
        trade_type : str
            'day_trade' ou 'swing_trade'.
        """
        self.model_pipeline = model_pipeline
        self.symbols = symbols
        self.max_position = max_position
        self.trade_type = trade_type

        self.risk = RiskManager(trade_type=trade_type)
        self.om = OrderManager()

        self.is_running = False

        logger.info("Engine inicializado. Modo: {} | Modalidade: {}", execution_config.mode.upper(), self.trade_type.upper())

    async def _process_symbol(self, symbol: str) -> None:
        """Lógica executada a cada iteração do polling para um ativo."""

        # 1. Checa circuit breakers (Se estourou PnL ou fora do horário)
        if not self.risk.can_trade():
            # Se a modalidade for Day Trade, fecha tudo ao bater o horário (ou circuit breaker)
            # Se for Swing Trade, só fecha se NÃO for motivo de horário (ou seja, foi circuit breaker de PnL)
            if self.trade_type == "day_trade" or "WINDOW" not in self.risk.halt_reason:
                if self.om.get_net_position(symbol) != 0:
                    logger.warning("Fechando posições para {} devido a: {}", symbol, self.risk.halt_reason)
                    self.om.close_positions(symbol)
            return

        try:
            # Requisita dados históricos para o cálculo de indicadores (ex: RSI, FracDiff precisam de janela)
            if execution_config.mode == "live":
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 500)
                if rates is None or len(rates) == 0:
                    return
                df_snapshot = pd.DataFrame(rates)
                df_snapshot['time'] = pd.to_datetime(df_snapshot['time'], unit='s')
                df_snapshot.set_index('time', inplace=True)
            else:
                df_snapshot = pd.DataFrame() # Simulado / mock

            # Se não há barra ou está incompleto, ignora
            if df_snapshot.empty or len(df_snapshot) < 50:
                logger.warning("Dados incompletos para {}. Aguardando mais barras.", symbol)
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

                # Sizing final (arredondado para o inteiro mais prximo e garantindo lote mnimo 1)
                target_volume = float(max(1, int(round(kelly_f * self.max_position))))

                # Valida usando a posição atual (caso close_positions tenha falhado parcialmente)
                if self.risk.validate_order(abs(self.om.get_net_position(symbol)), target_volume, self.max_position):
                    self.om.send_market_order(symbol, "buy", target_volume)

            elif alpha_side == -1 and net_position >= 0 and meta_label == 1 and kelly_f > 0:
                # Vender!
                self.om.close_positions(symbol)

                target_volume = float(max(1, int(round(kelly_f * self.max_position))))

                if self.risk.validate_order(abs(self.om.get_net_position(symbol)), target_volume, self.max_position):
                    self.om.send_market_order(symbol, "sell", target_volume)

        except Exception as e:
            audit.log_error("EngineLoop", f"Erro crítico processando {symbol}: {e}", critical=True)

    async def run_forever(self) -> None:
        """Loop infinito de sondagem (Polling loop asíncrono)."""
        if execution_config.mode == "live":
            if not mt5.initialize():
                logger.critical("Falha ao inicializar MT5. Erro: {}", mt5.last_error())
                return

        self.is_running = True
        logger.success("Iniciando Event Loop Assíncrono do TradeSystem5000...")

        last_heartbeat = time.time()

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

            # Heartbeat (A cada ~60s)
            current_time = time.time()
            if current_time - last_heartbeat >= 60.0:
                logger.debug("Heartbeat: Engine rodando. Último ciclo: {:.3f}s", elapsed)
                last_heartbeat = current_time

    def stop(self) -> None:
        """Interrompe o loop principal."""
        self.is_running = False
        logger.warning("Sinal de parada recebido. Motor sendo deligado...")
