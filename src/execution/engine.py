"""Motor de Execução Assíncrono (Event Loop) — TradeSystem5000.

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

from config.settings import execution_config, risk_config
from src.execution.audit import audit
from src.execution.order_manager import OrderManager
from src.execution.risk import RiskManager


class AsyncTradingEngine:
    """Motor de execução principal do TradeSystem5000."""

    def __init__(
        self,
        model_pipeline,
        symbols: list[str],
        max_position: float = 1.0,
        trade_type: str = "day_trade",
    ) -> None:
        """Inicializa o AsyncTradingEngine.

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

        # Rastreia a posição enviada mais recente por ativo (em lotes, com sinal).
        # Usado para detectar fechamentos por TP/SL no broker (posição some sem o engine ter enviado close).
        self._last_position: dict[str, float] = {}

        logger.info(
            "Engine inicializado. Modo: {} | Modalidade: {}",
            execution_config.mode.upper(),
            self.trade_type.upper(),
        )

    async def _process_symbol(self, symbol: str) -> None:
        """Lógica executada a cada iteração do polling para um ativo."""
        # 1. Checa circuit breakers (Se estourou PnL ou fora do horário)
        if not self.risk.can_trade(symbol):
            # Se a modalidade for Day Trade, fecha tudo ao bater o horário (ou circuit breaker)
            # Se for Swing Trade, só fecha se NÃO for motivo de horário (ou seja, foi circuit breaker de PnL)
            if self.trade_type == "day_trade" or "WINDOW" not in self.risk.halt_reason:
                if self.om.get_net_position(symbol) != 0:
                    logger.warning(
                        "Fechando posições para {} devido a: {}", symbol, self.risk.halt_reason
                    )
                    self.om.close_positions(symbol)
                    self._last_position[symbol] = 0.0
                    # notify_trade_closed é um no-op se system_state == HALTED_FOR_DAY;
                    # só tem efeito real em fechamentos por cool-down/window dentro do dia.
                    self.risk.notify_trade_closed(symbol)
            return

        try:
            # Requisita dados históricos para o cálculo de indicadores
            # (ex: RSI, FracDiff precisam de janela)
            # start_pos=1: pula o candle em formação (índice 0), garantindo que
            # apenas candles matematicamente fechados sejam usados — evita
            # repainting e sinais fantasmas.
            if execution_config.mode == "live":
                rates = mt5.copy_rates_from_pos(
                    symbol, mt5.TIMEFRAME_M5, 1, execution_config.live_bars
                )
                if rates is None or len(rates) == 0:
                    return
                df_snapshot = pd.DataFrame(rates)
                df_snapshot["time"] = pd.to_datetime(df_snapshot["time"], unit="s")
                df_snapshot.set_index("time", inplace=True)
            else:
                df_snapshot = pd.DataFrame()  # Simulado / mock

            # Se não há barra ou está incompleto, ignora
            if df_snapshot.empty or len(df_snapshot) < 50:
                logger.warning("Dados incompletos para {}. Aguardando mais barras.", symbol)
                return

            # Log de auditoria: timestamp do último candle fechado
            last_closed_ts = df_snapshot.index[-1]
            logger.info(
                "Audit [{}]: Predição usando último candle fechado em {}",
                symbol,
                last_closed_ts,
            )

            # 3. Predição Completa (Alpha -> MT5 Bar -> FeatureGen -> ML Meta)
            signal_data = self.model_pipeline(df_snapshot)

            # Formato esperado: {"side": 1/-1/0, "meta_prob": float, "kelly_fraction": float, "price": float}
            alpha_side = signal_data.get("side", 0)
            meta_prob = signal_data.get("meta_prob", 0.0)
            kelly_f = signal_data.get("kelly_fraction", 0.0)
            price = signal_data.get("price", 0.0)

            # Aplica limiar de convicção: probabilidade abaixo do threshold → kelly zerado
            # Isso substitui a conversão binária (meta_label == 1) por uma decisão contínua.
            # meta_prob abaixo do threshold indica ausência de edge suficiente; kelly_f já
            # deveria refletir isso se pipeline estiver correto, mas verificamos meta_prob
            # diretamente aqui como segunda linha de defesa.
            if meta_prob < risk_config.min_conviction_threshold:
                kelly_f = 0.0

            # Sizing proporcional: sem floor de 1 lote — zero é um resultado válido e esperado.
            target_volume = int(round(kelly_f * self.max_position))

            # Rastreia o sinal (incluindo lotes zero para análise de cobertura)
            if alpha_side != 0:
                audit.log_signal(
                    symbol,
                    alpha_side,
                    int(meta_prob >= risk_config.min_conviction_threshold),
                    kelly_f,
                    price,
                )

            # 4. Avalia Ação Direcional
            net_position = self.om.get_net_position(symbol)

            # Reconciliação TP/SL: detecta fechamento pelo broker (posição some sem o engine ter enviado close).
            # Se o engine registrou uma posição aberta (_last_position != 0) mas a posição real é 0,
            # significa que o broker fechou via TP ou SL. Inicia cool-down e pula a entrada neste tick.
            last_pos = self._last_position.get(symbol, 0.0)
            if last_pos != 0.0 and net_position == 0:
                logger.info(
                    "TP/SL detectado para {} (posição anterior: {:.1f} lotes). Iniciando cool-down.",
                    symbol,
                    last_pos,
                )
                self._last_position[symbol] = 0.0
                self.risk.notify_trade_closed(symbol)
                return  # Aguarda o cool-down expirar antes de abrir nova posição

            # Stop-and-Reverse proporcional:
            # IMPORTANTE: close_positions é chamado SOMENTE quando target_volume > 0.
            # Um sinal de baixa convicção (lote zero) NÃO fecha posições existentes —
            # isso evita churn e custos desnecessários em períodos de ruído.
            if alpha_side == 1 and net_position <= 0 and target_volume > 0:
                # Comprar! Fecha posição vendida antes, então abre long proporcional.
                self.om.close_positions(symbol)

                if self.risk.validate_order(
                    abs(self.om.get_net_position(symbol)), float(target_volume), self.max_position
                ):
                    self.om.send_market_order(symbol, "buy", float(target_volume))
                    self._last_position[symbol] = float(target_volume)

            elif alpha_side == -1 and net_position >= 0 and target_volume > 0:
                # Vender! Fecha posição comprada antes, então abre short proporcional.
                self.om.close_positions(symbol)

                if self.risk.validate_order(
                    abs(self.om.get_net_position(symbol)), float(target_volume), self.max_position
                ):
                    self.om.send_market_order(symbol, "sell", float(target_volume))
                    self._last_position[symbol] = -float(target_volume)

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
            await asyncio.gather(*tasks, return_exceptions=True)

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
