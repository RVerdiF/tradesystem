from unittest.mock import MagicMock, patch

import pytest

from src.execution.order_manager import OrderManager


# Simulação de constantes do MT5 para o teste
class MockMT5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1


@pytest.fixture
def om():
    with patch("src.execution.order_manager.execution_config") as mock_cfg:
        mock_cfg.magic_number = 5000
        mock_cfg.max_slippage_ticks = 5
        yield OrderManager()


class TestOrderManagerClose:
    def test_close_positions_paper_mode(self, om):
        """No modo paper, deve apenas logar e auditar, sem chamar mt5.positions_get."""
        with patch("src.execution.order_manager.execution_config") as mock_cfg:
            mock_cfg.mode = "paper"
            with patch("src.execution.order_manager.audit") as mock_audit:
                with patch("src.execution.order_manager.mt5") as mock_mt5:
                    om.close_positions("PETR4")

                    mock_mt5.positions_get.assert_not_called()
                    mock_audit.log_order.assert_called_once_with(
                        ticket=-1,
                        symbol="PETR4",
                        action="CLOSE_ALL_SIMULADA",
                        volume=0.0,
                        price=0.0,
                    )

    def test_close_positions_live_no_positions(self, om):
        """No modo live, se não houver posições, não deve fazer nada."""
        with patch("src.execution.order_manager.execution_config") as mock_cfg:
            mock_cfg.mode = "live"
            with patch("src.execution.order_manager.mt5") as mock_mt5:
                # Caso None
                mock_mt5.positions_get.return_value = None
                om.close_positions("PETR4")
                mock_mt5.positions_get.assert_called_with(symbol="PETR4")

                # Caso lista vazia
                mock_mt5.positions_get.return_value = []
                om.close_positions("PETR4")
                assert mock_mt5.positions_get.call_count == 2

    def test_close_positions_live_with_positions(self, om):
        """No modo live, deve fechar apenas posições com o magic_number correto."""
        with patch("src.execution.order_manager.execution_config") as mock_cfg:
            mock_cfg.mode = "live"
            # Importante: o magic_number no om é fixado no __init__
            om.magic_number = 5000

            # Mock positions
            pos1 = MagicMock()
            pos1.magic = 5000
            pos1.type = MockMT5.ORDER_TYPE_BUY
            pos1.ticket = 101
            pos1.volume = 1.0

            pos2 = MagicMock()
            pos2.magic = 9999  # Outro sistema
            pos2.type = MockMT5.ORDER_TYPE_BUY
            pos2.ticket = 102
            pos2.volume = 2.0

            pos3 = MagicMock()
            pos3.magic = 5000
            pos3.type = MockMT5.ORDER_TYPE_SELL
            pos3.ticket = 103
            pos3.volume = 0.5

            with patch("src.execution.order_manager.mt5") as mock_mt5:
                mock_mt5.positions_get.return_value = [pos1, pos2, pos3]
                mock_mt5.ORDER_TYPE_BUY = MockMT5.ORDER_TYPE_BUY
                mock_mt5.ORDER_TYPE_SELL = MockMT5.ORDER_TYPE_SELL

                with patch.object(om, "send_market_order") as mock_send:
                    om.close_positions("PETR4")

                    # Deve chamar send_market_order para pos1 e pos3, mas não pos2
                    assert mock_send.call_count == 2

                    # Chamada para pos1 (BUY -> SELL)
                    mock_send.assert_any_call("PETR4", action="sell", volume=1.0)
                    # Chamada para pos3 (SELL -> BUY)
                    mock_send.assert_any_call("PETR4", action="buy", volume=0.5)
