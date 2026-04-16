from unittest.mock import patch

import pytest

from src.data.mt5_connector import MT5ConnectionError, MT5Connector, mt5_session


@pytest.fixture
def mock_mt5():
    """Fixture to mock the MetaTrader5 library."""
    with patch("src.data.mt5_connector.mt5") as mock:
        with patch("src.data.mt5_connector.MT5_AVAILABLE", True):
            yield mock


class TestMT5Connector:
    def test_successful_connect(self, mock_mt5):
        """Test successful connection and login."""
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True

        connector = MT5Connector(login=12345, password="password", server="Server")
        connector.connect()

        assert connector.is_connected
        mock_mt5.initialize.assert_called_once()
        mock_mt5.login.assert_called_once_with(login=12345, password="password", server="Server")

    def test_connect_retry_logic(self, mock_mt5):
        """Test that connector retries on failure and eventually raises error."""
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = (-1, "Init failed")

        connector = MT5Connector(max_retries=3, retry_delay=0.01)

        with pytest.raises(MT5ConnectionError) as excinfo:
            connector.connect()

        assert "Não foi possível conectar" in str(excinfo.value)
        assert mock_mt5.initialize.call_count == 3

    def test_connect_failure_then_success(self, mock_mt5):
        """Test that connector succeeds after a failed attempt."""
        # Fail first, succeed second
        mock_mt5.initialize.side_effect = [False, True]
        mock_mt5.login.return_value = True

        connector = MT5Connector(max_retries=3, retry_delay=0.01)
        connector.connect()

        assert connector.is_connected
        assert mock_mt5.initialize.call_count == 2

    def test_disconnect(self, mock_mt5):
        """Test disconnection logic."""
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True

        connector = MT5Connector()
        connector.connect()
        connector.disconnect()

        assert not connector.is_connected
        mock_mt5.shutdown.assert_called_once()

    def test_mt5_session_context_manager(self, mock_mt5):
        """Test the mt5_session helper."""
        mock_mt5.initialize.return_value = True
        mock_mt5.login.return_value = True

        with mt5_session(login=999) as conn:
            assert conn.is_connected
            assert conn.login == 999

        assert not conn.is_connected
        mock_mt5.shutdown.assert_called_once()
