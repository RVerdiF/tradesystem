import sys
from unittest.mock import MagicMock

sys.modules["MetaTrader5"] = MagicMock()
