import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from src.optimization.run_opt import main


class TestRunOpt(unittest.TestCase):
    @patch("src.optimization.run_opt.argparse.ArgumentParser.parse_args")
    @patch("src.optimization.run_opt.fetch_mt5_data")
    @patch("src.optimization.run_opt.run_optimization")
    @patch("src.optimization.run_opt.save_optimized_params")
    @patch("src.optimization.run_opt.run_pipeline")
    def test_main_success(
        self, mock_run_pipeline, mock_save_params, mock_run_opt, mock_fetch_mt5, mock_parse_args
    ):
        # Setup mocks
        mock_args = MagicMock()
        mock_args.symbol = "PETR4"
        mock_args.n_bars = 1000
        mock_args.interval = "1h"
        mock_args.n_trials = 50
        mock_args.n_trials_phase1 = 30
        mock_args.n_trials_phase2 = 20
        mock_parse_args.return_value = mock_args

        mock_df = MagicMock(spec=pd.DataFrame)
        mock_fetch_mt5.return_value = mock_df

        mock_run_opt.return_value = {
            "params": {"param1": 1, "param2": 0.5},
            "metadata": {"dsr_score": 0.98},
        }

        mock_run_pipeline.return_value = {
            "sharpe": 2.0,
            "sharpe_alpha": 1.5,
            "sharpe_lift": 0.5,
            "calmar_ratio": 3.0,
            "n_trades": 50,
        }

        # Execute
        main()

        # Verify Argument Parsing
        mock_parse_args.assert_called_once()

        # Verify fetch_mt5_data call
        mock_fetch_mt5.assert_called_once_with(symbol="PETR4", n_bars=1000, interval="1h")

        # Verify run_optimization call
        mock_run_opt.assert_called_once_with(
            mock_df, interval="1h", n_trials=50, n_trials_phase1=30, n_trials_phase2=20
        )

        # Verify save_optimized_params call
        mock_save_params.assert_called_once_with(
            symbol="PETR4", params={"param1": 1, "param2": 0.5}, metadata={"dsr_score": 0.98}
        )

        # Verify run_pipeline call
        mock_run_pipeline.assert_called_once_with(
            mock_df, interval="1h", params={"param1": 1, "param2": 0.5}
        )

    @patch("src.optimization.run_opt.argparse.ArgumentParser.parse_args")
    @patch("src.optimization.run_opt.fetch_mt5_data")
    @patch("src.optimization.run_opt.logger")
    def test_main_exception_handling(self, mock_logger, mock_fetch_mt5, mock_parse_args):
        # Setup mocks to raise an exception
        mock_args = MagicMock()
        mock_args.symbol = "PETR4"
        mock_args.n_bars = 1000
        mock_args.interval = "1h"
        mock_args.n_trials = 50
        mock_args.n_trials_phase1 = 30
        mock_args.n_trials_phase2 = 20
        mock_parse_args.return_value = mock_args

        mock_fetch_mt5.side_effect = Exception("MT5 Error")

        # Execute
        main()

        # Verify exception was logged
        mock_logger.error.assert_called()
        args, _ = mock_logger.error.call_args
        self.assertIn("Erro durante a execução da otimização: MT5 Error", args[0])

    @patch("src.optimization.run_opt.argparse.ArgumentParser.parse_args")
    @patch("src.optimization.run_opt.fetch_mt5_data")
    @patch("src.optimization.run_opt.run_optimization")
    @patch("src.optimization.run_opt.save_optimized_params")
    @patch("src.optimization.run_opt.run_pipeline")
    def test_main_symbol_cleaning(
        self, mock_run_pipeline, mock_save_params, mock_run_opt, mock_fetch_mt5, mock_parse_args
    ):
        # Test if .SA is removed from symbol
        mock_args = MagicMock()
        mock_args.symbol = "PETR4.SA"
        mock_args.n_bars = 1000
        mock_args.interval = "1h"
        mock_args.n_trials = 50
        mock_args.n_trials_phase1 = 30
        mock_args.n_trials_phase2 = 20
        mock_parse_args.return_value = mock_args

        mock_df = MagicMock(spec=pd.DataFrame)
        mock_fetch_mt5.return_value = mock_df
        mock_run_opt.return_value = {"params": {}, "metadata": {"dsr_score": 0.5}}
        mock_run_pipeline.return_value = {}

        # Execute
        main()

        # Verify fetch_mt5_data call used cleaned symbol
        mock_fetch_mt5.assert_called_once_with(symbol="PETR4", n_bars=1000, interval="1h")


if __name__ == "__main__":
    unittest.main()
