import pytest

from src.backtest.cost_model import BrazilianCostModel, SlippageModel


def test_futures_win_costs_verification():
    """Verificação de modelagem de custos para Futuros B3.
    Rodar uma simulação de 1 trade de compra e venda de 1 minicontrato (WIN)
    com preço cravado em 130.000 pontos.
    Validar via asserção se o custo final simulado equivale exatamente
    à corretagem em reais + emolumentos fixos B3 + slippage.
    O valor total não pode passar de alguns poucos reais.
    """
    symbol = "WINJ26"
    price = 130000.0
    quantity = 1

    cost_model = BrazilianCostModel(symbol=symbol)
    slip_model = SlippageModel(symbol=symbol)

    # Custo de um round-trip (entrada e saída = 2 operações)
    # n_operations default é 2 em trade_cost
    tc = cost_model.trade_cost(price, quantity, n_operations=2)

    # O teste exige verificar corretagem (0), emolumentos fixos B3 (~0.25 por contrato), slippage (~1 tick = R$1.00)
    # Como são duas operações (round-trip), emolumentos são cobrados na entrada e na saída: 2 * 0.25 = 0.50
    assert tc == 0.50, f"Expected transaction cost of 0.50, got {tc}"

    # Slippage
    # Calculado por ordem, na verdade `estimate` retorna o slippage de UMA operação.
    # Mas num backtest a gente paga na entrada e na saída. Como testamos a unidade:
    sl = slip_model.estimate(price, quantity, avg_volume=1_000_000)

    # Slippage_ticks = 1.0; tick_size = 5.0; multiplier = 0.2
    # monetary_per_operation = 1.0 * 5.0 * 0.2 * 1 = 1.0
    assert 1.0 <= sl <= 2.0, f"Expected slippage between R$1 and R$2, got {sl}"

    total_round_trip = tc + (
        sl * 2
    )  # multiplicando slippage por 2 pq são duas operações (entrada/saida)

    # O valor total não pode passar de alguns poucos reais.
    assert total_round_trip < 5.0, f"Total round-trip cost is too high: {total_round_trip}"


def test_futures_wdo_costs_verification():
    """Teste similar para dólar."""
    symbol = "WDOJ26"
    price = 5000.0
    quantity = 1

    cost_model = BrazilianCostModel(symbol=symbol)
    slip_model = SlippageModel(symbol=symbol)

    tc = cost_model.trade_cost(price, quantity, n_operations=2)
    assert tc == 0.50

    sl = slip_model.estimate(price, quantity, avg_volume=1_000_000)

    # tick = 0.5, multiplier = 10.0, slippage_ticks = 1.0 -> 0.5 * 10.0 = 5.0 reais
    assert sl == pytest.approx(5.0, abs=1e-2)
