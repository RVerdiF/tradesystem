"""Exemplos de uso do KrakenClient — TradeSystem5000.

Execute com:
    python examples_kraken.py

Certifique-se de ter um arquivo .env com KRAKEN_API_KEY e KRAKEN_API_SECRET
(ou as variáveis exportadas no shell) antes de rodar exemplos privados.
"""

from dotenv import load_dotenv

load_dotenv()  # carrega .env antes de importar settings

from src.data.kraken_client import KrakenClient, KrakenError  # noqa: E402
from src.data.loaders import fetch_kraken_data  # noqa: E402

client = KrakenClient()


# ---------------------------------------------------------------------------
# 1. Ticker BTC/USD
# ---------------------------------------------------------------------------
def example_ticker() -> None:
    print("\n=== Ticker XBTUSD ===")
    result = client.ticker("XBTUSD")
    # Kraken retorna pelo nome canônico; pegamos o primeiro valor
    pair_data = next(iter(result.values()))
    print(f"  Ask:  {pair_data['a'][0]}")
    print(f"  Bid:  {pair_data['b'][0]}")
    print(f"  Last: {pair_data['c'][0]}")


# ---------------------------------------------------------------------------
# 2. OHLCV via loader
# ---------------------------------------------------------------------------
def example_ohlc() -> None:
    print("\n=== OHLCV XBTUSD (1h, últimas barras) ===")
    df = fetch_kraken_data("XBTUSD", interval="1h")
    print(df.tail(5).to_string())


# ---------------------------------------------------------------------------
# 3. Saldos e posições abertas (requer credenciais)
# ---------------------------------------------------------------------------
def example_account() -> None:
    print("\n=== Saldos ===")
    try:
        balances = client.balance()
        for asset, amount in balances.items():
            if float(amount) > 0:
                print(f"  {asset}: {amount}")

        print("\n=== Posições Abertas (margin) ===")
        positions = client.open_positions()
        if positions:
            for txid, pos in positions.items():
                print(f"  {txid}: {pos}")
        else:
            print("  Nenhuma posição de margem aberta (normal para spot puro).")

        print("\n=== Ordens Abertas ===")
        orders = client.open_orders()
        open_orders = orders.get("open", {})
        if open_orders:
            for txid, order in open_orders.items():
                desc = order["descr"]
                print(f"  {txid}: {desc['order']}")
        else:
            print("  Nenhuma ordem aberta.")

    except KrakenError as e:
        print(f"  Erro de API (verifique credenciais): {e}")


# ---------------------------------------------------------------------------
# 4. Ordem limit em dry-run (validate=True — não envia de verdade)
# ---------------------------------------------------------------------------
def example_dry_run_order() -> None:
    print("\n=== Ordem Limit BTC/USD (DRY-RUN) ===")
    try:
        result = client.add_order(
            pair="XBTUSD",
            side="buy",
            ordertype="limit",
            volume="0.001",
            price="10000",   # preço bem abaixo do mercado — seguro para teste
            validate=True,   # padrão: Kraken valida sem executar
        )
        print(f"  Resultado da validação: {result}")
        print("  Ordem NÃO enviada (validate=True). Passe validate=False para enviar.")
    except KrakenError as e:
        print(f"  Erro: {e}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    example_ticker()
    example_ohlc()
    example_account()
    example_dry_run_order()
