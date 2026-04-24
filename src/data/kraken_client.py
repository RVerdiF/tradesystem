"""Cliente REST para Kraken Spot API — TradeSystem5000.

Cobre market data (público) e account/order management (privado).
Credenciais via KRAKEN_API_KEY / KRAKEN_API_SECRET (env vars ou .env).

Referência: https://docs.kraken.com/api/docs/rest-api/
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
import urllib.parse
from typing import Any

import requests
from loguru import logger

from config.settings import kraken_config

_BASE_URL = "https://api.kraken.com"
_TIMEOUT = 10  # segundos


# ---------------------------------------------------------------------------
# Exceções
# ---------------------------------------------------------------------------
class KrakenError(Exception):
    """Erro retornado pela API Kraken (campo 'error' não-vazio)."""


class KrakenAuthError(KrakenError):
    """Credenciais ausentes ou inválidas."""


# ---------------------------------------------------------------------------
# Cliente
# ---------------------------------------------------------------------------
class KrakenClient:
    """Cliente para a Kraken Spot REST API.

    Uso básico::

        client = KrakenClient()                        # lê credenciais do env
        print(client.ticker("XBTUSD"))
        print(client.balance())
        client.add_order("XBTUSD", "buy", "limit", "0.001", price="30000", validate=True)
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        self._api_key = api_key or kraken_config.api_key
        self._api_secret = api_secret or kraken_config.api_secret
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "TradeSystem5000/1.0"})

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _nonce(self) -> str:
        return str(time.time_ns())

    def _sign(self, urlpath: str, data: dict) -> str:
        """Gera assinatura HMAC-SHA512 conforme especificação Kraken."""
        if not self._api_secret:
            raise KrakenAuthError("KRAKEN_API_SECRET não configurado.")
        post_data = urllib.parse.urlencode(data)
        encoded = (data["nonce"] + post_data).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        secret = base64.b64decode(self._api_secret)
        signature = hmac.new(secret, message, hashlib.sha512)
        return base64.b64encode(signature.digest()).decode()

    def _public(self, endpoint: str, params: dict | None = None) -> Any:
        url = f"{_BASE_URL}/0/public/{endpoint}"
        resp = self._session.get(url, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        return self._handle_response(resp.json())

    def _private(self, endpoint: str, data: dict | None = None) -> Any:
        if not self._api_key:
            raise KrakenAuthError("KRAKEN_API_KEY não configurado.")
        urlpath = f"/0/private/{endpoint}"
        payload = dict(data or {})
        payload["nonce"] = self._nonce()
        headers = {
            "API-Key": self._api_key,
            "API-Sign": self._sign(urlpath, payload),
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = self._session.post(
            f"{_BASE_URL}{urlpath}",
            data=payload,
            headers=headers,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return self._handle_response(resp.json())

    @staticmethod
    def _handle_response(body: dict) -> Any:
        """Kraken sempre retorna HTTP 200; erros ficam em body['error']."""
        errors = body.get("error", [])
        if errors:
            raise KrakenError(errors)
        return body.get("result")

    # ------------------------------------------------------------------
    # Market Data (público)
    # ------------------------------------------------------------------

    def server_time(self) -> dict:
        """Retorna hora do servidor Kraken."""
        return self._public("Time")

    def system_status(self) -> dict:
        """Retorna status do sistema: 'online', 'cancel_only', 'post_only'."""
        return self._public("SystemStatus")

    def asset_pairs(self, pair: str | None = None) -> dict:
        """Metadados de pares (decimais, ordermin, etc.)."""
        params = {"pair": pair} if pair else None
        return self._public("AssetPairs", params)

    def ticker(self, pair: str) -> dict:
        """Cotação atual (bid/ask/last/volume/24h stats)."""
        return self._public("Ticker", {"pair": pair})

    def ohlc(self, pair: str, interval: int = 1, since: int | None = None) -> dict:
        """Candles OHLCV.

        Parameters
        ----------
        interval : int
            Minutos: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600.
        since : int, optional
            Unix timestamp para buscar a partir de.
        """
        params: dict = {"pair": pair, "interval": interval}
        if since is not None:
            params["since"] = since
        return self._public("OHLC", params)

    def order_book(self, pair: str, count: int = 100) -> dict:
        """Book de ordens (bids e asks)."""
        return self._public("Depth", {"pair": pair, "count": count})

    def recent_trades(self, pair: str, since: int | None = None) -> dict:
        """Trades recentes."""
        params: dict = {"pair": pair}
        if since is not None:
            params["since"] = since
        return self._public("Trades", params)

    # ------------------------------------------------------------------
    # Account / Positions (privado)
    # ------------------------------------------------------------------

    def balance(self) -> dict:
        """Saldos spot por ativo."""
        return self._private("Balance")

    def balance_ex(self) -> dict:
        """Saldo estendido (inclui holds e staking)."""
        return self._private("BalanceEx")

    def trade_balance(self, asset: str = "ZUSD") -> dict:
        """Resumo de equity/margem."""
        return self._private("TradeBalance", {"asset": asset})

    def open_positions(self, txid: str | None = None, docalcs: bool = True) -> dict:
        """Posições de margem abertas (vazio se spot puro)."""
        data: dict = {"docalcs": docalcs}
        if txid:
            data["txid"] = txid
        return self._private("OpenPositions", data)

    def open_orders(self) -> dict:
        """Ordens abertas."""
        return self._private("OpenOrders")

    def closed_orders(
        self, start: int | None = None, end: int | None = None
    ) -> dict:
        """Ordens fechadas recentes."""
        data: dict = {}
        if start is not None:
            data["start"] = start
        if end is not None:
            data["end"] = end
        return self._private("ClosedOrders", data)

    def query_orders(self, txid: str) -> dict:
        """Consulta uma ou mais ordens por txid (separados por vírgula)."""
        return self._private("QueryOrders", {"txid": txid})

    # ------------------------------------------------------------------
    # Trading (privado)
    # ------------------------------------------------------------------

    def _check_system_online(self) -> None:
        status = self.system_status().get("status", "unknown")
        if status != "online":
            raise KrakenError(f"Sistema Kraken não está online: status={status!r}")

    def add_order(
        self,
        pair: str,
        side: str,
        ordertype: str,
        volume: str,
        price: str | None = None,
        price2: str | None = None,
        leverage: str | None = None,
        oflags: str | None = None,
        timeinforce: str | None = None,
        userref: int | None = None,
        validate: bool = True,
    ) -> dict:
        """Envia uma ordem.

        Parameters
        ----------
        pair : str
            Par de negociação, ex: "XBTUSD".
        side : str
            "buy" ou "sell".
        ordertype : str
            "market", "limit", "stop-loss", "take-profit", etc.
        volume : str
            Quantidade base (string para preservar precisão).
        price : str, optional
            Preço limite (obrigatório para limit/stop-loss/take-profit).
        validate : bool
            True (padrão) = dry-run — Kraken valida sem enviar.
            Passe validate=False explicitamente para enviar de verdade.
        """
        if not validate:
            self._check_system_online()

        data: dict = {
            "pair": pair,
            "type": side,
            "ordertype": ordertype,
            "volume": volume,
        }
        if price is not None:
            data["price"] = price
        if price2 is not None:
            data["price2"] = price2
        if leverage is not None:
            data["leverage"] = leverage
        if oflags is not None:
            data["oflags"] = oflags
        if timeinforce is not None:
            data["timeinforce"] = timeinforce
        if userref is not None:
            data["userref"] = userref
        if validate:
            data["validate"] = "true"

        result = self._private("AddOrder", data)
        mode = "DRY-RUN" if validate else "ENVIADA"
        logger.info("Ordem {} ({}) → {}", mode, pair, result)
        return result

    def amend_order(
        self,
        txid: str,
        volume: str | None = None,
        price: str | None = None,
        price2: str | None = None,
    ) -> dict:
        """Modifica preço ou volume de uma ordem aberta."""
        data: dict = {"txid": txid}
        if volume is not None:
            data["volume"] = volume
        if price is not None:
            data["price"] = price
        if price2 is not None:
            data["price2"] = price2
        return self._private("AmendOrder", data)

    def cancel_order(self, txid: str) -> dict:
        """Cancela uma ordem pelo txid."""
        result = self._private("CancelOrder", {"txid": txid})
        logger.info("Ordem {} cancelada.", txid)
        return result

    def cancel_all(self) -> dict:
        """Cancela todas as ordens abertas."""
        result = self._private("CancelAll")
        logger.warning("CancelAll executado — todas as ordens abertas canceladas.")
        return result

    def cancel_all_after(self, timeout: int) -> dict:
        """Dead-man's switch: cancela tudo após `timeout` segundos sem renovação."""
        return self._private("CancelAllOrdersAfter", {"timeout": timeout})
