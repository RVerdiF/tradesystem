"""
1.1 — Conectividade e Autenticação com MetaTrader 5.

Implementa um context manager para conexão segura ao terminal MT5,
com retry automático e backoff exponencial para resiliência de rede.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from loguru import logger

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore[assignment]

from config.settings import mt5_config


# ---------------------------------------------------------------------------
# Exceções customizadas
# ---------------------------------------------------------------------------
class MT5Error(Exception):
    """Erro base para operações MT5."""


class MT5ConnectionError(MT5Error):
    """Falha ao conectar ao terminal MT5."""


class MT5AuthError(MT5Error):
    """Falha na autenticação (login/senha/servidor)."""


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------
class MT5Connector:
    """
    Gerencia o ciclo de vida da conexão com o MetaTrader 5.

    Uso recomendado como context manager::

        with MT5Connector() as conn:
            info = conn.terminal_info()
            print(info)

    Também pode ser usado via connect()/disconnect() manuais.
    """

    def __init__(
        self,
        login: int | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ) -> None:
        self.login = login or mt5_config.login
        self.password = password or mt5_config.password
        self.server = server or mt5_config.server
        self.path = path or mt5_config.path
        self.timeout = timeout or mt5_config.timeout
        self.max_retries = max_retries if max_retries is not None else mt5_config.max_retries
        self.retry_delay = retry_delay if retry_delay is not None else mt5_config.retry_delay
        self._connected = False

    # ---- Context manager ----

    def __enter__(self) -> MT5Connector:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.disconnect()

    # ---- Conexão ----

    def connect(self) -> None:
        """Inicializa e autentica no MT5 com retry e backoff exponencial."""
        if not MT5_AVAILABLE:
            raise MT5ConnectionError(
                "Pacote MetaTrader5 não instalado. "
                "Instale com: pip install MetaTrader5 (apenas Windows)."
            )

        delay = self.retry_delay
        last_error = ""

        for attempt in range(1, self.max_retries + 1):
            logger.info("MT5 — tentativa de conexão {}/{}", attempt, self.max_retries)

            # Inicialização
            init_kwargs: dict = {"login": self.login, "timeout": self.timeout}
            if self.path:
                init_kwargs["path"] = self.path
            if self.server:
                init_kwargs["server"] = self.server
            if self.password:
                init_kwargs["password"] = self.password

            if not mt5.initialize(**init_kwargs):
                error_code = mt5.last_error()
                last_error = f"initialize falhou: {error_code}"
                logger.warning("MT5 — {}", last_error)
                time.sleep(delay)
                delay *= 2  # backoff exponencial
                continue

            # Login explícito (caso a inicialização não faça login automático)
            if self.login and self.password:
                authorized = mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server,
                )
                if not authorized:
                    error_code = mt5.last_error()
                    last_error = f"login falhou: {error_code}"
                    logger.warning("MT5 — {}", last_error)
                    mt5.shutdown()
                    time.sleep(delay)
                    delay *= 2
                    continue

            self._connected = True
            logger.success("MT5 — conectado com sucesso (conta {})", self.login)
            return

        # Esgotou tentativas
        raise MT5ConnectionError(
            f"Não foi possível conectar ao MT5 após {self.max_retries} tentativas. "
            f"Último erro: {last_error}"
        )

    def disconnect(self) -> None:
        """Encerra a conexão com o MT5."""
        if self._connected and MT5_AVAILABLE:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 — desconectado")

    # ---- Status ----

    @property
    def is_connected(self) -> bool:
        return self._connected

    def terminal_info(self) -> dict | None:
        """Retorna informações do terminal MT5."""
        self._ensure_connected()
        info = mt5.terminal_info()
        return info._asdict() if info else None

    def account_info(self) -> dict | None:
        """Retorna informações da conta logada."""
        self._ensure_connected()
        info = mt5.account_info()
        return info._asdict() if info else None

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise MT5ConnectionError("Não conectado ao MT5. Use connect() ou o context manager.")


# ---------------------------------------------------------------------------
# Atalho funcional
# ---------------------------------------------------------------------------
@contextmanager
def mt5_session(**kwargs) -> Generator[MT5Connector, None, None]:
    """
    Atalho para abrir uma sessão MT5::

        with mt5_session() as conn:
            print(conn.account_info())
    """
    conn = MT5Connector(**kwargs)
    try:
        conn.connect()
        yield conn
    finally:
        conn.disconnect()
