from __future__ import annotations

import httpx
import pytest

from deltav.config import DVT, ChainParams, Genesis
from deltav.crypto import KeyPair


@pytest.fixture
def params() -> ChainParams:
    return ChainParams(block_time=0.05)


@pytest.fixture
def alice() -> KeyPair:
    return KeyPair.from_seed_hex("11" * 32)


@pytest.fixture
def bob() -> KeyPair:
    return KeyPair.from_seed_hex("22" * 32)


@pytest.fixture
def carol() -> KeyPair:
    return KeyPair.from_seed_hex("33" * 32)


@pytest.fixture
def genesis(params, alice, bob, carol) -> Genesis:
    return Genesis(
        params=params,
        alloc={kp.address: 100_000 * DVT for kp in (alice, bob, carol)},
        stakes={alice.address: 10_000 * DVT, bob.address: 10_000 * DVT},
    )


class MultiTransport(httpx.AsyncBaseTransport):
    """Route requests to in-process ASGI apps by base URL — no sockets."""

    def __init__(self) -> None:
        self.routes: dict[str, httpx.ASGITransport] = {}

    def add(self, base_url: str, app) -> None:
        self.routes[base_url] = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}"
        if request.url.port:
            base += f":{request.url.port}"
        transport = self.routes.get(base)
        if transport is None:
            raise httpx.ConnectError(f"no route for {base}", request=request)
        return await transport.handle_async_request(request)
