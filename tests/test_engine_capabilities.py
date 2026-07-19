"""Nodes must PROVE what they can do, not declare it.

Two live failures motivated this: a node whose llama-server had died kept
answering /health with 200 and kept being routed jobs that could only 500,
and a model the catalog calls vision-capable was served by an engine started
without --mmproj, so images went to a node that could not read them.
"""
import httpx
import pytest

from deltav.compute.llamaserver import LlamaServerBackend
from deltav.compute.mock import MockBackend
from deltav.router.router import RouteError, SmartRouter, _why
from deltav.router.scoring import NodeView, score_node

ENGINE = "http://engine.test"


def _backend(health_code=200, props=None, props_code=200):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(health_code)
        if path == "/props":
            return httpx.Response(props_code, json=props or {})
        return httpx.Response(404)

    return LlamaServerBackend(base_url=ENGINE,
                              client=httpx.Client(transport=httpx.MockTransport(handler)))


# ------------------------------------------------------------ backend probe

def test_vision_comes_from_the_engine_not_the_catalog():
    """--mmproj is what makes it true, and only /props knows."""
    with_proj = _backend(props={"modalities": {"vision": True, "audio": False}})
    without = _backend(props={"modalities": {"vision": False}})
    assert with_proj.capabilities() == {"ready": True, "vision": True}
    assert without.capabilities() == {"ready": True, "vision": False}


def test_dead_engine_is_not_ready():
    assert _backend(health_code=503).capabilities()["ready"] is False


def test_unreachable_engine_is_not_ready():
    """A node that cannot reach its own engine must not claim it can serve."""
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("engine down")

    backend = LlamaServerBackend(base_url=ENGINE,
                                 client=httpx.Client(transport=httpx.MockTransport(boom)))
    assert backend.capabilities() == {"ready": False, "vision": False}


def test_engine_probe_is_cached():
    """/health is polled by every router refresh; that must not multiply
    into the engine while it is generating."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, json={"modalities": {"vision": True}})

    backend = LlamaServerBackend(base_url=ENGINE,
                                 client=httpx.Client(transport=httpx.MockTransport(handler)))
    for _ in range(5):
        backend.capabilities()
    assert calls.count("/health") == 1


def test_backend_without_a_probe_reports_its_declared_capability():
    caps = MockBackend().capabilities()
    assert caps["ready"] is True


# ------------------------------------------------------------ router uptake

def _node(addr="dv1a", vision=None, alive=True):
    return NodeView(address=addr, endpoint=f"http://{addr}", vram_mb=24000,
                    models=["repo::m.gguf"], reputation=0.8, stake=0, last_seen=0,
                    alive=alive, vision=vision)


async def _refresh_with(health_body: dict) -> NodeView:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/chain/nodes":
            return httpx.Response(200, json={"height": 10, "nodes": [{
                "address": "dv1a", "endpoint": "http://dv1a", "models": ["repo::m.gguf"],
                "reputation": 0.8, "stake": 0, "last_seen": 10,
                "hardware": {"vram_mb": 24000, "dynamic_models": False}}]})
        return httpx.Response(200, json=health_body)

    from deltav.crypto import KeyPair
    from deltav.router.catalog import Catalog
    router = SmartRouter(Catalog(), KeyPair.generate(),
                         httpx.AsyncClient(transport=httpx.MockTransport(handler)), 10)
    await router.refresh(["http://chain"])
    return router.nodes[0]


async def test_node_reporting_a_dead_engine_stops_being_a_candidate():
    """The daemon answering 200 says the daemon is up, nothing more."""
    node = await _refresh_with({"load": 0.0, "engine": {"ready": False, "vision": False}})
    assert node.alive is False
    assert score_node(node, "repo::m.gguf", 10) == float("-inf")


async def test_older_node_that_reports_nothing_is_still_trusted():
    """Absence of the field is not evidence of a dead engine — de-listing
    every node running an older build would take the network down."""
    node = await _refresh_with({"load": 0.0})
    assert node.alive is True
    assert node.vision is None


async def test_images_never_go_to_a_node_without_an_image_projector():
    from deltav.crypto import KeyPair
    from deltav.router.catalog import Catalog

    router = SmartRouter(Catalog(), KeyPair.generate(), httpx.AsyncClient(), 10)
    spec = next(s for s in router.catalog.specs if s.vision)
    router.nodes = [_node(vision=False)]
    router.nodes[0].models = [spec.ref]

    with pytest.raises(RouteError, match="mmproj"):
        await router.route("look", model=spec.ref, images=["data:image/png;base64,AAA"])


async def test_text_still_routes_to_a_node_without_vision():
    """Only image requests care; the same node answers text fine."""
    from deltav.crypto import KeyPair
    from deltav.router.catalog import Catalog

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "hi", "tokens_in": 1, "tokens_out": 1})

    router = SmartRouter(Catalog(), KeyPair.generate(),
                         httpx.AsyncClient(transport=httpx.MockTransport(handler)), 10)
    spec = next(s for s in router.catalog.specs if s.vision)
    node = _node(vision=False)
    node.models = [spec.ref]
    router.nodes = [node]
    assert (await router.route("hi", model=spec.ref)).text == "hi"


# ------------------------------------------------------------ benching

async def test_a_failed_node_is_not_resurrected_by_its_own_health_check():
    """The live loop this closes: an old-build node whose engine had died
    kept answering /health 200, so every refresh un-benched it and the next
    request hit the same 500. Forever."""
    from deltav.crypto import KeyPair
    from deltav.router.catalog import Catalog

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/chain/nodes":
            return httpx.Response(200, json={"height": 10, "nodes": [{
                "address": "dv1a", "endpoint": "http://dv1a", "models": ["repo::m.gguf"],
                "reputation": 0.8, "stake": 0, "last_seen": 10,
                "hardware": {"vram_mb": 24000, "dynamic_models": False}}]})
        return httpx.Response(200, json={"load": 0.0})      # old build: no engine field

    router = SmartRouter(Catalog(), KeyPair.generate(),
                         httpx.AsyncClient(transport=httpx.MockTransport(handler)), 10)
    await router.refresh(["http://chain"])
    router._bench(router.nodes[0])
    await router.refresh(["http://chain"])
    assert router.nodes[0].alive is False


async def test_the_bench_expires_so_a_repaired_node_returns_on_its_own():
    """Operators restart engines; nobody should have to restart the gateway."""
    from deltav.crypto import KeyPair
    from deltav.router.catalog import Catalog

    router = SmartRouter(Catalog(), KeyPair.generate(), httpx.AsyncClient(), 10)
    router._benched["dv1a"] = 0.0            # a deadline already in the past
    assert router._is_benched("dv1a") is False
    assert "dv1a" not in router._benched     # and the entry is reaped


# ------------------------------------------------------------ diagnosability

def test_route_errors_carry_the_node_reason_not_an_mdn_link():
    """'Server error 500 ... check MDN' told an operator nothing about which
    node broke or why."""
    resp = httpx.Response(500, json={"detail": "inference failed: llama-server 503"},
                          request=httpx.Request("POST", "http://dv1a/infer"))
    exc = httpx.HTTPStatusError("Server error '500'", request=resp.request, response=resp)
    why = _why(exc)
    assert "llama-server 503" in why
    assert "mozilla" not in why.lower()
