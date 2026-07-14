"""Smart routing: scoring, model resolution, failover."""
import httpx
import pytest

from deltav.crypto import KeyPair
from deltav.router import Catalog, RouteError, SmartRouter
from deltav.router.scoring import NodeView, score_node

MODEL_14B = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
MODEL_3B = "bartowski/Llama-3.2-3B-Instruct-GGUF::Llama-3.2-3B-Instruct-Q4_K_M.gguf"


def view(**kw) -> NodeView:
    base = dict(address="dv1node", endpoint="http://n:1", vram_mb=12282, models=[],
                reputation=0.5, stake=0, last_seen=0, active=True, load=0.0, alive=True)
    base.update(kw)
    return NodeView(**base)


def test_ready_model_beats_cold_start():
    ready = view(models=[MODEL_14B])
    cold = view(reputation=0.9)
    assert score_node(ready, MODEL_14B, 0) > score_node(cold, MODEL_14B, 0)


def test_dead_node_scores_neg_inf():
    assert score_node(view(alive=False), MODEL_14B, 0) == float("-inf")
    assert score_node(view(active=False), MODEL_14B, 0) == float("-inf")


def test_reputation_and_load_matter():
    good = view(reputation=0.9, load=0.1)
    bad = view(reputation=0.2, load=0.9)
    assert score_node(good, MODEL_14B, 0) > score_node(bad, MODEL_14B, 0)


def _router(handler) -> SmartRouter:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SmartRouter(Catalog(), KeyPair.generate(), client, price_per_token=10)


def _network_handler(fail_endpoints=frozenset()):
    """Fake network: one chain source + two 4070 nodes serving the 14B model."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/chain/nodes"):
            return httpx.Response(200, json={"height": 10, "nodes": [
                {"address": "dv1aaa", "endpoint": "http://a:1", "models": [MODEL_14B],
                 "hardware": {"vram_mb": 12282}, "reputation": 0.9, "stake": 10_000_000_000,
                 "last_seen": 10, "active": True},
                {"address": "dv1bbb", "endpoint": "http://b:1", "models": [MODEL_14B],
                 "hardware": {"vram_mb": 12282}, "reputation": 0.4, "stake": 1_000_000,
                 "last_seen": 5, "active": True},
            ]})
        if url.endswith("/health"):
            return httpx.Response(200, json={"load": 0.0})
        if url.endswith("/infer"):
            base = url.rsplit("/infer", 1)[0]
            if base in fail_endpoints:
                return httpx.Response(500, json={"detail": "boom"})
            node = "dv1aaa" if "//a:" in url else "dv1bbb"
            return httpx.Response(200, json={
                "text": f"answer from {node}", "tokens_in": 5, "tokens_out": 7,
                "receipt_tx": "rcpt", "node": node,
            })
        return httpx.Response(404)
    return handler


async def test_auto_resolves_best_fitting_model():
    router = _router(_network_handler())
    await router.refresh(["http://chain:1"])
    spec = router.resolve_model("auto")
    # both nodes are 12 GB / announce the 14B model -> auto picks the 14B
    assert spec.ref == MODEL_14B


async def test_route_prefers_high_reputation_node():
    router = _router(_network_handler())
    await router.refresh(["http://chain:1"])
    result = await router.route("hi", model="auto", max_tokens=16)
    assert result.node == "dv1aaa"
    assert result.attempts == 1


async def test_failover_to_second_node():
    router = _router(_network_handler(fail_endpoints={"http://a:1"}))
    await router.refresh(["http://chain:1"])
    result = await router.route("hi", model="auto", max_tokens=16)
    assert result.node == "dv1bbb"
    assert result.attempts == 2


async def test_all_nodes_down_raises():
    router = _router(_network_handler(fail_endpoints={"http://a:1", "http://b:1"}))
    await router.refresh(["http://chain:1"])
    with pytest.raises(RouteError, match="all candidate nodes failed"):
        await router.route("hi", model="auto", max_tokens=16)


async def test_unknown_model_raises():
    router = _router(_network_handler())
    await router.refresh(["http://chain:1"])
    with pytest.raises(RouteError, match="not in the catalog"):
        await router.route("hi", model="made/up-model")


async def test_no_live_nodes_raises():
    def handler(request):
        return httpx.Response(500)
    router = _router(handler)
    await router.refresh(["http://chain:1"])
    with pytest.raises(RouteError, match="no live nodes"):
        router.resolve_model("auto")
