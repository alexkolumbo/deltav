"""Hardware-aware model planner: exact KV math, context solving, ranking."""
import httpx

from deltav.config import ChainParams
from deltav.crypto import KeyPair
from deltav.gateway import GatewayDaemon
from deltav.router import Catalog
from deltav.router.catalog import kv_bytes_per_token
from deltav.router.planner import launch_hint, max_context_for, plan

RX_6600M = 8176
RTX_4070 = 12282

LLAMA_8B = Catalog().by_ref("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
LLAMA_3B = Catalog().by_ref("bartowski/Llama-3.2-3B-Instruct-GGUF")


def test_kv_math_is_exact():
    # 2 (K+V) x 32 layers x 8 kv-heads x 128 dim x 2 bytes = 128 KiB/token
    assert kv_bytes_per_token(LLAMA_8B, "f16") == 131072
    assert kv_bytes_per_token(LLAMA_8B, "q4_0") == int(131072 * 0.5625 / 2)


def test_3b_reaches_native_128k_on_8gb():
    assert max_context_for(LLAMA_3B, RX_6600M, "q4_0") == 131072


def test_8b_fp16_kv_is_context_bound_on_8gb():
    ctx = max_context_for(LLAMA_8B, RX_6600M, "f16")
    assert 8192 <= ctx <= 16384  # the goose lesson, now computed in advance
    # quantized KV more than doubles it
    assert max_context_for(LLAMA_8B, RX_6600M, "q4_0") >= 2 * ctx


def test_max_context_objective_tops_at_128k():
    options = plan(RX_6600M, objective="max_context")
    assert options[0].max_context == 131072
    assert options[0].params_b >= 3  # 3B beats 1B at equal context


def test_max_quality_objective_on_4070():
    options = plan(RTX_4070, objective="max_quality")
    assert options[0].quality >= 0.84  # 14B-class
    refs = {o.ref for o in options}
    assert not any("32B" in r or "70B" in r for r in refs)


def test_balanced_prefers_8b_on_8gb():
    options = plan(RX_6600M, objective="balanced")
    assert options[0].params_b >= 7
    assert options[0].max_context >= 16384


def test_launch_hint_includes_fa_for_quantized_kv():
    options = plan(RX_6600M, objective="max_context")
    quant_kv = next(o for o in options if o.kv_type != "f16")
    hint = launch_hint(quant_kv)
    assert "-fa on" in hint and f"-c {quant_kv.max_context}" in hint
    f16 = next((o for o in options if o.kv_type == "f16"), None)
    if f16:
        assert "-fa" not in launch_hint(f16)


async def test_gateway_plan_endpoint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)  # no live nodes — planner still works

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    gateway = GatewayDaemon(KeyPair.generate(), node_urls=["http://n:1"],
                            params=ChainParams(), client=client)
    api = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app),
                            base_url="http://gw")
    resp = await api.get("/v1/plan", params={"vram_mb": RX_6600M, "objective": "max_context"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["options"][0]["max_context"] == 131072
    assert "launch" in data["options"][0]
    assert data["options"][0]["already_served_on_network"] is False
    await api.aclose()
    await client.aclose()
