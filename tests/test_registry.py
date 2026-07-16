"""Model registry: HF discovery, persistence, ranking, merge with catalog."""
import json

import httpx
import pytest

from deltav.registry import DiscoveredModel, ModelRegistry
from deltav.router import Catalog

RX_6600M = 8176


def _hf_handler(models: list[dict], trees: dict[str, list], configs: dict | None = None):
    configs = configs or {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        path = request.url.path
        if path == "/api/models":
            return httpx.Response(200, json=models)
        if path.endswith("/tree/main"):
            repo = path[len("/api/models/"):-len("/tree/main")]
            return httpx.Response(200, json=trees.get(repo, []))
        if path.endswith("/config.json"):
            base = path[len("/"):-len("/resolve/main/config.json")]
            if base in configs:
                return httpx.Response(200, json=configs[base])
            return httpx.Response(404)
        return httpx.Response(404)
    return handler


def test_sync_discovers_and_persists(tmp_path):
    models = [{"id": "acme/Cool-7B-GGUF", "downloads": 5000, "likes": 10}]
    trees = {"acme/Cool-7B-GGUF": [
        {"path": "Cool-7B-Q4_K_M.gguf", "size": 4_600_000_000},
        {"path": "Cool-7B-Q8_0.gguf", "size": 8_000_000_000},
        {"path": "Cool-7B-MTP-Q4_K_M.gguf", "size": 4_700_000_000},  # avoided
    ]}
    configs = {"acme/Cool-7B": {"model_type": "qwen2", "num_hidden_layers": 28,
                                "num_attention_heads": 28, "num_key_value_heads": 4,
                                "hidden_size": 3584, "max_position_embeddings": 32768}}
    client = httpx.Client(transport=httpx.MockTransport(_hf_handler(models, trees, configs)))
    reg = ModelRegistry(path=tmp_path / "r.json", catalog=Catalog())
    added = reg.sync_from_hf(limit=10, client=client, now=1.0)
    assert added == 1
    m = next(iter(reg.discovered.values()))
    assert m.repo_id == "acme/Cool-7B-GGUF"
    assert "Q4_K_M" in m.filename and "MTP" not in m.filename  # preferred, non-MTP
    assert m.n_layers == 28 and m.n_kv_heads == 4               # arch from config.json
    assert m.downloads == 5000
    # persisted + reloadable
    reg2 = ModelRegistry(path=tmp_path / "r.json", catalog=Catalog())
    assert m.ref in reg2.discovered


def test_add_repo_with_vision_mmproj(tmp_path):
    trees = {"org/VL-8B-GGUF": [
        {"path": "VL-8B-Q4_K_M.gguf", "size": 5_000_000_000},
        {"path": "mmproj-VL-8B-F16.gguf", "size": 600_000_000},
    ]}
    client = httpx.Client(transport=httpx.MockTransport(_hf_handler([], trees)))
    reg = ModelRegistry(path=tmp_path / "r.json", catalog=Catalog())
    m = reg.add_repo("org/VL-8B-GGUF", client=client)
    assert m and m.vision is True                               # mmproj -> multimodal
    assert "mmproj" not in m.filename                           # served file isn't the projector


def test_rank_merges_catalog_and_discovered_and_marks_served(tmp_path):
    reg = ModelRegistry(path=tmp_path / "r.json", catalog=Catalog())
    reg.discovered["x/Custom-9B-GGUF::Custom-9B-Q4_K_M.gguf"] = DiscoveredModel(
        repo_id="x/Custom-9B-GGUF", filename="Custom-9B-Q4_K_M.gguf",
        file_mb=5200, quant="Q4_K_M", params_b=9.0, downloads=99999)
    served = {"bartowski/Meta-Llama-3.1-8B-Instruct-GGUF::Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"}
    ranked = reg.rank(RX_6600M, kind="chat", top=30, served=served)
    refs = [r["ref"] for r in ranked]
    assert any("Custom-9B" in r for r in refs)                  # discovered model included
    assert any(r["source"] == "catalog" for r in ranked)       # catalog too
    assert ranked[0]["served"]                                  # served model ranked first
    # everything ranked actually fits 8 GB
    assert all(r["max_context"] >= 2048 for r in ranked)


def test_rank_excludes_too_big(tmp_path):
    reg = ModelRegistry(path=tmp_path / "r.json", catalog=Catalog())
    reg.discovered["big/Huge-70B-GGUF::Huge-70B-Q4_K_M.gguf"] = DiscoveredModel(
        repo_id="big/Huge-70B-GGUF", filename="Huge-70B-Q4_K_M.gguf",
        file_mb=42000, quant="Q4_K_M", params_b=70.0)
    ranked = reg.rank(RX_6600M, kind="chat")
    assert all("Huge-70B" not in r["ref"] for r in ranked)     # 42 GB can't fit 8 GB
