"""A unified, auto-updating model database.

Three sources merged into one ranked list:
  * the curated catalog (hand-verified architecture facts);
  * models discovered from HuggingFace (a bot pulls trending GGUF repos,
    reads file sizes + config, estimates VRAM) — persisted and refreshable;
  * models actually announced by live nodes on the network.

`deltav registry sync` runs the HF bot; the gateway serves the merged DB
at /v1/registry; the setup wizard ranks from it for the operator's VRAM.
"""
from .registry import DiscoveredModel, ModelRegistry

__all__ = ["DiscoveredModel", "ModelRegistry"]
