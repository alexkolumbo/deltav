"""Diffusers image backend: registration, auto-selection safety, FLUX.1-schnell
tuning, and the draw path — exercised with a fake pipeline so the test suite
never needs torch/diffusers installed."""
import base64

import pytest

from deltav.compute import make_backend
from deltav.compute.base import BACKENDS, ImageRequest, InferRequest
from deltav.compute.diffusion import DEFAULT_IMAGE_MODEL, DiffusersBackend


def test_backend_is_registered():
    make_backend("auto")                       # triggers submodule registration
    assert any(c.name == "diffusers" for c in BACKENDS)


def test_auto_never_selects_the_draw_only_backend():
    """`auto` means a general-purpose node — handing it a draw-only engine
    would blow up on the first text job."""
    assert DiffusersBackend.text_capable is False
    assert make_backend("auto").name != "diffusers"


def test_default_model_is_the_commercially_usable_one():
    # FLUX.1-schnell is Apache-2.0; -dev and Ideogram are non-commercial, which
    # a paid inference network may not serve.
    assert DEFAULT_IMAGE_MODEL == "black-forest-labs/FLUX.1-schnell"


def test_schnell_tuning_respects_the_distillation():
    """schnell is timestep-distilled: no CFG, a handful of steps, 256-token
    prompts. Getting these wrong silently ruins output quality."""
    steps, guidance, max_seq = DiffusersBackend._tuning(
        "black-forest-labs/FLUX.1-schnell", 20)
    assert guidance == 0.0            # CFG on a distilled model produces mush
    assert steps <= 8                 # 20 steps is wasted compute, not "better"
    assert max_seq == 256
    # a non-distilled sibling keeps real guidance and a longer prompt budget
    d_steps, d_guidance, d_max = DiffusersBackend._tuning(
        "black-forest-labs/FLUX.1-dev", 30)
    assert d_guidance > 0 and d_steps == 30 and d_max == 512


class _FakeImage:
    def save(self, buf, format="PNG"):       # noqa: A002 - mirrors PIL's API
        buf.write(b"\x89PNG\r\n\x1a\n" + b"fake-pixels")


class _FakeOut:
    images = [_FakeImage()]


class _FakePipe:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return _FakeOut()


def test_generate_image_returns_base64_png_with_schnell_params(monkeypatch):
    backend = DiffusersBackend()
    pipe = _FakePipe()
    monkeypatch.setattr(backend, "load", lambda ref: None)
    monkeypatch.setattr(DiffusersBackend, "_generator", staticmethod(lambda seed: object()))
    backend._pipe = pipe

    result = backend.generate_image(ImageRequest(
        prompt="a cat", model_ref=DEFAULT_IMAGE_MODEL,
        width=768, height=512, steps=20, seed=7))

    assert result.backend == "diffusers" and result.seed == 7
    assert result.model_ref == DEFAULT_IMAGE_MODEL
    # diffusion sampling diverges across GPUs -> fuzzy spot-check path
    assert result.deterministic is False
    assert len(result.images) == 1
    assert base64.b64decode(result.images[0]).startswith(b"\x89PNG")
    # the distillation-critical params actually reached the pipeline
    assert pipe.kwargs["guidance_scale"] == 0.0
    assert pipe.kwargs["num_inference_steps"] <= 8
    assert pipe.kwargs["max_sequence_length"] == 256
    assert pipe.kwargs["width"] == 768 and pipe.kwargs["height"] == 512
    assert pipe.kwargs["prompt"] == "a cat"


def test_infer_refuses_text_with_a_clear_message():
    with pytest.raises(NotImplementedError, match="images only"):
        DiffusersBackend().infer(InferRequest(prompt="hi", model_ref="x"))


def test_is_available_reflects_missing_deps(monkeypatch):
    """No torch/diffusers on the box -> the backend reports itself unusable
    instead of failing later mid-job."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name in ("torch", "diffusers"):
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert DiffusersBackend.is_available() is False
