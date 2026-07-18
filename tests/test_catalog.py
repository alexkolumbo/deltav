"""Model catalog and VRAM-fit math."""
from deltav.router.catalog import CURATED_CATALOG, Catalog, estimate_vram_mb

RTX_4070_MB = 12282
RTX_3060_MB = 8192


def test_curated_catalog_sane():
    assert len(CURATED_CATALOG) >= 8
    for spec in CURATED_CATALOG:
        assert spec.file_mb > 0 and 0 < spec.quality <= 1
        # A GGUF spec names a concrete file inside the repo (repo::file). A
        # repo-style spec — a diffusers pipeline, an API relay — has no single
        # file and is referenced by repo id alone (see ModelSpec.ref).
        assert ("::" in spec.ref) == bool(spec.filename)


def test_4070_fits_14b_not_32b():
    catalog = Catalog()
    fitting = {s.repo_id for s in catalog.fitting(RTX_4070_MB)}
    assert "Qwen/Qwen2.5-14B-Instruct-GGUF" in fitting
    assert "Qwen/Qwen2.5-32B-Instruct-GGUF" not in fitting
    assert "bartowski/Llama-3.3-70B-Instruct-GGUF" not in fitting


def test_best_for_4070_is_highest_quality_fit():
    best = Catalog().best_for(RTX_4070_MB)
    assert best is not None
    assert best.params_b > 10  # a 14B-class model, not a 7B
    assert estimate_vram_mb(best) <= RTX_4070_MB


def test_8gb_picks_the_best_model_that_actually_fits():
    """Parameter count is NOT a size proxy any more: a ternary-compressed 27B
    (Bonsai, 3.6 GB) genuinely runs on an 8 GB card while a 14B Q4 does not.
    So assert the real constraint — it fits the VRAM — and that we didn't pick
    one of the uncompressed giants."""
    best = Catalog().best_for(RTX_3060_MB)
    assert best is not None
    assert estimate_vram_mb(best) <= RTX_3060_MB
    assert best.repo_id not in {
        "Qwen/Qwen2.5-32B-Instruct-GGUF",
        "bartowski/Llama-3.3-70B-Instruct-GGUF",
    }


def test_tiny_vram_still_gets_a_model():
    best = Catalog().best_for(2048)
    assert best is not None
    assert best.params_b <= 1.5


def test_by_ref_lookup():
    catalog = Catalog()
    spec = catalog.by_ref("Qwen/Qwen2.5-7B-Instruct-GGUF")
    assert spec is not None
    assert catalog.by_ref(spec.ref) == spec
    assert catalog.by_ref("nope/none") is None
