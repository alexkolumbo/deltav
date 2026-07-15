"""One-command node onboarding for non-technical operators.

`deltav setup` — an interactive wizard that takes someone from a bare
machine to a live, earning node, explaining every step in plain language.
"""
from .assets import LlamaAsset, resolve_llama_asset
from .custom import ModelAnalysis, analyze_model
from .wizard import SetupWizard, run_setup

__all__ = ["LlamaAsset", "resolve_llama_asset", "ModelAnalysis", "analyze_model",
           "SetupWizard", "run_setup"]
