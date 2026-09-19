"""Reuse the school-model implementation with a larger configured context."""

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parent.parent.parent / "archive" / "models" / "smLANGUAGE_en_SCH_001" / "model.py"
SPEC = importlib.util.spec_from_file_location("school_language_model", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

Config = MODULE.Config
LanguageModel = MODULE.LanguageModel
