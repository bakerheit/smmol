"""Reuse the school's dependency-free safetensors implementation."""

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parent.parent.parent / "archive" / "models" / "smLANGUAGE_en_SCH_001" / "safetensors_io.py"
SPEC = importlib.util.spec_from_file_location("school_safetensors_io", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

load_file = MODULE.load_file
save_file = MODULE.save_file
