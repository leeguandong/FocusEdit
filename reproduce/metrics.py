"""Load the pinned PnP evaluator while preserving its metric arithmetic."""
from pathlib import Path
import hashlib
import sys

EXPECTED_SHA256 = "4dee71873d6eadb0ce5eeed22c82ccd2690938c94cea105126ec11cfa5771343"


def load_calculator(pnp_root, clip_model):
    root = Path(pnp_root).resolve()
    math_file = root / "evaluation/matrics_calculator.py"
    digest = hashlib.sha256(math_file.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("PnP metric source differs from the pinned version; see docs/REPRODUCIBILITY.md")
    sys.path.insert(0, str(root))
    from evaluation import matrics_calculator as module
    original = module.CLIPScore
    def score(*args, **kwargs):
        from transformers import CLIPModel, CLIPProcessor
        def load_clip():
            return (CLIPModel.from_pretrained(clip_model),
                    CLIPProcessor.from_pretrained(clip_model, use_fast=False))
        kwargs["model_name_or_path"] = load_clip
        return original(*args, **kwargs)
    module.CLIPScore = score
    return module.MetricsCalculator, digest
