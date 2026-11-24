from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from .pipeline_naviedit import NaviEditPipeline
from .sd3_bundle import load_sd3_bundle


DEFAULT_MODEL_ROOT = "stabilityai/stable-diffusion-3-medium-diffusers"
DEFAULT_PRECISION = "fp16"
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_MAX_SEQUENCE_LENGTH = 77
DEFAULT_SEED = 123
DEFAULT_EDIT_CONFIG: Dict[str, Any] = {
    "num_inference_steps": 50,
    "noise_samples": 1,
    "cfg_scale": 13.5,
    "regularization_scale": 0.01,
    "regularization_dt_scale": True,
    "t_ref": 42,
    "t_edit": 50,
    "edit_dt": 1.0,
    "use_equiv_gain": True,
    "use_cfl": True,
    "cfl_tau": 8.0,
}


def resolve_model_root(model_root: Optional[str]) -> str:
    if model_root is None:
        return DEFAULT_MODEL_ROOT
    return str(model_root)


def normalize_precision(value: Optional[str]) -> str:
    precision = (value or DEFAULT_PRECISION).lower()
    mapping = {
        "fp32": "fp32",
        "float32": "fp32",
        "fp16": "fp16",
        "float16": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
    }
    if precision not in mapping:
        raise ValueError(f"Unsupported precision '{value}'.")
    return mapping[precision]


def default_device(device: Optional[str | torch.device] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_navi_pipeline(
    *,
    model_root: Optional[str],
    default_edit_config: Optional[Dict[str, Any]] = None,
    device: Optional[str | torch.device] = None,
    precision: Optional[str] = None,
    image_size: Optional[int] = None,
    use_center_crop: bool = True,
    max_sequence_length: Optional[int] = None,
    efficient_memory: bool = False,
) -> NaviEditPipeline:
    resolved_device = default_device(device)
    resolved_precision = normalize_precision(precision)
    resolved_model_root = resolve_model_root(model_root)
    resolved_image_size = int(image_size or DEFAULT_IMAGE_SIZE)
    resolved_sequence_length = int(max_sequence_length or DEFAULT_MAX_SEQUENCE_LENGTH)

    bundle = load_sd3_bundle(
        path=resolved_model_root,
        device=resolved_device,
        precision=resolved_precision,
        max_sequence_length=resolved_sequence_length,
        efficient_memory=bool(efficient_memory),
    )
    return NaviEditPipeline(
        bundle,
        default_edit_config=default_edit_config,
        image_size=resolved_image_size,
        device=resolved_device,
        use_center_crop=use_center_crop,
    )
