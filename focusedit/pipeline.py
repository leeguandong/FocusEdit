"""End-to-end FocusEdit inference with explicit encoding/probe/edit seeds."""
from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .config import default_config
from .attention import extract_edit_masks
from .guidance import FocusEditModel
from ._backbone.model import NaviFlowAlignModel
from ._backbone.pipeline_factory import build_navi_pipeline


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


@dataclass
class FocusEditOutput:
    image: Image.Image
    mask: Image.Image | None
    latents: torch.Tensor
    step_norms: list[float]


class FocusEditPipeline:
    """Attention localization followed by spatially modulated editing."""
    def __init__(self, backend, config=None):
        self.backend = backend
        self.config = default_config() if config is None else config

    @classmethod
    def from_pretrained(cls, model_id=None, *, config=None, device="cuda"):
        cfg = default_config() if config is None else config
        backend = build_navi_pipeline(
            model_root=model_id or cfg["model"], default_edit_config=cfg["edit"],
            precision=cfg["precision"], image_size=cfg["image_size"],
            max_sequence_length=cfg["max_sequence_length"], device=device,
        )
        return cls(backend, cfg)

    @torch.no_grad()
    def __call__(self, image, source_prompt, target_prompt, *, baseline=False):
        cfg, pipe = self.config, self.backend
        source_prompt = source_prompt.replace("[", "").replace("]", "")
        target_prompt = target_prompt.replace("[", "").replace("]", "")
        set_seed(cfg["seeds"]["encode"])
        x = pipe.vae_adapter.encode(pipe._prepare_image_tensor(image.convert("RGB")))
        mask_image = None
        if not baseline:
            alpha = float(cfg["background_guidance_floor"])
            if not 0 <= alpha <= 1:
                raise ValueError("background_guidance_floor must be in [0, 1]")
            set_seed(cfg["seeds"]["probe"])
            _, binary, info = extract_edit_masks(pipe, x, source_prompt, target_prompt,
                                                cfg=cfg["edit"], layers=cfg["attention_layers"])
            if cfg["mask_type"] == "binary":
                support = F.interpolate(binary, size=x.shape[-2:], mode="nearest")
            elif cfg["mask_type"] == "continuous":
                support = F.interpolate(info["continuous_unit"], size=x.shape[-2:],
                                        mode="bilinear", align_corners=False)
            else:
                raise ValueError("mask_type must be binary or continuous")
            weight = alpha + (1-alpha)*support
            model = FocusEditModel(pipe.transformer, pipe.scheduler, cfg["edit"]).to(x.device).eval()
            model.set_edit_mask(weight)
            mask_image = Image.fromarray((support[0,0].cpu().numpy()*255).astype(np.uint8))
        else:
            model = NaviFlowAlignModel(pipe.transformer, pipe.scheduler, cfg["edit"]).to(x.device).eval()
        source_embed = pipe.encode_prompt([source_prompt])
        target_embed = pipe.encode_prompt([target_prompt])
        set_seed(cfg["seeds"]["edit"])
        edited, steps = model(x, source_embed, target_embed)
        decoded = pipe.vae_adapter.decode(edited).detach().cpu().clamp(0,1)
        return FocusEditOutput(pipe._tensor_to_pil(decoded)[0], mask_image, edited.detach().cpu(), steps)
