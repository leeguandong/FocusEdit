from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .model import NaviFlowAlignModel


@dataclass
class NaviEditPipelineOutput:
    images: List[Image.Image] | torch.Tensor
    latents: torch.Tensor
    step_norms: List[float]


class _CenterSquareCropTransform:
    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width == height:
            return image
        target = min(width, height)
        try:
            resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            resample = Image.LANCZOS
        return ImageOps.fit(
            image,
            (target, target),
            method=resample,
            centering=(0.5, 0.5),
        )


class NaviEditPipeline:
    def __init__(
        self,
        bundle: Any,
        *,
        default_edit_config: Optional[Dict[str, Any]] = None,
        image_size: int = 1024,
        device: Optional[str | torch.device] = None,
        use_center_crop: bool = True,
    ) -> None:
        self.bundle = bundle
        self.transformer = bundle.transformer
        self.scheduler = bundle.scheduler
        self.vae_adapter = bundle.vae_adapter
        self.text_encode_fn = bundle.text_encode_fn
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.default_edit_config = default_edit_config or {}
        self.image_size = int(image_size)
        self._use_center_crop = bool(use_center_crop)
        self._image_transform = self._build_image_transform()

    def _build_image_transform(self):
        transform_steps: list[Any] = []
        if self._use_center_crop:
            transform_steps.append(_CenterSquareCropTransform())
        transform_steps.extend(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        return transforms.Compose(transform_steps)

    @staticmethod
    def _module_dtype(module) -> torch.dtype:
        return next(module.parameters()).dtype

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image | torch.Tensor,
        *,
        source_prompt: str,
        target_prompt: str,
        edit_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        output_type: str = "pil",
    ) -> NaviEditPipelineOutput:
        cfg = dict(self.default_edit_config)
        if edit_config:
            cfg.update(edit_config)

        seed_value = int(seed) if seed is not None else int(cfg.get("seed", 123))
        self._set_seed(seed_value)

        pixel_values = self._prepare_image_tensor(image)
        source_latents = self.vae_adapter.encode(pixel_values)
        source_embed = self.encode_prompt([source_prompt])
        target_embed = self.encode_prompt([target_prompt])

        model = NaviFlowAlignModel(self.transformer, self.scheduler, cfg).to(self._device)
        model.eval()
        edited_latents, step_norms = model(
            x_src=source_latents,
            source_embed=source_embed,
            target_embed=target_embed,
        )
        decoded = self.vae_adapter.decode(edited_latents).detach().cpu().clamp(0.0, 1.0)
        latents = edited_latents.detach().cpu()
        images = self._tensor_to_pil(decoded) if output_type == "pil" else decoded
        return NaviEditPipelineOutput(images=images, latents=latents, step_norms=step_norms)

    def encode_prompt(self, prompts: Sequence[str]):
        return self.text_encode_fn(list(prompts))

    def _prepare_image_tensor(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, Image.Image):
            tensor = self._image_transform(image)
        elif torch.is_tensor(image):
            tensor = image.float()
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim != 3:
                raise ValueError("Tensor inputs must have shape [C, H, W] or [1, C, H, W].")
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            if tensor.min().item() >= 0.0:
                tensor = tensor * 2.0 - 1.0
        else:
            raise TypeError("image must be a PIL.Image or torch.Tensor.")

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(device=self._device, dtype=self._module_dtype(self.bundle.vae))

    @staticmethod
    def _tensor_to_pil(images: torch.Tensor) -> List[Image.Image]:
        to_pil = transforms.ToPILImage()
        return [to_pil(image.float().clamp(0.0, 1.0)) for image in images]

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
