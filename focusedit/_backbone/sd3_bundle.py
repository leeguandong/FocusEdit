from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class StableDiffusion3Bundle:
    pipe: object
    transformer: object
    scheduler: object
    vae: object
    vae_adapter: object
    text_encode_fn: object


class StableDiffusion3VAEAdapter:
    def __init__(self, vae) -> None:
        self.vae = vae

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")

        first_param = next(self.vae.parameters())
        device = first_param.device
        dtype = first_param.dtype
        images = images.to(device=device, dtype=dtype)
        if images.min().item() >= 0.0 and images.max().item() <= 1.0:
            images = images * 2.0 - 1.0

        use_autocast = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
            latents = self.vae.encode(images).latent_dist.sample()

        scaling_factor = latents.new_tensor(float(getattr(self.vae.config, "scaling_factor", 1.0)))
        shift_factor = latents.new_tensor(float(getattr(self.vae.config, "shift_factor", 0.0)))
        latents = (latents - shift_factor) * scaling_factor
        return latents.to(device=device, dtype=dtype)

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 4:
            raise ValueError("latents must have shape [B, C, H, W]")

        first_param = next(self.vae.parameters())
        device = first_param.device
        dtype = first_param.dtype
        latents = latents.to(device=device, dtype=dtype)

        scaling_factor = latents.new_tensor(float(getattr(self.vae.config, "scaling_factor", 1.0)))
        shift_factor = latents.new_tensor(float(getattr(self.vae.config, "shift_factor", 0.0)))
        latents = latents / scaling_factor + shift_factor

        use_autocast = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
            decoded = self.vae.decode(latents, return_dict=False)[0]
        decoded = decoded.clamp(-1.0, 1.0)
        return (decoded + 1.0) / 2.0


def _resolve_dtype(precision: str) -> torch.dtype:
    normalized = precision.lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported precision: {precision}")
    return mapping[normalized]


def _prompt_list(prompts) -> list[str]:
    if isinstance(prompts, str):
        return [prompts]
    return [str(prompt) for prompt in prompts]


def _module_device(module) -> torch.device:
    return next(module.parameters()).device


def load_sd3_bundle(
    path: str,
    device: torch.device,
    precision: str = "fp16",
    max_sequence_length: int = 77,
    efficient_memory: bool = False,
) -> StableDiffusion3Bundle:
    from diffusers import StableDiffusion3Pipeline

    source = str(path)
    source_path = Path(source).expanduser()
    local_files_only = source_path.exists()
    if local_files_only:
        source = str(source_path.resolve())

    target_device = torch.device(device)
    dtype = _resolve_dtype(precision)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        source,
        local_files_only=local_files_only,
        torch_dtype=dtype,
    )
    pipe.set_progress_bar_config(disable=True)

    transformer = pipe.transformer.to(device=target_device, dtype=dtype).eval()
    transformer.requires_grad_(False)
    vae = pipe.vae.to(device=target_device, dtype=dtype).eval()
    vae.requires_grad_(False)
    vae_adapter = StableDiffusion3VAEAdapter(vae)

    text_device = torch.device("cpu") if efficient_memory else target_device
    text_dtype = torch.float32 if text_device.type == "cpu" else dtype

    for encoder in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        if encoder is None:
            continue
        encoder.to(device=text_device, dtype=text_dtype)
        encoder.eval()
        encoder.requires_grad_(False)

    max_length = int(max_sequence_length)

    def text_encode_fn(prompts):
        prompt_list = _prompt_list(prompts)
        target_dtype = next(transformer.parameters()).dtype
        target_device_inner = next(transformer.parameters()).device

        with torch.inference_mode():
            text_clip1_ids = pipe.tokenizer(
                prompt_list,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            text_clip1_out = pipe.text_encoder(
                text_clip1_ids.to(_module_device(pipe.text_encoder)),
                output_hidden_states=True,
            )
            pool_clip1_emb = text_clip1_out[0].to(device=target_device_inner, dtype=target_dtype)
            text_clip1_emb = text_clip1_out.hidden_states[-2].to(device=target_device_inner, dtype=target_dtype)

            text_clip2_ids = pipe.tokenizer_2(
                prompt_list,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            text_clip2_out = pipe.text_encoder_2(
                text_clip2_ids.to(_module_device(pipe.text_encoder_2)),
                output_hidden_states=True,
            )
            pool_clip2_emb = text_clip2_out[0].to(device=target_device_inner, dtype=target_dtype)
            text_clip2_emb = text_clip2_out.hidden_states[-2].to(device=target_device_inner, dtype=target_dtype)

            text_t5_ids = pipe.tokenizer_3(
                prompt_list,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids
            text_t5_emb = pipe.text_encoder_3(text_t5_ids.to(_module_device(pipe.text_encoder_3)))[0]
            text_t5_emb = text_t5_emb.to(device=target_device_inner, dtype=target_dtype)

        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        feature_delta = text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1]
        if feature_delta > 0:
            clip_prompt_emb = F.pad(clip_prompt_emb, (0, feature_delta))
        elif feature_delta < 0:
            clip_prompt_emb = clip_prompt_emb[..., : text_t5_emb.shape[-1]]

        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)
        return prompt_emb, pooled_prompt_emb

    return StableDiffusion3Bundle(
        pipe=pipe,
        transformer=transformer,
        scheduler=pipe.scheduler,
        vae=vae,
        vae_adapter=vae_adapter,
        text_encode_fn=text_encode_fn,
    )
