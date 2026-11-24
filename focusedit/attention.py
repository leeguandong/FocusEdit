"""FocusEdit localization from edited-token attention in SD3 blocks 8--13.

The text-only softmax makes the all-token mean spatially constant in exact
arithmetic. The edited-token attention is the effective localization signal.
The small legacy mean-difference term is retained for numerical reproducibility.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch
import torch.nn.functional as F

from ._backbone.model import NaviFlowAlignModel


class AttentionCapture:
    """Drop-in replacement for JointAttnProcessor2_0 that captures
    image-query → text-key attention maps (cross-attention sub-block)."""

    def __init__(self):
        self.attn_maps: list[torch.Tensor] = []
        self._capturing = False

    def start(self):
        self.attn_maps.clear()
        self._capturing = True

    def stop(self):
        self._capturing = False

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, *args, **kwargs):
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
            eq = eq.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ek = ek.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ev = ev.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_added_q is not None:
                eq = attn.norm_added_q(eq)
            if attn.norm_added_k is not None:
                ek = attn.norm_added_k(ek)

            n_img = query.shape[2]
            n_txt = ek.shape[2]
            query_full = torch.cat([query, eq], dim=2)
            key_full = torch.cat([key, ek], dim=2)
            value_full = torch.cat([value, ev], dim=2)

            if self._capturing:
                # Manual cross-attention: image_query × text_key
                # [B, H, n_img, d] × [B, H, n_txt, d]^T → [B, H, n_img, n_txt]
                scale = head_dim ** -0.5
                cross_scores = torch.matmul(
                    query.float(), ek.float().transpose(-2, -1)
                ) * scale
                cross_attn = cross_scores.softmax(dim=-1)  # [B, H, n_img, n_txt]
                self.attn_maps.append(cross_attn.detach().cpu())

            hidden_states = F.scaled_dot_product_attention(
                query_full, key_full, value_full, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
            hidden_states = hidden_states.to(query.dtype)
            hidden_states, encoder_hidden_states = (
                hidden_states[:, :residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)
            hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states


def _get_block_attn_key(block_idx: int) -> str:
    return f"transformer_blocks.{block_idx}.attn"


def _find_edited_token_indices(
    pipe, source_prompt: str, target_prompt: str
) -> tuple[list[int], bool]:
    """Find which token positions differ between source and target.
    Returns (indices, is_delete). For deletions (target shorter), returns the
    SOURCE positions of deleted words so attention can locate them.
    Uses CLIP-G tokenizer (tokenizer_2)."""
    tok = pipe.bundle.pipe.tokenizer_2
    src_ids = tok(source_prompt, padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").input_ids[0]
    tgt_ids = tok(target_prompt, padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").input_ids[0]
    src_len = (src_ids != tok.pad_token_id).sum().item()
    tgt_len = (tgt_ids != tok.pad_token_id).sum().item()

    diff = (src_ids != tgt_ids).nonzero(as_tuple=True)[0].tolist()
    # Added tokens in target.
    for i in range(src_len, tgt_len):
        if i not in diff:
            diff.append(i)

    is_delete = tgt_len < src_len
    if is_delete:
        # For deletions, use SOURCE positions of deleted tokens.
        del_positions = list(range(tgt_len, src_len))
        for i in diff:
            if i not in del_positions:
                del_positions.append(i)
        return sorted(del_positions), True

    return sorted(diff) if diff else list(range(min(5, 77))), False


@contextmanager
def _capture_attention(transformer, layers: Sequence[int]):
    """Temporarily replace attention processors in specified layers."""
    captures = {}
    original_processors = {}
    for idx in layers:
        key = _get_block_attn_key(idx)
        block = transformer.transformer_blocks[idx]
        original_processors[idx] = block.attn.processor
        cap = AttentionCapture()
        captures[idx] = cap
        block.attn.set_processor(cap)
        cap.start()
    try:
        yield captures
    finally:
        for idx in layers:
            captures[idx].stop()
            transformer.transformer_blocks[idx].attn.set_processor(original_processors[idx])


@torch.no_grad()
def extract_edit_masks(
    pipe,
    x_src: torch.Tensor,
    source_prompt: str,
    target_prompt: str,
    *,
    cfg: dict,
    layers: Sequence[int] = (8, 9, 10, 11, 12, 13),
    spatial_size: int | None = None,
    smooth_kernel: int = 5,
) -> torch.Tensor:
    """Run one FlowAlign step with attention capture, return edit mask [1,1,H,W]."""
    device = x_src.device
    model = NaviFlowAlignModel(pipe.transformer, pipe.scheduler, cfg).to(device).eval()
    timesteps_list, timestep_scale = model._prepare_timesteps(device)
    t_path, u_path = model._build_monotone_path(timesteps_list, timestep_scale, device)
    num_points = int(t_path.numel())
    start_idx = max(0, num_points - min(model.t_ref, num_points))
    p = float(start_idx) / float(num_points - 1) if num_points > 1 else 0.0
    p_t = torch.tensor(p, device=device, dtype=torch.float32)
    t_model, u_norm = model._interp_from_p(t_path, u_path, p_t)

    source_embed = pipe.encode_prompt([source_prompt])
    target_embed = pipe.encode_prompt([target_prompt])

    edited_tokens, is_delete = _find_edited_token_indices(pipe, source_prompt, target_prompt)

    # _estimate_updates runs 3 forwards per noise_sample:
    #   maps[0] = target_velocity(pt, target_embed)
    #   maps[1] = source_velocity(pt, source_embed)
    #   maps[2] = anchor_velocity(qt, source_embed)
    # Capture edited-token attention. The legacy all-token mean difference
    # below is numerically near zero under the text-only softmax.
    with _capture_attention(pipe.transformer, layers) as captures:
        model._estimate_updates(x_src, x_src.clone(), t_model, u_norm,
                                source_embed, target_embed)

    all_maps = []
    term_diagnostics = []
    for idx in layers:
        maps = captures[idx].attn_maps
        if len(maps) < 2:
            continue
        # maps[0] = target-conditioned, maps[1] = source-conditioned
        tgt_attn = maps[0]  # [B, H, n_img, n_txt]
        src_attn = maps[1]  # [B, H, n_img, n_txt]
        diff_term = (tgt_attn.mean(dim=3)-src_attn.mean(dim=3)).clamp(min=0)
        direct_term = (src_attn if is_delete else tgt_attn)[:, :, :, edited_tokens].mean(dim=3)
        term_diagnostics.append({"layer": int(idx), "diff_abs_max": float(diff_term.abs().max()),
            "diff_l2": float(diff_term.norm()), "direct_l2": float(direct_term.norm()),
            "text_tokens": int(tgt_attn.shape[-1])})

        if is_delete:
            # For deletions: high source attention at deleted tokens = object location.
            m = src_attn[:, :, :, edited_tokens].mean(dim=3) if edited_tokens else src_attn.mean(dim=3)
        else:
            # Preserve the original arithmetic; edited-token attention supplies
            # the meaningful spatial variation in this implementation.
            tgt_avg = tgt_attn.mean(dim=3)  # [B, H, n_img]
            src_avg = src_attn.mean(dim=3)  # [B, H, n_img]
            m = (tgt_avg - src_avg).clamp(min=0)  # only where target attends more
            # Also add: target attention at specifically edited tokens (direct signal).
            if edited_tokens:
                direct = tgt_attn[:, :, :, edited_tokens].mean(dim=3)  # [B, H, n_img]
                m = m + direct  # combine diff + direct signals

        m = m.mean(dim=1)  # average over heads → [B, n_img]
        all_maps.append(m)

    if not all_maps:
        raise RuntimeError("No attention maps captured; check SD3 attention processor compatibility.")

    agg = torch.stack(all_maps, dim=0).mean(dim=0)  # [B, n_img]
    n_img = agg.shape[-1]
    s = int(n_img ** 0.5)
    if spatial_size is None:
        spatial_size = s
    mask = agg.view(1, 1, s, s).float()
    if s != spatial_size:
        mask = F.interpolate(mask, size=(spatial_size, spatial_size), mode="bilinear", align_corners=False)
    mask = mask.to(device)

    # Normalize to [0,1].
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    # Spatial smoothing.
    if smooth_kernel > 1:
        mask = F.avg_pool2d(mask, kernel_size=smooth_kernel, stride=1,
                            padding=smooth_kernel // 2)

    # Continuous control: median shift and square-root remapping.
    # Preserve the historical floored return value; the controlled pipeline
    # applies its requested floor to continuous_unit exactly once.
    p50 = mask.median()
    mask_shifted = ((mask - p50) / (mask.max() - p50 + 1e-8)).clamp(0, 1)
    # Gamma curve: sharpen separation, keep a floor for residual guidance.
    floor = 0.15
    continuous = floor + (1.0 - floor) * mask_shifted.pow(0.5)  # sqrt = expand high end

    thr = mask.mean() + 0.5 * mask.std()
    binary = (mask > thr).float()
    k = 3
    binary = F.max_pool2d(binary, kernel_size=k, stride=1, padding=k // 2)

    return continuous, binary, {"saliency": mask, "continuous_unit": mask_shifted.pow(0.5),
        "edited_tokens": edited_tokens, "is_delete": is_delete, "terms": term_diagnostics}
