"""FocusEdit spatial guidance.

Only the target/source velocity difference is spatially weighted. The source
residual, trajectory regularizer and adaptive schedule are inherited from the
base editor. A zero guidance weight does not guarantee a frozen background.
"""
from __future__ import annotations

import torch

from ._backbone.model import NaviFlowAlignModel


class FocusEditModel(NaviFlowAlignModel):
    def __init__(self, transformer, scheduler, model_cfg: dict):
        super().__init__(transformer, scheduler, model_cfg)
        # Residual guidance kept in the background (0 = fully off → strongest
        # preservation; small positive keeps a little global guidance).
        self._bg_guidance_floor = float(model_cfg.get("bg_guidance_floor", 0.0))
        self._edit_mask: torch.Tensor | None = None  # [B,1,H,W] in latent res

    def set_edit_mask(self, mask: torch.Tensor | None) -> None:
        self._edit_mask = mask

    def _estimate_updates(
        self,
        x_src: torch.Tensor,
        x_edit: torch.Tensor,
        t_model: torch.Tensor,
        u_norm: torch.Tensor,
        source_embed: tuple[torch.Tensor, torch.Tensor],
        target_embed: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flow_update = torch.zeros_like(x_src)
        reg_update = torch.zeros_like(x_src)
        cfg_scale_tensor = x_src.new_tensor(self.cfg_scale)

        gw = None
        if self._edit_mask is not None:
            M = self._edit_mask.to(device=x_src.device, dtype=x_src.dtype)
            # Per-position effective guidance weight: floor + (1-floor) * M.
            gw = self._bg_guidance_floor + (1.0 - self._bg_guidance_floor) * M

        for _ in range(self.noise_samples):
            noise = torch.randn_like(x_src)
            qt = (1.0 - u_norm) * x_src + u_norm * noise
            pt = x_edit + qt - x_src

            target_velocity = self._predict_vector(pt, t_model, target_embed)
            source_velocity = self._predict_vector(pt, t_model, source_embed)
            delta = target_velocity - source_velocity
            if gw is not None:
                guided_velocity = source_velocity + cfg_scale_tensor * gw * delta
            else:
                guided_velocity = source_velocity + cfg_scale_tensor * delta
            anchor_velocity = self._predict_vector(qt, t_model, source_embed)

            flow_update = flow_update + (guided_velocity - anchor_velocity)
            reg_update = reg_update + (qt - u_norm * anchor_velocity - pt + u_norm * guided_velocity)

        scale = 1.0 / float(self.noise_samples)
        return flow_update * scale, reg_update * scale
