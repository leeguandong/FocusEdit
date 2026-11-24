from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class NaviFlowAlignModel(nn.Module):
    def __init__(self, transformer, scheduler, model_cfg: dict):
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

        self.num_inference_steps = int(model_cfg.get("num_inference_steps", model_cfg.get("n_steps", 33)))
        self.cfg_scale = float(model_cfg.get("cfg_scale", 13.5))
        self.regularization_scale = float(model_cfg.get("regularization_scale", 0.01))
        self.regularization_dt_scale = bool(model_cfg.get("regularization_dt_scale", False))
        self.noise_samples = int(max(1, model_cfg.get("noise_samples", 1)))

        self.t_ref = int(max(1, model_cfg.get("t_ref", self.num_inference_steps)))
        self.t_edit = int(max(1, model_cfg.get("t_edit", self.num_inference_steps)))
        self.edit_dt = float(model_cfg.get("edit_dt", 1.0))
        self.use_equiv_gain = bool(model_cfg.get("use_equiv_gain", True))

        self.use_cfl = bool(model_cfg.get("use_cfl", True))
        self.cfl_tau = float(model_cfg.get("cfl_tau", 8.0))

        self._asn_ema_beta = 0.90
        self._asn_tail_frac = 0.20
        self._asn_eps = 1e-8
        self._asn_p_clamp_eps = 1e-6

    def _retrieve_timesteps(self, device: torch.device) -> torch.Tensor:
        if hasattr(self.scheduler, "set_timesteps"):
            self.scheduler.set_timesteps(self.num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
        else:
            timesteps = getattr(self.scheduler, "timesteps", [])
        if torch.is_tensor(timesteps):
            return timesteps.to(device=device, dtype=torch.float32)
        return torch.tensor(list(timesteps), device=device, dtype=torch.float32)

    @staticmethod
    def _normalized_timestep(t, timestep_scale: float, device: torch.device) -> torch.Tensor:
        if torch.is_tensor(t):
            t_tensor = t.to(device=device, dtype=torch.float32)
        else:
            t_tensor = torch.tensor(t, device=device, dtype=torch.float32)
        if timestep_scale <= 1.0:
            return t_tensor
        return t_tensor / float(timestep_scale)

    def _prepare_timesteps(self, device: torch.device) -> tuple[torch.Tensor, float]:
        timesteps_list = self._retrieve_timesteps(device=device)
        if timesteps_list.numel() == 0:
            raise ValueError("Scheduler timesteps are empty.")

        max_timestep = float(timesteps_list.max().item())
        scheduler_config = getattr(self.scheduler, "config", None)
        timestep_scale = float(getattr(scheduler_config, "num_train_timesteps", max_timestep))
        if max_timestep <= 1.0:
            timestep_scale = 1.0
        return timesteps_list, timestep_scale

    def _build_monotone_path(
        self,
        timesteps_list: torch.Tensor,
        timestep_scale: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_list = timesteps_list.to(device=device, dtype=torch.float32)
        u_list = self._normalized_timestep(t_list, timestep_scale, device=device)
        order = torch.argsort(u_list, descending=True)
        return t_list[order], u_list[order]

    @staticmethod
    def _interp_from_p(
        t_path: torch.Tensor,
        u_path: torch.Tensor,
        p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_points = int(t_path.numel())
        if num_points <= 1:
            return t_path[0], u_path[0]

        p = p.clamp(0.0, 1.0)
        idx_f = p * float(num_points - 1)
        idx0 = torch.floor(idx_f).to(torch.long)
        idx1 = (idx0 + 1).clamp(max=num_points - 1)
        frac = idx_f - idx0.to(torch.float32)

        t0 = t_path[idx0]
        t1 = t_path[idx1]
        u0 = u_path[idx0]
        u1 = u_path[idx1]

        t_val = t0 * (1.0 - frac) + t1 * frac
        u_val = u0 * (1.0 - frac) + u1 * frac
        return t_val, u_val

    def _predict_vector(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        embeds: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        prompt_embeds, pooled_prompt_embeds = embeds
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=latents.device, dtype=torch.float32)
        else:
            timestep = timestep.to(device=latents.device, dtype=torch.float32)
        if timestep.dim() == 0:
            timestep = timestep.reshape(1)
        timestep = timestep.expand(latents.shape[0])
        return self.transformer(
            hidden_states=latents,
            timestep=timestep,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

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

        for _ in range(self.noise_samples):
            noise = torch.randn_like(x_src)
            qt = (1.0 - u_norm) * x_src + u_norm * noise
            pt = x_edit + qt - x_src

            target_velocity = self._predict_vector(pt, t_model, target_embed)
            source_velocity = self._predict_vector(pt, t_model, source_embed)
            guided_velocity = source_velocity + cfg_scale_tensor * (target_velocity - source_velocity)
            anchor_velocity = self._predict_vector(qt, t_model, source_embed)

            flow_update = flow_update + (guided_velocity - anchor_velocity)
            reg_update = reg_update + (qt - u_norm * anchor_velocity - pt + u_norm * guided_velocity)

        scale = 1.0 / float(self.noise_samples)
        return flow_update * scale, reg_update * scale

    def _asn_pick_p_next(
        self,
        p: float,
        i: int,
        t_edit: int,
        update_direction: torch.Tensor,
        prev_update_direction: Optional[torch.Tensor],
        mag_ema: Optional[float],
    ) -> tuple[float, float]:
        eps = self._asn_eps
        steps_left = max(1, t_edit - i)
        remaining = max(0.0, 1.0 - p)
        dp_min = remaining / float(steps_left)

        mag = update_direction.float().pow(2).mean(dim=1).mean(dim=(1, 2)).sqrt().mean().item()
        if mag_ema is None:
            mag_ema_new = mag
        else:
            mag_ema_new = self._asn_ema_beta * mag_ema + (1.0 - self._asn_ema_beta) * mag
        mag_norm = mag / (mag_ema_new + eps)

        osc = 0.0
        if prev_update_direction is not None:
            current = update_direction.float().flatten(1)
            previous = prev_update_direction.float().flatten(1)
            current = current / (current.norm(dim=1, keepdim=True) + eps)
            previous = previous / (previous.norm(dim=1, keepdim=True) + eps)
            cos = (current * previous).sum(dim=1).mean().clamp(-1.0, 1.0).item()
            osc = float(1.0 - cos)

        risk = float(mag_norm * (1.0 + osc))
        speed = float(torch.sigmoid(torch.tensor(-math.log(risk + eps))).item())

        tail_steps = max(2, int(round(self._asn_tail_frac * t_edit)))
        if i >= (t_edit - tail_steps):
            speed = 0.0

        mult = 0.5 + speed
        dp = dp_min * mult
        if i < (t_edit - 1):
            p_next = min(1.0 - self._asn_p_clamp_eps, p + dp)
        else:
            p_next = 1.0
        p_next = max(p, min(1.0, p_next))
        return p_next, mag_ema_new

    @torch.no_grad()
    def forward(
        self,
        x_src: torch.Tensor,
        source_embed: tuple[torch.Tensor, torch.Tensor],
        target_embed: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, list[float]]:
        device = x_src.device
        timesteps_list, timestep_scale = self._prepare_timesteps(device=device)
        t_path, u_path = self._build_monotone_path(timesteps_list, timestep_scale, device=device)

        num_points = int(t_path.numel())
        if num_points <= 0:
            raise ValueError("Scheduler timesteps are empty.")

        if num_points <= 1:
            p = 0.0
        else:
            start_idx = max(0, num_points - min(self.t_ref, num_points))
            p = float(start_idx) / float(num_points - 1)

        x_edit = x_src.clone()
        ds = float(self.edit_dt)
        mag_ema = None
        prev_flow_update: Optional[torch.Tensor] = None
        step_norms: list[float] = []

        for i in range(self.t_edit):
            p_t = torch.tensor(p, device=device, dtype=torch.float32)
            t_model, u_norm = self._interp_from_p(t_path, u_path, p_t)
            flow_update, reg_update = self._estimate_updates(
                x_src=x_src,
                x_edit=x_edit,
                t_model=t_model,
                u_norm=u_norm,
                source_embed=source_embed,
                target_embed=target_embed,
            )

            p_next, mag_ema = self._asn_pick_p_next(
                p=p,
                i=i,
                t_edit=self.t_edit,
                update_direction=flow_update,
                prev_update_direction=prev_flow_update,
                mag_ema=mag_ema,
            )
            _, u_norm_next = self._interp_from_p(
                t_path,
                u_path,
                torch.tensor(p_next, device=device, dtype=torch.float32),
            )

            dt_u = u_norm_next - u_norm
            if self.use_equiv_gain:
                gain = dt_u / ds
            else:
                gain = torch.ones_like(dt_u)

            flow_step = (ds * gain) * flow_update
            if self.regularization_dt_scale:
                reg_scale = x_src.new_tensor(self.regularization_scale) * dt_u.abs()
            else:
                reg_scale = x_src.new_tensor(self.regularization_scale)
            reg_step = reg_scale * reg_update
            total_step = flow_step + reg_step

            step_scale = 1.0
            if self.use_cfl:
                total_mag = total_step.float().pow(2).mean(dim=1).sum(dim=(1, 2)).sqrt().mean().item()
                if total_mag > 1e-8:
                    step_scale = min(1.0, self.cfl_tau / (total_mag + 1e-6))

            x_edit = x_edit.to(torch.float32)
            x_edit = x_edit + step_scale * total_step.to(torch.float32)
            x_edit = x_edit.to(dtype=x_src.dtype)

            step_norms.append(float(total_step.float().pow(2).mean().sqrt().item()))
            prev_flow_update = flow_update.detach()
            p = float(p_next)

        return x_edit, step_norms
