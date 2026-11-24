"""Configuration used by the released FocusEdit pipeline."""
from copy import deepcopy
from ._backbone.pipeline_factory import DEFAULT_EDIT_CONFIG

DEFAULT_CONFIG = {'model': 'stabilityai/stable-diffusion-3-medium-diffusers', 'image_size': 512, 'precision': 'bf16', 'max_sequence_length': 77, 'seeds': {'encode': 123, 'probe': 124, 'edit': 123}, 'attention_layers': [8, 9, 10, 11, 12, 13], 'mask_type': 'binary', 'background_guidance_floor': 0.2, 'edit': {'num_inference_steps': 50, 'noise_samples': 1, 'cfg_scale': 13.5, 'regularization_scale': 0.01, 'regularization_dt_scale': True, 't_ref': 42, 't_edit': 50, 'edit_dt': 1.0, 'use_equiv_gain': True, 'use_cfl': True, 'cfl_tau': 8.0}}

def default_config():
    return deepcopy(DEFAULT_CONFIG)
