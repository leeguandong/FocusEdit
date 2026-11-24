# Third-party acknowledgments

FocusEdit's contribution is attention-derived spatial localization and modulation of the target/source velocity difference. Its editing backbone builds on the public Navi-FlowAlign implementation from **NaviEdit** by Liangsi Lu and collaborators.

The four files in `focusedit/_backbone/` are adapted from NaviEdit commit `6728771938d3b6b9dc5b8a7b5c371b3c546a5b40`. Changes are limited to package-relative imports. The upstream MIT copyright and permission notice are preserved in `LICENSE`.

- NaviEdit: https://github.com/NaviEdit/NaviEdit
- Stable Diffusion 3 Medium: https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
- PIE-Bench and PnP evaluation: https://github.com/cure-lab/PnPInversion

Model weights, PIE-Bench images and the PnP implementation are obtained separately from their respective providers. Their licenses remain with those providers.
