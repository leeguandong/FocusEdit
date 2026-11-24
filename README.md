# FocusEdit

**Training-Free Spatially-Modulated Guidance for Background-Preserving Real Image Editing**

[中文说明](README_zh.md) · [Reproduce experiments](docs/REPRODUCIBILITY.md) · [Method](docs/METHOD.md)

FocusEdit edits a source image from a pair of source/target prompts. It extracts an edit region from the pretrained SD3 transformer's edited-token attention and spatially modulates the target/source velocity difference during editing. No additional localization network or model training is required.

The release contains the FocusEdit pipeline, attention localization, spatial guidance, a matched unmodulated baseline, the seven-configuration PIE-Bench experiment, and per-edit metrics that can be independently aggregated.

## How it works

1. **Locate:** capture image-query/text-key attention from SD3 blocks 8–13 at a reference noise scale and select changed prompt tokens.
2. **Construct the guidance map:** normalize and smooth the saliency, threshold and dilate the binary support, then apply a background floor: `M = alpha + (1 - alpha) * H`.
3. **Edit:** use `v_source + guidance_scale * M * (v_target - v_source)` within the flow-based editing process.

The default is a binary mask with `alpha = 0.2`. The underlying adaptive editing schedule and trajectory regularizer follow Navi-FlowAlign; see [acknowledgments](THIRD_PARTY_NOTICES.md). Spatial guidance suppresses direct background edits; source-state residuals and cross-region interactions can still move the background.

## Installation

Use Python 3.11 and a CUDA-capable PyTorch installation. Install the PyTorch/torchvision build appropriate for your GPU, then:

```bash
git clone https://github.com/leeguandong/FocusEdit.git
cd FocusEdit
python -m pip install -e .
```

Accept the SD3 Medium model license with its provider and authenticate using your usual model-hub tooling if required. Credentials are never supplied in a configuration file or command example in this repository.

## Edit an image

```bash
focusedit --image input.png \
  --source-prompt "A photo of a red car" \
  --target-prompt "A photo of a blue car" \
  --output outputs/edited.png
```

The default settings are **SD3 Medium, 512×512, BF16, 50 scheduler steps, 50 editing steps, reference depth 42, guidance scale 13.5**. The output includes the edited image and the predicted support mask. Non-square images are center-cropped. Set `--model` to a local model directory to use downloaded weights.

```python
from PIL import Image
from focusedit import FocusEditPipeline

editor = FocusEditPipeline.from_pretrained()
result = editor(Image.open("input.png"), "A red car", "A blue car")
result.image.save("edited.png")
```

Pass `--config configs/focusedit_512.json` to make every setting explicit. Pass `--baseline` to run the unmodulated Navi-FlowAlign backbone with the same resolution and random-number protocol. Experimental settings beyond the released 512 protocol should be identified separately when reporting results.

## Verified supplementary results

PIE-Bench: **700 edits**, including **556 edits with a nonempty background**. All methods below use the same 512/BF16 protocol. Background LPIPS is multiplied by 1,000; lower is better. Semantic scores use all 700 edits.

| Method | BG LPIPS, VAE reference ↓ | BG LPIPS, RGB reference ↓ | CLIP-edit ↑ |
|---|---:|---:|---:|
| Unmodulated Navi-FlowAlign | 125.353 | 128.903 | 23.086 |
| **FocusEdit, binary, alpha=0.2** | **53.365** | **59.544** | **22.823** |
| FocusEdit, binary, alpha=0 | 46.070 | 52.746 | 22.773 |
| FocusEdit, continuous, alpha=0.2 | 39.988 | 46.676 | 22.532 |
| Ground-truth support, alpha=0.2 | 17.519 | 25.259 | 23.263 |

The default reduces background LPIPS by **57.4%** under the VAE reference and **53.8%** under the RGB reference. The paired mean CLIP-edit difference is −0.264 (95% bootstrap interval [−0.456, −0.078]). Full results retain all seven configurations and both references in `results/`, including configurations omitted from this compact table.

These are fixed-protocol supplementary results. The originally submitted table reported 125.2 / 53.2 / 7.2 for the baseline / FocusEdit / oracle in a separate historical run. The original oracle and the new matched-floor oracle have different guidance-floor settings. The released supplementary records must not be relabeled as that historical run or as a reproduction of NaviEdit's published 34.49 score.

## Verify without a GPU

```bash
python results/recompute_means.py
```

This standard-library script reads 9,800 per-edit/configuration/reference rows and verifies all **84 means and valid counts**. See [reproduction instructions](docs/REPRODUCIBILITY.md) for generation, evaluation and paired confidence intervals.

## Repository layout

```text
focusedit/
  pipeline.py       FocusEdit public inference interface
  attention.py      Edited-token attention and support construction
  guidance.py       Spatial modulation of the velocity difference
  config.py         Explicit FocusEdit defaults
  _backbone/        Attributed Navi-FlowAlign and SD3 adapters
configs/            Experiment settings
reproduce/          Matched generation, evaluation and aggregation
results/            Verified per-edit records and independent replay
```

## Attribution and citation

FocusEdit builds on NaviEdit/Navi-FlowAlign and SD3; evaluation follows PIE-Bench/PnP. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source versions and acknowledgments. FocusEdit-specific code and documentation are presented separately from the attributed backbone. See [CITATION.cff](CITATION.cff) for manuscript metadata.
