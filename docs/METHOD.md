# FocusEdit method and implementation

## Localization

`focusedit.attention.extract_edit_masks` captures image-query/text-key attention in SD3 MM-DiT blocks 8–13. Prompt changes are located using CLIP-G token positions; deletions use source-side attention. The token difference is positional and can include shifted positions after a word insertion.

Attention scores are normalized over text keys. Consequently, averaging over **all** text tokens is constant in exact arithmetic. The legacy positive difference of those means is retained for numerical parity, but the edited-token attention supplies the effective spatial signal. This distinction matters when interpreting the localization mechanism.

The saliency is averaged over heads and layers, min/max normalized, smoothed by a 5×5 average pool, thresholded at mean + 0.5 standard deviation, and dilated by a 3×3 max pool. At 512 pixels, predicted supports are 32×32 and are lifted to the latent grid with nearest interpolation.

## Guidance

`focusedit.guidance.FocusEditModel` uses a spatial weight `M` in

```
guided_velocity = source_velocity + cfg_scale * M * (target_velocity - source_velocity)
flow_update = guided_velocity - anchor_velocity
```

The public pipeline constructs `M = alpha + (1-alpha)*H`, with binary support `H` and default `alpha=0.2`. It passes the final weight to the model with the model's internal floor set to zero; the floor is applied once. The backbone retains its regularizer and adaptive schedule.

Zero direct target/source guidance does not force zero total update: source and anchor are queried at different states after editing starts, and the transformer couples spatial regions. The implementation does not hard-replace background pixels or latents with the source.

## Continuous and oracle controls

The continuous support uses median subtraction, clipping to [0,1] and a square-root remapping. The controlled runner uses the pre-floor unit map from `continuous_unit` and applies the same requested floor as the binary method. The extractor's first return value preserves the historical 0.15-floor map for compatibility; the released pipeline does not use it.

For the oracle control, the ground-truth mask is area-downsampled to the same predicted support grid, thresholded at 0.5 and lifted with nearest interpolation. Ground-truth masks are used only for this explicit oracle and evaluation. The default FocusEdit inference uses no ground-truth region.

## Reproducibility scope

The released inference path preserves the frozen supplementary experiment's tensor arithmetic. Package-relative imports and command-line path parameters replace local orchestration. Predicted attention supports, output images, and mean aggregation are checked against saved records during release validation.
