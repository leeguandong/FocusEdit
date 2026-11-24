# Reproduce the fixed-protocol experiment

The archived experiment uses all 700 PIE-Bench edits, seven methods and two source references. It uses SD3 Medium at 512×512/BF16, scheduler/edit/reference settings 50/50/42, guidance 13.5, one noise probe per step, regularization 0.01 with timestep scaling, equivalent gain and CFL threshold 8. The full configuration is in `configs/focusedit_512.json`; historical metadata is in `results/protocol.json`.

## 1. Prepare inputs

Obtain PIE-Bench from its provider and SD3 Medium from its model provider. The PIE root must contain `mapping_file.json` and `annotation_images/`. The released mapping's SHA-256 is recorded in `results/protocol.json`.

```bash
python -m pip install -e '.[eval]'
git clone https://github.com/cure-lab/PnPInversion.git third_party/PnPInversion
git -C third_party/PnPInversion checkout 07f97f448150e2ca220bebd54c8f687c5c50c67a
```

The evaluator checks the metric-file SHA-256 before calculating scores. It downloads the CLIP, DINO and SqueezeNet assets through their normal public loaders; no private mirror is needed. `--clip-model` can point to a local CLIP model directory.

## 2. Generate

```bash
python -m reproduce.generate \
  --pie-root datasets/pie_bench \
  --output outputs/controlled --shard 0/1
```

Set `--model weights/sd3-medium` to use local weights. For three GPUs, launch shards `0/3`, `1/3`, `2/3` in separate processes with each process's visible device configured by your environment. Each shard sorts the full sample list, then takes every third ID. All seven methods share the source latent and text embeddings. Encoding, localization and editing use separate seeds 123, 124 and 123; editing noise is reset per method. Adaptive time coordinates can differ because updates differ.

`--max-samples 1 --smoke` is a diagnostic run; use a separate output directory. The mean-aggregation command intentionally rejects an incomplete 700-edit run. Resuming a shard with different IDs or settings is rejected.

The runner stores source references, seven edited PNGs, predicted/true masks, tensors, configuration records and timing metadata. Ten sample IDs chosen by category before generation also receive trajectory diagnostics. Timing was collected on shared GPUs and should be treated descriptively.

## 3. Evaluate

```bash
python -m reproduce.evaluate \
  --output outputs/controlled --shard 0 \
  --pnp-root third_party/PnPInversion
```

For a three-shard run, evaluate shards 0, 1 and 2. The evaluator reports structure distance, background PSNR/LPIPS/SSIM, CLIP-whole and CLIP-edit. It uses the same output under both VAE-reconstructed and original RGB references.

PIE RLE masks follow the PnP border convention. Background metrics zero the edited pixels in both inputs and evaluate the resulting full-frame images. Empty-background metrics remain undefined and are excluded. Thus background means use **556**, while structure/semantic means use **700**. Background LPIPS uses SqueezeNet, structure distance uses DINO ViT-B/8 and text similarity uses CLIP ViT-L/14 with slow/PIL preprocessing. A Transformers 5 adapter obtains the same projected CLIP features.

## 4. Aggregate

```bash
python -m reproduce.summarize --output outputs/controlled
python results/recompute_means.py
```

The first command summarizes your complete generated/evaluated run with paired bootstrap intervals (10,000 replicates, seed 20261124). The second independently verifies the **archived** CSV using only Python's standard library. The intervals are pointwise descriptive intervals and do not establish robustness across editing seeds.

## Version and historical-run scope

The supplementary generation environment used torch 2.8.0+cu128, torchvision 0.23.0+cu128, diffusers 0.35.2, transformers 5.2.0, accelerate 1.11.0, numpy 2.3.2, Pillow 12.2.0 and torchmetrics 1.8.2. Use matching versions for the closest numerical replication; CUDA kernels and hardware can affect bitwise equality.

This release exposes the supplementary seven-method pipeline. The original submission's six-shard run and the separately implemented gated Navi-FlowEdit comparison are distinct experiments; this package does not claim to reproduce their full tables. The released underlying Navi-FlowAlign implementation is pinned to the upstream commit listed in the notices. Its public default resolution is 1024/FP16, whereas the paired FocusEdit experiment is explicitly 512/BF16. This release does not claim to reproduce the upstream paper's 34.49 LPIPS score.
