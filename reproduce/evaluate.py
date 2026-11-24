"""Evaluate saved outputs with the original PnP implementation and BOTH references."""
import argparse
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def patch_clip(mc):
    import transformers
    if int(transformers.__version__.split('.')[0]) < 5:
        return
    model = mc.clip_metric_calculator.model
    def image_features(self, pixel_values, *a, **kw):
        return self.visual_projection(self.vision_model(pixel_values=pixel_values).pooler_output)
    def text_features(self, input_ids, attention_mask=None, *a, **kw):
        return self.text_projection(self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output)
    model.get_image_features = types.MethodType(image_features, model)
    model.get_text_features = types.MethodType(text_features, model)


def scalar(x):
    value = float(np.asarray(x).reshape(-1)[0])
    return value if math.isfinite(value) else None


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--pnp-root', type=Path, required=True)
    ap.add_argument('--clip-model', default='openai/clip-vit-large-patch14')
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = ap.parse_args()
    torch.set_num_threads(4)
    run = args.output
    from .metrics import load_calculator
    MetricsCalculator, evaluator_hash = load_calculator(args.pnp_root, args.clip_model)
    manifest = json.loads((run/f'manifest_{args.shard}.json').read_text())
    mc = MetricsCalculator(args.device)
    patch_clip(mc)
    for i, sid in enumerate(manifest['sample_ids']):
        folder = run/'samples'/sid
        if not (folder/'complete.json').exists():
            raise RuntimeError(f'Missing generated sample {sid}')
        dest = folder/'metrics.json'
        if dest.exists():
            continue
        record = json.loads((folder/'mask.json').read_text())
        payload = torch.load(folder/'tensors.pt', map_location='cpu', weights_only=True)
        mask = payload['gt_pixel'][0, 0].numpy()[:, :, None].repeat(3, axis=2)
        inv = 1-mask
        references = {name: Image.open(folder/f'source_{name}.png').convert('RGB') for name in ['rgb', 'vae']}
        all_metrics = {}
        for method in manifest['methods']:
            output = Image.open(folder/f'{method}.png').convert('RGB')
            semantic = {'clip_whole': scalar(mc.calculate_clip_similarity(output, record['target_prompt'], None)),
                        'clip_edit': scalar(mc.calculate_clip_similarity(output, record['target_prompt'], mask)) if mask.sum() else None}
            measures = {}
            for ref_name, src in references.items():
                values = {'struct': scalar(mc.calculate_structure_distance(src, output, None, None)), **semantic}
                for metric in ['psnr', 'lpips', 'ssim']:
                    values[metric] = scalar(getattr(mc, 'calculate_'+metric)(src, output, inv, inv)) if inv.sum() else None
                measures[ref_name] = values
            all_metrics[method] = measures
        result = {'sid': sid, 'category': record['category'], 'metrics': all_metrics,
                  'evaluator_sha256': evaluator_hash, 'references': ['original_rgb', 'vae_reconstruction'],
                  'background_valid': bool(inv.sum()), 'seeds': manifest['seeds'], 'device': args.device}
        temp = folder/'metrics.json.tmp'
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temp.replace(dest)
        print('METRICS', i+1, len(manifest['sample_ids']), sid, flush=True)
    (run/f'metrics_complete_{args.shard}.json').write_text(json.dumps({'n':len(manifest['sample_ids']), 'verified':True})+'\n')


if __name__ == '__main__':
    main()
