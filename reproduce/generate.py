"""Generate all seven matched FocusEdit variants on PIE-Bench at 512/BF16."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from focusedit._backbone.pipeline_factory import DEFAULT_EDIT_CONFIG, build_navi_pipeline
from focusedit._backbone.model import NaviFlowAlignModel
from focusedit.guidance import FocusEditModel as GuidanceModulatedModel
from focusedit.attention import extract_edit_masks as extract_edit_mask_diagnostics

ROOT = Path('.')
PIE = Path('.')
MODEL = 'stabilityai/stable-diffusion-3-medium-diffusers'
METHODS = ['baseline', 'attn_binary_a0', 'attn_binary_a02',
           'gt_binary_a0', 'gt_binary_a02', 'attn_continuous_a0', 'attn_continuous_a02']
SEEDS = {'encode': 123, 'probe': 124, 'edit': 123}
CFG = dict(DEFAULT_EDIT_CONFIG, num_inference_steps=50, t_ref=42, t_edit=50)


def seed(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def decode_mask(encoded):
    m = np.zeros(512*512, dtype=np.float32)
    for s, n in zip(encoded[::2], encoded[1::2]):
        m[s:min(s+n, len(m))] = 1
    m = m.reshape(512, 512)
    m[[0, -1], :] = 1
    m[:, [0, -1]] = 1
    return m


def overlap(pred, gt):
    pred, gt = pred.bool(), gt.bool()
    tp = int((pred & gt).sum())
    npred, ngt = int(pred.sum()), int(gt.sum())
    return {'iou': tp / max(int((pred | gt).sum()), 1),
            'precision': tp / npred if npred else None,
            'recall': tp / ngt if ngt else None,
            'predicted_fraction': npred / pred.numel(), 'gt_fraction': ngt / gt.numel()}


def masked_rms(x, mask):
    denom = float(mask.sum()) * x.shape[1]
    if denom == 0:
        return None
    return float(((x.float().square()*mask).sum()/denom).sqrt())


class ObservedGuidanceModel(GuidanceModulatedModel):
    """Observe the parent's exact arithmetic; never change its update outputs."""
    def __init__(self, *args, background, **kwargs):
        super().__init__(*args, **kwargs)
        self.background = background
        self.rows = []
        self.vectors = []
        self.previous_state = None

    def _predict_vector(self, *args, **kwargs):
        v = super()._predict_vector(*args, **kwargs)
        self.vectors.append(v)
        return v

    def _estimate_updates(self, x_src, x_edit, t_model, u_norm, source_embed, target_embed):
        self.vectors = []
        flow, reg = super()._estimate_updates(x_src, x_edit, t_model, u_norm, source_embed, target_embed)
        assert self.noise_samples == 1 and len(self.vectors) == 3
        target, source, anchor = self.vectors
        residual = source.float() - anchor.float()
        guidance = self.cfg_scale * self._edit_mask.float() * (target.float()-source.float())
        row = {'step': len(self.rows), 'u': float(u_norm), 'time_coordinate': float(t_model),
               'residual_bg_rms': masked_rms(residual, self.background),
               'direct_guidance_bg_rms': masked_rms(guidance, self.background),
               'flow_bg_rms': masked_rms(flow, self.background),
               'state_drift_bg_rms': masked_rms(x_edit-x_src, self.background)}
        if self.previous_state is not None:
            self.rows[-1]['actual_step_bg_rms'] = masked_rms(x_edit-self.previous_state, self.background)
        self.previous_state = x_edit.clone()
        self.rows.append(row)
        self.vectors = []
        return flow, reg


def prep(pipe, item):
    img = Image.open(PIE/'annotation_images'/item['image_path']).convert('RGB').resize((512, 512))
    src = item['original_prompt'].replace('[', '').replace(']', '')
    tgt = item['editing_prompt'].replace('[', '').replace(']', '')
    seed(SEEDS['encode'])
    x = pipe.vae_adapter.encode(pipe._prepare_image_tensor(img))
    return img, x, src, tgt


def record_mask(pipe, sid, item, out, parity=False):
    folder = out/'samples'/sid
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder/'mask.json'
    if record_path.exists() and not parity:
        return json.loads(record_path.read_text())
    torch.cuda.synchronize()
    t = time.perf_counter()
    img, x, src, tgt = prep(pipe, item)
    torch.cuda.synchronize()
    prep_s = time.perf_counter()-t
    seed(SEEDS['probe'])
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    continuous, binary, details = extract_edit_mask_diagnostics(pipe, x, src, tgt, cfg=CFG)
    torch.cuda.synchronize()
    probe_s = time.perf_counter()-t
    peak = torch.cuda.max_memory_allocated()
    gt = torch.from_numpy(decode_mask(item['mask']))[None, None].to(x.device)
    # Evaluate support at 512 pixels; create the controlled GT binary support
    # at the SAME 32x32 grid as the predicted mask, then lift both identically.
    gt_fraction = F.interpolate(gt, size=binary.shape[-2:], mode='area')
    gt_binary = (gt_fraction >= .5).float()
    pred_px = F.interpolate(binary, size=(512, 512), mode='nearest')
    binary_lat = F.interpolate(binary, size=x.shape[-2:], mode='nearest')
    gt_lat = F.interpolate(gt_binary, size=x.shape[-2:], mode='nearest')
    cont_lat = F.interpolate(details['continuous_unit'], size=x.shape[-2:], mode='bilinear', align_corners=False)
    payload = {'source_latent': x.cpu(), 'binary': binary_lat.cpu(), 'gt_binary': gt_lat.cpu(),
               'continuous': cont_lat.cpu(), 'saliency': details['saliency'].cpu(),
               'gt_pixel': gt.cpu(), 'source_prompt': src, 'target_prompt': tgt, 'seeds': SEEDS}
    tmp = folder/'tensors.tmp'
    torch.save(payload, tmp)
    tmp.replace(folder/'tensors.pt')
    img.save(folder/'source_rgb.png')
    pipe._tensor_to_pil(pipe.vae_adapter.decode(x).detach().cpu().clamp(0, 1))[0].save(folder/'source_vae.png')
    for name, m in [('mask_predicted', pred_px), ('mask_gt', gt)]:
        Image.fromarray((m[0, 0].cpu().numpy()*255).astype(np.uint8)).save(folder/f'{name}.png')
    record = {'sid': sid, 'category': item['image_path'].split('/')[0].split('_')[0],
              'image_path': item['image_path'], 'source_prompt': src, 'target_prompt': tgt,
              'pixel_localization': overlap(pred_px, gt), 'grid_localization': overlap(binary, gt_binary),
              'full_image_edit': bool(gt.bool().all()), 'attention_terms': details['terms'],
              'edited_tokens': details['edited_tokens'], 'is_delete': details['is_delete'],
              'preprocess_seconds': prep_s, 'probe_seconds': probe_s, 'probe_peak_allocated_bytes': peak,
              'extractor_validation': 'See release parity checks', 'seeds': SEEDS}
    atomic_json(record_path, record)
    return record


def generate(pipe, sid, out, observed=False, validate_model=False):
    folder = out/'samples'/sid
    p = torch.load(folder/'tensors.pt', map_location='cpu', weights_only=True)
    x = p['source_latent'].to('cuda')
    t = time.perf_counter()
    se = pipe.encode_prompt([p['source_prompt']])
    te = pipe.encode_prompt([p['target_prompt']])
    torch.cuda.synchronize()
    text_s = time.perf_counter()-t
    gt_bg = 1-F.interpolate(p['gt_pixel'].to('cuda'), size=x.shape[-2:], mode='area')
    records = {}
    for name in METHODS:
        path = folder/f'{name}.json'
        if path.exists() and (folder/f'{name}.png').exists():
            records[name] = json.loads(path.read_text())
            continue
        alpha = .2 if name.endswith('a02') else 0.
        support = None if name == 'baseline' else p['gt_binary' if name.startswith('gt_') else ('continuous' if 'continuous' in name else 'binary')].to(x.device)
        weight = torch.ones_like(p['binary']).to(x.device) if support is None else alpha+(1-alpha)*support
        if name == 'baseline':
            model = NaviFlowAlignModel(pipe.transformer, pipe.scheduler, CFG).to(x.device).eval()
        elif observed:
            model = ObservedGuidanceModel(pipe.transformer, pipe.scheduler, CFG, background=gt_bg).to(x.device).eval()
            model.set_edit_mask(weight)
        else:
            model = GuidanceModulatedModel(pipe.transformer, pipe.scheduler, CFG).to(x.device).eval()
            model.set_edit_mask(weight)
        seed(SEEDS['edit'])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        edited, steps = model(x, se, te)
        torch.cuda.synchronize()
        edit_s = time.perf_counter()-start
        start = time.perf_counter()
        image = pipe._tensor_to_pil(pipe.vae_adapter.decode(edited).detach().cpu().clamp(0, 1))[0]
        torch.cuda.synchronize()
        decode_s = time.perf_counter()-start
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        image.save(folder/f'{name}.png')
        info = {'sid': sid, 'method': name, 'alpha': alpha if support is not None else None,
                'seeds': SEEDS, 'edit_seconds': edit_s, 'decode_seconds': decode_s,
                'shared_text_encoding_seconds': text_s, 'peak_allocated_bytes': peak_alloc,
                'peak_reserved_bytes': peak_reserved, 'diagnostic_instrumentation': observed and name!='baseline',
                'timing_context': 'shared GPUs; descriptive only, not definitive overhead benchmark',
                'model_forward_calls_edit': 3*CFG['t_edit']*CFG['noise_samples'],
                'mean_guidance_weight': float(weight.mean()),
                'gt_background_mean_guidance_weight': float((weight*gt_bg).sum()/gt_bg.sum()) if gt_bg.sum() else None,
                'gt_edit_mean_guidance_weight': float((weight*(1-gt_bg)).sum()/(1-gt_bg).sum()) if (1-gt_bg).sum() else None,
                'final_bg_latent_drift_rms': masked_rms(edited-x, gt_bg), 'n_steps': len(steps)}
        if isinstance(model, ObservedGuidanceModel):
            model.rows[-1]['actual_step_bg_rms'] = masked_rms(edited-model.previous_state, gt_bg)
            atomic_json(folder/f'{name}_trajectory.json', model.rows)
            if validate_model:
                plain = GuidanceModulatedModel(pipe.transformer, pipe.scheduler, CFG).to(x.device).eval()
                plain.set_edit_mask(weight)
                seed(SEEDS['edit'])
                check, _ = plain(x, se, te)
                torch.testing.assert_close(edited, check, rtol=0, atol=0)
                info['observer_exact_parity'] = True
                del plain, check
        atomic_json(path, info)
        records[name] = info
        print(json.dumps({'sid': sid, 'method': name, 'edit_seconds': round(edit_s, 3)}, ensure_ascii=False), flush=True)
        del model, edited
    atomic_json(folder/'complete.json', {'sid': sid, 'methods': METHODS, 'verified_outputs': True})
    return records


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pie-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--model', default='stabilityai/stable-diffusion-3-medium-diffusers')
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--run', default='controlled')
    ap.add_argument('--stage', choices=['masks', 'generate', 'all'], default='all')
    ap.add_argument('--max-samples', type=int)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    global PIE, MODEL
    PIE, MODEL = args.pie_root, args.model
    torch.set_num_threads(4)
    i, n = map(int, args.shard.split('/'))
    assert 0 <= i < n
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    mapping_bytes = (PIE/'mapping_file.json').read_bytes()
    mapping = json.loads(mapping_bytes)
    all_ids = sorted(mapping)
    selected = all_ids[i::n]
    if args.max_samples:
        selected = selected[:args.max_samples]
    # A fixed one-per-category set chosen by sample ID before seeing outputs.
    diagnostic_ids = {next(s for s in all_ids if mapping[s]['image_path'].startswith(c+'_')) for c in sorted({r['image_path'].split('_')[0] for r in mapping.values()})}
    manifest = {'shard': args.shard, 'sample_ids': selected, 'diagnostic_ids': sorted(diagnostic_ids),
                'mapping_sha256': hashlib.sha256(mapping_bytes).hexdigest(), 'config': CFG, 'seeds': SEEDS,
                'methods': METHODS, 'source_manifest': {'release': 'focusedit-0.1.0', 'protocol': 'controlled-512'},
                'stage': args.stage,
                'model': 'stabilityai/stable-diffusion-3-medium-diffusers', 'precision': 'bf16', 'size': 512,
                'gt_control_definition': 'area downsample to predicted 32-grid; threshold >=0.5; nearest lift to latent grid',
                'mask_continuous_definition': 'original continuous unit before old 0.15 floor; shared alpha+(1-alpha)*unit'}
    manifest['instrumentation_hashes'] = {'run_controlled.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest_path = out/f'manifest_{i}.json'
    if manifest_path.exists():
        prev = json.loads(manifest_path.read_text())
        for k in ['sample_ids', 'mapping_sha256', 'config', 'seeds', 'source_manifest']:
            assert prev[k] == manifest[k], f'Cannot resume different manifest: {k}'
    atomic_json(manifest_path, manifest)
    print('LOADING', args.shard, len(selected), flush=True)
    pipe = build_navi_pipeline(model_root=MODEL, default_edit_config=CFG, precision='bf16', image_size=512)
    print('READY', torch.cuda.get_device_name(), flush=True)
    if args.stage in ['all', 'masks']:
        for j, sid in enumerate(selected):
            record_mask(pipe, sid, mapping[sid], out, parity=args.smoke)
            if (j+1)%10 == 0 or args.smoke:
                print('MASKS', j+1, len(selected), sid, flush=True)
        atomic_json(out/f'masks_complete_{i}.json', {'count': len(selected), 'sample_ids': selected})
    if args.stage in ['all', 'generate']:
        for j, sid in enumerate(selected):
            generate(pipe, sid, out, observed=(sid in diagnostic_ids or args.smoke), validate_model=args.smoke)
            print('GENERATED', j+1, len(selected), sid, flush=True)
        atomic_json(out/f'generation_complete_{i}.json', {'count': len(selected), 'sample_ids': selected})


if __name__ == '__main__':
    main()
