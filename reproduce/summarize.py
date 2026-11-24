"""Aggregate only a complete, matched controlled run; retain all methods/references."""
import argparse
import json
from pathlib import Path
import numpy as np

METHODS = ['baseline', 'attn_binary_a0', 'attn_binary_a02', 'gt_binary_a0',
           'gt_binary_a02', 'attn_continuous_a0', 'attn_continuous_a02']
METRICS = ['struct', 'psnr', 'lpips', 'ssim', 'clip_whole', 'clip_edit']
SCALE = {'struct': 1000, 'lpips': 1000, 'ssim': 100}
PAIRS = [('attn_binary_a02','baseline'), ('attn_binary_a0','baseline'),
         ('gt_binary_a02','attn_binary_a02'), ('gt_binary_a0','attn_binary_a0'),
         ('attn_continuous_a02','attn_binary_a02'), ('attn_continuous_a0','attn_binary_a0'),
         ('attn_binary_a02','attn_binary_a0'), ('gt_binary_a02','gt_binary_a0'),
         ('attn_continuous_a02','attn_continuous_a0')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run = parser.parse_args().output
    rows = [json.loads(p.read_text()) for p in sorted((run/'samples').glob('*/metrics.json'))]
    expected = set()
    for p in run.glob('manifest_*.json'):
        expected.update(json.loads(p.read_text())['sample_ids'])
    assert len(rows) == len({r['sid'] for r in rows}) == len(expected) == 700, 'Wait for all 700 matched records'
    assert {r['sid'] for r in rows} == expected
    rng = np.random.default_rng(20261124)
    out = {'n_samples': 700, 'bootstrap_replicates': 10000, 'bootstrap_seed': 20261124,
           'bootstrap_unit': 'paired edit/source image', 'reference_results': {}}
    markdown = ['# 固定协议补充实验：全 700 例', '',
                '保留七组配置及 RGB/VAE 两种参考口径。置信区间是描述性的逐项区间，未作多重比较校正；跨种子稳定性未由单种子设计验证。', '']
    for ref in ['rgb', 'vae']:
        values = {method: {metric: np.array([r['metrics'][method][ref][metric] if r['metrics'][method][ref][metric] is not None else np.nan for r in rows], dtype=float)*SCALE.get(metric,1)
                           for metric in METRICS} for method in METHODS}
        result = {'means': {}, 'paired_differences': []}
        markdown += [f'## 参考：{ref}', '', '| 方法 | 背景 LPIPS ×1000 | 背景 PSNR | 背景 SSIM ×100 | CLIP-edit |', '|---|---:|---:|---:|---:|']
        for method in METHODS:
            result['means'][method] = {metric: {'mean': float(np.nanmean(values[method][metric])),
                                               'n_valid': int(np.isfinite(values[method][metric]).sum())} for metric in METRICS}
            means = result['means'][method]
            markdown.append('| '+method+' | '+' | '.join(f"{means[m]['mean']:.4f} (n={means[m]['n_valid']})" for m in ['lpips','psnr','ssim','clip_edit'])+' |')
        for lhs, rhs in PAIRS:
            for metric in METRICS:
                a, b = values[lhs][metric], values[rhs][metric]
                valid = np.isfinite(a) & np.isfinite(b)
                d = a[valid]-b[valid]
                boots = []
                for _ in range(40):
                    ix = rng.integers(0, len(d), size=(250,len(d)))
                    boots.extend(d[ix].mean(axis=1))
                result['paired_differences'].append({'lhs':lhs,'rhs':rhs,'metric':metric,'n_pairs':len(d),
                    'delta':float(d.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist()})
        markdown += ['', '| 配对差（前者−后者） | 指标 | 均值差 | 95% 区间 |', '|---|---|---:|---|']
        for pair in result['paired_differences']:
            if pair['metric'] in ['lpips','clip_edit']:
                lo,hi=pair['ci95']
                markdown.append(f"| {pair['lhs']} − {pair['rhs']} | {pair['metric']} | {pair['delta']:.4f} | [{lo:.4f}, {hi:.4f}] |")
        markdown.append('')
        out['reference_results'][ref] = result
    target = run/'analysis'
    target.mkdir(exist_ok=True)
    (target/'controlled_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    (target/'controlled_results.md').write_text('\n'.join(markdown)+'\n')
    print('COMPLETE_MATCHED_SUMMARY', str(target/'controlled_results.md'), flush=True)


if __name__ == '__main__':
    main()
