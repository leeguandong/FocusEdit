"""Recompute all controlled-run means from the supplied CSV using only Python stdlib."""
from pathlib import Path
import argparse
import collections
import csv
import json
import math
import statistics

METRICS=['struct','psnr','lpips','ssim','clip_whole','clip_edit']

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--csv',type=Path,default=Path(__file__).with_name('controlled_per_edit_metrics.csv'))
 ap.add_argument('--protocol',type=Path,default=Path(__file__).with_name('protocol.json'))
 ap.add_argument('--expected',type=Path,default=Path(__file__).with_name('reported_means.json'))
 args=ap.parse_args()
 protocol=json.loads(args.protocol.read_text());expected=json.loads(args.expected.read_text())
 groups=collections.defaultdict(list);keys=set()
 with args.csv.open(newline='') as f:
  for row in csv.DictReader(f):
   key=(row['sid'],row['method'],row['reference'])
   assert key not in keys,key
   keys.add(key);groups[(row['reference'],row['method'])].append(row)
 assert len(keys)==9800 and len(groups)==14
 results={}
 for (ref,method),rows in sorted(groups.items()):
  assert len(rows)==700 and len({r['sid'] for r in rows})==700
  assert sum(int(r['background_valid']) for r in rows)==556
  values={}
  for metric in METRICS:
   data=[float(r[metric]) for r in rows if r[metric]!='']
   assert all(math.isfinite(x) for x in data)
   assert len(data)==(556 if metric in ['psnr','lpips','ssim'] else 700)
   value={'mean':statistics.mean(data)*protocol['report_scales'][metric],'n_valid':len(data)}
   target=expected[ref][method][metric]
   assert value['n_valid']==target['n_valid']
   assert math.isclose(value['mean'],target['mean'],rel_tol=1e-12,abs_tol=1e-10),(ref,method,metric)
   values[metric]=value
  results.setdefault(ref,{})[method]=values
 print(json.dumps({'status':'verified','rows':len(keys),'means_verified':84,'reference_results':results},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
