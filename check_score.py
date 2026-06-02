import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else 'eval_step10.json'
recs = [json.loads(l) for l in open(path)]
s = [x for r in recs for x in r['scores']]
print(f'{path}')
print(f'{len(recs)} prompts x {len(s)//len(recs)} samples = {len(s)} total')
print(f'pass@1  = {sum(1 for x in s if x>=1.0)/len(s):.4f}   (threshold 0.45)')
print(f'pass@16 = {sum(1 for r in recs if any(x>=1.0 for x in r["scores"]))/len(recs):.4f}')
print(f'mean reward = {sum(s)/len(s):.4f}')
