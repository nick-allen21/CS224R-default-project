import json
import sys
from math import comb
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'eval_step20.json'

def load(p):
    return [json.loads(l) for l in open(p)]

recs = load(path)
sft_recs = load('default_proj_sft/sft_eval.json')
ipo_recs = load('eval_jsons/ipo_eval.json')
N = len(recs[0]['scores'])  # samples per prompt (16)

# Chen et al. unbiased pass@k estimator
# pass@k = (1/P) * sum_i [1 - C(n - c_i, k) / C(n, k)]  where c_i = # correct samples for prompt i
def passk_one(c, n, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def passk(c_list, n, k):
    return sum(passk_one(c, n, k) for c in c_list) / len(c_list)

def c_list_of(rs):
    return [sum(1 for x in r['scores'] if x >= 1.0) for r in rs]

rloo_passk = [passk(c_list_of(recs), N, k) for k in range(1, N + 1)]
sft_passk = [passk(c_list_of(sft_recs), N, k) for k in range(1, N + 1)]
ipo_passk = [passk(c_list_of(ipo_recs), N, k) for k in range(1, N + 1)]

plt.figure(figsize=(6, 4))
plt.plot(range(1, N + 1), rloo_passk, marker='o', label='RLOO (fixed)', linewidth=2)
plt.plot(range(1, N + 1), sft_passk, marker='s', label='SFT', linewidth=1.5)
plt.plot(range(1, N + 1), ipo_passk, marker='^', label='IPO', linewidth=1.5)
plt.xlabel('k')
plt.ylabel('pass@k')
plt.title('Pass@k on held-out test set')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('passk_comparison.png', dpi=150, bbox_inches='tight')
print(f'saved passk_comparison.png')
print(f'RLOO pass@1  = {rloo_passk[0]:.4f}   pass@16 = {rloo_passk[-1]:.4f}')
print(f'SFT  pass@1  = {sft_passk[0]:.4f}   pass@16 = {sft_passk[-1]:.4f}')
print(f'IPO  pass@1  = {ipo_passk[0]:.4f}   pass@16 = {ipo_passk[-1]:.4f}')
