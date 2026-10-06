"""Within-category vs pooled (within tier) ranking: split the pooled spread into
selection inside the category and category mix (top and bottom groups hold different categories).

Outcome raw = forward annualized return; rel = forward return minus the (year, category, tier) average.
spread(raw) - spread(rel) = what the category mix of the top/bottom groups contributes.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import WORK  # noqa: E402

D = pd.read_pickle(os.path.join(WORK, 'mf_compare.pkl'))
H = D['H'].copy()
H['sig_t'] = np.where((H.tier == '资深') & H.t_5y.notna(), H.t_5y, H.t_3y)
H['sig_win'] = np.where((H.tier == '资深') & H.win_5y.notna(), H.win_5y, H.win_3y)
for h in (36, 60):
    c = f'fwd_ret_{h}'
    H[f'rel_{h}'] = H[c] - H.groupby('cell')[c].transform('mean')


def pct(sig, cell, min_cell=5):
    n = H.groupby(cell)[sig].transform('count')
    return H[sig].groupby(H[cell]).rank(pct=True).where(n >= min_cell)


def spread_by_year(p, out):
    d = pd.DataFrame({'p': p, 'y': H.year, 'o': H[out]}).dropna()
    return (d[d.p > 0.8].groupby('y').o.mean() - d[d.p <= 0.2].groupby('y').o.mean()).dropna()


rows = []
for sname, sig in [('类别指数稳定程度', 'sig_t'), ('月度胜率', 'sig_win'), ('多因子', 'mf_t')]:
    for how, cell in [('类内排', 'cell'), ('同档一起排', 'tiercell')]:
        p = pct(sig, cell)
        for h in (36, 60):
            raw = spread_by_year(p, f'fwd_ret_{h}')
            rel = spread_by_year(p, f'rel_{h}')
            mix = (raw - rel).dropna()
            rows.append(dict(signal=sname, how=how, h=h // 12, raw=raw.mean() * 100, raw_pos=f'{(raw > 0).sum()}/{len(raw)}',
                             within=rel.mean() * 100, within_pos=f'{(rel > 0).sum()}/{len(rel)}',
                             mix=mix.mean() * 100, mix_pos=f'{(mix > 0).sum()}/{len(mix)}'))
T = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(T.round(2).to_string(index=False))

print('\n== category mix of pooled top 20% / bottom 20% vs all (3y-window signal, share) ==')
for sname, sig in [('类别指数稳定程度', 'sig_t'), ('多因子', 'mf_t')]:
    p = pct(sig, 'tiercell')
    mixdf = pd.DataFrame({'all': H.cat.value_counts(normalize=True),
                          'top': H[p > 0.8].cat.value_counts(normalize=True),
                          'bottom': H[p <= 0.2].cat.value_counts(normalize=True)}).fillna(0)
    print(sname); print(mixdf.round(2).to_string())

print('\n== mix contribution by year, 类别指数稳定程度 pooled, 3y (pp) ==')
p = pct('sig_t', 'tiercell')
raw = spread_by_year(p, 'fwd_ret_36'); rel = spread_by_year(p, 'rel_36')
print(((raw - rel) * 100).round(1).to_dict())
p = pct('mf_t', 'tiercell')
raw = spread_by_year(p, 'fwd_ret_36'); rel = spread_by_year(p, 'rel_36')
print('多因子', ((raw - rel) * 100).round(1).to_dict())

# do categories with high average past signal beat in the next 3y? (is the mix effect systematic?)
print('\n== category level: average past signal vs next-3y category average return (rank corr by year) ==')
g = H.groupby(['year', 'tier', 'cat']).agg(n=('code', 'size'), t=('sig_t', 'mean'), mf=('mf_t', 'mean'), fwd=('fwd_ret_36', 'mean')).reset_index()
g = g[g.n >= 5]
for col in ['t', 'mf']:
    rc = g.groupby(['year', 'tier']).apply(lambda x: x[col].rank().corr(x.fwd.rank()) if len(x) >= 4 else np.nan, include_groups=False).dropna()
    print(col, 'mean rank corr', round(rc.mean(), 2), 'positive', f'{(rc > 0).sum()}/{len(rc)}')
pd.to_pickle(T, os.path.join(WORK, 'pool_vs_within.pkl'))
