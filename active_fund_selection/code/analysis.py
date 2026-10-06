"""Validation tests on yearly snapshots (2010-2023, January keys).

Signals use the tier's window: 新锐 -> 3y, 资深 -> 5y (资深 without a 5y value fall back to 3y).
Ranking is a percentile inside each (year, category, tier) cell with >= MIN_CELL funds;
groups pool those percentiles across cells. Outcomes are forward 3y/5y annualized returns
measured relative to the fund's (year, category, tier) cell average unless noted.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import WORK  # noqa: E402

MIN_CELL = 5
H = pd.read_pickle(os.path.join(WORK, 'snap_hist.pkl'))
H['year'] = H['T'].str[:4].astype(int)


def tier_col(df, base):
    v5, v3 = df.get(f'{base}_5y'), df.get(f'{base}_3y')
    if v5 is None:
        return v3
    return np.where((df.tier == '资深') & v5.notna(), v5, v3)


for base in ['t', 'win', 'ex', 'ret', 'dcap', 'martin', 'mdd', 'tall', 'exall']:
    H[f'sig_{base}'] = tier_col(H, base)

H['cell'] = H.year.astype(str) + '|' + H.cat + '|' + H.tier
cell_n = H.groupby('cell').code.transform('size')
for h in (36, 60):
    for k in ('ret', 'mdd'):
        c = f'fwd_{k}_{h}'
        H[f'rel_{k}_{h}'] = H[c] - H.groupby('cell')[c].transform('mean')
        H[f'relall_{k}_{h}'] = H[c] - H.groupby('year')[c].transform('mean')


def pct_in_cell(sig, ascending=True):
    s = H[sig] if ascending else -H[sig]
    p = s.groupby(H.cell).rank(pct=True)
    return p.where(cell_n >= MIN_CELL)


def spread(p, out, lo=0.2, hi=0.8, by_year=True):
    d = pd.DataFrame({'p': p, 'y': H.year, 'o': H[out]}).dropna()
    top = d[d.p > hi].groupby('y').o.mean()
    bot = d[d.p <= lo].groupby('y').o.mean()
    sp = (top - bot).dropna()
    return dict(top=top.mean(), bottom=bot.mean(), spread=sp.mean(),
                pos=f'{(sp > 0).sum()}/{len(sp)}',
                t=sp.mean() / sp.std(ddof=1) * np.sqrt(len(sp)) if len(sp) > 2 else np.nan,
                n_top=int((d.p > hi).sum()))


def fmt(d):
    return {k: (round(v * 100, 2) if isinstance(v, float) and k in ('top', 'bottom', 'spread') else
                (round(v, 2) if isinstance(v, float) else v)) for k, v in d.items()}


res = {}
print('== Test 1: persistence (top20% vs bottom20% within category x tier) ==')
for sig, asc in [('sig_t', True), ('sig_win', True), ('sig_ex', True), ('sig_ret', True)]:
    p = pct_in_cell(sig, asc)
    for out in ['fwd_ret_36', 'fwd_ret_60', 'fwd_mdd_36']:
        r = spread(p, out)
        res[('T1', sig, out)] = r
        print(sig, out, fmt(r))

print('== Test 1b: by tier (fwd_ret_36) ==')
for tier in ['新锐', '资深']:
    for sig in ['sig_t', 'sig_win', 'sig_ex']:
        p = pct_in_cell(sig).where(H.tier == tier)
        print(tier, sig, fmt(spread(p, 'fwd_ret_36')), '| 5y', fmt(spread(p, 'fwd_ret_60')))

print('== Test 2: classification useful? stability vs category index (within cell) '
      'vs stability vs all-market index (ranked within year x tier) ==')
p_cat = pct_in_cell('sig_t')
grp = H.year.astype(str) + '|' + H.tier
p_all = H.sig_tall.groupby(grp).rank(pct=True)
for out in ['fwd_ret_36', 'fwd_ret_60']:
    print('category  ', out, fmt(spread(p_cat, out)))
    print('no-classes', out, fmt(spread(p_all, out)))

print('== Test 3: category stability (same fund, consecutive years) ==')
prev = H[['code', 'year', 'cat']].copy()
prev['year'] += 1
M = H.merge(prev, on=['code', 'year'], suffixes=('', '_prev'))
M['same'] = M.cat == M.cat_prev
print('share unchanged', round(M.same.mean(), 3), 'n', len(M))
print(M.groupby('year').same.mean().round(2).to_dict())
print(pd.crosstab(M.cat_prev, M.cat, normalize='index').round(2).to_string())

print('== Test 4: drift and size effects (forward 3y return relative to cell average, pp) ==')
M['rel'] = M['rel_ret_36']
print('changed category:', round(M[~M.same].rel.mean() * 100, 2), 'n', (~M.same).sum(),
      '| unchanged:', round(M[M.same].rel.mean() * 100, 2), 'n', M.same.sum())
BROAD = {'全部', '中大盘', '小盘'}
M['ind_switch'] = ~M.same & ~(M.cat.isin(BROAD) & M.cat_prev.isin(BROAD))
print('industry-level switch:', round(M[M.ind_switch].rel.mean() * 100, 2), 'n', M.ind_switch.sum(),
      '| broad<->broad switch:', round(M[~M.same & ~M.ind_switch].rel.mean() * 100, 2), 'n', (~M.same & ~M.ind_switch).sum())
d = M[M.ind_switch | M.same].groupby(['year', 'ind_switch']).rel.mean().unstack()
print('industry switch minus unchanged by year:', ((d[True] - d[False]) * 100).round(2).to_dict())
print('by year changed-unchanged:', (M.groupby(['year', 'same']).rel.mean().unstack().pipe(lambda x: x[False] - x[True]) * 100).round(2).to_dict())
H['size_pct_cat'] = H.groupby(['year', 'cat'])['size'].rank(pct=True)
small = H[H.cat == '小盘']
print('小盘 top20% size:', round(small[small.size_pct_cat > 0.8].rel_ret_36.mean() * 100, 2), 'n', (small.size_pct_cat > 0.8).sum(),
      '| rest:', round(small[small.size_pct_cat <= 0.8].rel_ret_36.mean() * 100, 2))
for c in ['全部', '中大盘', '科技', '消费', '医药', '周期']:
    g = H[H.cat == c]
    print(c, 'top20% size rel', round(g[g.size_pct_cat > 0.8].rel_ret_36.mean() * 100, 2),
          'rest', round(g[g.size_pct_cat <= 0.8].rel_ret_36.mean() * 100, 2))
H['surge'] = (H['size'] / H.size_1y >= 3) & (H['size'] >= 10)
print('规模暴增:', round(H[H.surge].rel_ret_36.mean() * 100, 2), 'n', H.surge.sum(),
      '| others:', round(H[~H.surge].rel_ret_36.mean() * 100, 2))
print(H[H.surge].groupby('year').rel_ret_36.agg(['mean', 'size']).round(3).to_dict())

print('== Test 5: short lists (top 5 per cell, cells with >= 10 funds) ==')
crit = {'年化收益': ('sig_ret', False), '超额': ('sig_ex', False), '下跌捕获': ('sig_dcap', True),
        'Martin': ('sig_martin', False), '稳定程度': ('sig_t', False)}
rows = []
for name, (sig, asc) in crit.items():
    for cell, g in H.groupby('cell'):
        if len(g) < 10:
            continue
        pick = g.sort_values(sig, ascending=asc).head(5)
        rows.append(dict(crit=name, cell=cell, year=g.year.iloc[0],
                         rel_ret_36=pick.rel_ret_36.mean(), rel_mdd_36=pick.rel_mdd_36.mean(),
                         rel_ret_60=pick.rel_ret_60.mean(), ret_36=pick.fwd_ret_36.mean(),
                         mdd_36=pick.fwd_mdd_36.mean(), cell_ret_36=g.fwd_ret_36.mean(),
                         cell_mdd_36=g.fwd_mdd_36.mean()))
SL = pd.DataFrame(rows)
summ = SL.groupby('crit').agg(cells=('cell', 'size'), rel_ret_36=('rel_ret_36', 'mean'),
                              beat=('rel_ret_36', lambda x: (x > 0).mean()),
                              rel_mdd_36=('rel_mdd_36', 'mean'), rel_ret_60=('rel_ret_60', 'mean'),
                              ret_36=('ret_36', 'mean'), mdd_36=('mdd_36', 'mean'),
                              cell_ret_36=('cell_ret_36', 'mean'), cell_mdd_36=('cell_mdd_36', 'mean'))
print((summ * [1, 100, 1, 100, 100, 100, 100, 100, 100]).round(2).to_string())
by_year = SL.groupby(['crit', 'year']).rel_ret_36.mean().unstack(0) * 100
print(by_year.round(1).to_string())

pd.to_pickle(dict(H=H, M=M, SL=SL, summ=summ, res=res), os.path.join(WORK, 'analysis.pkl'))
