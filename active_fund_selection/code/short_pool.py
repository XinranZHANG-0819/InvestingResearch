"""Short lists (top 5 per cell, cells with >= 10 funds): does pre-filtering by monthly win rate help?

Variants per criterion X (年化, 下跌捕获, Martin):
  plain        top 5 by X in the whole cell (current chapter 7)
  win top50%   top 5 by X among funds in the cell's top half by monthly win rate
  win top20%   ... top 20% by win rate (falls back to fewer than 5 funds if the pool is small)
Plus: top 5 by monthly win rate alone. Outcomes relative to the cell average.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import WORK  # noqa: E402

A = pd.read_pickle(os.path.join(WORK, 'analysis.pkl'))
H = A['H']
H['pwin'] = H.groupby('cell').sig_win.rank(pct=True)
crit = {'年化收益／超额': ('sig_ret', False), '下跌捕获率': ('sig_dcap', True), 'Martin比率': ('sig_martin', False)}
rows = []
for cell, g in H.groupby('cell'):
    if len(g) < 10:
        continue
    yr = g.year.iloc[0]
    for name, (sig, asc) in list(crit.items()) + [('月度胜率', ('sig_win', False))]:
        pools = {'全类': g}
        if name != '月度胜率':
            pools['胜率前50%内'] = g[g.pwin > 0.5]
            pools['胜率前20%内'] = g[g.pwin > 0.8]
        for pname, pool in pools.items():
            pick = pool.dropna(subset=[sig]).sort_values(sig, ascending=asc).head(5)
            if not len(pick):
                continue
            rows.append(dict(crit=name, pool=pname, year=yr, cell=cell, n=len(pick),
                             rel3=pick.rel_ret_36.mean(), rel5=pick.rel_ret_60.mean(),
                             mdd3=pick.rel_mdd_36.mean()))
S = pd.DataFrame(rows)
by_year = S.groupby(['crit', 'pool', 'year'])[['rel3', 'rel5']].mean()
summ = S.groupby(['crit', 'pool']).agg(cells=('cell', 'size'), rel3=('rel3', 'mean'), beat3=('rel3', lambda x: (x > 0).mean()),
                                       rel5=('rel5', 'mean'), mdd3=('mdd3', 'mean'))
yrs = by_year.groupby(['crit', 'pool']).agg(pos3=('rel3', lambda x: f'{(x > 0).sum()}/{x.notna().sum()}'),
                                            pos5=('rel5', lambda x: f'{(x > 0).sum()}/{x.notna().sum()}'))
out = summ.join(yrs)
for c in ['rel3', 'rel5', 'mdd3']:
    out[c] = (out[c] * 100).round(2)
out['beat3'] = (out.beat3 * 100).round(0)
pd.set_option('display.width', 200)
print(out.to_string())
pd.to_pickle(dict(S=S, summ=out), os.path.join(WORK, 'short_pool.pkl'))
