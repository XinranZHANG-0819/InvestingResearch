"""Monthly win rate under the multi-factor classification (companion to mf_classify.py)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import WORK, load_panel  # noqa: E402

P = load_panel()
R, IDX = P['R'], P['IDX']
H = pd.read_pickle(os.path.join(WORK, 'mf_classify.pkl'))


def win(code, T, cat, n):
    iT = R.index.get_loc(pd.Period(T, 'M'))
    rr = R[code].iloc[iT - n + 1:iT + 1]
    bb = IDX[cat].reindex(rr.index)
    m = rr.notna() & bb.notna()
    return (rr[m] > bb[m]).mean() if m.sum() >= n * 0.9 else np.nan


def spread(sig, cell, out):
    n = H.groupby(cell)[sig].transform('count')
    p = H[sig].groupby(H[cell]).rank(pct=True).where(n >= 5)
    d = pd.DataFrame({'p': p, 'y': H.year, 'o': H[out]}).dropna()
    sp = (d[d.p > 0.8].groupby('y').o.mean() - d[d.p <= 0.2].groupby('y').o.mean()).dropna()
    return f'{sp.mean() * 100:+.2f}（{(sp > 0).sum()}/{len(sp)}）'


for k in ['mf_0.2_0.2', 'mf_0.3_0.3', 'mf_0.4_0.3', 'mf_0.5_0.5']:
    cat = H[f'cat_{k}']
    use5 = (H.tier == '资深') & H.t_5y.notna()
    H[f'win_{k}'] = [win(c, T, ct, 60 if u else 36) if isinstance(ct, str) else np.nan
                     for c, T, ct, u in zip(H.code, H['T'], cat, use5)]
    print(k, 'win 3y', spread(f'win_{k}', f'cell_{k}', 'fwd_ret_36'), '| 5y', spread(f'win_{k}', f'cell_{k}', 'fwd_ret_60'))
