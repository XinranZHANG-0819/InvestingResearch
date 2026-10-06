"""Build the monthly fund/index panel used by every later step.

Output: work/panel.pkl with
  funds    one row per main share class (code, name, type, status, company, inception)
  R        monthly total returns, index = month key (Period M of the NAV date, i.e. the
           first trading day of that month), columns = main share code
  IDX      monthly returns of the 8 category indices on the same month keys
  stints   merged manager tenures per fund (code, mgr_id, mgr, start, end[NaT = current])
  size     quarterly net assets (亿元) and stock ratio (%) per fund
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import kv, load  # noqa: E402

WORK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'work')
os.makedirs(WORK, exist_ok=True)

CATS = {
    '全部': ['H00985'], '中大盘': ['H00906'], '小盘': ['H00852'],
    '科技': ['H00993', 'H00994'], '医药': ['H00991'], '消费': ['H00989', 'H00990'],
    '金融地产': ['H00992'], '周期': ['H00986', 'H00987', 'H00988', 'H00995'],
}

# ---------- funds ----------
t34 = load('T34')
meta = t34.note.map(kv).apply(pd.Series)
t34 = pd.concat([t34[['code', 'name', 'value']], meta], axis=1)
t34['is_main'] = t34.code == t34['主份额']
funds = (t34.sort_values('is_main', ascending=False)
         .drop_duplicates('主份额')
         .rename(columns={'主份额': 'main', 'value': 'type', '状态': 'status',
                          '基金公司': 'company', '成立日': 'inception', '业绩比较基准': 'benchmark'}))
funds = funds[['main', 'name', 'type', 'status', 'company', 'inception', 'benchmark']].rename(columns={'main': 'code'})
funds['inception'] = pd.to_datetime(funds.inception, errors='coerce')
funds = funds.set_index('code')
MAIN = set(funds.index)

# ---------- NAV -> monthly total returns ----------
nav = load('T35_')
nav = nav[nav.code.isin(MAIN)]
nav['value'] = pd.to_numeric(nav.value, errors='coerce')
nav['date'] = pd.to_datetime(nav.date)
nav = nav.dropna(subset=['value'])
nav = nav[nav.value > 0]
nav['m'] = nav.date.dt.to_period('M')
nav = nav.sort_values(['code', 'date']).drop_duplicates(['code', 'm'], keep='first')

ev = load('T35b')
ev['date'] = pd.to_datetime(ev.date)
div = ev[ev.indicator == 'dividend_per_unit'].copy()
div['amt'] = pd.to_numeric(div.value, errors='coerce')
spl = ev[ev.indicator == 'split_ratio'].copy()
spl['k'] = spl.value.str.split(':').map(lambda x: float(x[1]) / float(x[0]) if len(x) == 2 else np.nan)
div = div.dropna(subset=['amt']).groupby('code')
spl = spl.dropna(subset=['k']).groupby('code')
div_codes, spl_codes = set(div.groups), set(spl.groups)

rows = []
for code, g in nav.groupby('code', sort=False):
    d = g.date.values
    v = g.value.values.astype(float)
    adj = v.copy()                      # value of one previous unit at this date
    if code in spl_codes:
        s = spl.get_group(code)
        for dt, k in zip(s.date.values, s.k.values):
            i = np.searchsorted(d, dt, side='left')   # first obs on/after split date
            if 0 < i < len(d):
                adj[i] = adj[i] * k
                # later observations are already in post-split units; carry the factor
                # only into this interval's return (handled below via ratio of adj/v)
    ratio_split = np.ones(len(d))
    ratio_split[1:] = adj[1:] / v[1:]
    cash = np.zeros(len(d))
    if code in div_codes:
        dd = div.get_group(code)
        for dt, a in zip(dd.date.values, dd.amt.values):
            i = np.searchsorted(d, dt, side='left')
            if 0 < i < len(d):
                cash[i] += a
    r = np.full(len(d), np.nan)
    r[1:] = (v[1:] * ratio_split[1:] + cash[1:] * ratio_split[1:]) / v[:-1] - 1
    rows.append(pd.DataFrame({'code': code, 'm': g.m.values, 'r': r}))
ret = pd.concat(rows).dropna()
R = ret.pivot(index='m', columns='code', values='r').sort_index()

# ---------- category indices ----------
b = load('T02')
ind = load('T11')
ix = pd.concat([b, ind])
ix['code'] = ix.code.str.replace('.CSI', '', regex=False)
ix = ix[ix.code.str.startswith('H')]
ix['close'] = pd.to_numeric(ix.close, errors='coerce')
ix = ix[ix.close > 0]
ix['date'] = pd.to_datetime(ix.date)
px = ix.pivot_table(index='date', columns='code', values='close').sort_index()
first_td = px.groupby(px.index.to_period('M')).head(1)
first_td.index = first_td.index.to_period('M')
ixr = first_td.pct_change()
IDX = pd.DataFrame({c: ixr[codes].mean(axis=1, skipna=False) for c, codes in CATS.items()})
IDX = IDX.dropna(how='all')

# ---------- managers: merge back-to-back team periods ----------
m = load('T36b')
st = m[m.indicator == 'fund_manager_start'].copy()
mm = st.note.map(kv).apply(pd.Series)
st = pd.concat([st[['code', 'value']], mm[['任职起始日', '离任日期', '经理ID']]], axis=1)
st.columns = ['code', 'mgr', 'start', 'end', 'mgr_id']
st['start'] = pd.to_datetime(st.start, errors='coerce')
st['end'] = pd.to_datetime(st.end.where(~st.end.str.contains('至今', na=False)), errors='coerce')
st = st.dropna(subset=['start']).sort_values(['code', 'mgr_id', 'start'])
merged = []
for (code, mid), g in st.groupby(['code', 'mgr_id'], sort=False):
    cur = None
    for s, e, name in zip(g.start, g.end, g.mgr):
        if cur is None:
            cur = [s, e, name]
        elif pd.isna(cur[1]) or s <= cur[1] + pd.Timedelta(days=7):
            if pd.isna(cur[1]) or pd.isna(e):
                cur[1] = pd.NaT if (pd.isna(cur[1]) or pd.isna(e)) else max(cur[1], e)
            else:
                cur[1] = max(cur[1], e)
        else:
            merged.append((code, mid, cur[2], cur[0], cur[1]))
            cur = [s, e, name]
    merged.append((code, mid, cur[2], cur[0], cur[1]))
stints = pd.DataFrame(merged, columns=['code', 'mgr_id', 'mgr', 'start', 'end'])
stints = stints[stints.code.isin(MAIN)]

# ---------- size and stock ratio ----------
s = load('T37_')
s = s[s.indicator.isin(['net_asset', 'stock_ratio']) & s.code.isin(MAIN)]
s['value'] = pd.to_numeric(s.value, errors='coerce')
s['date'] = pd.to_datetime(s.date)
size = s.pivot_table(index=['code', 'date'], columns='indicator', values='value').reset_index()

pickle.dump(dict(funds=funds, R=R, IDX=IDX, stints=stints, size=size),
            open(os.path.join(WORK, 'panel.pkl'), 'wb'))
print('funds', len(funds), 'R', R.shape, R.index.min(), R.index.max())
print('IDX', IDX.shape, IDX.index.min(), IDX.index.max())
print('stints', len(stints), 'current', stints.end.isna().sum())
print('size rows', len(size))
q = R.stack()
print('monthly r quantiles', q.quantile([0.0001, 0.001, 0.5, 0.999, 0.9999]).round(3).to_dict())
print('|r|>50%:', (q.abs() > 0.5).sum())
