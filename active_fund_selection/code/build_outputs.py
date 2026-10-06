"""Current lists, category index table, holdings diagnosis and the metrics workbook."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import CATS, WORK, load_panel  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else WORK
P = load_panel()
IDX = P['IDX']
C = pd.read_pickle(os.path.join(WORK, 'snap_current.pkl'))
A = pd.read_pickle(os.path.join(WORK, 'analysis.pkl'))
HOLD = ['110023', '002910', '007449', '004206', '001048', '001480', '005851', '006567', '001678']


def tw(df, base):
    v5 = df.get(f'{base}_5y')
    return np.where((df.tier == '资深') & v5.notna(), v5, df[f'{base}_3y'])


for b in ['t', 'win', 'ex', 'ret', 'dcap', 'martin', 'mdd', 'vol', 'beta']:
    C[b] = tw(C, b)
C['win_n'] = np.where(C.tier == '资深', 60, 36)
C['cell'] = C.cat + '|' + C.tier
C['rank_t'] = C.groupby('cell').t.rank(ascending=False, method='first').astype(int)
C['n_cell'] = C.groupby('cell').code.transform('size')
C['pct_t'] = C.groupby('cell').t.rank(pct=True)
C['pct_win'] = C.groupby('cell').win.rank(pct=True)
C['size_pct'] = C.groupby('cat')['size'].rank(pct=True)
C['big'] = C.size_pct > 0.8
C['surge'] = (C['size'] / C.size_1y >= 3) & (C['size'] >= 10)
ids = C.explode('mgr_ids')
dup = ids.groupby('mgr_ids').code.nunique()
C['same_mgr_n'] = C.mgr_ids.map(lambda s: max([dup.get(i, 1) for i in s] + [1]) - 1)
C['label'] = C.name.str.replace('证券投资基金', '', regex=False).str.replace('型', '', regex=False) + ' ' + C.code

# ---- category index table ----
rows = []
for c in CATS:
    s = IDX[c].dropna()
    def ann(x):
        return (1 + x).prod() ** (12 / len(x)) - 1
    nav = (1 + s).cumprod()
    mdd = (nav / nav.cummax() - 1).min()
    rows.append(dict(cat=c, full=ann(s), y5=ann(s.iloc[-60:]), y3=ann(s.iloc[-36:]), mdd=mdd,
                     vol=s.std() * np.sqrt(12), start=str(s.index[0])))
idx_tab = pd.DataFrame(rows)

# ---- long lists (top 10 by stability per cell) and short lists ----
long = C.sort_values(['cat', 'tier', 'rank_t'])
short = []
for cell, g in C.groupby('cell'):
    cat, tier = cell.split('|')
    for crit, col, asc in [('年化收益', 'ret', False), ('超额', 'ex', False),
                           ('下跌捕获率', 'dcap', True), ('Martin比率', 'martin', False)]:
        pick = g.dropna(subset=[col]).sort_values(col, ascending=asc).head(5)
        for i, (_, r) in enumerate(pick.iterrows(), 1):
            short.append(dict(cat=cat, tier=tier, crit=crit, rank=i, code=r.code, label=r.label,
                              value=r[col], t=r.t, rank_t=r.rank_t, n_cell=r.n_cell))
short = pd.DataFrame(short)

# ---- holdings ----
hold = C[C.code.isin(HOLD)].copy()
missing = sorted(set(HOLD) - set(hold.code))

C.to_pickle(os.path.join(WORK, 'current_enriched.pkl'))
pd.to_pickle(dict(idx_tab=idx_tab, long=long, short=short, hold=hold, missing=missing),
             os.path.join(WORK, 'outputs.pkl'))

# ---- workbook ----
nice = {
    'code': '代码', 'name': '基金', 'company': '公司', 'mgr': '现任经理', 'ten_start': '任职起点',
    'tenure': '任职年数', 'tier': '任职档', 'cat': '类别', 'te': '与类别指数跟踪误差',
    'size': '规模(亿)', 'size_1y': '一年前规模(亿)', 'stock_ratio': '近3年平均股票仓位(%)',
    'rank_t': '类内排名(稳定程度)', 'n_cell': '类内基金数', 'same_mgr_n': '同经理其他入池基金数',
    'big': '规模在本类前20%', 'surge': '规模暴增',
}
metric_cols = []
for w, wl in (('3y', '近3年'), ('5y', '近5年'), ('ten', '任职以来')):
    for k, kl in (('ret', '年化'), ('ex', '超额'), ('t', '稳定程度'), ('win', '月度胜率'),
                  ('vol', '波动率'), ('mdd', '最大回撤'), ('ulcer', '溃疡指数'), ('martin', 'Martin'),
                  ('calmar', '卡玛'), ('sharpe', '夏普'), ('beta', 'β'), ('dcap', '下跌捕获'),
                  ('ucap', '上涨捕获')):
        nice[f'{k}_{w}'] = f'{wl}{kl}'
        metric_cols.append(f'{k}_{w}')
base_cols = ['code', 'name', 'company', 'mgr', 'ten_start', 'tenure', 'tier', 'cat', 'te', 'size',
             'size_1y', 'stock_ratio', 'rank_t', 'n_cell', 'same_mgr_n', 'big', 'surge']
te_cols = [f'te_{c}' for c in CATS]
for c in CATS:
    nice[f'te_{c}'] = f'跟踪误差_{c}'
sheet = long[base_cols + metric_cols + te_cols].rename(columns=nice)
xl = os.path.join(OUT, '主动基金选基指标_2026-09.xlsx')
with pd.ExcelWriter(xl) as w:
    readme = pd.DataFrame({'说明': [
        '数据截至2026年9月初（月度净值，月度收益）；以基金为单位，只计现任经理任职以来。',
        '资格：现任经理任职满3年；规模≥2亿；近3年平均股票仓位≥60%（缺失不剔除）。',
        '类别：近3年月度收益与8个类别指数的跟踪误差最小者。',
        '排名窗口：新锐（任职3–5年）看近3年，资深（5年以上）看近5年；“类内排名”按稳定程度。',
        '稳定程度=月度超额均值÷月度超额标准差×√月数；超额=基金年化−类别指数年化。',
        '下跌捕获=类别指数下跌月份中基金平均收益÷指数平均收益；Martin=年化÷溃疡指数。',
        '夏普按无风险利率1.5%计；比率与百分比均为小数（0.12即12%）。']})
    readme.to_excel(w, sheet_name='说明', index=False)
    sheet.to_excel(w, sheet_name='全部基金', index=False)
    for c in CATS:
        sub = sheet[sheet['类别'] == c]
        if len(sub):
            sub.to_excel(w, sheet_name=c, index=False)
    short.rename(columns={'cat': '类别', 'tier': '任职档', 'crit': '准则', 'rank': '名次', 'code': '代码',
                          'label': '基金', 'value': '准则数值', 't': '稳定程度', 'rank_t': '类内排名(稳定程度)',
                          'n_cell': '类内基金数'}).to_excel(w, sheet_name='短名单', index=False)
    idx_tab.to_excel(w, sheet_name='类别指数', index=False)
print('wrote', xl, sheet.shape)
print(pd.crosstab(C.cat, C.tier, margins=True))
print(idx_tab.round(3).to_string())
print('missing holdings', missing)
cols = ['code', 'label', 'cat', 'tier', 'rank_t', 'n_cell', 'pct_t', 'pct_win', 't', 'win', 'ex', 'ret', 'dcap', 'mdd', 'size', 'big', 'surge']
print(hold[cols].round(2).to_string())
