"""Render report tables as markdown snippets in work/md/."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from metrics import WORK, CATS
O = pd.read_pickle(os.path.join(WORK, 'outputs.pkl'))
C = pd.read_pickle(os.path.join(WORK, 'current_enriched.pkl'))
MD = os.path.join(WORK, 'md'); os.makedirs(MD, exist_ok=True)
HOLD = set(['110023', '002910', '007449', '004206', '001048', '001480', '005851', '006567', '001678'])
def short(name):
    s = re.sub(r'(证券投资基金|型|发起式|\(LOF\)|（LOF）)', '', name)
    return s.strip()
C['nm'] = C.name.map(short) + ' ' + C.code
pc = lambda x: '—' if pd.isna(x) else f'{x*100:.0f}%'
pc1 = lambda x: '—' if pd.isna(x) else f'{x*100:.1f}%'
d2 = lambda x: '—' if pd.isna(x) else f'{x:.2f}'
def flags(r):
    f = []
    if r.code in HOLD: f.append('持有')
    if r.big: f.append('规模大')
    if r.surge: f.append('暴增')
    if r.size_stale: f.append('规模数据旧')
    return '、'.join(f)
# counts
ct = pd.crosstab(C.cat, C.tier).reindex(CATS).fillna(0).astype(int)
ct['合计'] = ct.sum(axis=1)
lines = ['| 类别 | 新锐 | 资深 | 合计 |', '| --- | --- | --- | --- |']
for c, r in ct.sort_values('合计', ascending=False).iterrows():
    lines.append(f"| {c} | {r.get('新锐', 0)} | {r.get('资深', 0)} | {r['合计']} |")
lines.append(f"| 合计 | {ct['新锐'].sum()} | {ct['资深'].sum()} | {ct['合计'].sum()} |")
open(f'{MD}/counts.md', 'w').write('\n'.join(lines))
I = O['idx_tab'].set_index('cat').reindex(CATS)
lines = ['| 类别 | 2005年2月以来年化 | 近5年年化 | 近3年年化 | 最大回撤 | 年化波动 |', '| --- | --- | --- | --- | --- | --- |']
for c, r in I.iterrows():
    lines.append(f'| {c} | {pc1(r.full)} | {pc1(r.y5)} | {pc1(r.y3)} | {pc(r.mdd)} | {pc(r.vol)} |')
open(f'{MD}/idx.md', 'w').write('\n'.join(lines))
# long lists
for cat in CATS:
    g = C[C.cat == cat].sort_values(['tier', 'rank_t'], ascending=[False, True])
    lines = ['| 档 | 名次 | 基金 | 经理 | 规模(亿) | 稳定程度 | 月度胜率 | 年化 | 超额 | 下跌捕获 | 标注 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for tier in ['资深', '新锐']:
        gg = g[g.tier == tier].sort_values('rank_t').head(10)
        for r in gg.itertuples():
            lines.append(f'| {tier} | {r.rank_t}/{r.n_cell} | {r.nm} | {r.mgr} | {r.size:.0f} | {d2(r.t)} | {pc(r.win)} | {pc(r.ret)} | {pc(r.ex)} | {d2(r.dcap)} | {flags(r)} |')
    open(f'{MD}/long_{cat}.md', 'w').write('\n'.join(lines))
# short lists: one fund per row, with annualized return and max drawdown for every window
SH = O['short'].merge(C[['code', 'nm']], on='code')   # short already carries ret/mdd per window
neg = lambda x: '—' if pd.isna(x) else f'{x*100:.0f}%'.replace('-', '−')
for cat in CATS:
    g = SH[SH.cat == cat]
    if not len(g):
        continue
    lines = ['| 档 | 准则 | 基金 | 准则数值 | 近3年年化 | 近3年回撤 | 近5年年化 | 近5年回撤 | 任职年化 | 任职回撤 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for tier in ['资深', '新锐']:
        gt = g[g.tier == tier]
        first_tier = True
        for crit, lab, fmt in [('年化收益', '年化收益／超额', None), ('下跌捕获率', '下跌捕获率', lambda x: d2(x).replace('-', '−')), ('Martin比率', 'Martin比率', lambda x: f'{x:.1f}'.replace('-', '−'))]:
            gg = gt[gt.crit == crit].sort_values('rank')
            ex = gt[gt.crit == '超额'].set_index('code').value
            for i, r in enumerate(gg.itertuples()):
                val = f'{neg(r.value)}／{neg(ex.get(r.code))}' if crit == '年化收益' else fmt(r.value)
                lines.append(f"| {tier if first_tier else ''} | {lab if i == 0 else ''} | {r.nm} | {val} | "
                             f"{neg(r.ret_3y)} | {neg(r.mdd_3y)} | {neg(r.ret_5y)} | {neg(r.mdd_5y)} | {neg(r.ret_ten)} | {neg(r.mdd_ten)} |")
                first_tier = False
    open(f'{MD}/short_{cat}.md', 'w').write('\n'.join(lines))
# check 年化 and 超额 picks identical
same = all(set(SH[(SH.cat == c) & (SH.tier == t) & (SH.crit == '年化收益')].code) == set(SH[(SH.cat == c) & (SH.tier == t) & (SH.crit == '超额')].code)
           for c in CATS for t in ['资深', '新锐'])
print('年化 and 超额 short lists identical in every cell:', same)
# holdings
Hd = C[C.code.isin(HOLD)].sort_values(['cat', 'rank_t'])
lines = ['| 基金 | 经理 | 类别·档 | 类内名次 | 稳定程度 | 月度胜率 | 年化 | 超额 | 下跌捕获 | 规模(亿) | 一年前规模(亿) | 标注 |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
for r in Hd.itertuples():
    f = flags(r).replace('持有', '').strip('、')
    s1 = '—' if (pd.isna(r.size_1y) or r.size_stale) else f'{r.size_1y:.0f}'
    lines.append(f'| {r.nm} | {r.mgr} | {r.cat}·{r.tier} | {r.rank_t}/{r.n_cell} | {d2(r.t)} | {pc(r.win)} | {pc(r.ret)} | {pc(r.ex)} | {d2(r.dcap)} | {r.size:.0f} | {s1} | {f} |')
open(f'{MD}/hold.md', 'w').write('\n'.join(lines))
print(open(f'{MD}/counts.md').read()); print(open(f'{MD}/idx.md').read()); print(open(f'{MD}/hold.md').read())
print('stale size share', round(C.size_stale.mean(), 3))
