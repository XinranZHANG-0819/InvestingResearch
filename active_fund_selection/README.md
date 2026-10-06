# 主动基金选基研究（2026-09）

报告：《A股主动基金选基研究》（Claude Docs）。本目录是计算代码和结果表格。

- `主动基金选基指标_2026-09.xlsx`：1,408只符合资格基金的全部指标（近3年、近5年、任职以来；另有原报告口径的多因子选股超额和同档名次），每类一张表，另有短名单（含各窗口年化和最大回撤）和类别指数表。
- 数据包另需 T33（中证800价值/成长指数，供多因子模型用）。
- `code/`：从 Google Drive 选基数据包（T34、T35月末净值、T35b、T36b、T37）和指数数据包（T02、T11）计算。
  1. `decode_dl.py` 解码下载文件到 `dl/`；
  2. `panel.py` 生成月度收益面板；
  3. `run_snapshots.py` 计算当前和2010–2023年每年初的快照（`metrics.py` 为核心口径）；
  4. `analysis.py`、`side_checks.py` 为检验；
  5. `mf_compare.py`、`mf_extra.py` 与原报告多因子方法对比（基金/经理、类内/同档）；`pool_vs_within.py` 把一起排的分数拆成挑基金和类别搭配；`short_pool.py` 检验短名单先按月度胜率筛选是否有用；`mf_classify.py`、`mf_classify_win.py` 检验按多因子暴露分类；
  6. `build_outputs.py`、`make_md.py` 生成表格和报告用表（先跑一次 `build_outputs.py`，再跑 `mf_extra.py`，再跑一次 `build_outputs.py` 以并入多因子列）。
