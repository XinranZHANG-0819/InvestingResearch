# 主动基金选基研究（2026-09）

报告：《A股主动基金选基研究》（Claude Docs）。本目录是计算代码和结果表格。

- `主动基金选基指标_2026-09.xlsx`：1,408只符合资格基金的全部指标（近3年、近5年、任职以来），每类一张表，另有短名单和类别指数表。
- `code/`：从 Google Drive 选基数据包（T34、T35月末净值、T35b、T36b、T37）和指数数据包（T02、T11）计算。
  1. `decode_dl.py` 解码下载文件到 `dl/`；
  2. `panel.py` 生成月度收益面板；
  3. `run_snapshots.py` 计算当前和2010–2023年每年初的快照（`metrics.py` 为核心口径）；
  4. `analysis.py`、`side_checks.py` 为检验；
  5. `build_outputs.py`、`make_md.py` 生成表格和报告用表。
