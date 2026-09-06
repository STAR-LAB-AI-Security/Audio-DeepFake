#!/usr/bin/env bash
# =============================================================================
# 防守方终端脚本 —— 语音隐匿与反欺骗基准（Baseline：LFCC-GMM 反欺骗检测器）
#
# 在新机器上以默认参数运行：
#     pip install -r requirements.txt
#     bash run_defense.sh
#
# 输出：EER_clean / AUC_clean / EER_robust / AUC_robust / FRR / FAR / DefenseScore
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 指定解释器：默认 python，可用 PYTHON 环境变量覆盖
PY="${PYTHON:-python}"

echo ">>> [run_defense] 防守方 Baseline 评测（LFCC-GMM vs 固定攻击池 A0）"
"$PY" test.py --stage defense
