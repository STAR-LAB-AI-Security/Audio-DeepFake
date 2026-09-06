#!/usr/bin/env bash
# =============================================================================
# 攻击方终端脚本 —— 语音隐匿与反欺骗基准（Baseline：CodecAttack 有损编解码隐匿）
#
# 在新机器上以默认参数运行：
#     pip install -r requirements.txt
#     bash run_attack.sh
#
# 输出：EER_clean / AUC_clean / 每攻击 ASR / CS / 合法率 / 保真度 / AttackScore
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# 指定解释器：默认 python，可用 PYTHON 环境变量覆盖
PY="${PYTHON:-python}"

echo ">>> [run_attack] 攻击方 Baseline 评测（CodecAttack 等固定攻击池 vs LFCC-GMM）"
"$PY" test.py --stage attack
