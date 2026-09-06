# 语音隐匿与反欺骗 —— 代码说明

本目录是"语音隐匿与反欺骗"（DeepFake 语音隐匿与反欺骗攻防）题目的标准代码包，
实现攻防双方 Baseline 与统一评测基准（Benchmark）。全部代码仅依赖
`numpy` / `scipy` / `scikit-learn`，在 CPU 上数秒~数十秒即可跑通，无需 GPU
与网络（数据集为程序化合成的轻量语音集）。

## 目录结构

```
code/
├── dataset.py        # 数据集：生成/缓存 DeepGuard-Synth 语音反欺骗评测集
├── attack.py         # 攻击 Baseline：CodecAttack（有损编解码隐匿）+ 固定池
├── defense.py        # 防御 Baseline：LFCC-GMM 反欺骗检测器
├── test.py           # Benchmark 评测：训练→干净检测→攻击→鲁棒检测→评分
├── run_attack.sh     # 攻击方评测入口（python test.py --stage attack）
├── run_defense.sh    # 防守方评测入口（python test.py --stage defense）
├── requirements.txt  # 最小依赖
└── README.md         # 本说明
```

## 快速开始

```bash
pip install -r requirements.txt
bash run_attack.sh      # 攻击方 Baseline 评测
bash run_defense.sh     # 防守方 Baseline 评测
```

指标输出到 `results/metrics.json`（含干净/鲁棒检测 EER、AUC、误报率、每攻击
ASR/CS/合法率/保真度、AttackScore、DefenseScore 等）。

## 接口契约（对齐赛题）

- 攻击方（赛题 §5.4）：`attack(sample) -> {"audio", "sample_rate"}`，
  其中 `sample = {"sample_id", "audio": float32 mono, "sample_rate": 16000}`。
  默认实现为交付 Baseline **CodecAttack**（WAV → MP3/AAC/Opus 类有损编解码
  往返模拟：低通带限 + PQMF 子带量化 + 内部采样率往返重采样）；
  另提供固定攻击池成员 `A_noise`（粉噪叠加，SNR 20 dB）、
  `A_channel`（混响 + 往返重采样 + 增益 + μ律压缩）。
- 防御方（赛题 §7.3）：`defend(request) -> {"spoof_probability": 0~1}`，
  `0.0`=真人语音（bonafide）、`1.0`=伪造语音（spoof）。默认实现为交付
  Baseline **LFCC-GMM**：LFCC（20 阶 + delta + delta-delta，60 维逐帧特征）
  + 双对角协方差 GMM（各 32 分量）+ LLR → sigmoid 概率映射。
- 统一音频格式：16 kHz、单声道、float32、取值 [-1,1]、时长 3.0 s。

## 各文件独立用法

```bash
python dataset.py                       # 生成/加载基准数据并打印规模
python attack.py --attack A_codec       # 对测试集 spoof 执行隐匿攻击并保存
python defense.py                       # 训练/加载模型，打印测试集干净检测统计
python test.py --stage all              # 完整攻防评测（--stage attack|defense|all）
```

## 评测要点

- 固定随机种子（默认 42）、固定数据划分（训练 64 / 测试 64，测试集不参与
  防御训练），结果可复现；
- 攻击成功判定阈值 τ=0.5：`ASR = #(合法 ∧ p_s(x')<0.5) / N_spoof`；
- 合法性裁判：时长比 [0.95,1.05]、STOI≥0.45、SI-SDR≥-5 dB、削波率≤0.10、
  静音比≤0.80，任一不满足则该样本攻击无效（赛题 §6）；
- 最终评分：`AttackScore = 100×CoreASR×Q`；`DefenseScore = 100×[0.5(1-EER_robust)
  +0.25(1-EER_clean)+0.15×AUC_robust+0.10×Efficiency]`。
