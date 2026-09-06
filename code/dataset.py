#!/usr/bin/env python
"""数据集代码 —— 语音隐匿与反欺骗基准（DeepGuard-Synth 语音反欺骗评测集）。

职责：
1. 程序化生成"类语音"信号：干净谐波语音段（bonafide）与叠加 DeepFake
   伪影的语音段（spoof）。信号模型与赛题材料 deepguard_backend/data_gen.py
   一致（基频谐波堆叠 + 共振峰整形；spoof 额外注入声码器嗡嗡声/子带量化
   噪声/相位不连续等取证伪影），保证在无 GPU、无网络环境下端到端可跑；
2. 首次运行生成 128 条样本并落盘 data/synth_cache.npz，之后直接读取本地
   缓存，划分完全一致、结果可复现；
3. 统一输出规范（基准输入输出规范，采样率 16000 Hz、单声道、float32、
   取值 [-1,1]、时长 3.0 s）：

       sample = {"sample_id": str, "audio": [T] float32 mono,
                 "sample_rate": 16000}
       训练/测试集 = {"audios": [N, 48000] float32, "labels": [N] int64,
                      "sample_ids": [N] str, "attack_ids": [N] str,
                      "speaker_ids": [N] str}

4. 基准固定划分为训练 64 条（bonafide 32 + spoof 32）与测试 64 条
   （bonafide 32 + spoof 32），随机种子 42；测试集全程不参与防御训练，
   模拟"隐藏测试集"。

用法：
    python dataset.py                 # 生成/加载基准数据并打印规模
"""

from __future__ import annotations

import os

import numpy as np
from scipy.signal import lfilter

SAMPLE_RATE = 16000
DURATION = 3.0
N_SAMPLES = int(DURATION * SAMPLE_RATE)  # 48000
CACHE_DIR = "data"

LABEL_BONAFIDE = 0
LABEL_SPOOF = 1


# --------------------------------------------------------------------------- #
# 信号模型（移植自材料 deepguard_backend/data_gen.py，仅将一阶滤波向量化）
# --------------------------------------------------------------------------- #
def _voice_segment(rng, n, f0, formants, sr):
    """一段浊音段：基频谐波堆叠通过共振峰谐振器整形。

    与材料一致：基频加微抖动/颤音，取前 7 次谐波（幅度 1/h），每个共振峰
    用一阶低通（极点 a=exp(-2*pi*bw/sr)）近似，再叠加调制幅度呼气噪声。
    """
    t = np.arange(n) / sr
    vib = 1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t)
    f0_inst = f0 * vib * (1.0 + 0.01 * rng.standard_normal(n))
    phase = 2 * np.pi * np.cumsum(f0_inst) / sr
    src = np.zeros(n)
    for h in range(1, 8):
        src += (1.0 / h) * np.sin(h * phase)
    out = src.copy()
    for _fc, bw, gain in formants:
        # y[n] = b*x[n] + a*y[n-1]  <->  lfilter([b], [1, -a], x)
        a = np.exp(-2 * np.pi * bw / sr)
        b = 1 - a
        out = out + gain * lfilter([b], [1.0, -a], src)
    amp = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    out += 0.02 * amp * rng.standard_normal(n)
    return out


def _generate_utterance(rng, sr=SAMPLE_RATE, duration=DURATION,
                        spoof=False, attack_id="clean"):
    """生成一条语音。spoof=True 时按 attack_id 注入 DeepFake 伪影。"""
    n = int(duration * sr)
    n_seg = int(rng.integers(4, 7))
    bounds = np.sort(rng.choice(np.arange(1, n_seg) * (n // n_seg),
                                size=n_seg - 1, replace=False))
    bounds = np.concatenate([[0], bounds, [n]])
    sig = np.zeros(n, dtype=np.float64)
    formant_sets = [
        [(700, 90, 1.0), (1100, 120, 0.6), (2500, 180, 0.3)],
        [(500, 80, 1.0), (1600, 130, 0.5), (2700, 200, 0.25)],
        [(300, 70, 1.0), (1900, 140, 0.45), (2400, 160, 0.3)],
    ]
    for s in range(n_seg):
        a, b = bounds[s], bounds[s + 1]
        seg_len = b - a
        if seg_len < int(0.08 * sr):
            continue
        f0 = float(rng.uniform(90, 180))
        formants = formant_sets[int(rng.integers(0, len(formant_sets)))]
        seg = _voice_segment(rng, seg_len, f0, formants, sr)
        env = np.ones(seg_len)
        ramp = int(0.02 * sr)
        if ramp > 0 and ramp < seg_len:
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
        sig[a:b] = seg * env
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.8
    if spoof:
        sig = _inject_spoof_artefacts(sig, sr, attack_id, rng)
    return sig.astype(np.float32)


def _inject_spoof_artefacts(sig, sr, attack_id, rng):
    """注入 DeepFake 风格伪影（与材料 data_gen.py 一致）。

    - voc/tts/vc : 声码器嗡嗡声（110 Hz 方波弱分量）+ 梳状滤波强调
    - codec/neural: 子带量化噪声 + 平滑滤波（神经声码器残差）
    - phase       : 随机段符号翻转造成的相位不连续
    - 最后统一做高带频谱轻微平坦化（lfilter([1,-0.3],[1,-0.1])）
    """
    from scipy.signal import fftconvolve

    method = attack_id if attack_id != "clean" else "voc"
    if method in ("voc", "tts", "vc"):
        t = np.arange(len(sig)) / sr
        buzz = np.sign(np.sin(2 * np.pi * 110 * t)) * 0.05
        sig = sig + buzz
        sig = fftconvolve(sig, np.array([1.0, -0.6, 0.3, -0.15]), mode="same")
    elif method in ("codec", "neural"):
        noise = rng.standard_normal(len(sig)) * 0.04
        sig = sig + noise
        sig = fftconvolve(sig, np.array([0.4, 0.3, 0.2, 0.1]), mode="same")
    elif method == "phase":
        n_flips = int(rng.integers(3, 8))
        idx = np.sort(rng.choice(len(sig) - 1, size=n_flips, replace=False))
        idx = np.concatenate([[0], idx, [len(sig)]])
        sign = 1.0
        for k in range(len(idx) - 1):
            sig[idx[k]:idx[k + 1]] *= sign
            sign *= -1
        sig *= 0.7
    sig = lfilter([1.0, -0.3], [1.0, -0.1], sig)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.8
    return sig.astype(np.float32)


# --------------------------------------------------------------------------- #
# 基准数据集生成与划分
# --------------------------------------------------------------------------- #
def generate_dataset(n_bonafide: int = 64, n_spoof: int = 64,
                     seed: int = 42, speakers: int = 4) -> dict:
    """生成 bonafide + spoof 语音集，返回含样本字典列表的 dict。"""
    rng = np.random.default_rng(seed)
    spoof_kinds = ["voc", "tts", "vc", "codec", "neural", "phase"]
    samples = []
    for i in range(n_bonafide):
        sid = f"bona_{i:04d}"
        spk = f"spk{int(rng.integers(0, speakers))}"
        audio = _generate_utterance(rng, spoof=False)
        samples.append({"sample_id": sid, "audio": audio, "label": LABEL_BONAFIDE,
                        "attack_id": "human", "speaker_id": spk})
    for i in range(n_spoof):
        sid = f"spoof_{i:04d}"
        spk = f"spk{int(rng.integers(0, speakers))}"
        attack_id = spoof_kinds[i % len(spoof_kinds)]
        audio = _generate_utterance(rng, spoof=True, attack_id=attack_id)
        samples.append({"sample_id": sid, "audio": audio, "label": LABEL_SPOOF,
                        "attack_id": attack_id, "speaker_id": spk})
    return samples


def _stack(samples):
    """把样本字典列表整理为统一的 {audios, labels, ...} 数组字典。"""
    audios = np.stack([s["audio"] for s in samples]).astype(np.float32)
    labels = np.array([s["label"] for s in samples], dtype=np.int64)
    return {
        "audios": audios,
        "labels": labels,
        "sample_ids": [s["sample_id"] for s in samples],
        "attack_ids": [s["attack_id"] for s in samples],
        "speaker_ids": [s["speaker_id"] for s in samples],
        "sample_rate": SAMPLE_RATE,
    }


def load_benchmark(train_size: int = 32, test_size: int = 32,
                   seed: int = 42, cache_dir: str = CACHE_DIR,
                   use_cache: bool = True) -> tuple[dict, dict]:
    """加载基准训练集与测试集，返回 (train, test) 各为 {audios, labels, ...}。

    优先读取 data/synth_cache.npz 缓存；无缓存时生成并按 seed 分层随机划分
    （保证两集合均覆盖各类伪影与说话人），随后落盘缓存。
    """
    train_path = os.path.join(cache_dir, "synth_cache.npz")
    if use_cache and os.path.exists(train_path):
        d = np.load(train_path, allow_pickle=True)
        train = {k: d[k] for k in ("audios", "labels", "sample_ids",
                                   "attack_ids", "speaker_ids")}
        test = {k: d[k] for k in ("audios", "labels", "sample_ids",
                                  "attack_ids", "speaker_ids")}
        train["sample_ids"] = list(train["sample_ids"])
        test["sample_ids"] = list(test["sample_ids"])
        train["attack_ids"] = list(train["attack_ids"])
        test["attack_ids"] = list(test["attack_ids"])
        train["speaker_ids"] = list(train["speaker_ids"])
        test["speaker_ids"] = list(test["speaker_ids"])
        print(f"[dataset] 命中本地缓存: {train_path}")
        return train, test

    samples = generate_dataset(n_bonafide=train_size + test_size,
                               n_spoof=train_size + test_size, seed=seed)
    bona = [s for s in samples if s["label"] == LABEL_BONAFIDE]
    spoof = [s for s in samples if s["label"] == LABEL_SPOOF]
    rng = np.random.default_rng(seed)
    bona_tr = [bona[i] for i in rng.permutation(len(bona))[:train_size]]
    spoof_tr = [spoof[i] for i in rng.permutation(len(spoof))[:train_size]]
    bona_te = [s for s in bona if s not in bona_tr]
    spoof_te = [s for s in spoof if s not in spoof_tr]
    # 测试集裁剪到 test_size（保证确定性）
    bona_te = bona_te[:test_size]
    spoof_te = spoof_te[:test_size]
    train = _stack(bona_tr + spoof_tr)
    test = _stack(bona_te + spoof_te)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        train_path,
        audios=train["audios"], labels=train["labels"],
        sample_ids=np.array(train["sample_ids"]),
        attack_ids=np.array(train["attack_ids"]),
        speaker_ids=np.array(train["speaker_ids"]),
    )
    print(f"[dataset] 已保存基准数据缓存: {train_path}")
    return train, test


if __name__ == "__main__":
    tr, te = load_benchmark()
    print(f"train.audios: {tr['audios'].shape}  dtype={tr['audios'].dtype}  "
          f"值域 [{tr['audios'].min():.3f}, {tr['audios'].max():.3f}]")
    print(f"train.labels: {tr['labels'].shape}  类别 {sorted(set(tr['labels'].tolist()))}")
    print(f"test.audios:  {te['audios'].shape}  test.labels: {te['labels'].shape}")
    print(f"spoof 伪影类型(train/test): "
          f"{sorted(set(tr['attack_ids']))} / {sorted(set(te['attack_ids']))}")
