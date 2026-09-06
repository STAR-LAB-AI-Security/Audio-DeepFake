#!/usr/bin/env python
"""防御代码 —— Baseline：LFCC-GMM 反欺骗检测器（DeepFake 语音检测）。

原理（对齐赛题 §15 D0-1 与 ASVspoof 2021 官方 LFCC-GMM baseline，
材料 deepguard_backend/defenses/lfcc_gmm.py）：
1. 前端：LFCC（线性频率倒谱系数）——预加重 -> 分帧(30ms/15ms) -> 汉明窗
   -> |FFT|^2 -> 线性三角滤波器组 -> log -> DCT-II，取前 20 阶，再加
   delta 与 delta-delta，得到 60 维逐帧特征；
2. 后端：对 bonafide 帧与 spoof 帧分别训练一个对角协方差高斯混合模型
   （GMM，32 个分量），得到 P(x|bona) 与 P(x|spoof)；
3. 打分：一条语音的对数似然比 LLR = score_bona(x) - score_spoof(x)，
   训练集上标定 LLR 的均值/方差后经 sigmoid 映射为
   spoof_probability ∈ [0,1]（0=真人语音，1=伪造语音）。

接口（对齐赛题 §7.3）：
    defend(request: {"sample_id", "audio": float32 mono, "sample_rate": 16000})
        -> {"spoof_probability": 0.0~1.0}

用法：
    python defense.py                 # 在基准训练集上训练模型并保存
                                      # data/defense_model.pkl，打印测试集
                                      # 干净检测 EER/AUC
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
from scipy.fft import dct
from scipy.signal import lfilter
from sklearn.mixture import GaussianMixture

SAMPLE_RATE = 16000
MODEL_PATH = os.path.join("data", "defense_model.pkl")


# --------------------------------------------------------------------------- #
# LFCC 前端（numpy/scipy 自实现，移植自材料 features/lfcc.py）
# --------------------------------------------------------------------------- #
def _pre_emphasis(sig: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return lfilter([1.0, -coeff], [1.0], sig)


def _framing(sig: np.ndarray, fs: int, win_len: float = 0.03,
             win_hop: float = 0.015):
    frame_length = int(round(win_len * fs))
    frame_step = int(round(win_hop * fs))
    n = len(sig)
    if n <= frame_length:
        frames = np.pad(sig, (0, max(0, frame_length - n)))
        return frames.reshape(1, -1), frame_length
    n_frames = 1 + int(np.ceil((n - frame_length) / frame_step))
    pad_len = (n_frames - 1) * frame_step + frame_length
    padded = np.pad(sig, (0, pad_len - n))
    indices = (np.arange(frame_length)[None, :]
               + np.arange(n_frames)[:, None] * frame_step)
    return padded[indices], frame_length


def _linear_filterbanks(nfilts: int, nfft: int, fs: int,
                        low_freq: float, high_freq: float) -> np.ndarray:
    """线性三角滤波器组（等带宽，与 LFCC 定义一致），形状 (nfilts, nfft//2+1)。"""
    f_min = max(0.0, low_freq)
    f_max = min(fs / 2.0, high_freq) if high_freq else fs / 2.0
    n_bins = nfft // 2 + 1
    bin_freqs = np.linspace(0.0, fs / 2.0, n_bins)
    band_edges = np.linspace(f_min, f_max, nfilts + 2)
    fb = np.zeros((nfilts, n_bins), dtype=np.float64)
    for i in range(nfilts):
        left, center, right = band_edges[i], band_edges[i + 1], band_edges[i + 2]
        l_idx = (bin_freqs >= left) & (bin_freqs <= center)
        if center > left:
            fb[i, l_idx] = (bin_freqs[l_idx] - left) / (center - left)
        r_idx = (bin_freqs >= center) & (bin_freqs <= right)
        if right > center:
            fb[i, r_idx] = (right - bin_freqs[r_idx]) / (right - center)
    energy = fb.sum(axis=1, keepdims=True)
    energy[energy == 0] = 1.0
    return fb / energy  # 单位面积归一，各频带贡献均衡


def _deltas(x: np.ndarray, width: int = 3) -> np.ndarray:
    """时间轴上的一阶差分（帧间），边缘复制填充。"""
    hlen = int(np.floor(width / 2))
    win = list(range(hlen, -hlen - 1, -1))
    pad = np.pad(x, ((hlen, hlen), (0, 0)), mode="edge")
    d = lfilter(win, 1, pad, axis=0)
    return d[hlen * 2:]


def lfcc(sig: np.ndarray, fs: int = 16000, num_ceps: int = 20,
         order_deltas: int = 2, nfilts: int = 70, nfft: int = 512,
         win_len: float = 0.030, win_hop: float = 0.015) -> np.ndarray:
    """计算 LFCC 特征，返回 (num_frames, num_ceps*(1+order_deltas)) float32。"""
    sig = np.asarray(sig, dtype=np.float64).reshape(-1)
    if sig.size == 0:
        return np.zeros((1, num_ceps * (1 + order_deltas)), dtype=np.float32)
    sig = _pre_emphasis(sig)
    frames, frame_length = _framing(sig, fs, win_len, win_hop)
    frames = frames * np.hamming(frame_length)[None, :]
    spec = np.abs(np.fft.rfft(frames, n=nfft)) ** 2
    fb = _linear_filterbanks(nfilts, nfft, fs, 0.0, 0.0)
    fbank = spec @ fb.T
    fbank = np.log10(fbank + 2.220446049250313e-16)
    feats = dct(fbank, type=2, norm="ortho", axis=1)[:, :num_ceps]
    if order_deltas > 0:
        out = [feats]
        for _ in range(order_deltas):
            out.append(_deltas(out[-1]))
        feats = np.concatenate(out, axis=1)
    return feats.astype(np.float32, copy=False)


def extract_lfcc_array(sig: np.ndarray, fs: int = 16000,
                       num_ceps: int = 20, with_delta: bool = True) -> np.ndarray:
    """便捷封装：默认 20 阶 + delta + delta-delta -> 60 维逐帧特征。"""
    return lfcc(sig, fs=fs, num_ceps=num_ceps,
                order_deltas=2 if with_delta else 0)


# --------------------------------------------------------------------------- #
# LFCC-GMM 反欺骗检测器（交付 Baseline，D_GMM）
# --------------------------------------------------------------------------- #
# [IMPORTANT] Replace this line with real attack/defense code.

# --------------------------------------------------------------------------- #
# 统一接口与 CLI
# --------------------------------------------------------------------------- #
_MODEL = None


def _get_model() -> Defense:
    """懒加载：优先读缓存模型；否则在基准训练集上训练并保存。"""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if os.path.exists(MODEL_PATH):
        _MODEL = Defense()
        _MODEL.load(MODEL_PATH)
        return _MODEL
    from dataset import load_benchmark, LABEL_BONAFIDE, LABEL_SPOOF
    train, _ = load_benchmark()
    bona = train["audios"][train["labels"] == LABEL_BONAFIDE]
    spoof = train["audios"][train["labels"] == LABEL_SPOOF]
    _MODEL = Defense().train(list(bona), list(spoof))
    _MODEL.save(MODEL_PATH)
    return _MODEL


def defend(request: dict) -> dict:
    """赛题 §7.3 防御方统一入口（默认使用交付 Baseline：LFCC-GMM）。"""
    return _get_model().defend(request)


def _quick_metrics(bp, sp, threshold=0.5):
    """CLI 用简单检测统计（EER 用 DET 曲线最小 |FRR-FAR| 处近似）。"""
    scores = np.concatenate([bp, sp])
    labels = np.concatenate([np.zeros(len(bp)), np.ones(len(sp))])
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    frr = np.concatenate([np.atleast_1d(0.0), np.cumsum(labels) / max(n_pos, 1)])
    far = np.concatenate([np.atleast_1d(1.0),
                          (n_neg - (np.arange(1, len(labels) + 1) - np.cumsum(labels)))
                          / max(n_neg, 1)])
    idx = int(np.argmin(np.abs(frr - far)))
    eer = float(np.mean([frr[idx], far[idx]]))
    acc = float(np.mean((scores >= threshold) == labels))
    return eer, acc


if __name__ == "__main__":
    from dataset import load_benchmark, LABEL_BONAFIDE, LABEL_SPOOF

    parser = argparse.ArgumentParser(description="LFCC-GMM 反欺骗检测器（Baseline）")
    parser.add_argument("--retrain", action="store_true", help="强制重新训练")
    parser.add_argument("--n-components", type=int, default=None)
    args = parser.parse_args()

    if args.retrain and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    model = Defense(**( {"n_components": args.n_components}
                        if args.n_components else {}))
    if not os.path.exists(MODEL_PATH):
        tr, te = load_benchmark()
        model.train(list(tr["audios"][tr["labels"] == LABEL_BONAFIDE]),
                    list(tr["audios"][tr["labels"] == LABEL_SPOOF]))
        model.save(MODEL_PATH)
        print(f"[defense] 已训练并保存模型: {MODEL_PATH}（耗时 {model.train_seconds:.1f}s）")
    else:
        model.load(MODEL_PATH)
        _, te = load_benchmark()

    bp = np.array([model.defend({"audio": a, "sample_rate": SAMPLE_RATE})["spoof_probability"]
                   for a in te["audios"][te["labels"] == LABEL_BONAFIDE]])
    sp = np.array([model.defend({"audio": a, "sample_rate": SAMPLE_RATE})["spoof_probability"]
                   for a in te["audios"][te["labels"] == LABEL_SPOOF]])
    eer, acc = _quick_metrics(bp, sp)
    print(f"[defense] 测试集干净检测: EER≈{eer:.4f}  Acc@0.5={acc:.4f}  "
          f"mean_p(bona)={bp.mean():.4f}  mean_p(spoof)={sp.mean():.4f}")
