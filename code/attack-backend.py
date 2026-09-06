#!/usr/bin/env python
"""攻击代码 —— Baseline：有损编解码隐匿攻击（Codec Attack）。

原理：真实世界的 DeepFake 语音会经过社交平台压缩、重采样、均衡器、背景
噪声等信道后处理，这些处理在不明显影响内容与可懂度的前提下会破坏反欺骗
检测器所依赖的取证特征（子带量化噪声、相位细节、神经声码器痕迹）。
CodecAttack 模拟 "WAV -> MP3/AAC/Opus -> WAV 16 kHz" 的有损编解码往返
（赛题 §14 A0-1）：先按码率做低通带限，再用 PQMF 风格 DCT 子带量化引入
"带形量化噪声"，最后按编码器内部采样率往返重采样。无 ffmpeg 环境时用该
模拟器忠实复现真实编解码的取证特征破坏方式。

本文件同时提供赛题固定攻击池的另两个 CPU 轻量成员：
    A_noise   噪声叠加（白/粉/棕/起伏噪声，按目标 SNR 缩放）
    A_channel 信道模拟链（混响 + 带宽限制往返重采样 + 随机增益 + μ律压缩）

接口（对齐赛题 §5.4）：
    attack(sample: {"sample_id", "audio": float32 mono, "sample_rate": 16000})
        -> {"audio": float32 mono, "sample_rate": 16000}

用法：
    python attack.py --attack A_codec --output data/attacked.npz
    # 读取基准测试集 spoof 样本，执行攻击并保存攻击后音频
"""

from __future__ import annotations

import argparse
import os
from math import gcd

import numpy as np
from scipy.signal import resample_poly, butter, lfilter, fftconvolve
from scipy.fft import dct, idct

SAMPLE_RATE = 16000


# --------------------------------------------------------------------------- #
# 公共基类（对齐材料 attacks/base.py：tanh 软限幅、NaN 清理）
# --------------------------------------------------------------------------- #
class BaseAttack:
    """攻击接口。attack(sample) -> {audio, sample_rate}，不含任何防御查询。"""

    name: str = "base"
    params: dict = {}

    def __init__(self, **params):
        self.params = {**self.params, **params}
        self.rng = np.random.default_rng(self.params.get("seed", 0))

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        raise NotImplementedError

    def attack(self, sample: dict) -> dict:
        audio = np.asarray(sample["audio"], dtype=np.float32).reshape(-1)
        sr = int(sample.get("sample_rate", SAMPLE_RATE))
        out = self.process(audio, sr)
        out = np.asarray(out, dtype=np.float64).reshape(-1)
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.tanh(out)  # 软限幅（峰值保护）
        return {"audio": out.astype(np.float32), "sample_rate": sr}


# --------------------------------------------------------------------------- #
# A_codec —— 交付 Baseline：有损编解码隐匿攻击
# --------------------------------------------------------------------------- #
def _quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """对称中平量化器。"""
    levels = 2 ** bits
    step = 2.0 / levels
    return np.round(x / step) * step


def _lowpass(x, cutoff, sr, order=6):
    nyq = 0.5 * sr
    b, a = butter(order, cutoff / nyq, btype="low")
    return lfilter(b, a, x)


def _pqmf_quantize(x: np.ndarray, n_bands: int, bits: int) -> np.ndarray:
    """伪 QMF 子带量化：DCT 分帧 -> 按子带量化（高频带更粗）-> 重叠相加。

    引入真实编解码器同类的"带形量化噪声"（材料 codec_attack.py 同款）。
    """
    n = len(x)
    if n == 0:
        return x
    frame, hop = 1024, 512
    window = np.hanning(frame)
    out = np.zeros_like(x)
    norm = np.zeros_like(x)
    i = 0
    while i < n:
        end = min(i + frame, n)
        L = end - i
        seg = x[i:end]
        if L < frame:
            seg = np.pad(seg, (0, frame - L))
        segw = seg * window
        coeffs = dct(segw, type=2, norm="ortho")
        band_size = max(1, len(coeffs) // n_bands)
        q_coeffs = coeffs.copy()
        for b in range(n_bands):
            s = b * band_size
            e = min(len(coeffs), s + band_size)
            if e <= s:
                continue
            band_bits = max(1, bits - (b // max(1, n_bands // 4)))
            step = 2.0 / (2 ** band_bits)
            q_coeffs[s:e] = np.round(coeffs[s:e] / step) * step
        rec = idct(q_coeffs, type=2, norm="ortho")
        out[i:end] += rec[:L] * window[:L]
        norm[i:end] += window[:L] ** 2
        i += hop
    norm[norm == 0] = 1.0
    return out / norm


# [IMPORTANT] Replace this line with real attack/defense code.

# --------------------------------------------------------------------------- #
# A_noise —— 噪声叠加攻击
# --------------------------------------------------------------------------- #
def _generate_noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "pink":
        white = rng.standard_normal(n)
        pink = np.zeros(n)
        rows = 16
        for r in range(rows):
            stride = 1 << r
            idx = np.arange(0, n, stride)
            vals = rng.standard_normal(len(idx))
            pink += np.repeat(vals, stride)[:n]
        return pink / rows
    if kind == "brown":
        return np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
    if kind == "babble":
        white = rng.standard_normal(n)
        b = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        a = np.array([1.0, -0.5, 0.2])
        return lfilter(b, a, white)
    return rng.standard_normal(n)


class NoiseAttack(BaseAttack):
    """噪声叠加攻击（固定池 A_noise）：按目标 SNR 缩放噪声叠加到语音上。"""

    name = "A_noise"
    params = {"kind": "pink", "snr_db": 20.0, "seed": 0}

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        x = audio.astype(np.float64)
        noise = _generate_noise(self.params["kind"], len(x), self.rng)
        sig_p = float(np.mean(x ** 2)) + 1e-12
        noise_p = float(np.mean(noise ** 2)) + 1e-12
        scale = float(np.sqrt(sig_p / noise_p * 10 ** (-self.params["snr_db"] / 10.0)))
        return (x + scale * noise).astype(np.float32)


# --------------------------------------------------------------------------- #
# A_channel —— 信道模拟攻击
# --------------------------------------------------------------------------- #
def _reverb(x, sr, rt60, rng):
    if rt60 <= 0:
        return x
    length = int(sr * rt60)
    if length < 8:
        return x
    decay = np.exp(-3.5 * np.arange(length) / max(length, 1))
    ir = rng.standard_normal(length) * decay
    ir[0] = 1.0
    ir /= (np.linalg.norm(ir) + 1e-9)
    y = fftconvolve(x, ir, mode="full")[:len(x)]
    return 0.7 * x + 0.3 * y


def _mulaw_compress(x, mu=255.0):
    x = np.clip(x, -1.0, 1.0)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


class ChannelAttack(BaseAttack):
    """信道模拟攻击（固定池 A_channel）：混响 + 往返重采样带限 + 随机增益
    + μ律轻度动态压缩，模拟社交平台传输链。"""

    name = "A_channel"
    params = {"rt60": 0.25, "resample_to": 8000, "gain_db": 1.5,
              "compress": True, "seed": 0}

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        x = audio.astype(np.float64)
        x = _reverb(x, sample_rate, self.params["rt60"], self.rng)
        rs = self.params.get("resample_to")
        if rs and rs < sample_rate:
            g1 = gcd(int(sample_rate), int(rs))
            x = resample_poly(x, int(rs) // g1, int(sample_rate) // g1)
            g2 = gcd(int(rs), int(sample_rate))
            x = resample_poly(x, int(sample_rate) // g2, int(rs) // g2)
        g = float(10 ** (self.params["gain_db"] / 20.0))
        if self.rng.random() < 0.5:
            g = 1.0 / g
        x = x * g
        if self.params["compress"]:
            x = _mulaw_compress(x)
        return x.astype(np.float32)


# --------------------------------------------------------------------------- #
# 注册表与统一接口
# --------------------------------------------------------------------------- #
ATTACK_REGISTRY = {
    "A_codec": CodecAttack,
    "A_noise": NoiseAttack,
    "A_channel": ChannelAttack,
}


def build_attack(name: str, **params) -> BaseAttack:
    if name not in ATTACK_REGISTRY:
        raise KeyError(f"Unknown attack '{name}'. Available: {list(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name](**params)


def attack(sample: dict, **params) -> dict:
    """赛题 §5.4 攻击方统一入口：默认使用交付 Baseline（CodecAttack）。"""
    return build_attack("A_codec", **params).attack(sample)


def default_pool() -> list[BaseAttack]:
    """基准固定攻击池 A0 = {A_codec, A_noise, A_channel}（CPU 轻量子集）。"""
    return [build_attack(n) for n in ("A_codec", "A_noise", "A_channel")]


if __name__ == "__main__":
    from dataset import load_benchmark, LABEL_SPOOF

    parser = argparse.ArgumentParser(description="语音隐匿攻击（Baseline: CodecAttack）")
    parser.add_argument("--attack", choices=sorted(ATTACK_REGISTRY), default="A_codec")
    parser.add_argument("--output", default="data/attacked.npz", help="输出路径")
    args = parser.parse_args()

    _, test = load_benchmark()
    atk = build_attack(args.attack)
    idx = np.where(test["labels"] == LABEL_SPOOF)[0]
    out_audios = []
    for i in idx:
        out = atk.attack({"sample_id": test["sample_ids"][i],
                          "audio": test["audios"][i], "sample_rate": SAMPLE_RATE})
        out_audios.append(out["audio"])
    out_audios = np.stack(out_audios).astype(np.float32)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, audios=out_audios,
                        sample_ids=np.array([test["sample_ids"][i] for i in idx]))
    print(f"[attack] {args.attack} 完成：对 {len(idx)} 条测试 spoof 语音执行隐匿，"
          f"输出 {args.output}（shape={out_audios.shape}）")
