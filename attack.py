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

角色分工（槽位与官方基线互不耦合）：
    ① 槽位（学生提交，契约见 attack-blank.py）：class CodecAttack(BaseAttack)
        被测攻击主体（name="A_codec" 固定、params 类属性、__init__(**params)、
        process(audio, sample_rate) -> np.ndarray），由 test.py 直接构造评测；
        学生可完全自定义实现。
    ② 后端自带官方基线 BaselineCodecAttack（同名有损编解码算法）：
        注册表/固定攻击池 A0 的 A_codec 成员/模块级 attack() 兼容层均指向它，
        与槽位学生代码无关。

用法：
    python attack.py --attack A_codec --output data/attacked.npz
    # 读取基准测试集 spoof 样本，执行官方基线攻击并保存攻击后音频
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


class BaselineCodecAttack(BaseAttack):
    """有损编解码隐匿攻击（官方基线 A0-1，后端自带）。

    默认参数：codec=mp3, bitrate_kbps=64, bits=8, n_bands=32,
    lowpass_hz=8000。步骤：① 低通带限（模拟编解码带宽限制）；
    ② PQMF 子带量化（bitrate 越低、量化位惩罚越大，64 kbps -> 惩罚 2 位）；
    ③ 按编解码器内部采样率往返重采样（mp3@64k -> 16000 Hz，opus -> 16000，
    aac -> 22050）。
    """

    name = "A_codec"
    params = {
        "codec": "mp3",
        "bitrate_kbps": 64,
        "bits": 8,
        "n_bands": 32,
        "lowpass_hz": 8000,
        "seed": 0,
    }

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        codec = self.params["codec"]
        x = audio.astype(np.float64)
        lp = self.params["lowpass_hz"]
        if lp and lp < sample_rate / 2:
            x = _lowpass(x, lp, sample_rate)
        bits = max(2, int(self.params["bits"]) - self._bit_penalty(codec,
                                                                  self.params["bitrate_kbps"]))
        x = _pqmf_quantize(x, self.params["n_bands"], bits)
        if codec == "opus":
            internal = 16000
        elif codec == "aac":
            internal = 22050
        else:  # mp3
            internal = 16000 if self.params["bitrate_kbps"] <= 64 else 22050
        if internal != sample_rate:
            g = gcd(int(sample_rate), int(internal))
            x = resample_poly(x, int(internal) // g, int(sample_rate) // g)
            g2 = gcd(int(internal), int(sample_rate))
            x = resample_poly(x, int(sample_rate) // g2, int(internal) // g2)
        return x.astype(np.float32)

    @staticmethod
    def _bit_penalty(codec: str, bitrate: int) -> int:
        if bitrate <= 32:
            return 3
        if bitrate <= 64:
            return 2
        if bitrate <= 96:
            return 1
        return 0


import os
import shutil
import subprocess
import tempfile

import numpy as np


class CodecAttack(BaseAttack):
    """有损编解码隐匿攻击。

    真实路径: WAV -> MP3/AAC/Opus (ffmpeg) -> WAV 16kHz
    回退路径: 低通 + 降采样 + PQMF 子带量化（模拟有损编解码的频带截断与量化噪声）

    固定接口:
        name = "A_codec"
        __init__(**params)
        process(audio, sample_rate) -> np.ndarray
    """

    name = "A_codec"
    params = {
        "codec": "mp3",        # "mp3" | "aac" | "opus" | "sim"
        "bitrate_kbps": 64,
        "bits": 8,             # 回退模拟路径的子带量化位数
        "n_bands": 32,         # 回退模拟路径的 PQMF 子带数
        "lowpass_hz": 8000,
        "seed": 0,
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._codec = str(self.params.get("codec", "mp3")).lower()
        self._bitrate = int(self.params.get("bitrate_kbps", 64))
        self._bits = int(self.params.get("bits", 8))
        self._n_bands = int(self.params.get("n_bands", 32))
        self._lowpass_hz = float(self.params.get("lowpass_hz", 8000))
        self._rng = np.random.default_rng(int(self.params.get("seed", 0)))
        self._ffmpeg = None if self._codec == "sim" else shutil.which("ffmpeg")

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x.copy()
        n_in = x.size

        if self._ffmpeg is not None and self._try_real_codec(x, sample_rate):
            y = self._transcode_ffmpeg(x, sample_rate)
        else:
            y = self._simulate_codec(x, sample_rate)

        # 严格时长对齐（评测要求 0.95 <= T'/T <= 1.05，这里做到样本级一致）
        if y.size > n_in:
            y = y[:n_in]
        elif y.size < n_in:
            y = np.pad(y, (0, n_in - y.size), mode="reflect" if y.size > 1 else "constant")

        # 轻度去直流，避免引入缓慢漂移被检测器当作伪影
        y = y - np.mean(y)
        return np.asarray(y, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 路径一：真实 ffmpeg 编解码
    # ------------------------------------------------------------------ #
    def _try_real_codec(self, x: np.ndarray, sr: int) -> bool:
        return self._ffmpeg is not None

    def _transcode_ffmpeg(self, x: np.ndarray, sr: int) -> np.ndarray:
        codec_args = {
            "mp3":  ["-c:a", "libmp3lame", "-b:a", f"{self._bitrate}k"],
            "aac":  ["-c:a", "aac", "-b:a", f"{self._bitrate}k"],
            "opus": ["-c:a", "libopus", "-b:a", f"{self._bitrate}k"],
        }.get(self._codec, ["-c:a", "libmp3lame", "-b:a", f"{self._bitrate}k"])

        with tempfile.TemporaryDirectory() as td:
            f_in = os.path.join(td, "in.wav")
            f_mid = os.path.join(td, f"mid.{ 'm4a' if self._codec=='aac' else self._codec }")
            f_out = os.path.join(td, "out.wav")
            self._write_wav(f_in, x, sr)
            try:
                # encode
                subprocess.run(
                    [self._ffmpeg, "-y", "-v", "error", "-i", f_in, *codec_args, f_mid],
                    check=True, capture_output=True, timeout=60,
                )
                # decode back to 16k mono pcm
                subprocess.run(
                    [self._ffmpeg, "-y", "-v", "error", "-i", f_mid,
                     "-ar", str(sr), "-ac", "1", "-f", "wav", f_out],
                    check=True, capture_output=True, timeout=60,
                )
                y, _ = self._read_wav(f_out)
                if y.size > 0:
                    return y.astype(np.float32)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
                pass
        # ffmpeg 失败则退回模拟路径
        return self._simulate_codec(x, sr)

    @staticmethod
    def _write_wav(path: str, x: np.ndarray, sr: int) -> None:
        """免 soundfile/scipy 依赖，写 16-bit PCM WAV。"""
        import struct, wave
        xi = np.clip(x, -1.0, 1.0)
        xi = (xi * 32767.0).astype("<i2")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(xi.tobytes())

    @staticmethod
    def _read_wav(path: str):
        import wave
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            n = w.getnframes()
            raw = w.readframes(n)
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        return x, sr

    # ------------------------------------------------------------------ #
    # 路径二：numpy 模拟有损编解码
    # ------------------------------------------------------------------ #
    def _simulate_codec(self, x: np.ndarray, sr: int) -> np.ndarray:
        y = x.copy()

        # 1) 低通（优先用 backend 的 _lowpass，否则自实现 FFT 截断）
        lp = getattr(self, "_lowpass", None)
        if callable(lp):
            y = np.asarray(lp(y, sr, self._lowpass_hz), dtype=np.float32)
        else:
            y = self._fft_lowpass(y, sr, self._lowpass_hz)

        # 2) 降采样再升回（模拟带宽受限 + 重采样伪影）
        target_sr = int(min(sr, max(4000, 2 * int(self._lowpass_hz))))
        if target_sr < sr:
            y = self._resample(y, sr, target_sr)
            y = self._resample(y, target_sr, sr)

        # 3) PQMF 子带量化（优先用 backend 的 _pqmf_quantize / _quantize）
        pqmf = getattr(self, "_pqmf_quantize", None)
        if callable(pqmf):
            y = np.asarray(pqmf(y, n_bands=self._n_bands, bits=self._bits),
                           dtype=np.float32)
        else:
            y = self._fft_subband_quantize(y, self._n_bands, self._bits)

        return np.asarray(y, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 本地工具（backend 不存在时的兜底实现）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fft_lowpass(x: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
        n = x.size
        if n < 8:
            return x
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        # 6Hz 过渡带，避免吉布斯振铃过强
        mask = freqs <= cutoff
        trans = (freqs > cutoff) & (freqs <= cutoff + 400.0)
        ramp = 0.5 * (1.0 + np.cos(np.pi * (freqs[trans] - cutoff) / 400.0))
        X[trans] *= ramp
        X[~mask & ~trans] = 0.0
        y = np.fft.irfft(X, n=n)
        return y.astype(np.float32)

    @staticmethod
    def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        """线性插值重采样（足够模拟信道重采样，不产生 ringing）。"""
        if sr_in == sr_out or x.size < 2:
            return x
        n_out = int(round(x.size * sr_out / sr_in))
        t_in = np.arange(x.size, dtype=np.float64)
        t_out = np.linspace(0.0, x.size - 1.0, n_out)
        y = np.interp(t_out, t_in, x)
        return y.astype(np.float32)

    @staticmethod
    def _fft_subband_quantize(x: np.ndarray, n_bands: int, bits: int) -> np.ndarray:
        """频域分带 + 每带按峰值比例量化，模拟感知编码的量化噪声。"""
        n = x.size
        if n < 16 or n_bands <= 0 or bits <= 0:
            return x
        X = np.fft.rfft(x)
        mags = np.abs(X)
        # 将 rfft bin 均匀分成 n_bands 个子带
        edges = np.linspace(0, X.size, n_bands + 1).astype(int)
        levels = float(2 ** bits - 1)
        for b in range(n_bands):
            lo, hi = edges[b], edges[b + 1]
            if hi <= lo:
                continue
            peak = mags[lo:hi].max()
            if peak < 1e-8:
                continue
            # 每带独立缩放后量化，模拟 scalefactor + 量化
            band = X[lo:hi] / peak
            band_q = np.round(band * levels) / levels
            # 量化步长带来的失真按感知加权：高频带放宽（保留噪声掩蔽效果）
            keep = 1.0 - 0.5 * (b / max(1, n_bands - 1)) * 0.2
            X[lo:hi] = peak * (keep * band_q + (1 - keep) * band)
        y = np.fft.irfft(X, n=n)
        return y.astype(np.float32)


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
    "A_codec": BaselineCodecAttack,
    "A_noise": NoiseAttack,
    "A_channel": ChannelAttack,
}


def build_attack(name: str, **params) -> BaseAttack:
    if name not in ATTACK_REGISTRY:
        raise KeyError(f"Unknown attack '{name}'. Available: {list(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name](**params)


def attack(sample: dict, **params) -> dict:
    """赛题 §5.4 攻击方统一入口（模块级兼容层）：默认官方基线 BaselineCodecAttack；
    与槽位学生代码无关（学生类由 test.py 直接构造评测）。"""
    return build_attack("A_codec", **params).attack(sample)


def default_pool() -> list[BaseAttack]:
    """官方固定攻击池 A0 = {A_codec, A_noise, A_channel}（CPU 轻量子集），
    全部为后端自带官方攻击；不含槽位学生代码。"""
    return [build_attack(n) for n in ("A_codec", "A_noise", "A_channel")]


if __name__ == "__main__":
    from dataset import load_benchmark, LABEL_SPOOF

    parser = argparse.ArgumentParser(description="语音隐匿攻击（官方基线；槽位攻击见 attack-blank.py）")
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
