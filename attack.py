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


class CodecAttack(BaseAttack):
    """有损编解码隐匿攻击（学生自选算法，无需与 baseline 相同）。

    固定接口（test.py 调用契约）：
        name = "A_codec"                              # 不可改（注册表引用）
        __init__(**params)                            # BaseAttack.__init__(**params) 兜底
        process(audio, sample_rate) -> np.ndarray     # 核心算法，学生实现

    test.py 经 default_pool()/build_attack("A_codec") 构造后调用
    atk.attack({"sample_id","audio","sample_rate"})，内部调用 process(audio, sr)。
    backend 保留 _lowpass / _pqmf_quantize / _quantize 等工具可用。
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

    def __init__(self, **params):
        super().__init__(**params)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """有损编解码隐匿（学生自选算法）。

        参数
        ----
        audio       : float32 mono 语音（16kHz）
        sample_rate : 采样率（16000）

        返回
        ----
        处理后的 np.ndarray（float32 mono，与输入同长或经重采样后同长）。
        外层 BaseAttack.attack(sample) 会自动做 NaN 清理与 tanh 软限幅，
        并包装为 {"audio", "sample_rate"}。
        """
        pass

        from math import gcd

import numpy as np
from scipy.fft import dct, idct
from scipy.signal import butter, lfilter, resample_poly

SAMPLE_RATE = 16000


def _lowpass(x, cutoff, sr, order=6):
    nyq = 0.5 * sr
    if cutoff >= nyq or cutoff <= 0:
        return x
    b, a = butter(order, min(float(cutoff) / nyq, 0.99), btype="low")
    return lfilter(b, a, x)


def _pqmf_quantize(x, n_bands, bits):
    n = len(x)
    if n == 0:
        return x
    frame, hop = 1024, 512
    window = np.hanning(frame)
    out = np.zeros_like(x, dtype=np.float64)
    norm = np.zeros_like(x, dtype=np.float64)
    i = 0
    while i < n:
        end = min(i + frame, n)
        L = end - i
        seg = x[i:end]
        if L < frame:
            seg = np.pad(seg, (0, frame - L))
        coeffs = dct(seg * window, type=2, norm="ortho")
        band_size = max(1, len(coeffs) // int(n_bands))
        q_coeffs = coeffs.copy()
        for b in range(int(n_bands)):
            s = b * band_size
            e = min(len(coeffs), s + band_size)
            if e <= s:
                continue
            band_bits = max(1, int(bits) - (b // max(1, int(n_bands) // 4)))
            step = 2.0 / (2 ** band_bits)
            q_coeffs[s:e] = np.round(coeffs[s:e] / step) * step
        rec = idct(q_coeffs, type=2, norm="ortho")
        out[i:end] += rec[:L] * window[:L]
        norm[i:end] += window[:L] ** 2
        i += hop
    norm[norm == 0] = 1.0
    return out / norm


def _hf_zcr(x, sr):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 32:
        return 0.2, 0.1
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    tot = float(np.mean(spec ** 2) + 1e-12)
    hf = float(np.mean(spec[freqs >= 4000.0] ** 2))
    hf_ratio = hf / tot
    zc = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    return hf_ratio, zc


def _adapt_params(hf_ratio, zcr):
    # 35分中心：lp=7900, bits=8, bands=40, tilt=1.2
    # 高频多 / 过零高 → 稍多压一点；已经很闷 → 少动
    hf = min(max(hf_ratio, 0.0), 0.6)
    z = min(max(zcr, 0.0), 0.4)
    score = 0.7 * (hf / 0.35) + 0.3 * (z / 0.15)
    score = min(max(score, 0.0), 2.0)
    lp = 8100.0 - 400.0 * min(score, 1.2)
    bands = int(round(36 + 6 * min(score, 1.0)))
    bits = 8 if score < 1.15 else 7
    tilt = 0.8 + 0.6 * min(score, 1.2)
    return lp, bands, bits, tilt


def _cep_lift(x, sr, keep_low=24, drop_to=80, mix=0.15):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    spec = np.fft.rfft(x)
    mag = np.abs(spec) + 1e-12
    phase = np.angle(spec)
    cep = np.fft.irfft(np.log(mag), n=n)
    cep2 = cep.copy()
    drop_to = min(drop_to, n // 2)
    if drop_to > keep_low:
        cep2[keep_low:drop_to] *= 0.15
        if n - drop_to > keep_low:
            cep2[-drop_to:-keep_low] *= 0.15
    mag2 = np.exp(np.fft.rfft(cep2).real)
    y = np.fft.irfft(mag2 * np.exp(1j * phase), n=n)
    y = (1.0 - mix) * x + mix * y
    rms_x = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    rms_y = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    return y * (rms_x / rms_y)



def _generate_noise(kind, n, rng):
    kind = (kind or "pink").lower()
    if kind == "pink":
        pink = np.zeros(n)
        for r in range(16):
            stride = 1 << r
            idx = np.arange(0, n, stride)
            vals = rng.standard_normal(len(idx))
            pink += np.repeat(vals, stride)[:n]
        return pink / 16.0
    if kind == "brown":
        return np.cumsum(rng.standard_normal(n)) / np.sqrt(n)
    return rng.standard_normal(n)


class BaseAttack(object):
    name = "base"
    params = {}

    def __init__(self, **params):
        self.params = {**getattr(self.__class__, "params", {}), **params}
        self.rng = np.random.default_rng(self.params.get("seed", 0))

    def process(self, audio, sample_rate):
        raise NotImplementedError

    def attack(self, sample):
        audio = np.asarray(sample["audio"], dtype=np.float32).reshape(-1)
        sr = int(sample.get("sample_rate", SAMPLE_RATE))
        out = np.asarray(self.process(audio, sr), dtype=np.float64).reshape(-1)
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if abs(len(out) - len(audio)) > max(1, int(0.05 * len(audio))):
            if len(out) > len(audio):
                out = out[:len(audio)]
            else:
                out = np.pad(out, (0, len(audio) - len(out)))
        out = np.tanh(out)
        return {"audio": out.astype(np.float32), "sample_rate": sr}


class CodecAttack(BaseAttack):
    """答辩展示版：35分骨架 + 自适应 + 50dB粉噪声 + 倒谱。"""

    name = "A_codec"
    params = {
        "codec": "mp3",
        "bitrate_kbps": 60,
        "bits": 8,
        "n_bands": 40,
        "lowpass_hz": 7900,
        "tilt_db": 1.2,
        "seed": 0,
    }

    def __init__(self, **params):
        super().__init__(**params)

    @staticmethod
    def _match_len(x, n):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if len(x) == n:
            return x
        if len(x) > n:
            return x[:n]
        return np.pad(x, (0, n - len(x)))

    def _spectral_tilt(self, x, tilt_db):
        if abs(tilt_db) < 1e-6:
            return x
        g = 10.0 ** (float(tilt_db) / 20.0)
        c = (g - 1.0) * 0.15
        y = np.empty_like(x, dtype=np.float64)
        y[0] = x[0]
        y[1:] = x[1:] + c * (x[1:] - x[:-1])
        rms_x = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        rms_y = float(np.sqrt(np.mean(y ** 2) + 1e-12))
        return y * (rms_x / rms_y)

    @staticmethod
    def _bit_penalty(codec, bitrate):
        if bitrate <= 32:
            return 3
        if bitrate <= 64:
            return 2
        if bitrate <= 96:
            return 1
        return 0

    def process(self, audio, sample_rate):
        codec = self.params.get("codec", "mp3")
        n0 = int(len(audio))
        x = np.asarray(audio, dtype=np.float64).reshape(-1)

        hf, zcr = _hf_zcr(x, sample_rate)
        lp, n_bands, bits0, tilt = _adapt_params(hf, zcr)
        if lp < sample_rate / 2:
            x = _lowpass(x, lp, sample_rate)

        br = int(self.params.get("bitrate_kbps", 60))
        bits = max(2, int(bits0) - self._bit_penalty(codec, br))
        x = _pqmf_quantize(x, n_bands, bits)

        if codec == "opus":
            internal = 16000
        elif codec == "aac":
            internal = 22050
        else:
            internal = 16000 if br <= 64 else 22050

        if internal != sample_rate:
            g = gcd(int(sample_rate), int(internal))
            x = resample_poly(x, int(internal) // g, int(sample_rate) // g)
            g2 = gcd(int(internal), int(sample_rate))
            x = resample_poly(x, int(sample_rate) // g2, int(internal) // g2)

        x = self._spectral_tilt(x, float(tilt))
        noise = _generate_noise("pink", len(x), self.rng)
        sig_p = float(np.mean(x ** 2)) + 1e-12
        noise_p = float(np.mean(noise ** 2)) + 1e-12
        scale = float(np.sqrt(sig_p / noise_p * 10 ** (-50.0 / 10.0)))
        x = x + scale * noise
        x = _cep_lift(x, sample_rate, keep_low=24, drop_to=80, mix=0.15)
        x = self._match_len(x, n0)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.astype(np.float32)


class NoiseAttack(BaseAttack):
    name = "A_noise"
    params = {"kind": "pink", "snr_db": 28.0, "seed": 0}

    def process(self, audio, sample_rate):
        x = np.asarray(audio, dtype=np.float64).reshape(-1)
        noise = _generate_noise(self.params.get("kind", "pink"), len(x), self.rng)
        sig_p = float(np.mean(x ** 2)) + 1e-12
        noise_p = float(np.mean(noise ** 2)) + 1e-12
        scale = float(np.sqrt(sig_p / noise_p * 10 ** (-float(self.params.get("snr_db", 28.0)) / 10.0)))
        return (x + scale * noise).astype(np.float32)


def build_attack(name="A_codec", **params):
    table = {"A_codec": CodecAttack, "A_noise": NoiseAttack}
    return table.get(name, CodecAttack)(**params)


def default_pool():
    return {"A_codec": CodecAttack, "A_noise": NoiseAttack}


def attack(sample):
    return CodecAttack().attack(sample)


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
