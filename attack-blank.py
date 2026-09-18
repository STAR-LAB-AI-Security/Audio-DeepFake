# attack-blank.py  （官方模板：只提供 CodecAttack，BaseAttack 由评测 backend 提供）
import numpy as np
from scipy.fft import dct, idct
from scipy.signal import butter, lfilter


def _hf_zcr(x, sr):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = len(x)
    if n < 32:
        return 0.2, 0.1
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    tot = float(np.mean(spec ** 2) + 1e-12)
    hf = float(np.mean(spec[freqs >= 4000.0] ** 2))
    zc = float(np.mean(np.abs(np.diff(np.signbit(x)))))
    return hf / tot, zc


def _adapt_params(hf_ratio, zcr):
    hf = min(max(hf_ratio, 0.0), 0.6)
    z = min(max(zcr, 0.0), 0.4)
    score = 0.7 * (hf / 0.35) + 0.3 * (z / 0.15)
    score = min(max(score, 0.0), 2.0)
    lp = 8100.0 - 400.0 * min(score, 1.2)
    bands = int(round(36 + 6 * min(score, 1.0)))
    bits = 8 if score < 1.15 else 7
    tilt = 0.8 + 0.6 * min(score, 1.2)
    return lp, bands, bits, tilt


def _pink(n, rng):
    pink = np.zeros(n)
    for r in range(16):
        stride = 1 << r
        idx = np.arange(0, n, stride)
        pink += np.repeat(rng.standard_normal(len(idx)), stride)[:n]
    return pink / 16.0


def _cep_lift(x, keep_low=24, drop_to=80, mix=0.15):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    spec = np.fft.rfft(x)
    mag = np.abs(spec) + 1e-12
    phase = np.angle(spec)
    cep = np.fft.irfft(np.log(mag), n=n)
    drop_to = min(drop_to, n // 2)
    if drop_to > keep_low:
        cep[keep_low:drop_to] *= 0.15
        if n - drop_to > keep_low:
            cep[-drop_to:-keep_low] *= 0.15
    mag2 = np.exp(np.fft.rfft(cep).real)
    y = np.fft.irfft(mag2 * np.exp(1j * phase), n=n)
    y = (1.0 - mix) * x + mix * y
    rms_x = float(np.sqrt(np.mean(x ** 2) + 1e-12))
    rms_y = float(np.sqrt(np.mean(y ** 2) + 1e-12))
    return y * (rms_x / rms_y)


def _local_lowpass(x, cutoff, sr, order=6):
    nyq = 0.5 * sr
    if cutoff >= nyq or cutoff <= 0:
        return x
    b, a = butter(order, min(float(cutoff) / nyq, 0.99), btype="low")
    return lfilter(b, a, x)


def _local_pqmf(x, n_bands, bits):
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
    def _bit_penalty(codec, bitrate):
        if bitrate <= 32:
            return 3
        if bitrate <= 64:
            return 2
        if bitrate <= 96:
            return 1
        return 0

    def _tilt(self, x, tilt_db):
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

    def process(self, audio, sample_rate):
        n0 = int(len(audio))
        x = np.asarray(audio, dtype=np.float64).reshape(-1)

        hf, zcr = _hf_zcr(x, sample_rate)
        lp, n_bands, bits0, tilt = _adapt_params(hf, zcr)

        lp_fn = globals().get("_lowpass", _local_lowpass)
        pqmf_fn = globals().get("_pqmf_quantize", _local_pqmf)
        if lp and lp < sample_rate / 2:
            x = lp_fn(x, lp, sample_rate)

        br = int(self.params.get("bitrate_kbps", 60))
        bits = max(2, int(bits0) - self._bit_penalty(self.params.get("codec", "mp3"), br))
        x = pqmf_fn(x, n_bands, bits)
        x = self._tilt(x, float(tilt))

        rng = getattr(self, "rng", np.random.default_rng(self.params.get("seed", 0)))
        noise = _pink(len(x), rng)
        sig_p = float(np.mean(x ** 2)) + 1e-12
        noise_p = float(np.mean(noise ** 2)) + 1e-12
        x = x + noise * np.sqrt(sig_p / noise_p * 10 ** (-50.0 / 10.0))
        x = _cep_lift(x, keep_low=24, drop_to=80, mix=0.15)

        if len(x) != n0:
            x = x[:n0] if len(x) > n0 else np.pad(x, (0, n0 - len(x)))
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x.astype(np.float32)
