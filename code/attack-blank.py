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
