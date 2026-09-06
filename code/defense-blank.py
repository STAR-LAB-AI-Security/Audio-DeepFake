class Defense:
    """反欺骗检测器（学生自选算法，不指定特定防御方法）。

    固定接口（test.py 调用契约）：
        __init__(**params)                         # test.py 只传 seed：Defense(seed=seed)
        train(bonafide_samples, spoof_samples, sample_rate=16000) -> self
        defend(request: dict) -> {"spoof_probability": float 0.0~1.0}   # 0=真人, 1=伪造

    可选（模块级懒加载/CLI 使用，不实现则无法缓存模型）：
        save(path) / load(path)

    backend 保留 LFCC 前端工具（lfcc / extract_lfcc_array 等）可用。
    """

    name = "defense"

    def __init__(self, **params):
        self.params = dict(params)

    def train(self, bonafide_samples, spoof_samples, sample_rate=SAMPLE_RATE):
        pass

    def defend(self, request: dict) -> dict:
        pass

    def save(self, path: str):
        pass

    def load(self, path: str):
        pass
