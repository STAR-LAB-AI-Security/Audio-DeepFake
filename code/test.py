#!/usr/bin/env python
"""测试代码 —— 语音隐匿与反欺骗基准（Benchmark）指标计算。

流程（与基准文档一致）：
    防御训练  : 在训练集（bonafide 32 + spoof 32）上训练 LFCC-GMM 检测器
    干净检测  : 测试集原始 bonafide + spoof -> EER_clean / AUC_clean / FRR / FAR
    攻击阶段  : 对测试集 spoof 执行固定攻击池 A0 = {A_codec, A_noise, A_channel}
                -> 保真度裁判判定合法攻击 -> 防御打分 -> 每攻击 ASR / CS /
                合法率 / 保真度 / AttackScore
    鲁棒评测  : 测试集原始 bonafide + 全部攻击后 spoof -> EER_robust / AUC_robust
                -> DefenseScore

指标公式（与基准文档一致，p_s 为防御输出的 spoof_probability，τ=0.5）：
    EER  = 使 FRR 与 FAR 相等（DET 曲线上 |FRR-FAR| 最小处取平均）的错误率
    AUC  = ROC 曲线下面积（Mann-Whitney U 秩统计）
    ASR  = #(valid ∧ p_s(x') < τ) / N_spoof
    CS   = mean_i max(0, p_s(x_i) - p_s(x_i'))
    Q    = 质量系数（STOI/SI-SDR/时长/削波/内容保持 加权）
    AttackScore  = 100 * CoreASR * Q
    DefenseScore = 100 * [0.5(1-EER_robust) + 0.25(1-EER_clean)
                          + 0.15*AUC_robust + 0.10*Efficiency]

用法：
    python test.py --stage attack      # 攻击方 Baseline 评测（默认）
    python test.py --stage defense     # 防守方 Baseline 评测
    python test.py --stage all         # 完整攻防评测
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.signal import stft

from attack import build_attack, default_pool
from dataset import load_benchmark, SAMPLE_RATE, LABEL_BONAFIDE, LABEL_SPOOF
from defense import Defense

THRESHOLD = 0.5  # 攻击成功判定阈值 τ（赛题 §八）


# =========================================================================== #
# 一、检测指标（移植自材料 metrics/detection.py，即 ASVspoof 2021 eval_metrics）
# =========================================================================== #
def compute_det_curve(target_scores, nontarget_scores):
    """DET 曲线。target=bonafide（正类），返回 (FRR, FAR, thresholds)。"""
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size),
                             np.zeros(nontarget_scores.size)))
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = (nontarget_scores.size
                            - (np.arange(1, n_scores + 1) - tar_trial_sums))
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1),
                          nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds


def compute_eer(bonafide_scores, spoof_scores):
    """EER（ASVspoof 约定：高分=更像真人，这里分数为 1-spoof_probability）。"""
    frr, far, thresholds = compute_det_curve(bonafide_scores, spoof_scores)
    min_index = int(np.argmin(np.abs(frr - far)))
    eer = float(np.mean((frr[min_index], far[min_index])))
    return eer, float(thresholds[min_index])


def auc_from_spoof_prob(bonafide_prob, spoof_prob):
    """ROC-AUC（spoof 为正类、高分=更像伪造），Mann-Whitney U 秩统计。"""
    bonafide_prob = np.asarray(bonafide_prob, dtype=np.float64)
    spoof_prob = np.asarray(spoof_prob, dtype=np.float64)
    scores = np.concatenate([spoof_prob, bonafide_prob])
    labels = np.concatenate([np.ones(len(spoof_prob)), np.zeros(len(bonafide_prob))])
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):  # 并列分数取平均秩
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def fpr_fnr_at_threshold(bonafide_prob, spoof_prob, threshold=THRESHOLD):
    """固定阈值 τ 下的误报率：
    FRR（bonafide→spoof）= P(p_s ≥ τ | bonafide)
    FAR（spoof→bonafide）= P(p_s <  τ | spoof)
    """
    bonafide_prob = np.asarray(bonafide_prob, dtype=np.float64)
    spoof_prob = np.asarray(spoof_prob, dtype=np.float64)
    frr = float(np.mean(bonafide_prob >= threshold))
    far = float(np.mean(spoof_prob < threshold))
    return {"FRR": frr, "FAR": far, "threshold": threshold}


def all_detection_metrics(bonafide_prob, spoof_prob, threshold=THRESHOLD):
    """完整防御指标束：EER / AUC / FRR / FAR（spoof_probability 约定）。"""
    eer, eer_thr = compute_eer(1.0 - np.asarray(bonafide_prob),
                               1.0 - np.asarray(spoof_prob))
    auc = auc_from_spoof_prob(bonafide_prob, spoof_prob)
    fr_fn = fpr_fnr_at_threshold(bonafide_prob, spoof_prob, threshold)
    acc = float(np.mean((np.concatenate([bonafide_prob, spoof_prob]) >= threshold)
                        == np.concatenate([np.zeros(len(bonafide_prob)),
                                           np.ones(len(spoof_prob))])))
    return {
        "EER": eer,
        "EER_threshold": float(1.0 - eer_thr),  # 换算回 spoof_probability 阈值
        "AUC": auc,
        "Accuracy@0.5": acc,
        "FRR": fr_fn["FRR"],
        "FAR": fr_fn["FAR"],
        "threshold": threshold,
        "n_bonafide": int(np.asarray(bonafide_prob).size),
        "n_spoof": int(np.asarray(spoof_prob).size),
    }


# =========================================================================== #
# 二、保真度指标（移植自材料 metrics/fidelity.py，SI-SDR / STOI 均自实现）
# =========================================================================== #
def si_sdr(ref, est):
    """尺度不变信噪比 SI-SDR（Le Roux et al., 2019），单位 dB。"""
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    est = np.asarray(est, dtype=np.float64).reshape(-1)
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    eps = 1e-8
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = float(np.dot(est, ref) / (np.dot(ref, ref) + eps))
    target = alpha * ref
    noise = est - target
    return float(10.0 * np.log10(np.dot(target, target)
                                 / (np.dot(noise, noise) + eps) + eps))


def _third_octave_edges(fmin, fmax, n):
    ratio = 2.0 ** (1.0 / 3.0)
    edges = [fmin]
    f = fmin
    while f < fmax and len(edges) <= n + 1:
        f *= ratio
        edges.append(min(f, fmax))
    return np.array(edges[:n + 1])


def _stoi_numpy(ref, est, fs):
    """窄带 STOI 近似（Taal et al., 2011）：1/3 倍频程带能量 + 滑动窗相关性。"""
    nfft = 512
    frame_shift = int(0.01 * fs)
    n_bands = 15
    j_sample = int(0.26 * fs / frame_shift)

    f, _, zref = stft(ref, fs=fs, window="hann", nperseg=nfft,
                      noverlap=nfft - frame_shift)
    _, _, zest = stft(est, fs=fs, window="hann", nperseg=nfft,
                      noverlap=nfft - frame_shift)
    n_frames = min(zref.shape[1], zest.shape[1])
    zref, zest = zref[:, :n_frames], zest[:, :n_frames]

    band_edges = _third_octave_edges(150.0, min(3800.0, fs / 2.0), n_bands)
    band_idx = []
    for k in range(len(band_edges) - 1):
        lo = int(np.searchsorted(f, band_edges[k]))
        hi = int(np.searchsorted(f, band_edges[k + 1]))
        if hi <= lo:
            hi = lo + 1
        band_idx.append((lo, min(hi, len(f))))
    band_idx = [(lo, hi) for lo, hi in band_idx if hi > lo]

    x_ref = np.zeros((len(band_idx), n_frames))
    x_est = np.zeros((len(band_idx), n_frames))
    for bi, (lo, hi) in enumerate(band_idx):
        x_ref[bi] = np.sqrt(np.sum(np.abs(zref[lo:hi]) ** 2, axis=0))
        x_est[bi] = np.sqrt(np.sum(np.abs(zest[lo:hi]) ** 2, axis=0))

    win = 15
    d = np.zeros((len(band_idx), max(0, n_frames - j_sample + 1)))
    for bi in range(len(band_idx)):
        for m in range(d.shape[1]):
            seg_ref = x_ref[bi, m:m + j_sample]
            seg_est = x_est[bi, m:m + j_sample]
            sigma = np.sqrt(np.mean(seg_ref ** 2)) + 1e-9
            seg_ref_n = seg_ref / sigma
            seg_est_n = np.clip(seg_est / sigma, -np.sqrt(10.0), np.sqrt(10.0))
            num = np.sum(seg_ref_n * seg_est_n)
            den = (np.linalg.norm(seg_ref_n) * np.linalg.norm(seg_est_n) + 1e-12)
            d[bi, m] = num / den
    if d.size == 0:
        return float("nan")
    return float(np.clip(np.mean(d), -1.0, 1.0))


def stoi(ref, est, fs=SAMPLE_RATE):
    """短时客观可懂度 STOI，取值约 [0,1]，越高越保真。"""
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    est = np.asarray(est, dtype=np.float64).reshape(-1)
    n = min(len(ref), len(est))
    return _stoi_numpy(ref[:n], est[:n], fs)


def rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x ** 2) + 1e-12)) if x.size else 0.0


def clipping_ratio(x, thr=0.99):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= thr))


def silence_ratio(x, fs=SAMPLE_RATE, frame_len=0.03, rms_thr=0.01):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    fl = int(frame_len * fs)
    n_frames = max(1, len(x) // fl)
    frames = x[:n_frames * fl].reshape(n_frames, fl)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return float(np.mean(frame_rms < rms_thr))


def duration_ratio(ref, est):
    return float(len(np.asarray(est)) / max(1, len(np.asarray(ref))))


def fidelity_bundle(ref, est, fs=SAMPLE_RATE):
    """(原始, 攻击后) 语音对的保真度指标束。"""
    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    est = np.asarray(est, dtype=np.float32).reshape(-1)
    return {
        "SI_SDR": si_sdr(ref, est),
        "STOI": stoi(ref, est, fs),
        "RMS_ratio": float(rms(est) / (rms(ref) + 1e-12)),
        "clipping_ratio_est": clipping_ratio(est),
        "silence_ratio_est": silence_ratio(est, fs),
        "duration_ratio": duration_ratio(ref, est),
    }


# =========================================================================== #
# 三、合法攻击裁判（赛题 §6：格式/时长/质量/内容保持）
# =========================================================================== #
class FidelityJudge:
    """判定合法攻击：所有硬性阈值满足才 valid=1（赛题 §6.1-6.4）。

    默认阈值（与材料 config.yaml 一致）：时长比 [0.95,1.05]、STOI ≥ 0.45、
    SI-SDR ≥ -5 dB、削波率 ≤ 0.10、静音比 ≤ 0.80。
    """

    def __init__(self, duration_min=0.95, duration_max=1.05,
                 stoi_min=0.45, si_sdr_min=-5.0,
                 clipping_max=0.10, silence_max=0.80):
        self.duration_min = duration_min
        self.duration_max = duration_max
        self.stoi_min = stoi_min
        self.si_sdr_min = si_sdr_min
        self.clipping_max = clipping_max
        self.silence_max = silence_max

    def judge(self, ref_audio, est_audio, sample_rate=SAMPLE_RATE) -> dict:
        fid = fidelity_bundle(ref_audio, est_audio, sample_rate)
        reasons = []
        dur = fid["duration_ratio"]
        if not (self.duration_min <= dur <= self.duration_max):
            reasons.append(f"duration_ratio_{dur:.2f}")
        if fid["clipping_ratio_est"] > self.clipping_max:
            reasons.append("severe_clipping")
        if fid["silence_ratio_est"] > self.silence_max:
            reasons.append("excessive_silence")
        if np.isfinite(fid["STOI"]) and fid["STOI"] < self.stoi_min:
            reasons.append(f"stoi_low_{fid['STOI']:.2f}")
        if np.isfinite(fid["SI_SDR"]) and fid["SI_SDR"] < self.si_sdr_min:
            reasons.append(f"sdr_low_{fid['SI_SDR']:.2f}")
        return {"valid": len(reasons) == 0, "reasons": reasons,
                "fidelity": fid}


# =========================================================================== #
# 四、攻击指标与最终评分（赛题 §8 §11 §12，移植自材料 metrics/scoring.py）
# =========================================================================== #
def attack_success_rate(p_before, p_after, valid, threshold=THRESHOLD):
    """ASR(A,D) = #(valid ∧ p_s(x') < τ) / N_spoof。"""
    p_after = np.asarray(p_after, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    n = len(p_after)
    if n == 0:
        return float("nan")
    success = valid & (p_after < threshold)
    return float(np.mean(success))


def confidence_suppression(p_before, p_after):
    """CS(A,D) = mean_i max(0, p_s(x_i) - p_s(x_i'))。"""
    p_before = np.asarray(p_before, dtype=np.float64)
    p_after = np.asarray(p_after, dtype=np.float64)
    if len(p_before) == 0:
        return float("nan")
    return float(np.mean(np.maximum(0.0, p_before - p_after)))


def quality_coefficient(fid_agg: dict) -> float:
    """Q(a) ∈ [0,1]：STOI/SI-SDR/时长/削波/内容保持 的加权组合（赛题 §11.3）。

    无 ASR 转写时内容保持轴用 STOI 代理（与材料一致）。
    """
    thr = {"STOI_min": 0.45, "SI_SDR_min": -5.0,
           "duration_min": 0.95, "duration_max": 1.05, "clipping_max": 0.1}
    stoi_q = float(np.clip((fid_agg.get("STOI", 0.0) - thr["STOI_min"])
                           / (1.0 - thr["STOI_min"]), 0.0, 1.0))
    sdr_q = float(np.clip((fid_agg.get("SI_SDR", 0.0) - thr["SI_SDR_min"])
                          / (20.0 - thr["SI_SDR_min"]), 0.0, 1.0))
    dur = fid_agg.get("duration_ratio", 1.0)
    dur_q = 1.0 if thr["duration_min"] <= dur <= thr["duration_max"] else 0.0
    clip_q = float(np.clip(1.0 - fid_agg.get("clipping_ratio_est", 0.0)
                           / max(thr["clipping_max"], 1e-6), 0.0, 1.0))
    content_q = stoi_q  # 无 ASR 时的内容保持代理
    q = (0.35 * stoi_q + 0.25 * sdr_q + 0.25 * content_q
         + 0.10 * clip_q + 0.05 * dur_q)
    return float(np.clip(q, 0.0, 1.0))


def attack_score(core_asr, q):
    """AttackScore = 100 * CoreASR * Q（赛题 §11.4）。"""
    if np.isnan(core_asr):
        return 0.0
    return float(100.0 * core_asr * float(np.clip(q, 0.0, 1.0)))


def efficiency(model_size_mb, avg_inference_ms,
               ref_size=100.0, ref_time=200.0):
    """Efficiency ∈ [0,1]：模型越小、推理越快越接近 1。"""
    s = float(np.clip(ref_size / (model_size_mb + 1e-6), 0.0, 1.0))
    t = float(np.clip(ref_time / (avg_inference_ms + 1e-6), 0.0, 1.0))
    return float(np.clip(0.5 * t + 0.5 * s, 0.0, 1.0))


def defense_score(eer_robust, eer_clean, auc_robust, eff=1.0):
    """DefenseScore = 100*[0.5(1-EER_robust)+0.25(1-EER_clean)
    +0.15*RobustAUC+0.10*Efficiency]（赛题 §十二）。"""
    robust_score = 1.0 - eer_robust
    clean_score = 1.0 - eer_clean
    auc = float(np.clip(auc_robust if not np.isnan(auc_robust) else 0.5, 0.0, 1.0))
    eff = float(np.clip(eff, 0.0, 1.0))
    return float(100.0 * (0.50 * robust_score + 0.25 * clean_score
                          + 0.15 * auc + 0.10 * eff))


# =========================================================================== #
# 五、评测主流程
# =========================================================================== #
def _defend_many(model, audios):
    return np.array([model.defend({"audio": a, "sample_rate": SAMPLE_RATE})["spoof_probability"]
                     for a in audios], dtype=np.float64)


def evaluate(stage: str = "attack", seed: int = 42, threshold: float = THRESHOLD,
             output_dir: str = "results"):
    train, test = load_benchmark(seed=seed)
    t0 = time.time()

    # ---------------- 1) 训练防御（交付 Baseline：LFCC-GMM） ----------------
    print("\n[评测] 训练 LFCC-GMM 反欺骗检测器 ...")
    model = Defense(seed=seed).train(
        list(train["audios"][train["labels"] == LABEL_BONAFIDE]),
        list(train["audios"][train["labels"] == LABEL_SPOOF]))
    print(f"[评测] 训练完成（耗时 {model.train_seconds:.1f}s）")

    # ---------------- 2) 干净检测（测试集原始语音） ----------------
    bona_idx = np.where(test["labels"] == LABEL_BONAFIDE)[0]
    spoof_idx = np.where(test["labels"] == LABEL_SPOOF)[0]
    t_inf = time.time()
    bp = _defend_many(model, test["audios"][bona_idx])
    sp = _defend_many(model, test["audios"][spoof_idx])
    inf_sec = time.time() - t_inf
    clean = all_detection_metrics(bp, sp, threshold)
    print(f"[评测] 干净检测: EER={clean['EER']:.4f}  AUC={clean['AUC']:.4f}  "
          f"Acc@0.5={clean['Accuracy@0.5']:.4f}  FRR={clean['FRR']:.4f}  "
          f"FAR={clean['FAR']:.4f}")

    # ---------------- 3) 攻击池 A0 = {A_codec, A_noise, A_channel} ----------------
    judge = FidelityJudge()
    attacks = default_pool()
    per_attack = []
    all_p_after = []   # 全部攻击后 spoof 的 p_s（鲁棒评测用）
    all_valid = []
    for atk in attacks:
        t1 = time.time()
        p_after_list, valid_list, fid_list = [], [], []
        for i in spoof_idx:
            out = atk.attack({"sample_id": test["sample_ids"][i],
                              "audio": test["audios"][i],
                              "sample_rate": SAMPLE_RATE})
            verdict = judge.judge(test["audios"][i], out["audio"])
            p = model.defend({"audio": out["audio"], "sample_rate": SAMPLE_RATE})["spoof_probability"]
            p_after_list.append(p)
            valid_list.append(int(verdict["valid"]))
            fid_list.append(verdict["fidelity"])
        p_after = np.array(p_after_list, dtype=np.float64)
        valid = np.array(valid_list, dtype=bool)
        all_p_after.append(p_after)
        all_valid.append(valid)

        asr = attack_success_rate(sp, p_after, valid, threshold)
        cs = confidence_suppression(sp, p_after)
        valid_rate = float(np.mean(valid))
        fid_agg = {k: float(np.mean([f[k] for f in fid_list if np.isfinite(f[k])]))
                   for k in ("SI_SDR", "STOI", "duration_ratio",
                             "clipping_ratio_est", "silence_ratio_est")}
        q = quality_coefficient(fid_agg)
        robust = all_detection_metrics(bp, p_after, threshold)
        core_asr = asr  # CoreASR = ASR(a, D0)，D0={D_GMM}（材料后端同以固定池近似隐藏池）
        per_attack.append({
            "attack": atk.name,
            "ASR": asr,
            "CS": cs,
            "valid_attack_rate": valid_rate,
            "mean_p_before": float(np.mean(sp)),
            "mean_p_after": float(np.mean(p_after)),
            "EER_robust": robust["EER"],
            "AUC_robust": robust["AUC"],
            "fidelity": fid_agg,
            "Q": q,
            "CoreASR": core_asr,
            "AttackScore": attack_score(core_asr, q),
            "wall_time_s": round(time.time() - t1, 2),
        })
        print(f"[评测] {atk.name}: ASR={asr:.4f}  CS={cs:.4f}  "
              f"valid_rate={valid_rate:.3f}  Q={q:.4f}  "
              f"AttackScore={per_attack[-1]['AttackScore']:.2f}  "
              f"EER_robust={robust['EER']:.4f}  ({time.time()-t1:.1f}s)")

    # ---------------- 4) 鲁棒检测与最终评分 ----------------
    pooled_p_after = np.concatenate(all_p_after)
    robust = all_detection_metrics(bp, pooled_p_after, threshold)

    n_params = 0
    for g in (model._gmm_bona, model._gmm_spoof):
        n_params += int(g.means_.size + g.covariances_.size + g.weights_.size)
    model_size_mb = n_params * 8 / 1e6
    avg_inf_ms = inf_sec / (len(bona_idx) + len(spoof_idx)) * 1000.0
    eff = efficiency(model_size_mb, avg_inf_ms)
    d_score = defense_score(robust["EER"], clean["EER"], robust["AUC"], eff)

    result = {
        "benchmark": "语音隐匿与反欺骗",
        "stage": stage,
        "seed": seed,
        "threshold": threshold,
        "dataset": {"train": int(len(train["labels"])),
                    "test": int(len(test["labels"])),
                    "n_bonafide_test": int(len(bona_idx)),
                    "n_spoof_test": int(len(spoof_idx)),
                    "sample_rate": SAMPLE_RATE},
        "defense": {"name": model.name,
                    "n_components": model.params["n_components"],
                    "num_ceps": model.params["num_ceps"],
                    "feat_dim": model.params["num_ceps"] * 3,
                    "train_seconds": round(model.train_seconds, 2),
                    "model_size_mb": round(model_size_mb, 4),
                    "avg_inference_ms": round(avg_inf_ms, 3),
                    "Efficiency": eff},
        "clean": clean,
        "robust": robust,
        "per_attack": per_attack,
        "scores": {
            "CoreASR": float(np.mean([p["CoreASR"] for p in per_attack])),
            "AttackScore_mean": float(np.mean([p["AttackScore"] for p in per_attack])),
            "DefenseScore": d_score,
        },
        "wall_time_s": round(time.time() - t0, 1),
    }
    return result


def print_report(res: dict):
    print("\n" + "=" * 68)
    print(f"基准: {res['benchmark']}  |  阶段: {res['stage']}")
    print("=" * 68)
    print(f"数据: 训练 {res['dataset']['train']} 条 / 测试 {res['dataset']['test']} 条 "
          f"(bona {res['dataset']['n_bonafide_test']} + spoof {res['dataset']['n_spoof_test']})  "
          f"@{res['dataset']['sample_rate']} Hz")
    print(f"防御: {res['defense']['name']}  (GMM 分量 {res['defense']['n_components']}, "
          f"特征 {res['defense']['feat_dim']} 维)")
    print("-" * 68)
    print(f"干净检测   : EER={res['clean']['EER']:.4f}  AUC={res['clean']['AUC']:.4f}  "
          f"Acc@0.5={res['clean']['Accuracy@0.5']:.4f}  "
          f"FRR={res['clean']['FRR']:.4f}  FAR={res['clean']['FAR']:.4f}")
    print(f"鲁棒检测   : EER={res['robust']['EER']:.4f}  AUC={res['robust']['AUC']:.4f}  "
          f"FRR={res['robust']['FRR']:.4f}  FAR={res['robust']['FAR']:.4f}")
    print("-" * 68)
    print(f"{'攻击':10s} {'ASR':>7s} {'CS':>7s} {'valid':>6s} {'Q':>6s} "
          f"{'EER_rob':>8s} {'AttackScore':>12s}")
    for p in res["per_attack"]:
        print(f"{p['attack']:10s} {p['ASR']:7.4f} {p['CS']:7.4f} "
              f"{p['valid_attack_rate']:6.3f} {p['Q']:6.4f} "
              f"{p['EER_robust']:8.4f} {p['AttackScore']:12.2f}")
    print("-" * 68)
    print(f"最终评分   : CoreASR(均值)={res['scores']['CoreASR']:.4f}  "
          f"AttackScore(均值)={res['scores']['AttackScore_mean']:.2f}  "
          f"DefenseScore={res['scores']['DefenseScore']:.2f}  "
          f"Efficiency={res['defense']['Efficiency']:.3f}")
    print(f"总耗时     : {res['wall_time_s']}s")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="语音隐匿与反欺骗基准：Baseline 指标计算",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--stage", choices=["attack", "defense", "all"],
                        default="attack", help="评测阶段")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="攻击成功判定阈值 τ")
    parser.add_argument("--output", default="results/metrics.json",
                        help="指标输出路径")
    args = parser.parse_args()

    res = evaluate(stage=args.stage, seed=args.seed,
                   threshold=args.threshold)
    print_report(res)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n[test] 指标已保存: {args.output}")
