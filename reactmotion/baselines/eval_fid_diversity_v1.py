#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_gen_fid_diversity.py

评估 run_eval_gen_dump.sh 生成的 motion diversity 和 FID。

约定:
  - 真实分布 (real):
      * 使用 new_data/test.csv 中的 raw_file_name
      * 截取下划线前缀, 例如 000477_2 -> 000477 -> zero-pad 成 6 位
      * 从 <dataset_dir>/HumanML3D/new_joint_vecs/<motion_id>.npy 读取 joint 向量
      * 对时间维做平均, 得到 263 维 feature

  - 生成分布 (gen):
      * 从 gen_root 下的 group=*/ *.motion_codes.npy 读取离散 codes
      * 使用 motion VQ-VAE (HumanVQVAE.forward_decoder) 解码为 joints
      * 同样对时间维做平均, 得到 263 维 feature

  - 然后在这 263 维特征空间上计算:
      * FID(real, gen)
      * Diversity(real), Diversity(gen)

不再依赖 T2M 的 dataloader/evaluator wrapper。

用法示例:

python -m eval.eval_gen_fid_diversity \\
  --gen_root /ibex/project/c2191/luoc/results/eval_results/eval_gen_dump_10000-ape/cond=a+e/ckpt=d72daf190e \\
  --dataset_dir /ibex/project/c2191/luoc/dataset/A2R \\
  --pairs_csv ./new_data/test.csv \\
  --vqvae_ckpt ./motion_VQVAE/net_last.pth
"""

import os
import glob
import argparse

import numpy as np
import torch
from torch import cuda
from scipy import linalg
from tqdm import tqdm

import models.vqvae as vqvae
import pandas as pd


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(activations: np.ndarray):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation: np.ndarray, diversity_times: int):
    assert len(activation.shape) == 2
    n = activation.shape[0]
    if n < 2:
        return 0.0

    diversity_times = int(min(diversity_times, n - 1))
    diversity_times = max(diversity_times, 1)

    first_indices = np.random.choice(n, diversity_times, replace=False)
    second_indices = np.random.choice(n, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return float(dist.mean())


def load_real_joint_features(pairs_csv: str, dataset_dir: str, split: str = "test", max_frames: int = 196) -> np.ndarray:
    """
    使用 new_data/test.csv + HumanML3D/new_joint_vecs 作为真实分布:
      - 从 raw_file_name 解析 motion_id (下划线前缀, zero-pad 6 位)
      - 读取 joint vecs, 对时间维做平均 -> 263 维 feature
    """
    if os.path.isdir(pairs_csv):
        csv_path = os.path.join(pairs_csv, f"{split}.csv")
    else:
        csv_path = pairs_csv
    if not os.path.isfile(csv_path):
        raise RuntimeError(f"Missing csv: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8")
    if "raw_file_name" not in df.columns:
        raise RuntimeError(f"csv missing column `raw_file_name`, have: {list(df.columns)}")

    joint_root = os.path.join(dataset_dir, "HumanML3D", "new_joint_vecs")
    feats = []
    seen = set()
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Real joints"):
        raw = str(r["raw_file_name"]).strip()
        if not raw:
            continue
        mid = raw.split("_", 1)[0]
        mid = str(mid).zfill(6)
        if mid in seen:
            continue
        seen.add(mid)
        npy_path = os.path.join(joint_root, f"{mid}.npy")
        if not os.path.isfile(npy_path):
            continue
        arr = np.load(npy_path, allow_pickle=False)  # [T,263] 或 [T,D]
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        T, D = arr.shape
        if D != 263:
            # 如果维度异常, 跳过该样本
            continue
        T_use = min(T, max_frames)
        feat = arr[:T_use].mean(axis=0).astype(np.float32)  # [263]
        feats.append(feat)

    if not feats:
        raise RuntimeError("No valid joint vecs found for real motions.")

    return np.stack(feats, axis=0)


@torch.no_grad()
def collect_gen_joint_features(gen_root: str, vae, device, max_frames: int = 196) -> np.ndarray:
    """
    从 run_eval_gen_dump.sh 输出的 group=*/xxx.motion_codes.npy 中提取生成 joints feature:
      - VQ-VAE 解码为 joints
      - 对时间维做平均 -> 263 维 feature
    """
    feats = []

    group_dirs = sorted(glob.glob(os.path.join(gen_root, "group=*")))
    if not group_dirs:
        raise RuntimeError(f"No group=* dirs found under {gen_root}")

    for gdir in tqdm(group_dirs, desc="Generated motions (groups)"):
        code_paths = sorted(glob.glob(os.path.join(gdir, "*.motion_codes.npy")))
        if not code_paths:
            continue

        codes_batch = []
        max_len = 0
        for p in code_paths:
            codes = np.load(p, allow_pickle=False).astype(np.int64).reshape(-1)  # [T]
            max_len = max(max_len, codes.shape[0])
            codes_batch.append(codes)

        bs = len(codes_batch)
        index_motion = np.zeros((bs, max_len), dtype=np.int64)
        m_lens = np.zeros((bs,), dtype=np.int64)
        for i, c in enumerate(codes_batch):
            L = c.shape[0]
            index_motion[i, :L] = c
            m_lens[i] = L

        index_motion = torch.from_numpy(index_motion).to(device)

        # VQ-VAE decode -> joint motions
        pred_pose = vae.forward_decoder(index_motion)  # [B, T, D]
        B, T, D = pred_pose.shape
        T_use = min(T, max_frames)
        arr = pred_pose[:, :T_use].mean(dim=1).cpu().numpy().astype(np.float32)  # [B,263]
        feats.append(arr)

    if not feats:
        raise RuntimeError("No motion_codes loaded; check gen_root.")

    feats_np = np.concatenate(feats, axis=0)
    return feats_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gen_root",
        type=str,
        required=True,
        help="run_eval_gen_dump.sh 输出的 ckpt 目录 (包含 group=* 子目录)",
    )
    ap.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="A2R 数据根目录 (包含 HumanML3D/new_joint_vecs)",
    )
    ap.add_argument(
        "--pairs_csv",
        type=str,
        required=True,
        help="new_data/test.csv 路径或包含 train/val/test.csv 的目录",
    )
    ap.add_argument(
        "--vqvae_ckpt",
        type=str,
        required=True,
        help="运动 VQ-VAE checkpoint, 如 ./motion_VQVAE/net_last.pth",
    )
    ap.add_argument(
        "--diversity_samples",
        type=int,
        default=300,
        help="计算 diversity 时采样的对数上限 (实际会被截到 N-1)",
    )

    args_cli = ap.parse_args()

    device = torch.device("cuda" if cuda.is_available() else "cpu")

    # ---- VQ-VAE (用于把 motion_codes 解到 joints) ----
    # 构造一个最小 args, 对应你训练 VQ-VAE 的默认配置 (t2m, 22 关节)
    class DummyArgs:
        def __init__(self):
            self.dataname = "t2m"        # => 22 joints in HumanVQVAE
            self.code_dim = 512
            self.nb_code = 512
            self.output_emb_width = 512
            self.down_t = 2
            self.stride_t = 2
            self.width = 512
            self.depth = 3
            self.dilation_growth_rate = 3
            self.vq_act = "relu"
            self.vq_norm = None
            # quantizer 相关, 对应 option_vq 的默认
            self.quantizer = "ema_reset"
            self.beta = 1.0
            self.mu = 0.99

    base_args = DummyArgs()
    vae = vqvae.HumanVQVAE(
        base_args,
        nb_code=base_args.nb_code,
        code_dim=base_args.code_dim,
        output_emb_width=base_args.output_emb_width,
        down_t=base_args.down_t,
        stride_t=base_args.stride_t,
        width=base_args.width,
        depth=base_args.depth,
        dilation_growth_rate=base_args.dilation_growth_rate,
        activation=base_args.vq_act,
        norm=base_args.vq_norm,
    )
    ckpt = torch.load(args_cli.vqvae_ckpt, map_location="cpu")
    vae.load_state_dict(ckpt["net"], strict=True)
    vae = vae.to(device).eval()

    # ---- Real joints from new_joint_vecs ----
    print("[Real] collecting joint features from new_joint_vecs ...")
    motion_real_np = load_real_joint_features(args_cli.pairs_csv, args_cli.dataset_dir, split="test")
    print(f"[Real] motions: {motion_real_np.shape[0]} samples, dim={motion_real_np.shape[1]}")

    # ---- Generated joints from motion_codes ----
    print("[Gen] collecting joint features from gen_root =", args_cli.gen_root)
    motion_gen_np = collect_gen_joint_features(args_cli.gen_root, vae, device)
    print(f"[Gen] motions: {motion_gen_np.shape[0]} samples, dim={motion_gen_np.shape[1]}")

    # ---- FID ----
    gt_mu, gt_cov = calculate_activation_statistics(motion_real_np)
    mu, cov = calculate_activation_statistics(motion_gen_np)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # ---- Diversity ----
    div_real = calculate_diversity(
        motion_real_np,
        args_cli.diversity_samples if motion_real_np.shape[0] > args_cli.diversity_samples else max(motion_real_np.shape[0] - 1, 1),
    )
    div_gen = calculate_diversity(
        motion_gen_np,
        args_cli.diversity_samples if motion_gen_np.shape[0] > args_cli.diversity_samples else max(motion_gen_np.shape[0] - 1, 1),
    )

    print("========== FID & Diversity ==========")
    print(f"FID (gen vs real): {fid:.4f}")
    print(f"Diversity (real):  {div_real:.4f}")
    print(f"Diversity (gen):   {div_gen:.4f}")


if __name__ == "__main__":
    main()

