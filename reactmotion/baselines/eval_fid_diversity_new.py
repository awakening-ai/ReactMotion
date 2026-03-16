#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_gen_fid_diversity_eval.py

使用 T2M 的 EvaluatorModelWrapper.get_motion_embeddings 作为特征空间，
对 run_eval_gen_dump.sh 生成的 motion 做 FID 和 Diversity 评估。

Real:
  - 使用 new_data/test.csv + HumanML3D/new_joint_vecs
  - 从 raw_file_name 提取 motion_id（下划线前缀，zero-pad 6 位），
    在 <dataset_dir>/HumanML3D/new_joint_vecs/<motion_id>.npy 读取 joints
  - 调用 EvaluatorModelWrapper.get_motion_embeddings 得到 motion embedding

Gen:
  - 从 gen_root 下 group=*/ *.motion_codes.npy 读取离散 codes
  - 用 motion VQ-VAE (HumanVQVAE.forward_decoder) 解码为 joints
  - 同样调用 get_motion_embeddings 得到 embedding

然后在该 embedding 空间上计算:
  - FID(real, gen)
  - Diversity(real), Diversity(gen)

用法示例:

python -m eval.eval_gen_fid_diversity_eval \\
  --gen_root /ibex/project/c2191/luoc/results/eval_results/eval_gen_dump_10000-ape/cond=a+e/ckpt=d72daf190e \\
  --dataset_dir /ibex/project/c2191/luoc/dataset/A2R \\
  --pairs_csv ./new_data/test.csv \\
  --t2m_opt ./checkpoints/t2m/Comp_v6_KLD005/opt.txt \\
  --vqvae_ckpt /ibex/project/c2191/luoc/backup/sub-a/code_for_a2rm/motion_VQVAE/net_last.pth \\
  --mean_path ./dataset/HumanML3D/Mean.npy \\
  --std_path ./dataset/HumanML3D/Std.npy

注意:
  - --mean_path / --std_path 是 HumanML3D 263-dim 的均值/标准差文件
    (用于归一化 new_joint_vecs 的原始特征, 与 VQ-VAE/T2M evaluator 训练时保持一致)
  - 如果没有这两个文件, 可以不传 (会直接使用原始值, FID 可能偏高)
"""

import os
import glob
import argparse

import numpy as np
import torch
from torch import cuda
from scipy import linalg
from tqdm import tqdm
import pandas as pd

from external.T2M_GPT.options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.vqvae as vqvae


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

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


def calculate_diversity(activation: np.ndarray) -> float:
    """Average L2 distance over all n*(n-1)/2 unique pairs."""
    from scipy.spatial.distance import pdist
    if activation.shape[0] < 2:
        return 0.0
    return float(pdist(activation, metric="euclidean").mean())


@torch.no_grad()
def collect_real_embeddings(
    pairs_csv: str,
    dataset_dir: str,
    eval_wrapper,
    device,
    split: str = "test",
    max_frames: int = 196,
    mean: np.ndarray = None,
    std: np.ndarray = None,
) -> np.ndarray:
    """
    从 new_data/test.csv + new_joint_vecs 中提取真实 motion embedding (EvaluatorModelWrapper 空间)。
    mean/std: HumanML3D 263-dim 均值/标准差，用于归一化原始特征。
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
    motions = []
    lens = []
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
        arr = np.load(npy_path, allow_pickle=False).astype(np.float32)  # [T, 263]
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        T, D = arr.shape
        if D != 263:
            continue
        T_use = min(T, max_frames)
        arr = arr[:T_use]
        # Normalize to the same space as the T2M evaluator / VQ-VAE training
        if mean is not None and std is not None:
            arr = (arr - mean) / (std + 1e-8)
        motions.append(arr)
        lens.append(T_use)

    if not motions:
        raise RuntimeError("No valid joint vecs found for real motions.")

    feats = []
    BATCH = 64
    for i in tqdm(range(0, len(motions), BATCH), desc="Real embeddings (batches)"):
        batch_m = motions[i:i + BATCH]
        batch_l = lens[i:i + BATCH]
        max_T = max(batch_l)
        B = len(batch_m)
        x = np.zeros((B, max_T, 263), dtype=np.float32)
        l_arr = np.zeros((B,), dtype=np.int64)
        for j, (m, L) in enumerate(zip(batch_m, batch_l)):
            x[j, :L] = m
            l_arr[j] = L
        x_t = torch.from_numpy(x).to(device)
        l_t = torch.from_numpy(l_arr).to(device)
        em = eval_wrapper.get_motion_embeddings(x_t, l_t)  # [B,D]
        feats.append(em.cpu().numpy().astype(np.float32))

    return np.concatenate(feats, axis=0)


@torch.no_grad()
def collect_gen_embeddings(gen_root: str, vae, eval_wrapper, device, max_frames: int = 196) -> np.ndarray:
    """
    从 run_eval_gen_dump.sh 输出的 motion_codes 解码 joints, 再用 EvaluatorModelWrapper 提取 embedding。
    VQ-VAE 训练时使用了归一化特征, 其 decoder 输出也处于归一化空间, 因此无需额外归一化。
    """
    feats = []

    group_dirs = sorted(glob.glob(os.path.join(gen_root, "group=*")))
    if not group_dirs:
        raise RuntimeError(f"No group=* dirs found under {gen_root}")

    # Accumulate decoded joints across all groups, then batch-embed at the end
    all_joints = []
    all_lens = []

    for gdir in tqdm(group_dirs, desc="Generated motions (groups)"):
        code_paths = sorted(glob.glob(os.path.join(gdir, "*.motion_codes.npy")))
        if not code_paths:
            continue

        for p in code_paths:
            codes = np.load(p, allow_pickle=False).astype(np.int64).reshape(-1)  # [T_codes]
            # forward_decoder collapses batch dim: must decode one sample at a time
            idx = torch.from_numpy(codes).unsqueeze(0).to(device)  # [1, T_codes]
            pose = vae.forward_decoder(idx)                          # [1, T_dec, 263]
            pose = pose[0]                                            # [T_dec, 263]
            T_use = min(pose.shape[0], max_frames)
            all_joints.append(pose[:T_use].cpu())
            all_lens.append(T_use)

    if not all_joints:
        raise RuntimeError("No motion_codes loaded; check gen_root.")

    # Batch and embed in chunks of 64
    BATCH = 64
    for i in range(0, len(all_joints), BATCH):
        batch_j = all_joints[i:i + BATCH]
        batch_l = all_lens[i:i + BATCH]
        max_T = max(batch_l)
        B = len(batch_j)
        x = torch.zeros(B, max_T, 263)
        for k, (j, L) in enumerate(zip(batch_j, batch_l)):
            x[k, :L] = j
        x = x.to(device)
        l_t = torch.tensor(batch_l, dtype=torch.long, device=device)
        em = eval_wrapper.get_motion_embeddings(x, l_t)
        feats.append(em.cpu().numpy().astype(np.float32))

    return np.concatenate(feats, axis=0)


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
        "--t2m_opt",
        type=str,
        required=True,
        help="T2M opt.txt 路径 (用于 EvaluatorModelWrapper)",
    )
    ap.add_argument(
        "--vqvae_ckpt",
        type=str,
        required=True,
        help="运动 VQ-VAE checkpoint, 如 ./motion_VQVAE/net_last.pth",
    )
    ap.add_argument(
        "--mean_path",
        type=str,
        default=None,
        help="HumanML3D 263-dim Mean.npy 路径 (用于归一化 new_joint_vecs 的原始特征)",
    )
    ap.add_argument(
        "--std_path",
        type=str,
        default=None,
        help="HumanML3D 263-dim Std.npy 路径 (用于归一化 new_joint_vecs 的原始特征)",
    )

    args = ap.parse_args()

    device = torch.device("cuda" if cuda.is_available() else "cpu")

    # Load normalization stats (critical: new_joint_vecs are raw, evaluator expects normalized)
    mean_np = std_np = None
    if args.mean_path and args.std_path:
        if os.path.isfile(args.mean_path) and os.path.isfile(args.std_path):
            mean_np = np.load(args.mean_path).astype(np.float32)  # [263]
            std_np = np.load(args.std_path).astype(np.float32)    # [263]
            print(f"[Norm] loaded Mean from {args.mean_path}, Std from {args.std_path}")
        else:
            print(f"[Norm] WARNING: mean/std path not found — using raw features (FID may be inflated)")
    else:
        print("[Norm] WARNING: --mean_path / --std_path not provided — real joints NOT normalized. FID will likely be inflated.")

    # Evaluator wrapper
    eval_opt = get_opt(args.t2m_opt, device)
    eval_wrapper = EvaluatorModelWrapper(eval_opt)

    # VQ-VAE
    # 这里假设你训练 VQ-VAE 时用的是 t2m 配置
    class DummyArgs:
        def __init__(self):
            self.dataname = "t2m"
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
            self.quantizer = "ema_reset"
            self.beta = 1.0
            self.mu = 0.99

    vq_args = DummyArgs()
    vae = vqvae.HumanVQVAE(
        vq_args,
        nb_code=vq_args.nb_code,
        code_dim=vq_args.code_dim,
        output_emb_width=vq_args.output_emb_width,
        down_t=vq_args.down_t,
        stride_t=vq_args.stride_t,
        width=vq_args.width,
        depth=vq_args.depth,
        dilation_growth_rate=vq_args.dilation_growth_rate,
        activation=vq_args.vq_act,
        norm=vq_args.vq_norm,
    )
    ckpt = torch.load(args.vqvae_ckpt, map_location="cpu")
    vae.load_state_dict(ckpt["net"], strict=True)
    vae = vae.to(device).eval()

    # Real
    print("[Real] collecting motion embeddings (Evaluator) ...")
    real_emb = collect_real_embeddings(
        args.pairs_csv, args.dataset_dir, eval_wrapper, device,
        split="test", mean=mean_np, std=std_np,
    )
    print(f"[Real] motions: {real_emb.shape[0]} samples, dim={real_emb.shape[1]}")

    # Gen
    print("[Gen] collecting motion embeddings from gen_root =", args.gen_root)
    gen_emb = collect_gen_embeddings(args.gen_root, vae, eval_wrapper, device)
    print(f"[Gen] motions: {gen_emb.shape[0]} samples, dim={gen_emb.shape[1]}")

    # FID
    gt_mu, gt_cov = calculate_activation_statistics(real_emb)
    mu, cov = calculate_activation_statistics(gen_emb)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    # Diversity (all pairs)
    div_real = calculate_diversity(real_emb)
    div_gen  = calculate_diversity(gen_emb)

    print("========== FID & Diversity (Evaluator space) ==========")
    print(f"FID (gen vs real): {fid:.4f}")
    print(f"Diversity (real):  {div_real:.4f}")
    print(f"Diversity (gen):   {div_gen:.4f}")


if __name__ == "__main__":
    main()

