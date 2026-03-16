#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_gt_fid_diversity.py

计算 GT（Ground Truth）VQ codes 经过 VQ-VAE 解码后的 FID 和 Diversity，
作为评估生成模型的上界基准。

GT VQ codes 路径: <dataset_dir>/HumanML3D/VQVAE/<motion_id>.npy
Real joint vecs: <dataset_dir>/HumanML3D/new_joint_vecs/<motion_id>.npy
两者都通过 EvaluatorModelWrapper.get_motion_embeddings 得到 embedding，
然后计算:
  - FID(gt_vq_decoded, real_joints)   -- VQ-VAE 本身的重建损失上界
  - Diversity(real), Diversity(gt_vq)

用法:

python -m eval.eval_gt_fid_diversity \\
  --dataset_dir /ibex/project/c2191/luoc/dataset/A2R \\
  --pairs_csv ./new_data/test.csv \\
  --t2m_opt ./checkpoints/t2m/Comp_v6_KLD005/opt.txt \\
  --vqvae_ckpt /ibex/project/c2191/luoc/backup/sub-a/code_for_a2rm/motion_VQVAE/net_last.pth \\
  --mean_path /home/luoc/hub/luoc/projects/audio2reactivemotion/reactmotion/external/T2M_GPT/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \\
  --std_path  /home/luoc/hub/luoc/projects/audio2reactivemotion/reactmotion/external/T2M_GPT/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy
"""

import os
import argparse

import numpy as np
import torch
from torch import cuda
from scipy import linalg
from tqdm import tqdm
import pandas as pd

from external.T2M-GPT.options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.vqvae as vqvae_module


# ---------------------------------------------------------------------------
# Metric helpers (same as eval_gen_fid_diversity_eval.py)
# ---------------------------------------------------------------------------

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


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


# ---------------------------------------------------------------------------
# Feature collection
# ---------------------------------------------------------------------------

def _motion_ids_from_csv(pairs_csv: str, split: str = "test"):
    """Return deduplicated list of motion_ids from CSV raw_file_name column."""
    if os.path.isdir(pairs_csv):
        csv_path = os.path.join(pairs_csv, f"{split}.csv")
    else:
        csv_path = pairs_csv
    if not os.path.isfile(csv_path):
        raise RuntimeError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    if "raw_file_name" not in df.columns:
        raise RuntimeError(f"CSV missing column `raw_file_name`, have: {list(df.columns)}")
    seen = []
    visited = set()
    for raw in df["raw_file_name"].dropna().astype(str):
        mid = str(raw.strip().split("_", 1)[0]).zfill(6)
        if mid not in visited:
            visited.add(mid)
            seen.append(mid)
    return seen


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
    """Embed real joint vecs (new_joint_vecs) via EvaluatorModelWrapper."""
    motion_ids = _motion_ids_from_csv(pairs_csv, split)
    joint_root = os.path.join(dataset_dir, "HumanML3D", "new_joint_vecs")

    motions, lens = [], []
    for mid in tqdm(motion_ids, desc="Real joints"):
        npy = os.path.join(joint_root, f"{mid}.npy")
        if not os.path.isfile(npy):
            continue
        arr = np.load(npy, allow_pickle=False).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 263:
            continue
        T_use = min(arr.shape[0], max_frames)
        arr = arr[:T_use]
        if mean is not None and std is not None:
            arr = (arr - mean) / (std + 1e-8)
        motions.append(arr)
        lens.append(T_use)

    if not motions:
        raise RuntimeError("No real joint vecs found.")

    feats = []
    BATCH = 64
    for i in tqdm(range(0, len(motions), BATCH), desc="Real embeddings"):
        bm, bl = motions[i:i + BATCH], lens[i:i + BATCH]
        max_T = max(bl)
        B = len(bm)
        x = np.zeros((B, max_T, 263), dtype=np.float32)
        for j, (m, L) in enumerate(zip(bm, bl)):
            x[j, :L] = m
        x_t = torch.from_numpy(x).to(device)
        l_t = torch.tensor(bl, dtype=torch.long, device=device)
        feats.append(eval_wrapper.get_motion_embeddings(x_t, l_t).cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def collect_gt_vq_embeddings(
    pairs_csv: str,
    dataset_dir: str,
    vae,
    eval_wrapper,
    device,
    split: str = "test",
    max_frames: int = 196,
) -> np.ndarray:
    """
    Read GT VQ codes from <dataset_dir>/HumanML3D/VQVAE/<motion_id>.npy,
    decode via VQ-VAE, embed via EvaluatorModelWrapper.
    VQ-VAE decoder output is already in normalized feature space.
    """
    motion_ids = _motion_ids_from_csv(pairs_csv, split)
    vqvae_root = os.path.join(dataset_dir, "HumanML3D", "VQVAE")

    all_joints, all_lens = [], []
    missing = 0
    for mid in tqdm(motion_ids, desc="GT VQ codes"):
        npy = os.path.join(vqvae_root, f"{mid}.npy")
        if not os.path.isfile(npy):
            missing += 1
            continue
        codes = np.load(npy, allow_pickle=False).astype(np.int64).reshape(-1)  # [T_codes]
        idx = torch.from_numpy(codes).unsqueeze(0).to(device)   # [1, T_codes]
        pose = vae.forward_decoder(idx)[0]                       # [T_dec, 263]
        T_use = min(pose.shape[0], max_frames)
        all_joints.append(pose[:T_use].cpu())
        all_lens.append(T_use)

    if missing:
        print(f"[GT VQ] {missing}/{len(motion_ids)} motion IDs not found in VQVAE dir — skipped.")
    if not all_joints:
        raise RuntimeError(f"No GT VQ codes found under {vqvae_root}")

    feats = []
    BATCH = 64
    for i in tqdm(range(0, len(all_joints), BATCH), desc="GT VQ embeddings"):
        bj, bl = all_joints[i:i + BATCH], all_lens[i:i + BATCH]
        max_T = max(bl)
        B = len(bj)
        x = torch.zeros(B, max_T, 263)
        for k, (j, L) in enumerate(zip(bj, bl)):
            x[k, :L] = j
        l_t = torch.tensor(bl, dtype=torch.long, device=device)
        feats.append(eval_wrapper.get_motion_embeddings(x.to(device), l_t).cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="GT VQ-code FID & Diversity (upper-bound baseline)")
    ap.add_argument("--dataset_dir", type=str, required=True,
                    help="A2R 数据根目录 (包含 HumanML3D/VQVAE 和 HumanML3D/new_joint_vecs)")
    ap.add_argument("--pairs_csv", type=str, required=True,
                    help="new_data/test.csv 路径或包含 test.csv 的目录")
    ap.add_argument("--t2m_opt", type=str, required=True,
                    help="T2M opt.txt 路径 (用于 EvaluatorModelWrapper)")
    ap.add_argument("--vqvae_ckpt", type=str, required=True,
                    help="运动 VQ-VAE checkpoint (net_last.pth)")
    ap.add_argument("--mean_path", type=str, default=None,
                    help="HumanML3D 263-dim Mean.npy (归一化 new_joint_vecs)")
    ap.add_argument("--std_path", type=str, default=None,
                    help="HumanML3D 263-dim Std.npy (归一化 new_joint_vecs)")
    ap.add_argument("--split", type=str, default="test",
                    help="使用的数据分割 (默认 test)")
    ap.add_argument("--max_frames", type=int, default=196)
    args = ap.parse_args()

    device = torch.device("cuda" if cuda.is_available() else "cpu")

    # Normalization stats for real joints
    mean_np = std_np = None
    if args.mean_path and args.std_path:
        if os.path.isfile(args.mean_path) and os.path.isfile(args.std_path):
            mean_np = np.load(args.mean_path).astype(np.float32)
            std_np  = np.load(args.std_path).astype(np.float32)
            print(f"[Norm] Mean={args.mean_path}")
        else:
            print("[Norm] WARNING: mean/std files not found — real joints will NOT be normalized.")
    else:
        print("[Norm] WARNING: --mean_path / --std_path not provided — real joints NOT normalized.")

    # EvaluatorModelWrapper
    eval_opt = get_opt(args.t2m_opt, device)
    eval_wrapper = EvaluatorModelWrapper(eval_opt)

    # VQ-VAE
    class DummyArgs:
        dataname          = "t2m"
        code_dim          = 512
        nb_code           = 512
        output_emb_width  = 512
        down_t            = 2
        stride_t          = 2
        width             = 512
        depth             = 3
        dilation_growth_rate = 3
        vq_act            = "relu"
        vq_norm           = None
        quantizer         = "ema_reset"
        beta              = 1.0
        mu                = 0.99

    vq_args = DummyArgs()
    net = vqvae_module.HumanVQVAE(
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
    net.load_state_dict(ckpt["net"], strict=True)
    net = net.to(device).eval()

    # --- Real embeddings ---
    print("\n[Real] collecting joint embeddings ...")
    real_emb = collect_real_embeddings(
        args.pairs_csv, args.dataset_dir, eval_wrapper, device,
        split=args.split, max_frames=args.max_frames,
        mean=mean_np, std=std_np,
    )
    print(f"[Real] {real_emb.shape[0]} samples, dim={real_emb.shape[1]}")

    # --- GT VQ embeddings ---
    print("\n[GT VQ] collecting GT VQ-decoded embeddings ...")
    gt_emb = collect_gt_vq_embeddings(
        args.pairs_csv, args.dataset_dir, net, eval_wrapper, device,
        split=args.split, max_frames=args.max_frames,
    )
    print(f"[GT VQ] {gt_emb.shape[0]} samples, dim={gt_emb.shape[1]}")

    # --- Metrics ---
    real_mu, real_cov = calculate_activation_statistics(real_emb)
    gt_mu,   gt_cov   = calculate_activation_statistics(gt_emb)
    fid = calculate_frechet_distance(real_mu, real_cov, gt_mu, gt_cov)

    div_real = calculate_diversity(real_emb)
    div_gt   = calculate_diversity(gt_emb)

    print("\n========== GT VQ FID & Diversity (Evaluator space) ==========")
    print(f"FID  (gt_vq vs real):  {fid:.4f}   ← VQ-VAE 重建上界")
    print(f"Diversity (real):       {div_real:.4f}")
    print(f"Diversity (gt_vq):      {div_gt:.4f}")


if __name__ == "__main__":
    main()
