#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eval_fid_humanml3d.py

用完整 HumanML3D 数据集作为 real 参考分布，计算生成动作的 FID 和 Diversity。
这衡量的是动作的"自然性/真实性"（motion naturalness），而非任务特异性。

Real 来源（三种方式，优先级依次降低）:
  1. --split_txt_dir 指定含 train.txt/val.txt/test.txt 的目录（标准 HumanML3D 划分）
  2. --split_txt     直接指定单个 split .txt 文件（每行一个 motion ID）
  3. 不指定 split 文件 → 扫描 --joint_vecs_dir 下所有 .npy（用全部文件）

Gen 来源（与 eval_gen_fid_diversity_eval.py 相同）:
  eval_dump  → gen_root/group=*/*.motion_codes.npy
  qwen_t2m   → gen_root/tokens/<key>/m*.npy

用法示例:

  # 用 HumanML3D 全量（扫描所有 .npy）
  python -m eval.eval_fid_humanml3d \\
    --joint_vecs_dir /ibex/project/c2191/luoc/dataset/A2R/HumanML3D/new_joint_vecs \\
    --gen_root /ibex/project/c2191/luoc/results/.../cond=a+e/ckpt=xxx \\
    --t2m_opt ./checkpoints/t2m/Comp_v6_KLD005/opt.txt \\
    --vqvae_ckpt /path/to/motion_VQVAE/t2m.pth \\
    --mean_path /path/to/mean.npy \\
    --std_path  /path/to/std.npy

  # 只用标准 test split
  python -m eval.eval_fid_humanml3d \\
    --joint_vecs_dir /ibex/project/c2191/luoc/dataset/A2R/HumanML3D/new_joint_vecs \\
    --split_txt /path/to/HumanML3D/test.txt \\
    --gen_root ... \\
    --t2m_opt ... --vqvae_ckpt ... --mean_path ... --std_path ...
"""

import os
import glob
import argparse

import numpy as np
import torch
from torch import cuda
from scipy import linalg
from scipy.spatial.distance import pdist
from tqdm import tqdm

from external.T2M-GPT.options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.vqvae as vqvae_module


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape and sigma1.shape == sigma2.shape
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + np.eye(sigma1.shape[0]) * eps).dot(sigma2 + np.eye(sigma2.shape[0]) * eps))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def calculate_activation_statistics(act: np.ndarray):
    return np.mean(act, axis=0), np.cov(act, rowvar=False)


def calculate_diversity(act: np.ndarray) -> float:
    """Average L2 distance over all n*(n-1)/2 unique pairs."""
    if act.shape[0] < 2:
        return 0.0
    return float(pdist(act, metric="euclidean").mean())


# ---------------------------------------------------------------------------
# Collect real HumanML3D embeddings
# ---------------------------------------------------------------------------

def _resolve_motion_ids(joint_vecs_dir: str, split_txt: str = None, split_txt_dir: str = None):
    """
    Return a list of absolute .npy paths to load.
    Priority: split_txt_dir > split_txt > scan all.
    """
    if split_txt_dir and os.path.isdir(split_txt_dir):
        ids = set()
        for fname in ("train.txt", "val.txt", "test.txt"):
            p = os.path.join(split_txt_dir, fname)
            if os.path.isfile(p):
                with open(p) as f:
                    ids.update(l.strip() for l in f if l.strip())
        print(f"[HumanML3D] split_txt_dir: loaded {len(ids)} motion IDs from {split_txt_dir}")
    elif split_txt and os.path.isfile(split_txt):
        with open(split_txt) as f:
            ids = {l.strip() for l in f if l.strip()}
        print(f"[HumanML3D] split_txt: loaded {len(ids)} motion IDs from {split_txt}")
    else:
        # Scan all .npy in joint_vecs_dir
        all_files = sorted(glob.glob(os.path.join(joint_vecs_dir, "*.npy")))
        print(f"[HumanML3D] scanning all: found {len(all_files)} .npy files in {joint_vecs_dir}")
        return all_files

    paths = []
    missing = 0
    for mid in sorted(ids):
        p = os.path.join(joint_vecs_dir, f"{mid}.npy")
        if os.path.isfile(p):
            paths.append(p)
        else:
            missing += 1
    if missing:
        print(f"[HumanML3D] WARNING: {missing} motion IDs not found in {joint_vecs_dir}")
    return paths


@torch.no_grad()
def collect_humanml3d_embeddings(
    joint_vecs_dir: str,
    eval_wrapper,
    device,
    max_frames: int = 196,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    split_txt: str = None,
    split_txt_dir: str = None,
) -> np.ndarray:
    """Load all HumanML3D joint vecs, normalize, embed via EvaluatorModelWrapper."""
    npy_paths = _resolve_motion_ids(joint_vecs_dir, split_txt, split_txt_dir)
    if not npy_paths:
        raise RuntimeError(f"No .npy files found in {joint_vecs_dir}")

    motions, lens = [], []
    skipped = 0
    for p in tqdm(npy_paths, desc="Loading HumanML3D joints"):
        arr = np.load(p, allow_pickle=False).astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 263)
        if arr.ndim != 2 or arr.shape[1] != 263:
            skipped += 1
            continue
        T_use = min(arr.shape[0], max_frames)
        arr = arr[:T_use]
        if mean is not None and std is not None:
            arr = (arr - mean) / (std + 1e-8)
        motions.append(arr)
        lens.append(T_use)

    if skipped:
        print(f"[HumanML3D] skipped {skipped} files (wrong shape)")
    print(f"[HumanML3D] {len(motions)} valid motions loaded")

    feats = []
    BATCH = 64
    for i in tqdm(range(0, len(motions), BATCH), desc="Embedding HumanML3D"):
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


# ---------------------------------------------------------------------------
# Collect generated embeddings (same as eval_gen_fid_diversity_eval.py)
# ---------------------------------------------------------------------------

def _collect_code_paths(gen_root: str, gen_format: str):
    if gen_format == "eval_dump":
        group_dirs = sorted(glob.glob(os.path.join(gen_root, "group=*")))
        if not group_dirs:
            raise RuntimeError(f"[eval_dump] No group=* dirs under {gen_root}")
        paths = []
        for gdir in group_dirs:
            paths.extend(sorted(glob.glob(os.path.join(gdir, "*.motion_codes.npy"))))
        return paths

    if gen_format == "qwen_t2m":
        tokens_root = os.path.join(gen_root, "tokens")
        search_root = tokens_root if os.path.isdir(tokens_root) else gen_root
        key_dirs = sorted(d for d in glob.glob(os.path.join(search_root, "*")) if os.path.isdir(d))
        if not key_dirs:
            raise RuntimeError(f"[qwen_t2m] No sub-dirs under {search_root}")
        paths = []
        for kdir in key_dirs:
            paths.extend(sorted(glob.glob(os.path.join(kdir, "m*.npy"))))
        return paths

    raise ValueError(f"Unknown gen_format: {gen_format!r}")


@torch.no_grad()
def collect_gen_embeddings(gen_root, vae, eval_wrapper, device, max_frames=196, gen_format="eval_dump"):
    code_paths = _collect_code_paths(gen_root, gen_format)
    if not code_paths:
        raise RuntimeError(f"No motion code .npy files found under {gen_root} (format={gen_format})")
    print(f"[Gen] {len(code_paths)} code files (format={gen_format})")

    all_joints, all_lens = [], []
    for p in tqdm(code_paths, desc="Decoding gen motions"):
        codes = np.load(p, allow_pickle=False).astype(np.int64).reshape(-1)
        idx = torch.from_numpy(codes).unsqueeze(0).to(device)
        pose = vae.forward_decoder(idx)[0]                        # [T_dec, 263]
        T_use = min(pose.shape[0], max_frames)
        all_joints.append(pose[:T_use].cpu())
        all_lens.append(T_use)

    feats = []
    BATCH = 64
    for i in range(0, len(all_joints), BATCH):
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
    ap = argparse.ArgumentParser(description="FID/Diversity against full HumanML3D (motion naturalness)")

    # Real: HumanML3D
    ap.add_argument("--joint_vecs_dir", type=str, required=True,
                    help="HumanML3D new_joint_vecs 目录 (所有 .npy 文件)")
    ap.add_argument("--split_txt", type=str, default=None,
                    help="单个 HumanML3D split .txt 文件 (每行一个 motion ID)")
    ap.add_argument("--split_txt_dir", type=str, default=None,
                    help="含 train.txt/val.txt/test.txt 的目录 (用全部 split)")

    # Gen
    ap.add_argument("--gen_root", type=str, required=True,
                    help="生成结果根目录")
    ap.add_argument("--gen_format", type=str, default="eval_dump",
                    choices=["eval_dump", "qwen_t2m"],
                    help="eval_dump: group=*/*.motion_codes.npy; qwen_t2m: tokens/<key>/m*.npy")

    # Model
    ap.add_argument("--t2m_opt", type=str, required=True,
                    help="T2M Comp_v6_KLD005/opt.txt 路径")
    ap.add_argument("--vqvae_ckpt", type=str, required=True,
                    help="运动 VQ-VAE checkpoint (net_last.pth / t2m.pth)")
    ap.add_argument("--mean_path", type=str, default=None,
                    help="HumanML3D 263-dim Mean.npy (用于归一化 joint vecs)")
    ap.add_argument("--std_path", type=str, default=None,
                    help="HumanML3D 263-dim Std.npy")
    ap.add_argument("--max_frames", type=int, default=196)

    args = ap.parse_args()
    device = torch.device("cuda" if cuda.is_available() else "cpu")

    # Normalization
    mean_np = std_np = None
    if args.mean_path and args.std_path:
        if os.path.isfile(args.mean_path) and os.path.isfile(args.std_path):
            mean_np = np.load(args.mean_path).astype(np.float32)
            std_np  = np.load(args.std_path).astype(np.float32)
            print(f"[Norm] loaded Mean/Std")
        else:
            print("[Norm] WARNING: mean/std files not found — joints NOT normalized")
    else:
        print("[Norm] WARNING: --mean_path/--std_path not provided — joints NOT normalized")

    # EvaluatorModelWrapper
    eval_opt = get_opt(args.t2m_opt, device)
    eval_wrapper = EvaluatorModelWrapper(eval_opt)

    # VQ-VAE
    class DummyArgs:
        dataname = "t2m"; code_dim = 512; nb_code = 512; output_emb_width = 512
        down_t = 2; stride_t = 2; width = 512; depth = 3
        dilation_growth_rate = 3; vq_act = "relu"; vq_norm = None
        quantizer = "ema_reset"; beta = 1.0; mu = 0.99

    vq_args = DummyArgs()
    net = vqvae_module.HumanVQVAE(
        vq_args, nb_code=vq_args.nb_code, code_dim=vq_args.code_dim,
        output_emb_width=vq_args.output_emb_width, down_t=vq_args.down_t,
        stride_t=vq_args.stride_t, width=vq_args.width, depth=vq_args.depth,
        dilation_growth_rate=vq_args.dilation_growth_rate,
        activation=vq_args.vq_act, norm=vq_args.vq_norm,
    )
    ckpt = torch.load(args.vqvae_ckpt, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    net = net.to(device).eval()

    # --- Real: full HumanML3D ---
    print("\n[Real] loading HumanML3D reference embeddings ...")
    real_emb = collect_humanml3d_embeddings(
        args.joint_vecs_dir, eval_wrapper, device,
        max_frames=args.max_frames, mean=mean_np, std=std_np,
        split_txt=args.split_txt, split_txt_dir=args.split_txt_dir,
    )
    print(f"[Real] {real_emb.shape[0]} samples, dim={real_emb.shape[1]}")

    # --- Gen ---
    print("\n[Gen] loading generated motion embeddings ...")
    gen_emb = collect_gen_embeddings(
        args.gen_root, net, eval_wrapper, device,
        max_frames=args.max_frames, gen_format=args.gen_format,
    )
    print(f"[Gen] {gen_emb.shape[0]} samples, dim={gen_emb.shape[1]}")

    # --- Metrics ---
    real_mu, real_cov = calculate_activation_statistics(real_emb)
    gen_mu,  gen_cov  = calculate_activation_statistics(gen_emb)
    fid = calculate_frechet_distance(real_mu, real_cov, gen_mu, gen_cov)
    div_real = calculate_diversity(real_emb)
    div_gen  = calculate_diversity(gen_emb)

    print("\n========== FID & Diversity (HumanML3D reference) ==========")
    print(f"Real samples:      {real_emb.shape[0]}")
    print(f"Gen  samples:      {gen_emb.shape[0]}")
    print(f"FID  (gen vs HumanML3D): {fid:.4f}   ← motion naturalness")
    print(f"Diversity (HumanML3D):   {div_real:.4f}")
    print(f"Diversity (gen):         {div_gen:.4f}")


if __name__ == "__main__":
    main()
