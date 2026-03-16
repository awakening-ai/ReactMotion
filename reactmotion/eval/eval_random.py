#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/eval_random.py

Random baseline: 对每个 test 样本，从
  /ibex/project/.../HumanML3D/VQVAE/*.npy
中随机抽取一个 VQ code 文件作为 listener motion 预测，
然后计算 FID & Diversity，并保存 joint_vecs / motion_tokens
以便后续用 run_eval_retrieval_win_rates.sh 计算 Win↑/Gen@3↑。

用法:
  python -m eval.eval_random \\
    --dataset_dir  /ibex/project/c2191/luoc/dataset/A2R \\
    --pairs_csv    ./new_data \\
    --t2m_opt      ./checkpoints/t2m/Comp_v6_KLD005/opt.txt \\
    --vqvae_ckpt   /home/luoc/.../t2m.pth \\
    --mean_path    /home/luoc/.../mean.npy \\
    --std_path     /home/luoc/.../std.npy \\
    --out_dir      ./out_random \\
    --seed         42
"""

import os
import json
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import linalg
from scipy.spatial.distance import pdist
from tqdm import tqdm

from external.T2M_GPT.options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.vqvae as vqvae_module


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def frechet_distance(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    cm, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(cm).all():
        cm = linalg.sqrtm(
            (s1 + np.eye(s1.shape[0]) * eps).dot(s2 + np.eye(s2.shape[0]) * eps))
    if np.iscomplexobj(cm):
        cm = cm.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(cm))


def all_pair_diversity(act: np.ndarray) -> float:
    if act.shape[0] < 2:
        return 0.0
    return float(pdist(act, metric="euclidean").mean())


# ─────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────

def motion_id_from_raw(raw):
    x = "" if pd.isna(raw) else str(raw).strip()
    mid = x.split("_", 1)[0] if x else ""
    return str(mid).zfill(6) if mid else ""


def read_test_csv(pairs_csv: str, split: str = "test") -> pd.DataFrame:
    if os.path.isdir(pairs_csv):
        p = os.path.join(pairs_csv, f"{split}.csv")
        return pd.read_csv(p, encoding="utf-8")
    df = pd.read_csv(pairs_csv, encoding="utf-8")
    if "split" in df.columns:
        df = df[df["split"].str.lower().str.strip() == split].copy()
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# VQ-VAE
# ─────────────────────────────────────────────────────────────

def build_vqvae(ckpt_path: str, device) -> torch.nn.Module:
    class _A:
        dataname = "t2m"; quantizer = "ema_reset"
        beta = 1.0; mu = 0.99

    ckpt   = torch.load(ckpt_path, map_location="cpu")
    net_sd = ckpt.get("net", ckpt)
    cb_key = next((k for k in net_sd if "codebook" in k and k.endswith(".weight")), None)
    nb_code, code_dim = 512, 512
    if cb_key:
        nb_code  = int(net_sd[cb_key].shape[0])
        code_dim = int(net_sd[cb_key].shape[1])

    net = vqvae_module.HumanVQVAE(
        _A(), nb_code=nb_code, code_dim=code_dim,
        output_emb_width=512, down_t=2, stride_t=2,
        width=512, depth=3, dilation_growth_rate=3,
        activation="relu", norm=None,
    )
    net.load_state_dict(net_sd, strict=True)
    print(f"[VQ-VAE] nb_code={nb_code}, code_dim={code_dim}")
    return net.to(device).eval()


# ─────────────────────────────────────────────────────────────
# Scan VQVAE dir
# ─────────────────────────────────────────────────────────────

def scan_vqvae_dir(vqvae_dir: str) -> List[str]:
    """Return sorted list of all .npy paths in the VQVAE directory."""
    paths = []
    for fname in os.listdir(vqvae_dir):
        if fname.endswith(".npy"):
            paths.append(os.path.join(vqvae_dir, fname))
    paths.sort()
    print(f"[VQVAEDir] {len(paths)} .npy files found in {vqvae_dir}")
    return paths


# ─────────────────────────────────────────────────────────────
# Embed real test motions
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def embed_real_motions(
    test_ids: List[str],
    joint_vecs_dir: str,
    eval_wrapper: EvaluatorModelWrapper,
    device,
    max_frames: int = 196,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    batch_size: int = 64,
) -> np.ndarray:
    motions, lens = [], []
    for mid in tqdm(test_ids, desc="Loading real joint vecs"):
        p = os.path.join(joint_vecs_dir, f"{mid}.npy")
        if not os.path.isfile(p):
            continue
        arr = np.load(p, allow_pickle=False).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 263:
            continue
        T = min(arr.shape[0], max_frames)
        arr = arr[:T].copy()
        if mean is not None and std is not None:
            arr = (arr - mean) / (std + 1e-8)
        motions.append(arr)
        lens.append(T)

    print(f"  real: {len(motions)} motions")
    feats = []
    for i in range(0, len(motions), batch_size):
        bm, bl = motions[i:i + batch_size], lens[i:i + batch_size]
        max_T = max(bl)
        B     = len(bm)
        x     = np.zeros((B, max_T, 263), dtype=np.float32)
        for j, (m, L) in enumerate(zip(bm, bl)):
            x[j, :L] = m
        xt = torch.from_numpy(x).to(device)
        lt = torch.tensor(bl, dtype=torch.long, device=device)
        feats.append(eval_wrapper.get_motion_embeddings(xt, lt).cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


# ─────────────────────────────────────────────────────────────
# Decode random VQ codes + save + embed
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_and_embed(
    sampled_paths: List[str],         # random VQVAE .npy paths, one per test query
    query_ids: List[str],             # test motion IDs used as output filenames
    vae: torch.nn.Module,
    eval_wrapper: EvaluatorModelWrapper,
    device,
    max_frames: int = 196,
    batch_size: int = 64,
    save_joint_vecs_dir: Optional[str] = None,
    save_motion_tokens_dir: Optional[str] = None,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> np.ndarray:
    if save_joint_vecs_dir:
        os.makedirs(save_joint_vecs_dir,    exist_ok=True)
    if save_motion_tokens_dir:
        os.makedirs(save_motion_tokens_dir, exist_ok=True)

    all_joints, all_lens = [], []

    for qid, vq_path in tqdm(zip(query_ids, sampled_paths),
                              total=len(sampled_paths),
                              desc="Decoding random VQ codes"):
        if not os.path.isfile(vq_path):
            dummy = torch.zeros(4, 263)
            all_joints.append(dummy); all_lens.append(4)
            if save_joint_vecs_dir:
                np.save(os.path.join(save_joint_vecs_dir, f"{qid}.npy"),
                        dummy.numpy().astype(np.float32))
            if save_motion_tokens_dir:
                np.save(os.path.join(save_motion_tokens_dir, f"{qid}.npy"),
                        np.zeros(1, dtype=np.int32))
            continue

        codes  = np.load(vq_path, allow_pickle=False).astype(np.int64).reshape(-1)
        idx    = torch.from_numpy(codes).unsqueeze(0).to(device)
        pose   = vae.forward_decoder(idx)[0]              # [T, 263] normalized
        T      = min(pose.shape[0], max_frames)
        pose_t = pose[:T].cpu()
        all_joints.append(pose_t)
        all_lens.append(T)

        # Save denormalized joint vectors
        if save_joint_vecs_dir:
            j_np = pose_t.numpy().astype(np.float32)
            if mean is not None and std is not None:
                j_np = j_np * std + mean               # denormalize → original space
            np.save(os.path.join(save_joint_vecs_dir,    f"{qid}.npy"), j_np)

        # Save motion token codes
        if save_motion_tokens_dir:
            np.save(os.path.join(save_motion_tokens_dir, f"{qid}.npy"),
                    codes.astype(np.int32))

    # Batch-embed for FID (normalized space is correct for EvaluatorModelWrapper)
    feats = []
    for i in range(0, len(all_joints), batch_size):
        bj, bl = all_joints[i:i + batch_size], all_lens[i:i + batch_size]
        max_T  = max(bl)
        B      = len(bj)
        x      = torch.zeros(B, max_T, 263)
        for k, (j, L) in enumerate(zip(bj, bl)):
            x[k, :L] = j
        lt = torch.tensor(bl, dtype=torch.long, device=device)
        feats.append(eval_wrapper.get_motion_embeddings(x.to(device), lt).cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Random motion baseline eval")

    # Data
    ap.add_argument("--dataset_dir", required=True,
                    help="A2R dataset root (contains HumanML3D/VQVAE/ and new_joint_vecs/)")
    ap.add_argument("--pairs_csv",   required=True,
                    help="dir with test.csv or single csv with split column")
    ap.add_argument("--test_split",  default="test")
    ap.add_argument("--vqvae_pool_dir", default="",
                    help="Dir to sample random VQ .npy files from. "
                         "Default: {dataset_dir}/HumanML3D/VQVAE")

    # Models
    ap.add_argument("--t2m_opt",    required=True,
                    help="T2M opt.txt for EvaluatorModelWrapper")
    ap.add_argument("--vqvae_ckpt", required=True,
                    help="VQ-VAE checkpoint for decoding")

    # Normalization
    ap.add_argument("--mean_path", default=None)
    ap.add_argument("--std_path",  default=None)

    # Output
    ap.add_argument("--out_dir",   default="./out_random",
                    help="Root output dir:\n"
                         "  {out_dir}/random/joint_vecs/{test_id}.npy\n"
                         "  {out_dir}/random/motion_tokens/{test_id}.npy\n"
                         "  {out_dir}/random/metrics.json\n"
                         "  {out_dir}/summary.json")

    # Misc
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_frames", type=int, default=196)
    ap.add_argument("--seed",       type=int, default=42,
                    help="Random seed. Run with multiple seeds and average for stability.")
    ap.add_argument("--num_seeds",  type=int, default=1,
                    help="Run this many seeds (42..42+num_seeds-1) and report mean±std")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    joint_vecs_dir = os.path.join(args.dataset_dir, "HumanML3D", "new_joint_vecs")
    vqvae_pool_dir = args.vqvae_pool_dir or os.path.join(args.dataset_dir, "HumanML3D", "VQVAE")

    # Normalization
    mean_np = std_np = None
    if args.mean_path and args.std_path:
        if os.path.isfile(args.mean_path) and os.path.isfile(args.std_path):
            mean_np = np.load(args.mean_path).astype(np.float32)
            std_np  = np.load(args.std_path).astype(np.float32)
            print(f"[Norm] loaded Mean/Std")

    # Models
    eval_opt     = get_opt(args.t2m_opt, device)
    eval_wrapper = EvaluatorModelWrapper(eval_opt)
    print("[VQ-VAE] loading ...")
    vae = build_vqvae(args.vqvae_ckpt, device)

    # Scan VQVAE pool
    all_vq_paths = scan_vqvae_dir(vqvae_pool_dir)
    if not all_vq_paths:
        raise RuntimeError(f"No .npy files found in {vqvae_pool_dir}")

    # Load test set (unique listener motions)
    df = read_test_csv(args.pairs_csv, args.test_split)
    df["_mid"] = df["raw_file_name"].apply(motion_id_from_raw)
    df = df[df["_mid"] != "000000"].drop_duplicates("_mid").reset_index(drop=True)
    test_ids = df["_mid"].tolist()
    N_test   = len(test_ids)
    print(f"[Data] test unique motions = {N_test}")
    print(f"[Pool] {len(all_vq_paths)} VQ files available for random sampling")

    # Real test embeddings
    print("\n[Real] building test motion embeddings ...")
    real_emb = embed_real_motions(
        test_ids, joint_vecs_dir, eval_wrapper, device,
        max_frames=args.max_frames, mean=mean_np, std=std_np,
        batch_size=args.batch_size,
    )
    print(f"[Real] {real_emb.shape[0]} samples, dim={real_emb.shape[1]}")
    real_mu  = np.mean(real_emb, axis=0)
    real_cov = np.cov(real_emb, rowvar=False)
    div_real = all_pair_diversity(real_emb)

    # ── Run over seeds ────────────────────────────────────────
    all_fid  = []
    all_div  = []
    seed_results = {}

    for seed_offset in range(args.num_seeds):
        seed = args.seed + seed_offset
        rng  = np.random.default_rng(seed)

        # Randomly sample one VQ .npy path per test query
        sampled_paths = rng.choice(all_vq_paths, size=N_test, replace=True).tolist()
        n_unique = len(set(sampled_paths))
        print(f"\n[Seed {seed}] sampling {N_test} motions  ({n_unique} distinct)")

        # For the first seed: save joint_vecs + motion_tokens
        if seed_offset == 0 and args.out_dir:
            jv_dir = os.path.join(args.out_dir, "random", "joint_vecs")
            mt_dir = os.path.join(args.out_dir, "random", "motion_tokens")
        else:
            jv_dir = mt_dir = None

        gen_emb = decode_and_embed(
            sampled_paths, test_ids, vae, eval_wrapper, device,
            max_frames=args.max_frames, batch_size=args.batch_size,
            save_joint_vecs_dir=jv_dir, save_motion_tokens_dir=mt_dir,
            mean=mean_np, std=std_np,
        )

        gen_mu  = np.mean(gen_emb, axis=0)
        gen_cov = np.cov(gen_emb, rowvar=False)
        fid     = frechet_distance(real_mu, real_cov, gen_mu, gen_cov)
        div_gen = all_pair_diversity(gen_emb)

        print(f"  FID (random vs real):      {fid:.4f}")
        print(f"  Diversity (real):           {div_real:.4f}")
        print(f"  Diversity (random):         {div_gen:.4f}")

        all_fid.append(fid)
        all_div.append(div_gen)
        seed_results[f"seed_{seed}"] = dict(fid=fid, div_gen=div_gen, div_real=div_real)

        if jv_dir:
            print(f"  [Saved] joint_vecs    → {jv_dir}")
            print(f"  [Saved] motion_tokens → {mt_dir}")

    # ── Summary ──────────────────────────────────────────────
    fid_mean = float(np.mean(all_fid))
    fid_std  = float(np.std(all_fid))
    div_mean = float(np.mean(all_div))
    div_std  = float(np.std(all_div))

    print("\n\n========== Random Baseline Summary ==========")
    print(f"  Seeds:            {args.seed} … {args.seed + args.num_seeds - 1}")
    print(f"  Pool size:        {len(all_vq_paths)} VQ files")
    print(f"  FID  (mean±std):  {fid_mean:.4f} ± {fid_std:.4f}")
    print(f"  Div  (mean±std):  {div_mean:.4f} ± {div_std:.4f}")
    print(f"  Div  (real):      {div_real:.4f}")

    summary = dict(
        fid_mean=fid_mean, fid_std=fid_std,
        div_gen_mean=div_mean, div_gen_std=div_std,
        div_real=div_real,
        num_seeds=args.num_seeds,
        pool_size=len(all_vq_paths),
        per_seed=seed_results,
    )
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        # Per-mode metrics.json (compatible with run_eval_retrieval_win_rates.sh)
        mode_dir = os.path.join(args.out_dir, "random")
        os.makedirs(mode_dir, exist_ok=True)
        with open(os.path.join(mode_dir, "metrics.json"), "w") as f:
            json.dump({"fid": fid_mean, "div_gen": div_mean, "div_real": div_real}, f, indent=2)
        # Full summary
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Saved] {args.out_dir}/summary.json")
        print(f"[Saved] {args.out_dir}/random/metrics.json")
        if args.num_seeds == 1:
            print(f"\nRun with --num_seeds 3 for stable mean±std:")
            print(f"  python -m eval.eval_random ... --num_seeds 3 --seed 42")
        print(f"\nGet Win↑/Gen@3↑ (edit RETRIEVAL_ROOT in the script):")
        print(f"  RETRIEVAL_ROOT={args.out_dir}")
        print(f"  bash run_eval_retrieval_win_rates.sh")


if __name__ == "__main__":
    main()
