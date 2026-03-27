#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/evaluate.py — Unified evaluation entry point for ReactMotion.

Two evaluation pipelines:
  1) Win-rate / Gen@3 / nDCG:  generate motion candidates, then score with JudgeNetwork
  2) FID / Diversity:  compare generated vs real motion in evaluator embedding space

Usage:
  # Run both pipelines:
  python -m reactmotion.eval.evaluate \
      --pipeline all \
      --gen_ckpt /path/to/generator_ckpt \
      --judge_ckpt /path/to/judge_ckpt \
      --dataset_dir /path/to/A2R \
      --pairs_csv /path/to/new_data \
      --audio_code_dir /path/to/audio_codes \
      --out_dir ./eval_output \
      --t2m_opt ./checkpoints/t2m/Comp_v6_KLD005/opt.txt \
      --vqvae_ckpt /path/to/motion_VQVAE/net_last.pth

  # Win-rate only:
  python -m reactmotion.eval.evaluate --pipeline winrate ...

  # FID/Diversity only:
  python -m reactmotion.eval.evaluate --pipeline fid ...
"""

import argparse
import os
import sys
import subprocess


def run_winrate(args):
    """
    Pipeline 1:  Generation  →  Judge scoring  →  Win-rate / Gen@3 / nDCG metrics.

    Step A: eval_reactmotion.py   — generate num_gen motion samples per group
    Step B: eval_reactmotion_with_judge.py — score gen dump, compute group metrics
    """
    gen_out = os.path.join(args.out_dir, "gen_dump")
    os.makedirs(gen_out, exist_ok=True)

    # --- Step A: generate ---
    gen_cmd = [
        sys.executable, "-m", "reactmotion.eval.eval_reactmotion",
        "--gen_ckpt", args.gen_ckpt,
        "--dataset_dir", args.dataset_dir,
        "--pairs_csv", args.pairs_csv,
        "--only_split", args.eval_split,
        "--cond_mode", args.cond_mode,
        "--audio_mode", args.audio_mode,
        "--num_gen", str(args.num_gen),
        "--source_len", str(args.source_len),
        "--gen_max_length", str(args.gen_max_length),
        "--out_dir", gen_out,
        "--seed", str(args.seed),
    ]
    if args.use_emotion:
        gen_cmd.append("--use_emotion")
    if args.audio_code_dir:
        gen_cmd += ["--audio_code_dir", args.audio_code_dir]
    if args.wav_dir:
        gen_cmd += ["--wav_dir", args.wav_dir]
    if args.audio_token_level:
        gen_cmd += ["--audio_token_level", args.audio_token_level]

    print("=" * 60)
    print("[Step 1/2] Generating motion candidates ...")
    print(f"  cmd: {' '.join(gen_cmd)}")
    print("=" * 60)
    ret = subprocess.run(gen_cmd, cwd=os.getcwd())
    if ret.returncode != 0:
        print(f"[ERROR] Generation failed with code {ret.returncode}")
        return

    # Find the gen dump index file
    gen_dump = os.path.join(gen_out, f"index_{args.eval_split}.flat.jsonl")
    if not os.path.isfile(gen_dump):
        # fallback: try group-level jsonl
        gen_dump = os.path.join(gen_out, f"index_{args.eval_split}.jsonl")
    if not os.path.isfile(gen_dump):
        print(f"[ERROR] Gen dump not found at {gen_dump}")
        return

    # --- Step B: score with JudgeNetwork ---
    judge_out = os.path.join(args.out_dir, "judge_metrics")
    os.makedirs(judge_out, exist_ok=True)

    judge_cmd = [
        sys.executable, "-m", "reactmotion.eval.eval_reactmotion_with_judge",
        "--ckpt", args.judge_ckpt,
        "--gen_dump", gen_dump,
        "--pairs_csv", args.pairs_csv,
        "--dataset_dir", args.dataset_dir,
        "--audio_code_dir", args.audio_code_dir,
        "--out_dir", judge_out,
        "--eval_split", args.eval_split,
        "--cond_head", args.cond_head,
        "--batch_size", str(args.batch_size),
        "--k_gold", str(args.k_gold),
        "--k_silver", str(args.k_silver),
        "--k_neg", str(args.k_neg),
        "--win_mode", args.win_mode,
        "--seed", str(args.seed),
    ]
    if args.fixed_mode:
        judge_cmd += ["--fixed_mode", args.fixed_mode]

    print("=" * 60)
    print("[Step 2/2] Scoring with JudgeNetwork ...")
    print(f"  cmd: {' '.join(judge_cmd)}")
    print("=" * 60)
    ret = subprocess.run(judge_cmd, cwd=os.getcwd())
    if ret.returncode != 0:
        print(f"[ERROR] Judge scoring failed with code {ret.returncode}")
        return

    # Print summary from group_metrics.csv
    metrics_csv = os.path.join(judge_out, "group_metrics.csv")
    if os.path.isfile(metrics_csv):
        try:
            import pandas as pd
            df = pd.read_csv(metrics_csv)
            if len(df) > 0:
                def _mean(col):
                    x = pd.to_numeric(df[col], errors="coerce")
                    return float(x.mean()) if x.notna().any() else float("nan")

                print("=" * 60)
                print("[Win-rate Results]")
                print(f"  Groups evaluated: {len(df)}")
                for col in ["win_gen_vs_neg", "win_gen_vs_silver", "win_gen_vs_gold", "gen_at3", "ndcg5"]:
                    if col in df.columns:
                        print(f"  {col:25s} = {_mean(col):.4f}")
                print("=" * 60)
            else:
                print("[WARN] group_metrics.csv is empty — no win-rate metrics computed.")
        except ImportError:
            print("[WARN] pandas not available; skipping metrics summary. See:", metrics_csv)
    else:
        print("[WARN] group_metrics.csv not found at:", metrics_csv)

    print("=" * 60)
    print("[Win-rate pipeline done]")
    print(f"  Gen dump:       {gen_out}")
    print(f"  Judge metrics:  {judge_out}")
    print("=" * 60)


def run_fid(args):
    """
    Pipeline 2:  FID / Diversity evaluation.

    Uses pre-generated motion codes (from gen dump) + real joint vecs
    to compute FID and Diversity in T2M evaluator embedding space.
    """
    gen_root = args.gen_root
    if not gen_root:
        # Default: look for gen dump from winrate pipeline
        gen_root = os.path.join(args.out_dir, "gen_dump")

    fid_cmd = [
        sys.executable, "-m", "reactmotion.eval.eval_fid_diversity",
        "--gen_root", gen_root,
        "--gen_format", args.gen_format,
        "--dataset_dir", args.dataset_dir,
        "--pairs_csv", args.pairs_csv,
        "--t2m_opt", args.t2m_opt,
        "--vqvae_ckpt", args.vqvae_ckpt,
        "--real_split", args.real_split,
    ]
    if args.mean_path:
        fid_cmd += ["--mean_path", args.mean_path]
    if args.std_path:
        fid_cmd += ["--std_path", args.std_path]

    print("=" * 60)
    print("[FID/Diversity] Computing metrics ...")
    print(f"  cmd: {' '.join(fid_cmd)}")
    print("=" * 60)
    ret = subprocess.run(fid_cmd, cwd=os.getcwd())
    if ret.returncode != 0:
        print(f"[ERROR] FID/Diversity eval failed with code {ret.returncode}")
        return

    print("=" * 60)
    print("[FID/Diversity pipeline done]")
    print("  (FID & Diversity values printed above by the sub-process)")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser(
        description="Unified ReactMotion evaluation: win-rate/Gen@3 + FID/Diversity"
    )

    ap.add_argument(
        "--pipeline", type=str, default="all",
        choices=["all", "winrate", "fid"],
        help="Which pipeline(s) to run. 'all' runs winrate then fid.",
    )
    ap.add_argument("--out_dir", type=str, default="./eval_output")

    # --- shared ---
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_split", type=str, default="test", choices=["val", "test"])

    # --- winrate pipeline ---
    ap.add_argument("--gen_ckpt", type=str, default="", help="Generator checkpoint path")
    ap.add_argument("--judge_ckpt", type=str, default="", help="JudgeNetwork checkpoint path")
    ap.add_argument("--audio_code_dir", type=str, default="")
    ap.add_argument("--wav_dir", type=str, default="")
    ap.add_argument("--cond_mode", type=str, default="t+e",
                    choices=["t", "t+e", "a", "a+e", "t+a", "t+a+e"])
    ap.add_argument("--audio_mode", type=str, default="none", choices=["none", "code", "wav"])
    ap.add_argument("--audio_token_level", type=str, default="base", choices=["base", "all", "rand"])
    ap.add_argument("--use_emotion", action="store_true")
    ap.add_argument("--num_gen", type=int, default=3)
    ap.add_argument("--source_len", type=int, default=512)
    ap.add_argument("--gen_max_length", type=int, default=256)
    ap.add_argument("--cond_head", type=str, default="fused", choices=["fused", "text", "audio", "emo"])
    ap.add_argument("--fixed_mode", type=str, default="")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--k_gold", type=int, default=3)
    ap.add_argument("--k_silver", type=int, default=2)
    ap.add_argument("--k_neg", type=int, default=5)
    ap.add_argument("--win_mode", type=str, default="mean", choices=["mean", "best", "worst"])

    # --- fid pipeline ---
    ap.add_argument("--gen_root", type=str, default="",
                    help="Gen motion root dir for FID. If empty, uses out_dir/gen_dump.")
    ap.add_argument("--gen_format", type=str, default="eval_dump", choices=["eval_dump", "qwen_t2m"])
    ap.add_argument("--t2m_opt", type=str, default="", help="T2M opt.txt for EvaluatorModelWrapper")
    ap.add_argument("--vqvae_ckpt", type=str, default="", help="Motion VQ-VAE checkpoint")
    ap.add_argument("--mean_path", type=str, default="")
    ap.add_argument("--std_path", type=str, default="")
    ap.add_argument("--real_split", type=str, default="test", choices=["train", "val", "test", "all"])

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.pipeline in ("all", "winrate"):
        if not args.gen_ckpt:
            ap.error("--gen_ckpt is required for winrate pipeline")
        if not args.judge_ckpt:
            ap.error("--judge_ckpt is required for winrate pipeline")
        run_winrate(args)

    if args.pipeline in ("all", "fid"):
        if not args.t2m_opt:
            ap.error("--t2m_opt is required for fid pipeline")
        if not args.vqvae_ckpt:
            ap.error("--vqvae_ckpt is required for fid pipeline")
        run_fid(args)

    print("\n[All done]")


if __name__ == "__main__":
    main()
