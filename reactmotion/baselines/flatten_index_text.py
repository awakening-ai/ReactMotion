#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flatten_index_text.py

把 run_eval_gen_dump.sh 生成的 index_test.jsonl (text-only 版本, 结构如):

  {
    "cond_mode": "t",
    "gen_ckpt": "...",
    "ckpt_hash": "40bfcbd983",
    "split": "test",
    "key": "12142",
    "group_hash": "c22c28b2e802",
    "num_gen": 3,
    "items": [
      {
        "motion_codes_npy": "...gen00.motion_codes.npy",
        "raw_output_txt": "...gen00.raw_model_output.txt",
        "caption_txt": "...gen00.caption.txt"
      },
      ...
    ]
  }

展平成一行一条生成样本, 并把 caption_txt 从“路径”展开成实际文本, 同时从
new_data/test.csv 里根据 key(=group_id) 取 speaker sayings:

输出 (一行一个 JSON):
  {
    "key": "12142",
    "ckpt_hash": "40bfcbd983",
    "cond_mode": "t",
    "group_hash": "c22c28b2e802",
    "speaker_sayings": "A call came through saying ...",
    "caption_txt": "a person moves their hand to their mouth then lowers it",
    "motion_codes_npy": "/.../gen00.motion_codes.npy"
  }
"""

import argparse
import json
import os
from typing import Any, Dict

import pandas as pd


def _read_caption(path: str) -> str:
    """Read a caption txt file safely; return original path on failure."""
    try:
        if not path or not isinstance(path, str):
            return path
        if not os.path.isfile(path):
            return path
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        return txt
    except Exception:
        return path


def _load_sayings_map(test_csv: str) -> Dict[str, str]:
    df = pd.read_csv(test_csv, encoding="utf-8")
    if "group_id" not in df.columns or "sayings" not in df.columns:
        raise RuntimeError(
            f"CSV {test_csv} 必须包含 'group_id' 和 'sayings' 两列, 但现在列为: {list(df.columns)}"
        )
    # group_id 统一转成字符串
    mp: Dict[str, str] = {}
    for _, r in df.iterrows():
        gid = str(r["group_id"])
        mp[gid] = str(r["sayings"])
    return mp


def process(index_in: str, test_csv: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    gid2sayings = _load_sayings_map(test_csv)
    n_in = n_out = 0

    with open(index_in, "r", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj: Dict[str, Any] = json.loads(line)
            except Exception:
                # 非 JSON 行直接跳过
                continue

            cond_mode = str(obj.get("cond_mode", ""))
            ckpt_hash = str(obj.get("ckpt_hash", ""))
            split = str(obj.get("split", ""))
            key = str(obj.get("key", ""))
            group_hash = str(obj.get("group_hash", ""))

            # 从 CSV 查 sayings
            speaker_sayings = gid2sayings.get(key, "")

            items = obj.get("items", [])
            if not isinstance(items, list):
                items = []

            for it in items:
                motion_codes_npy = str(it.get("motion_codes_npy", ""))
                cap_path = str(it.get("caption_txt", ""))
                caption_txt = _read_caption(cap_path)

                rec = {
                    "key": key,
                    "ckpt_hash": ckpt_hash,
                    "cond_mode": cond_mode,
                    "split": split,
                    "group_hash": group_hash,
                    "speaker_sayings": speaker_sayings,
                    "caption_txt": caption_txt,
                    "motion_codes_npy": motion_codes_npy,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_out += 1

    print(
        f"[Done] read {n_in} index lines from {index_in}, "
        f"wrote {n_out} flattened samples to {out_path}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index_in",
        type=str,
        required=True,
        help="Path to index_test.jsonl (run_eval_gen_dump 的概览文件).",
    )
    ap.add_argument(
        "--test_csv",
        type=str,
        required=True,
        help="new_data/test.csv (包含 group_id 和 sayings 列).",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="输出的 flattened jsonl 路径.",
    )
    args = ap.parse_args()
    process(args.index_in, args.test_csv, args.out)


if __name__ == "__main__":
    main()

