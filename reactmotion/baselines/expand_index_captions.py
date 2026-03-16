#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expand_index_captions.py

Usage:
  python -m eval.expand_index_captions \
    --index_in  /path/to/index_test.flat.jsonl \
    --index_out /path/to/index_test.flat.with_captions.jsonl

功能:
  逐行读取 index_*.flat.jsonl，把每条记录里的
    "caption_txt": "/path/to/file.txt"
  替换成:
    "caption_txt": "<file content>"

  其余字段保持不变。如果 caption_txt 不是字符串路径，或者文件不存在，则原样保留。
"""

import argparse
import json
import os
from typing import Any, Dict


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


def process(index_in: str, index_out: str) -> None:
    os.makedirs(os.path.dirname(index_out) or ".", exist_ok=True)
    n_in = n_out = 0

    with open(index_in, "r", encoding="utf-8") as fin, open(
        index_out, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                obj: Dict[str, Any] = json.loads(line)
            except Exception:
                # just copy raw line if it's not valid JSON
                fout.write(line + "\n")
                n_out += 1
                continue

            cap_field = obj.get("caption_txt", None)
            if isinstance(cap_field, str):
                obj["caption_txt"] = _read_caption(cap_field)

            # Reorder fields so that:
            #   ... other fields ...,
            #   "caption_txt": "...",
            #   "emotion": ...,
            #   "audio_code_path": ...,
            #   "motion_codes_npy": ...
            # and emotion / audio / motion are never placed before caption_txt.
            order_after = ["emotion", "audio_code_path", "motion_codes_npy"]
            new_obj: Dict[str, Any] = {}

            # 1) all keys except the ones we want to control explicitly
            for k, v in obj.items():
                if k in ("caption_txt", "emotion", "audio_code_path", "motion_codes_npy"):
                    continue
                new_obj[k] = v

            # 2) caption_txt
            if "caption_txt" in obj:
                new_obj["caption_txt"] = obj["caption_txt"]

            # 3) emotion / audio_code_path / motion_codes_npy (if present)
            for k in order_after:
                if k in obj:
                    new_obj[k] = obj[k]

            fout.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[Done] processed {n_in} lines -> {index_out} ({n_out} written)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--index_in",
        type=str,
        required=True,
        help="Path to index_test.flat.jsonl",
    )
    ap.add_argument(
        "--index_out",
        type=str,
        required=True,
        help="Output jsonl path with caption_txt expanded to actual text.",
    )
    args = ap.parse_args()

    process(args.index_in, args.index_out)


if __name__ == "__main__":
    main()

