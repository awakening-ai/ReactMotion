#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train/train_joint_judge.py

Train JointJudge with:
- motion: HumanML3D new_joint_vecs (263-dim), normalized by mean/std
- text : T5 tokenizer + T5 encoder inside model
- audio: audio codes (.npz with key 'codes'), shape (K,T) or (T,K)
- emotion: hashed ids

Loss: in-batch InfoNCE for fused + per-modality heads (masked by has_*)

Val metrics: acc1/acc3/acc5/mrr/ndcg5 computed from fused logits
"""

import os
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import json
import math
import argparse
import random
from os.path import join as pjoin
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

from models.joint_judge_model import JointJudge, JointCondBatch


# --------------------------
# Helpers
# --------------------------
def normalize_text(x: Any) -> str:
    s = "" if pd.isna(x) else str(x)
    return " ".join(s.strip().split())


def motion_id_from_raw(raw_file_name: Any) -> str:
    x = "" if pd.isna(raw_file_name) else str(raw_file_name).strip()
    mid = x.split("_", 1)[0] if x else ""
    return str(mid).zfill(6) if mid else ""


def read_split_csv(pairs_csv_or_dir: str, split: str) -> pd.DataFrame:
    if os.path.isdir(pairs_csv_or_dir):
        path = pjoin(pairs_csv_or_dir, f"{split}.csv")
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing split csv: {path}")
        return pd.read_csv(path, encoding="utf-8")
    df = pd.read_csv(pairs_csv_or_dir, encoding="utf-8")
    if "split" not in df.columns:
        raise RuntimeError("pairs_csv is file but has no `split` column.")
    sp = df["split"].astype(str).str.lower().str.strip()
    return df[sp == split].copy()


def clean_audio_stem(x: Any) -> str:
    s = "" if pd.isna(x) else str(x).strip()
    if s.lower().endswith(".wav"):
        s = s[:-4]
    if s.lower().endswith(".mp3"):
        s = s[:-4]
    return s


def pick_audio_code_path(audio_code_dir: str, stem: str) -> str:
    """
    Try {stem}.npz or {stem}.npy.
    Return "" if not found.
    """
    stem = str(stem).strip()
    if not stem:
        return ""
    p1 = pjoin(audio_code_dir, stem + ".npz")
    if os.path.isfile(p1):
        return p1
    p2 = pjoin(audio_code_dir, stem + ".npy")
    if os.path.isfile(p2):
        return p2
    # sometimes input already contains extension
    if os.path.isfile(pjoin(audio_code_dir, stem)):
        return pjoin(audio_code_dir, stem)
    return ""


def load_audio_codes_np(path: str) -> np.ndarray:
    """
    Your npz keys: ['codes', 'num_codebooks', ...]
    codes shape example: (8, 96) int32
    Return: np.int64 array [T,K]
    """
    obj = np.load(path, allow_pickle=False)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if "codes" in obj.files:
            arr = obj["codes"]
        else:
            arr = obj[obj.files[0]]
        obj.close()
    else:
        arr = obj

    a = np.asarray(arr)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    elif a.ndim == 2:
        pass
    else:
        a = a.reshape(a.shape[0], -1)

    # heuristic to [T,K]
    # common cases:
    #   (K,T) -> transpose
    #   (T,K) -> keep
    if a.shape[0] <= 16 and a.shape[1] > a.shape[0]:
        # likely (K,T)
        a = a.transpose(1, 0)

    return a.astype(np.int64)


def pad_audio_codes(a_tk: np.ndarray, max_len: int, codebooks: int, pad_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    a_tk: [T,K] (K may be != codebooks)
    Returns:
      codes: [max_len, codebooks]
      pad_mask: [max_len] True=pad
    """
    a = np.asarray(a_tk)
    if a.ndim != 2:
        a = a.reshape(a.shape[0], -1)
    T, K = a.shape

    # adjust K to codebooks
    if K < codebooks:
        padk = np.full((T, codebooks - K), pad_id, dtype=np.int64)
        a = np.concatenate([a, padk], axis=1)
    elif K > codebooks:
        a = a[:, :codebooks]

    T_use = min(T, max_len)
    out = np.full((max_len, codebooks), pad_id, dtype=np.int64)
    out[:T_use] = a[:T_use]
    pad_mask = np.ones((max_len,), dtype=np.bool_)
    pad_mask[:T_use] = False
    return out, pad_mask


# --------------------------
# Dataset
# --------------------------
class JudgeDataset(Dataset):
    """
    Each sample = one group:
      - sayings + emotion
      - one gold motion id -> joint vec
      - one audio code file (npz/npy)
    Negatives: in-batch (other samples' motions)
    """

    def __init__(
        self,
        split: str,
        pairs_csv: str,
        dataset_dir: str,
        audio_code_dir: str,
        key_by: str = "group_id",
        max_motion_len: int = 196,
        max_text_len: int = 64,
    ):
        super().__init__()
        assert split in ["train", "val", "test"]
        assert key_by in ["group_id", "sayings_emotion"]

        self.split = split
        self.dataset_dir = dataset_dir
        self.joint_dir = pjoin(dataset_dir, "HumanML3D", "new_joint_vecs")
        self.audio_code_dir = audio_code_dir
        self.max_motion_len = int(max_motion_len)
        self.max_text_len = int(max_text_len)
        self.epoch = 0

        # motion norm stats (same as your previous script)
        self.motion_meta_dir = "/ibex/project/c2191/luoc/dataset/A2R/HumanML3D/"
        self.motion_mean = np.load(pjoin(self.motion_meta_dir, "mean.npy"))
        self.motion_std = np.load(pjoin(self.motion_meta_dir, "std.npy"))

        df = read_split_csv(pairs_csv, split).copy()
        need_cols = ["label", "sayings", "emotion", "raw_file_name", "generated_wav_name"]
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing columns: {missing}. Found: {list(df.columns)}")
        if key_by == "group_id" and "group_id" not in df.columns:
            raise RuntimeError("key_by='group_id' but csv missing `group_id` column")

        df["sayings"] = df["sayings"].map(normalize_text)
        df["emotion"] = df["emotion"].astype(str)
        df["motion_id"] = df["raw_file_name"].apply(motion_id_from_raw)
        df["audio_stem"] = df["generated_wav_name"].map(clean_audio_stem)

        if key_by == "group_id":
            df["group_key"] = df["group_id"].astype(str)
        else:
            df["group_key"] = df["sayings"] + "|||" + df["emotion"]

        groups: Dict[str, Dict[str, Any]] = {}
        for _, r in df.iterrows():
            k = str(r["group_key"])
            g = groups.get(k)
            if g is None:
                g = dict(
                    sayings=str(r["sayings"]),
                    emotion=str(r["emotion"]),
                    audio_stems=[],
                    gold_motion_ids=[],
                )
                groups[k] = g

            st = str(r["audio_stem"]).strip()
            if st:
                g["audio_stems"].append(st)

            if str(r["label"]).strip().lower() in {"gold", "pos", "positive", "gt", "true", "1"}:
                mid = str(r["motion_id"])
                if mid:
                    g["gold_motion_ids"].append(mid)

        self.items: List[Dict[str, Any]] = []
        drops = {"no_gold": 0, "no_audio": 0}
        for k, g in groups.items():
            # unique stems
            stems = list(dict.fromkeys([s for s in g["audio_stems"] if str(s).strip()]))
            audio_paths = [pick_audio_code_path(self.audio_code_dir, s) for s in stems]
            audio_paths = [p for p in audio_paths if p]
            audio_paths = list(dict.fromkeys(audio_paths))

            if not g["gold_motion_ids"]:
                drops["no_gold"] += 1
                continue
            if len(audio_paths) == 0:
                # allow missing audio: has_a will be false
                drops["no_audio"] += 1

            self.items.append(dict(
                key=k,
                sayings=g["sayings"],
                emotion=g["emotion"],
                audio_paths=audio_paths,   # may be empty
                gold_motion_ids=g["gold_motion_ids"],
            ))

        print(f"[JudgeDataset] split={split} groups={len(self.items)} drops={drops}")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def forward_transform(self, data):
        return (data - self.motion_mean) / self.motion_std

    def __len__(self) -> int:
        return len(self.items)

    def _load_joint(self, motion_id: str) -> Tuple[torch.Tensor, int]:
        path = pjoin(self.joint_dir, f"{motion_id}.npy")
        if not os.path.isfile(path):
            raise RuntimeError(f"Missing joint vec: {path}")
        arr = np.load(path)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        T, D = arr.shape
        if D != 263:
            raise RuntimeError(f"Expected dim=263, got {arr.shape}")
        T_use = min(T, self.max_motion_len)
        out = np.zeros((self.max_motion_len, D), dtype=np.float32)
        out[:T_use] = arr[:T_use]
        # out = self.forward_transform(out)
        return torch.from_numpy(out), T_use

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        rng = random.Random(self.epoch * 1_000_003 ^ idx)

        mid = rng.choice(it["gold_motion_ids"])
        joint_seq, T_use = self._load_joint(mid)

        audio_path = ""
        if len(it["audio_paths"]) > 0:
            audio_path = rng.choice(it["audio_paths"])

        return dict(
            key=it["key"],
            sayings=it["sayings"],
            emotion=it["emotion"],
            audio_code_path=audio_path,
            joint_seq=joint_seq,
            joint_len=T_use,
        )


# --------------------------
# Collator
# --------------------------
class JudgeCollator:
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        max_text_len: int = 64,
        emo_vocab: int = 128,
        audio_codebooks: int = 8,
        audio_vocab: int = 2048,
        audio_pad_id: int = 2048,
        max_audio_len: int = 512,
    ):
        self.tok = tokenizer
        self.max_text_len = int(max_text_len)
        self.emo_vocab = int(emo_vocab)

        self.audio_codebooks = int(audio_codebooks)
        self.audio_vocab = int(audio_vocab)
        self.audio_pad_id = int(audio_pad_id)
        self.max_audio_len = int(max_audio_len)

    def encode_emotion(self, emo: str) -> int:
        return hash(str(emo).strip().lower()) % self.emo_vocab

    @staticmethod
    def _repair_only_e(has_t: torch.Tensor, has_a: torch.Tensor, has_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        only_e = has_e & (~has_t) & (~has_a)
        if only_e.any():
            has_t = has_t | only_e
        none = (~has_t) & (~has_a) & (~has_e)
        if none.any():
            has_t = has_t | none
        return has_t, has_a, has_e

    def __call__(self, batch: List[Dict[str, Any]]) -> Tuple[JointCondBatch, List[str]]:
        B = len(batch)

        # ---- text ----
        sayings = [b["sayings"] for b in batch]
        tok = self.tok(
            sayings,
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        # ---- emotion ----
        emotion_ids = torch.tensor([self.encode_emotion(b["emotion"]) for b in batch], dtype=torch.long)

        # ---- audio codes ----
        audio_list, mask_list, has_a_list = [], [], []
        for b in batch:
            p = str(b.get("audio_code_path", "")).strip()
            if p and os.path.isfile(p):
                a = load_audio_codes_np(p)  # [T,K] int64
                codes, pad_mask = pad_audio_codes(
                    a, max_len=self.max_audio_len, codebooks=self.audio_codebooks, pad_id=self.audio_pad_id
                )
                audio_list.append(torch.from_numpy(codes))
                mask_list.append(torch.from_numpy(pad_mask))
                has_a_list.append(True)
            else:
                codes = np.full((self.max_audio_len, self.audio_codebooks), self.audio_pad_id, dtype=np.int64)
                pad_mask = np.ones((self.max_audio_len,), dtype=np.bool_)
                audio_list.append(torch.from_numpy(codes))
                mask_list.append(torch.from_numpy(pad_mask))
                has_a_list.append(False)

        audio_codes = torch.stack(audio_list, dim=0)          # [B,Ta,K]
        audio_pad_mask = torch.stack(mask_list, dim=0)        # [B,Ta] True=pad

        # ---- motion joints ----
        joint_seqs = torch.stack([b["joint_seq"] for b in batch], dim=0)  # [B,Tm,263]
        joint_lens = torch.tensor([b["joint_len"] for b in batch], dtype=torch.long)

        # ---- modality flags ----
        has_t = torch.ones((B,), dtype=torch.bool)            # always have text
        has_e = torch.ones((B,), dtype=torch.bool)            # always have emotion
        has_a = torch.tensor(has_a_list, dtype=torch.bool)

        has_t, has_a, has_e = self._repair_only_e(has_t, has_a, has_e)

        cb = JointCondBatch(
            has_t=has_t,
            has_a=has_a,
            has_e=has_e,
            text_input_ids=tok["input_ids"],
            text_attn_mask=tok["attention_mask"],
            emotion_ids=emotion_ids,
            audio_codes=audio_codes,
            audio_pad_mask=audio_pad_mask,
            joint_seq=joint_seqs,
            joint_lens=joint_lens,
        )

        keys = [b["key"] for b in batch]
        return cb, keys


def _move_cb(cb: JointCondBatch, device: torch.device) -> JointCondBatch:
    return JointCondBatch(
        has_t=cb.has_t.to(device, non_blocking=True),
        has_a=cb.has_a.to(device, non_blocking=True),
        has_e=cb.has_e.to(device, non_blocking=True),
        text_input_ids=cb.text_input_ids.to(device, non_blocking=True),
        text_attn_mask=cb.text_attn_mask.to(device, non_blocking=True),
        emotion_ids=cb.emotion_ids.to(device, non_blocking=True),
        audio_codes=cb.audio_codes.to(device, non_blocking=True),
        audio_pad_mask=cb.audio_pad_mask.to(device, non_blocking=True),
        joint_seq=cb.joint_seq.to(device, non_blocking=True),
        joint_lens=cb.joint_lens.to(device, non_blocking=True),
    )


# --------------------------
# Metrics (in-batch)
# --------------------------
@torch.no_grad()
def inbatch_metrics(logits: torch.Tensor) -> Dict[str, float]:
    """
    logits: [B,B] diagonal correct
    """
    B = logits.size(0)
    device = logits.device
    labels = torch.arange(B, device=device)

    def acc_at(k: int) -> float:
        kk = min(k, B)
        topk = torch.topk(logits, k=kk, dim=1).indices
        hits = topk.eq(labels.unsqueeze(1)).any(dim=1).float()
        return hits.mean().item()

    # MRR
    order = logits.argsort(dim=1, descending=True)
    ranks = (order == labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    mrr = (1.0 / ranks.float()).mean().item()

    # NDCG@5, single positive per query
    k5 = min(5, B)
    top5 = torch.topk(logits, k=k5, dim=1).indices
    hit5 = top5.eq(labels.unsqueeze(1)).float()
    denom = torch.log2(torch.arange(k5, dtype=torch.float32, device=device) + 2.0)
    idcg = 1.0 / math.log2(2.0)
    ndcg5 = (hit5 / denom).sum(dim=1).div(idcg).mean().item()

    return dict(acc1=acc_at(1), acc3=acc_at(3), acc5=acc_at(5), mrr=mrr, ndcg5=ndcg5)


# --------------------------
# Loss compute
# --------------------------
def compute_loss_and_logits(
    model: JointJudge,
    cb: JointCondBatch,
    w_fused: float,
    w_text: float,
    w_audio: float,
    w_emo: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z_t, z_a, z_e, z_f = model.encode_modalities(cb)
    z_m = model.encode_motion(cb.joint_seq, cb.joint_lens)

    loss = torch.tensor(0.0, device=z_f.device)
    w_sum = 0.0

    # fused always (keep all)
    if w_fused > 0:
        loss = loss + w_fused * model.info_nce_loss(z_f, z_m, keep_mask=None)
        w_sum += w_fused

    # masked per modality
    if w_text > 0:
        lt = model.info_nce_loss(z_t, z_m, keep_mask=cb.has_t.to(z_t.device))
        loss = loss + w_text * lt
        w_sum += w_text

    if w_audio > 0:
        la = model.info_nce_loss(z_a, z_m, keep_mask=cb.has_a.to(z_a.device))
        loss = loss + w_audio * la
        w_sum += w_audio

    if w_emo > 0:
        le = model.info_nce_loss(z_e, z_m, keep_mask=cb.has_e.to(z_e.device))
        loss = loss + w_emo * le
        w_sum += w_emo

    if w_sum > 0:
        loss = loss / w_sum

    logits = model._scale() * (z_f @ z_m.t())
    return loss, logits


@torch.no_grad()
def run_eval(
    model: JointJudge,
    loader: DataLoader,
    device: torch.device,
    w_fused: float,
    w_text: float,
    w_audio: float,
    w_emo: float,
) -> Dict[str, float]:
    model.eval()
    losses, a1, a3, a5, mrrs, ndcgs = [], [], [], [], [], []

    for cb, _keys in loader:
        cb = _move_cb(cb, device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
            loss, logits = compute_loss_and_logits(model, cb, w_fused, w_text, w_audio, w_emo)
        losses.append(float(loss.item()))
        m = inbatch_metrics(logits.float())
        a1.append(m["acc1"]); a3.append(m["acc3"]); a5.append(m["acc5"])
        mrrs.append(m["mrr"]); ndcgs.append(m["ndcg5"])

    def mean(xs):
        xs = [x for x in xs if x == x]
        return float(np.mean(xs)) if xs else float("nan")

    return dict(
        val_loss=mean(losses),
        acc1=mean(a1),
        acc3=mean(a3),
        acc5=mean(a5),
        mrr=mean(mrrs),
        ndcg5=mean(ndcgs),
    )


# --------------------------
# LR schedule
# --------------------------
def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--audio_code_dir", type=str, required=True)
    ap.add_argument("--save_dir", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)

    # model / tokenizer
    ap.add_argument("--t5_name", type=str, default="google-t5/t5-base")
    ap.add_argument("--freeze_text", action="store_true", default=True)
    ap.add_argument("--no_freeze_text", dest="freeze_text", action="store_false")

    # audio code params
    ap.add_argument("--audio_vocab", type=int, default=2048)
    ap.add_argument("--audio_pad_id", type=int, default=2048)
    ap.add_argument("--audio_codebooks", type=int, default=8)
    ap.add_argument("--max_audio_len", type=int, default=512)
    ap.add_argument("--audio_downsample", type=int, default=1)

    # dataset lengths
    ap.add_argument("--max_text_len", type=int, default=64)
    ap.add_argument("--max_motion_len", type=int, default=196)

    # modality dropout
    ap.add_argument("--cond_dropout", type=float, default=0.20)

    # loss weights
    ap.add_argument("--w_fused", type=float, default=1.0)
    ap.add_argument("--w_text", type=float, default=0.5)
    ap.add_argument("--w_audio", type=float, default=0.5)
    ap.add_argument("--w_emo", type=float, default=0.2)

    # motion ckpt
    ap.add_argument("--motion_ckpt_path", type=str,
                    default="./checkpoints/t2m/Comp_v6_KLD005/text_mot_match/model/finest.tar")
    ap.add_argument("--train_motion_encoder", action="store_true", default=True)
    ap.add_argument("--no_train_motion_encoder", dest="train_motion_encoder", action="store_false")

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    # datasets
    print("[Dataset] building ...")
    train_ds = JudgeDataset(
        split="train",
        pairs_csv=args.pairs_csv,
        dataset_dir=args.dataset_dir,
        audio_code_dir=args.audio_code_dir,
        max_motion_len=args.max_motion_len,
        max_text_len=args.max_text_len,
    )
    val_ds = JudgeDataset(
        split="val",
        pairs_csv=args.pairs_csv,
        dataset_dir=args.dataset_dir,
        audio_code_dir=args.audio_code_dir,
        max_motion_len=args.max_motion_len,
        max_text_len=args.max_text_len,
    )

    # tokenizer + collator
    print("[Tokenizer] load T5 tokenizer ...")
    tok = T5Tokenizer.from_pretrained(args.t5_name)
    collator = JudgeCollator(
        tokenizer=tok,
        max_text_len=args.max_text_len,
        emo_vocab=128,
        audio_codebooks=args.audio_codebooks,
        audio_vocab=args.audio_vocab,
        audio_pad_id=args.audio_pad_id,
        max_audio_len=args.max_audio_len,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False,
    )

    # model
    print("[Model] init JointJudge ...")
    model = JointJudge(
        t5_name=args.t5_name,
        freeze_text=bool(args.freeze_text),
        audio_vocab=args.audio_vocab,
        audio_pad_id=args.audio_pad_id,
        audio_codebooks=args.audio_codebooks,
        audio_max_len=args.max_audio_len,
        audio_downsample=args.audio_downsample,
        cond_dropout=args.cond_dropout,
        motion_ckpt_path=args.motion_ckpt_path,
        train_motion_encoder=bool(args.train_motion_encoder),
        device=device,
    )

    # collect trainable
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Params] trainable={n_trainable/1e6:.2f}M / total={n_total/1e6:.2f}M")

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    sch = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    print(f"[Schedule] warmup={warmup_steps} total={total_steps} peak_lr={args.lr:.2e}")

    best_ndcg5 = -1.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_ds.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for cb, _keys in pbar:
            global_step += 1
            cb = _move_cb(cb, device)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                loss, logits = compute_loss_and_logits(
                    model, cb, args.w_fused, args.w_text, args.w_audio, args.w_emo
                )
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            sch.step()

            with torch.no_grad():
                m = inbatch_metrics(logits.float())
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc1=f"{m['acc1']:.3f}",
                    ndcg5=f"{m['ndcg5']:.3f}",
                    lr=f"{sch.get_last_lr()[0]:.2e}",
                    ha=f"{cb.has_a.float().mean().item():.2f}",
                )

            if global_step % args.eval_every == 0:
                metrics = run_eval(model, val_loader, device, args.w_fused, args.w_text, args.w_audio, args.w_emo)
                print(
                    f"\n[VAL step={global_step}] "
                    f"loss={metrics['val_loss']:.4f} "
                    f"acc1={metrics['acc1']:.3f} "
                    f"acc3={metrics['acc3']:.3f} "
                    f"acc5={metrics['acc5']:.3f} "
                    f"mrr={metrics['mrr']:.3f} "
                    f"ndcg5={metrics['ndcg5']:.3f}"
                )

                ckpt = dict(
                    step=global_step,
                    epoch=epoch,
                    model_state=model.state_dict(),
                    opt_state=opt.state_dict(),
                    sch_state=sch.state_dict(),
                    best_ndcg5=best_ndcg5,
                    val_metrics=metrics,
                    args=vars(args),
                )

                tmp = pjoin(args.save_dir, "joint_judge_curr.pt.tmp")
                cur = pjoin(args.save_dir, "joint_judge_curr.pt")
                torch.save(ckpt, tmp)
                os.replace(tmp, cur)

                with open(pjoin(args.save_dir, "cur_metrics.json"), "w") as f:
                    json.dump({"epoch": epoch, "step": global_step, **metrics}, f, indent=2)

                if metrics["ndcg5"] > best_ndcg5:
                    best_ndcg5 = metrics["ndcg5"]
                    ckpt["best_ndcg5"] = best_ndcg5
                    best_path = pjoin(args.save_dir, "joint_judge_best.pt")
                    torch.save(ckpt, best_path)
                    print(f"[Best] ndcg5={best_ndcg5:.4f} -> saved {best_path}")

                model.train()

    print(f"[Done] best_ndcg5={best_ndcg5:.4f}")


if __name__ == "__main__":
    main()