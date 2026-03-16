#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reactmotion/train/train_judge.py

Training script for JudgeNetwork (multi-modal scorer/ranker for best-of-K selection).
"""

import os, json, argparse
from collections import deque
from os.path import join as pjoin
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import T5Tokenizer

from reactmotion.models.judge_network import (
    MODES_FULL, MODE2ID,
    DEFAULT_AUDIO_VOCAB, DEFAULT_AUDIO_PAD, DEFAULT_MOTION_VOCAB,
    seed_everything, read_split_csv,
    CondBatch, GroupBatch, move_cb_to,
    JudgeGroupDataset, GroupCollator, JudgeNetwork,
    group_infonce_loss, group_infonce_loss_with_bank,
    in_group_order_margin_loss, alignment_reg,
    acc_at_k_any_gold, ndcg_at_k, run_eval,
)


# =========================================================
# Generic bank (store motion embeddings)
# =========================================================

class GenericMotionBank:
    """
    Stores motion embeddings z_m as generic negatives.
    Keep it diverse; don't let a single template dominate.
    """
    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.buf = deque()

    def __len__(self):
        return len(self.buf)

    @torch.no_grad()
    def add(self, z: torch.Tensor):
        z = z.detach().float().cpu()
        for i in range(z.size(0)):
            self.buf.append(z[i])
        while len(self.buf) > self.max_size:
            self.buf.popleft()

    @torch.no_grad()
    def get(self, device: torch.device) -> Optional[torch.Tensor]:
        if len(self.buf) == 0:
            return None
        z = torch.stack(list(self.buf), dim=0)
        return z.to(device, non_blocking=True)


# =========================================================
# Training
# =========================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--audio_code_dir", type=str, required=True)
    ap.add_argument("--key_by", type=str, default="group_id", choices=["group_id", "sayings_emotion"])
    ap.add_argument("--t5_encoder", type=str, default="google-t5/t5-base")

    ap.add_argument("--k_gold", type=int, default=3)
    ap.add_argument("--k_silver", type=int, default=2)
    ap.add_argument("--k_neg", type=int, default=5)
    ap.add_argument("--require_audio", action="store_true")

    # whether to treat silver as positive (NOT recommended for best-of-K reranking)
    ap.add_argument("--use_silver_as_pos", action="store_true")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--max_text_len", type=int, default=128)
    ap.add_argument("--max_audio_len", type=int, default=512)
    ap.add_argument("--max_motion_len", type=int, default=196)

    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--output_dim", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=12)
    ap.add_argument("--enc_layers", type=int, default=6)
    ap.add_argument("--ff_dim", type=int, default=3072)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07)

    # training steps control
    ap.add_argument("--epochs", type=int, default=2000)  # kept for compatibility
    ap.add_argument("--max_steps", type=int, default=200000)  # stop by steps first
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=200)

    # forced single modality ratio (important for scorer robustness)
    ap.add_argument("--force_single_ratio", type=float, default=0.4,
                    help="Prob of forcing a sample to be single-modality (t-only or a-only).")

    # ablation disables
    ap.add_argument("--disable_text", action="store_true")
    ap.add_argument("--disable_audio", action="store_true")
    ap.add_argument("--disable_emo", action="store_true")

    # modality loss weights
    ap.add_argument("--w_fused", type=float, default=1.0)
    ap.add_argument("--w_text", type=float, default=0.3)
    ap.add_argument("--w_audio", type=float, default=0.3)
    ap.add_argument("--w_emo", type=float, default=0.1)

    # alignment (agreement) regularizer weight
    ap.add_argument("--w_align", type=float, default=0.01)

    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--freeze_text", action="store_true")

    # in-group ordering loss
    ap.add_argument("--w_ord", type=float, default=0.1)
    ap.add_argument("--m_gs", type=float, default=0.2)
    ap.add_argument("--m_sn", type=float, default=0.2)

    # bank negatives
    ap.add_argument("--bank_size", type=int, default=4096)
    ap.add_argument("--bank_warmup", type=int, default=1000)
    ap.add_argument("--bank_update_k", type=int, default=8)
    ap.add_argument("--bank_topk", type=int, default=5,
                    help="For each sample, pick candidates from topK negatives by score, then RANDOM pick 1.")
    ap.add_argument("--bank_alpha", type=float, default=1.0,
                    help="Strength multiplier on bank logits only (denominator).")

    args = ap.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    print("[Device]", device)

    # emotion vocab from train+val
    df_tr = read_split_csv(args.pairs_csv, "train")
    df_va = read_split_csv(args.pairs_csv, "val")
    emos = sorted(set([str(x).strip().lower() for x in pd.concat([df_tr["emotion"], df_va["emotion"]]).tolist()
                       if str(x).strip()]))
    emo2id = {"<unk>": 0}
    for e in emos:
        if e not in emo2id:
            emo2id[e] = len(emo2id)
    print("[Emotion] size =", len(emo2id))

    train_set = JudgeGroupDataset("train", args.pairs_csv, args.dataset_dir, args.audio_code_dir,
                                 key_by=args.key_by, seed=args.seed,
                                 k_gold=args.k_gold, k_silver=args.k_silver, k_neg=args.k_neg,
                                 require_audio=bool(args.require_audio))
    val_set = JudgeGroupDataset("val", args.pairs_csv, args.dataset_dir, args.audio_code_dir,
                               key_by=args.key_by, seed=args.seed + 999,
                               k_gold=args.k_gold, k_silver=args.k_silver, k_neg=args.k_neg,
                               require_audio=bool(args.require_audio))

    tok = T5Tokenizer.from_pretrained(args.t5_encoder)

    train_collate = GroupCollator(
        tok, emo2id,
        max_text_len=args.max_text_len,
        max_audio_len=args.max_audio_len,
        max_motion_len=args.max_motion_len,
        seed=args.seed,
        deterministic_mode=False,
        force_single_ratio=args.force_single_ratio,
        disable_text=args.disable_text,
        disable_audio=args.disable_audio,
        disable_emo=args.disable_emo,
    )
    val_collate = GroupCollator(
        tok, emo2id,
        max_text_len=args.max_text_len,
        max_audio_len=args.max_audio_len,
        max_motion_len=args.max_motion_len,
        seed=args.seed + 999,
        deterministic_mode=True,
        force_single_ratio=0.0,
        disable_text=args.disable_text,
        disable_audio=args.disable_audio,
        disable_emo=args.disable_emo,
    )

    # weighted sampling by group_w
    group_weights = torch.tensor([float(g.get("group_w", 1.0)) for g in train_set.groups], dtype=torch.double)
    sampler = WeightedRandomSampler(weights=group_weights, num_samples=len(train_set), replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collate
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_collate
    )

    model = JudgeNetwork(
        t5_name_or_path=args.t5_encoder,
        num_emotions=len(emo2id),
        d_model=args.d_model,
        output_dim=args.output_dim,
        nhead=args.nhead,
        enc_layers=args.enc_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        audio_vocab=DEFAULT_AUDIO_VOCAB,
        audio_pad_id=DEFAULT_AUDIO_PAD,
        audio_codebooks=8,
        max_audio_len=args.max_audio_len,
        motion_vocab=DEFAULT_MOTION_VOCAB,
        max_motion_len=args.max_motion_len,
        temperature=args.temperature,
    ).to(device)

    if args.freeze_text:
        model.freeze_text_encoder(True)
        print("[Freeze] text_enc frozen")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
    print(f"[Opt] trainable params: {sum(p.numel() for p in trainable_params)/1e6:.2f}M")

    global_step = 0
    best_val = float("inf")
    bank = GenericMotionBank(max_size=args.bank_size)

    print("[Train] start")
    for ep in range(1, args.epochs + 1):
        model.train()
        train_set.set_epoch(ep)
        val_set.set_epoch(ep)

        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")
        for gb in pbar:
            global_step += 1
            if global_step > int(args.max_steps):
                print(f"[Stop] reached max_steps={args.max_steps}")
                print("[Done]")
                return

            cb = move_cb_to(gb.cb, device)
            B, C, Tm = gb.motion_codes.shape
            mc = gb.motion_codes.view(B*C, Tm).to(device, non_blocking=True)
            mp = gb.motion_pad.view(B*C, Tm).to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                zs = model.encode_condition(cb)
                zf = zs["z_f"]
                zt = zs["z_t"]
                za = zs["z_a"]
                ze = zs["z_e"]

                zm = model.encode_motion(mc, mp).view(B, C, -1)

                lab = gb.label.to(device, non_blocking=True)
                msk = gb.cand_mask.to(device, non_blocking=True)
                gw = gb.group_w.to(device, non_blocking=True)

                # bank negatives
                z_bank = None
                if (global_step >= args.bank_warmup) and (len(bank) > 0):
                    z_bank = bank.get(device)

                # fused loss (with bank in denominator)
                loss_f, logits = group_infonce_loss_with_bank(
                    zf, zm, lab, msk, model.scale(),
                    z_bank=z_bank,
                    bank_alpha=float(args.bank_alpha),
                    use_silver_as_pos=bool(args.use_silver_as_pos),
                    sample_w=gw,
                )

                # update generic bank: pick from negatives, but RANDOM within topK (avoid template domination)
                with torch.no_grad():
                    if args.bank_size > 0:
                        neg_mask = (lab == 0) & msk
                        if neg_mask.any():
                            neg_logits = logits.masked_fill(~neg_mask, float("-inf"))  # [B,C]
                            topk = min(int(args.bank_topk), neg_logits.size(1))
                            idx_topk = torch.topk(neg_logits, k=topk, dim=1).indices  # [B,topk]
                            picked_logits = torch.gather(neg_logits, 1, idx_topk)      # [B,topk]
                            valid = torch.isfinite(picked_logits)                      # [B,topk]
                            if valid.any():
                                # random pick 1 per sample from valid topk
                                pick_j = []
                                for b in range(B):
                                    vb = valid[b].nonzero(as_tuple=False).view(-1)
                                    if vb.numel() == 0:
                                        pick_j.append(-1)
                                    else:
                                        j = vb[torch.randint(low=0, high=vb.numel(), size=(1,), device=device)].item()
                                        pick_j.append(j)
                                pick_j = torch.tensor(pick_j, device=device, dtype=torch.long)  # [B]
                                ok = pick_j >= 0
                                if ok.any():
                                    idx = idx_topk[torch.arange(B, device=device), pick_j.clamp_min(0)]  # [B]
                                    zb_pick = zm.detach()[torch.arange(B, device=device), idx]           # [B,D]
                                    zb_pick = zb_pick[ok]
                                    if zb_pick.size(0) > int(args.bank_update_k):
                                        perm = torch.randperm(zb_pick.size(0), device=device)[: int(args.bank_update_k)]
                                        zb_pick = zb_pick[perm]
                                    bank.add(zb_pick)

                # optional ordering loss (small)
                loss_ord = loss_f * 0.0
                if args.w_ord > 0:
                    loss_ord = in_group_order_margin_loss(
                        logits=logits,
                        label=lab,
                        cand_mask=msk,
                        m_gs=float(args.m_gs),
                        m_sn=float(args.m_sn),
                        sample_w=gw,
                    )

                # per-modality losses (only count samples where modality exists)
                loss_t = loss_f * 0.0
                loss_a = loss_f * 0.0
                loss_e = loss_f * 0.0

                if args.w_text > 0:
                    w = gw * cb.has_t.float()
                    loss_t, _ = group_infonce_loss(
                        zt, zm, lab, msk, model.scale(),
                        use_silver_as_pos=bool(args.use_silver_as_pos),
                        sample_w=w,
                    )
                if args.w_audio > 0:
                    w = gw * cb.has_a.float()
                    loss_a, _ = group_infonce_loss(
                        za, zm, lab, msk, model.scale(),
                        use_silver_as_pos=bool(args.use_silver_as_pos),
                        sample_w=w,
                    )
                if args.w_emo > 0:
                    w = gw * cb.has_e.float()
                    loss_e, _ = group_infonce_loss(
                        ze, zm, lab, msk, model.scale(),
                        use_silver_as_pos=bool(args.use_silver_as_pos),
                        sample_w=w,
                    )

                # alignment regularizer (agreement)
                align = loss_f * 0.0
                if args.w_align > 0:
                    align = alignment_reg(zt, za, ze, cb.has_t, cb.has_a, cb.has_e)

                loss = (
                    float(args.w_fused) * loss_f +
                    float(args.w_text)  * loss_t +
                    float(args.w_audio) * loss_a +
                    float(args.w_emo)   * loss_e +
                    float(args.w_align) * align +
                    float(args.w_ord)   * loss_ord
                )

            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            opt.step()

            with torch.no_grad():
                acc1 = acc_at_k_any_gold(logits, lab, msk, k=1)
                ndcg5 = ndcg_at_k(logits, lab, msk, k=5)
                ht = float(cb.has_t.float().mean().item())
                ha = float(cb.has_a.float().mean().item())
                he = float(cb.has_e.float().mean().item())
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lf=f"{loss_f.item():.3f}",
                    lt=f"{loss_t.item():.3f}",
                    la=f"{loss_a.item():.3f}",
                    le=f"{loss_e.item():.3f}",
                    align=f"{align.item():.3f}",
                    ord=f"{loss_ord.item():.3f}",
                    acc1=f"{acc1:.3f}",
                    ndcg5=f"{ndcg5:.3f}",
                    ht=f"{ht:.2f}", ha=f"{ha:.2f}", he=f"{he:.2f}",
                    scale=f"{model.scale().item():.2f}",
                    bank=f"{len(bank)}",
                )

            if (global_step % args.eval_every) == 0:
                metrics = run_eval(model, val_loader, device, use_silver_as_pos=bool(args.use_silver_as_pos))
                print(
                    f"\n[VAL step={global_step}] "
                    f"loss={metrics['val_loss']:.4f} "
                    f"acc1={metrics['acc1']:.3f} acc3={metrics['acc3']:.3f} acc5={metrics['acc5']:.3f} "
                    f"ndcg5={metrics['ndcg5']:.3f}"
                )

                ckpt = dict(
                    step=global_step, epoch=ep, best_val=best_val,
                    model=model.state_dict(), opt=opt.state_dict(),
                    emo2id=emo2id, args=vars(args), val_metrics=metrics
                )

                # save cur
                tmp = pjoin(args.save_dir, "cur.pt.tmp")
                out = pjoin(args.save_dir, "cur.pt")
                torch.save(ckpt, tmp)
                os.replace(tmp, out)
                with open(pjoin(args.save_dir, "cur_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump({"epoch": ep, "step": global_step, **metrics}, f, indent=2)

                # save best
                if metrics["val_loss"] < best_val:
                    best_val = metrics["val_loss"]
                    ckpt["best_val"] = best_val
                    torch.save(ckpt, pjoin(args.save_dir, "best.pt"))
                    print(f"[Saved best] {pjoin(args.save_dir, 'best.pt')}")

    print("[Done]")


if __name__ == "__main__":
    main()
