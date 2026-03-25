#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_reactmotion.py

- Supports:
  - resume from checkpoint safely (tokenizer+model loaded from ckpt to avoid vocab mismatch)
  - auto-resume from output_dir (get_last_checkpoint)
  - modality dropout in collator/trainer
  - diversity anti-template regularizer in trainer
  - optional group-wise weight (from csv score / constant / none)  [safe: only used if Trainer supports it]
  - optional batch-level template suppression (within-batch duplicate penalty) [safe: only used if Trainer supports it]
  - wandb online/offline/disabled
"""

import argparse
import os
import time
import inspect
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import wandb
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from reactmotion.train.callback_diversity_early_stop import (
    DiversityEarlyStopCallback, DiversityEarlyStopConfig
)
from reactmotion.dataset.reactmotionnet_dataset import ReactMotionNet
from reactmotion.dataset.collator import ReactMotionCollator
from reactmotion.train.trainer_reactmotion import ReactMotionTrainer
from reactmotion.train.callback_diversity_eval import DiversitySimpleCallback


# -------------------------
# utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def can_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return major >= 8  # Ampere+


def build_run_name(args) -> str:
    ts = time.strftime("%m%d-%H%M")
    base = os.path.basename(args.model_name.rstrip("/"))
    loss_tag = getattr(args, "loss_type", "multi_ce_rank")
    ifw_tag = f"_ifw{args.freq_alpha:g}" if getattr(args, "use_inverse_freq_reweight", False) else ""
    return (
        f"reactmotion_{base}"
        f"_cond{args.cond_mode}_a{args.audio_mode}"
        f"_drop{args.cond_dropout:g}"
        f"_div{args.diversity_w:g}"
        f"_lr{args.learning_rate:g}_bs{args.batch_size}"
        f"_kg{args.k_gold}_m{args.rank_margin:g}_wr{args.w_rank:g}_wgn{args.w_gn:g}"
        f"_loss{loss_tag}{ifw_tag}_{ts}"
    )


def resolve_resume_checkpoint(args) -> str:
    """
    Priority:
      1) --resume_from_checkpoint if provided
      2) if --auto_resume, find last checkpoint under output_dir
      3) else ""
    """
    if args.resume_from_checkpoint:
        return args.resume_from_checkpoint

    if not args.auto_resume:
        return ""

    last_ckpt = get_last_checkpoint(args.output_dir)
    return last_ckpt or ""


def load_model_and_tokenizer(
    model_name: str = "google-t5/t5-base",
    motion_codebook_size: int = 512,
    resume_ckpt: str = "",
    mimi_codebooks: int = 8,
    mimi_cardinality: int = 2048,
) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    """
    IMPORTANT:
    - If resuming, load tokenizer+model from checkpoint to guarantee vocab consistency (avoid size mismatch)
    - If fresh, load from model_name then add motion/audio/emotion tokens and resize embeddings.
    """
    if resume_ckpt:
        tok = T5Tokenizer.from_pretrained(resume_ckpt)
        model = T5ForConditionalGeneration.from_pretrained(resume_ckpt)
        return tok, model

    tok = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Motion tokens (one per VQ code) + wrapper
    new_tokens = [f"<Motion Token {i}>" for i in range(motion_codebook_size)]
    new_tokens += ["<Motion Tokens>", "</Motion Tokens>"]

    # Emotion wrapper
    new_tokens += ["<Emotion>", "</Emotion>"]

    # Audio wrapper
    new_tokens += ["<Audio Tokens>", "</Audio Tokens>"]

    # Audio code tokens:
    #  - 1D case:  "<Audio Token i>"
    #  - multi-level codes: "<Audio Level lv Token i>" for lv in [0, mimi_codebooks)
    # These match dataset_A2RM_gsn_flex._format_audio_tokens / collator_gsn_flex._codes_to_audio_text.
    # NOTE: this adds mimi_cardinality * (1 + mimi_codebooks) tokens, e.g. 2048 * 9 ≈ 18k new tokens.
    audio_base = [f"<Audio Level 0 Token {i}>" for i in range(mimi_cardinality)]
    audio_level = [
        f"<Audio Level {lv} Token {i}>"
        for lv in range(1, mimi_codebooks)
        for i in range(mimi_cardinality)
    ]
    new_tokens += audio_base + audio_level

    tok.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tok))
    return tok, model


def make_seq2seq_args(**kwargs):
    """
    Make Seq2SeqTrainingArguments with backward/forward compatibility.

    - Handles evaluation_strategy vs eval_strategy differences
    - Drops unsupported kwargs automatically (e.g., save_safetensors on older transformers)
    - Removes eval_steps when None
    """
    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    params = sig.parameters

    # (1) compat: evaluation_strategy <-> eval_strategy
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in params and "eval_strategy" in params:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    # (2) remove eval_steps if None (some versions dislike it)
    if kwargs.get("eval_steps", "KEEP") is None:
        kwargs.pop("eval_steps", None)

    # (3) drop unsupported args (e.g., save_safetensors)
    filtered = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs.keys()) - set(filtered.keys()))
    if dropped:
        print(f"[WARN] Seq2SeqTrainingArguments: dropped unsupported args = {dropped}")

    return Seq2SeqTrainingArguments(**filtered)


def safe_init(cls_or_fn, kwargs: Dict[str, Any], name: str):
    """
    Some of your local modules (Trainer/Collator) might not yet accept new args.
    This helper tries to construct with kwargs; if TypeError occurs, it will drop
    unknown kwargs and retry (so the script does not crash from extra arguments).
    """
    try:
        return cls_or_fn(**kwargs)
    except TypeError as e:
        msg = str(e)
        # best-effort: filter kwargs by signature
        try:
            sig = inspect.signature(cls_or_fn)
            allowed = set(sig.parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            print(f"[WARN] {name}: dropped unsupported args = {sorted(set(kwargs.keys()) - allowed)}")
            return cls_or_fn(**filtered)
        except Exception:
            # if still failing, re-raise original
            raise


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name", type=str, default="google-t5/t5-base")
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--pairs_csv", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./output-a2rm-gsn-rank")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=200_000)
    ap.add_argument("--save_steps", type=int, default=5000)
    ap.add_argument("--save_total_limit", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source_len", type=int, default=256)
    ap.add_argument("--target_len", type=int, default=256)

    # conditioning
    ap.add_argument("--cond_mode", type=str, default="t+e",
                    choices=["t", "t+e", "a", "a+e", "t+a", "t+a+e"])
    ap.add_argument("--use_emotion", action="store_true")
    ap.add_argument("--key_by", type=str, default="group_id",
                    choices=["group_id", "sayings_emotion", "sayings_only"])

    # loss type: ranking vs pure cross-entropy (single gold)
    ap.add_argument(
        "--loss_type",
        type=str,
        default="multi_ce_rank",
        choices=["multi_ce_rank", "ce", "multi_ce", "rank"],
        help="multi_ce_rank: CE (multi-gold LSE) + ranking; "
             "ce: pure CE on 1 gold; "
             "multi_ce: CE on multi-gold (avg via LSE), no ranking; "
             "rank: ranking margin loss only (no CE).",
    )

    # modality dropout
    ap.add_argument("--cond_dropout", type=float, default=0.30,
                    help="Drop prob for each available modality when cond_mode has >=2 modalities.")
    ap.add_argument("--drop_e_only_when_multi", action="store_true",
                    help="If set: emotion can be dropped only when there is >=2 modalities; otherwise keep e when mode is t+e or a+e.")

    # diversity anti-template regularizer (global cache-based)
    ap.add_argument("--diversity_w", type=float, default=0.02,
                    help="Weight for anti-template penalty (0 disables).")
    ap.add_argument("--diversity_cache_size", type=int, default=50000,
                    help="LRU size for sequence signature counts.")

    # audio
    ap.add_argument("--audio_mode", type=str, default="none", choices=["none", "code", "wav"])
    ap.add_argument("--audio_key", type=str, default="motion_id", choices=["motion_id", "raw_file_name", "audio_stem"])
    ap.add_argument("--audio_code_dir", type=str, default=None)
    ap.add_argument("--wav_dir", type=str, default=None)
    ap.add_argument("--audio_token_level", type=str, default="base", choices=["base", "all", "rand"])
    ap.add_argument("--audio_sr", type=int, default=24000)
    ap.add_argument("--mimi_codebooks", type=int, default=8)
    ap.add_argument("--mimi_cardinality", type=int, default=2048)
    ap.add_argument("--mimi_chunk_frames", type=int, default=32)
    ap.add_argument("--enable_audio_aug", action="store_true")

    # gold sampling
    ap.add_argument("--k_gold", type=int, default=2)
    ap.add_argument("--sample_gold", type=str, default="random", choices=["random", "first"])
    ap.add_argument("--normalize_logsumexp", action="store_true")

    # rank loss
    ap.add_argument("--rank_margin", type=float, default=1.0)
    ap.add_argument("--w_rank", type=float, default=0.5)
    ap.add_argument("--w_gn", type=float, default=0.5)

    # inverse-frequency reweighting
    ap.add_argument("--use_inverse_freq_reweight", action="store_true",
                    help="Enable inverse-frequency token reweighting in loss (upweight rare motion tokens).")
    ap.add_argument("--freq_alpha", type=float, default=1.0,
                    help="Exponent for inverse-frequency weight: w(t) = 1/freq(t)^alpha. "
                         "Higher alpha = stronger emphasis on rare tokens.")

    # eval
    ap.add_argument("--do_eval", action="store_true")
    ap.add_argument("--eval_steps", type=int, default=5000)
    ap.add_argument("--eval_max_samples", type=int, default=0)

    # resume
    ap.add_argument("--resume_from_checkpoint", type=str, default=None)
    ap.add_argument("--auto_resume", action="store_true")

    # wandb
    ap.add_argument("--wandb_project", type=str, default="A2RM-T5-Pairs")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--wandb_tags", type=str, default="gsn_rank,fixedK,promptv3")

    # perf
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)



    # ----------------------------------------
    # OPTIONAL: group-wise weight (safe: only used if Trainer supports)
    # ----------------------------------------
    ap.add_argument("--group_w_mode", type=str, default="none",
                    choices=["none", "from_csv", "constant"],
                    help="Group-wise sample weight mode.")
    ap.add_argument("--group_w_col", type=str, default="group_w",
                    help="CSV column name for group weight, e.g. group_w/score/item_w.")
    ap.add_argument("--group_w_agg", type=str, default="mean",
                    choices=["mean", "max", "first"],
                    help="Aggregation over rows within a group when using from_csv.")
    ap.add_argument("--group_w_clip_min", type=float, default=0.2)
    ap.add_argument("--group_w_clip_max", type=float, default=5.0)
    ap.add_argument("--group_w_const", type=float, default=1.0)


    # ----------------------------------------
    # OPTIONAL: batch-level template suppression (safe: only used if Trainer supports)
    # ----------------------------------------
    ap.add_argument("--batch_template_w", type=float, default=0.0,
                    help="Weight for within-batch duplicate-signature penalty (0 disables).")
    ap.add_argument("--batch_template_power", type=float, default=1.0,
                    help="Penalty exponent for within-batch duplicates.")

    # ----------------------------------------
    # OPTIONAL: T2M co-training (HumanML3D text-to-motion)
    # ----------------------------------------
    ap.add_argument("--enable_t2m", action="store_true",
                    help="Enable joint training with HumanML3D text-to-motion task.")
    ap.add_argument("--t2m_ratio", type=float, default=0.33,
                    help="Fraction of T2M samples in each epoch (default 0.33 = 1:2 ratio).")
    ap.add_argument("--t2m_loss_weight", type=float, default=1.0,
                    help="Weight for T2M loss relative to ReactMotion loss.")
    ap.add_argument("--humanml3d_dir", type=str, default=None,
                    help="Path to HumanML3D directory (default: {dataset_dir}/HumanML3D).")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # -------------------------
    # wandb env
    # -------------------------
    if args.wandb_mode == "disabled":
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_MODE"] = args.wandb_mode

    run_name = build_run_name(args)
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    if args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=tags,
            config=vars(args),
        )

    # -------------------------
    # resume -> load tokenizer/model
    # -------------------------
    resume_ckpt = resolve_resume_checkpoint(args)
    if resume_ckpt:
        print("[Resume] resolved checkpoint:", resume_ckpt)

    tokenizer, model = load_model_and_tokenizer(
        args.model_name,
        motion_codebook_size=512,
        resume_ckpt=resume_ckpt,
        mimi_codebooks=args.mimi_codebooks,
        mimi_cardinality=args.mimi_cardinality,
    )

    # dataset flags inferred
    use_transcription = ("t" in args.cond_mode)
    # emotion only if user enables AND mode has +e
    use_emotion = bool(args.use_emotion) and ("+e" in args.cond_mode)

    # -------------------------
    # datasets
    # -------------------------
    rm_train_ds = ReactMotionNet(
        split="train",
        dataset_dir=args.dataset_dir,
        pairs_csv=args.pairs_csv,
        use_transcription=use_transcription,
        use_emotion=use_emotion,
        key_by=args.key_by,
        audio_mode=args.audio_mode,
        audio_token_level=args.audio_token_level,
        audio_code_dir=args.audio_code_dir,
        wav_dir=args.wav_dir,
        min_gold=1, min_silver=1, min_neg=1,
        min_audio=1 if args.audio_mode != "none" else 0,
        group_w_mode=args.group_w_mode,
        group_w_col=args.group_w_col,
        group_w_agg=args.group_w_agg,
        group_w_const=args.group_w_const,
        group_w_clip_min=args.group_w_clip_min,
        group_w_clip_max=args.group_w_clip_max,
    )

    train_ds = rm_train_ds
    train_sampler = None

    if args.enable_t2m:
        from reactmotion.dataset.humanml3d_dataset import HumanML3DDataset
        from reactmotion.dataset.joint_dataset import JointDataset, build_weighted_sampler

        hml_dir = args.humanml3d_dir or os.path.join(args.dataset_dir, "HumanML3D")
        t2m_train_ds = HumanML3DDataset(
            split="train",
            dataset_dir=args.dataset_dir,
            humanml3d_dir=hml_dir,
        )
        train_ds = JointDataset(rm_train_ds, t2m_train_ds)
        train_sampler = build_weighted_sampler(
            rm_size=len(rm_train_ds),
            t2m_size=len(t2m_train_ds),
            t2m_ratio=args.t2m_ratio,
        )
        print(
            f"[JointTraining] ReactMotion={len(rm_train_ds)} + T2M={len(t2m_train_ds)} "
            f"= {len(train_ds)} total, t2m_ratio={args.t2m_ratio:.2f}"
        )

    val_ds = None
    if args.do_eval:
        val_ds = ReactMotionNet(
            split="val",
            dataset_dir=args.dataset_dir,
            pairs_csv=args.pairs_csv,
            use_transcription=use_transcription,
            use_emotion=use_emotion,
            key_by=args.key_by,
            audio_mode=args.audio_mode,
            audio_token_level=args.audio_token_level,
            audio_code_dir=args.audio_code_dir,
            wav_dir=args.wav_dir,
            min_gold=1, min_silver=1, min_neg=1,
            min_audio=1 if args.audio_mode != "none" else 0,
            group_w_mode=args.group_w_mode,
            group_w_col=args.group_w_col,
            group_w_agg=args.group_w_agg,
            group_w_const=args.group_w_const,
            group_w_clip_min=args.group_w_clip_min,
            group_w_clip_max=args.group_w_clip_max,
        )
        if args.eval_max_samples and args.eval_max_samples > 0:
            idxs = list(range(len(val_ds)))
            random.Random(args.seed).shuffle(idxs)
            idxs = idxs[: args.eval_max_samples]
            val_ds = torch.utils.data.Subset(val_ds, idxs)
            print(f"[Eval] using subset: {len(val_ds)} samples")

    # -------------------------
    # collator (safe init)
    # -------------------------
    collator_kwargs = dict(
        tokenizer=tokenizer,
        source_len=args.source_len,
        target_len=args.target_len,
        fixed_k_gold=args.k_gold,
        sample_gold=args.sample_gold,
        force_first_motion=True,
        one_gold=True,  # always expand on a single gold; K is controlled by fixed_k_gold
        cond_mode=args.cond_mode,
        audio_mode=args.audio_mode,
        audio_sr=args.audio_sr,
        mimi_codebooks=args.mimi_codebooks,
        mimi_cardinality=args.mimi_cardinality,
        mimi_chunk_frames=args.mimi_chunk_frames,
        audio_token_level=args.audio_token_level,
        enable_audio_aug=args.enable_audio_aug,
        device="cuda" if torch.cuda.is_available() else "cpu",
        cond_dropout=args.cond_dropout,
        drop_e_only_when_multi=args.drop_e_only_when_multi,
    )

    # for pure cross-entropy training: use only one fixed gold as ground truth
    if args.loss_type == "ce":
        collator_kwargs["fixed_k_gold"] = 1
        collator_kwargs["sample_gold"] = "first"

    if args.enable_t2m:
        from reactmotion.dataset.joint_collator import JointCollator
        collator = safe_init(JointCollator, collator_kwargs, "JointCollator")
    else:
        collator = safe_init(ReactMotionCollator, collator_kwargs, "ReactMotionCollator")

    # -------------------------
    # training args
    # -------------------------
    use_bf16 = can_bf16()
    use_fp16 = torch.cuda.is_available() and (not use_bf16)

    train_args = make_seq2seq_args(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        evaluation_strategy="steps" if args.do_eval else "no",
        eval_steps=args.eval_steps if args.do_eval else None,
        report_to=(["wandb"] if args.wandb_mode != "disabled" else []),
        run_name=run_name,
        predict_with_generate=False,
        remove_unused_columns=False,
        save_safetensors=False,
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=10.0,
        warmup_steps=1000,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        fp16=use_fp16,
        bf16=use_bf16,
    )

    # -------------------------
    # callbacks config
    # -------------------------
    div_cfg = DiversityEarlyStopConfig(
        patience=3,
        min_unique_ratio=0.35,
        max_top1_freq=0.08,
        eval_batches=20,
        max_new_tokens=args.target_len,
        num_beams=1,
        do_sample=False,
        log_distinct=True,
    )


    # -------------------------
    # trainer (safe init)
    # -------------------------
    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,

        loss_type=args.loss_type,

        normalize_logsumexp=args.normalize_logsumexp,
        rank_margin=args.rank_margin,
        w_rank=args.w_rank,
        w_gn=args.w_gn,

        # wav prompt build params (only relevant when audio_mode="wav")
        enable_wav=(args.audio_mode == "wav"),
        source_len=args.source_len,
        cond_mode=args.cond_mode,
        audio_sr=args.audio_sr,
        mimi_codebooks=args.mimi_codebooks,
        mimi_cardinality=args.mimi_cardinality,
        mimi_chunk_frames=args.mimi_chunk_frames,
        audio_level=args.audio_token_level,
        enable_audio_aug=args.enable_audio_aug,

        # inverse-frequency reweighting
        use_inverse_freq_reweight=args.use_inverse_freq_reweight,
        freq_alpha=args.freq_alpha,

        # modality dropout + diversity
        cond_dropout=args.cond_dropout,
        drop_e_only_when_multi=args.drop_e_only_when_multi,
        diversity_w=args.diversity_w,
        diversity_cache_size=args.diversity_cache_size,

        # in-batch template suppression
        batch_template_w=args.batch_template_w,
        batch_template_power=args.batch_template_power,

        # T2M co-training
        t2m_loss_weight=getattr(args, "t2m_loss_weight", 1.0),
    )

    trainer = safe_init(ReactMotionTrainer, trainer_kwargs, "ReactMotionTrainer")

    # Override sampler for joint training (weighted random sampling)
    if train_sampler is not None:
        trainer._get_train_sampler = lambda _dataset=None: train_sampler

    trainer.add_callback(DiversityEarlyStopCallback(div_cfg))
    div_cb = DiversitySimpleCallback(
    eval_batches=20,
    num_return_sequences=4,
    max_new_tokens=args.target_len,
    do_sample=True,
    temperature=1.0,
    top_p=0.95,
    sig_max_len=64,
    prefix="div",
    )
    div_cb.trainer = trainer          # bind trainer reference
    trainer.add_callback(div_cb)


    # -------------------------
    # train
    # -------------------------
    if resume_ckpt:
        print("[Train] resume_from_checkpoint =", resume_ckpt)
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        print("[Train] fresh training")
        trainer.train()

    if args.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main()
