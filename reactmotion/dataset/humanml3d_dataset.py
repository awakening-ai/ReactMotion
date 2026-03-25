#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HumanML3D Text-to-Motion dataset for co-training with ReactMotion.

Loads motion captions from HumanML3D texts/ directory and reformulates them
as speaker dialogue prompts so the model learns general motion generation
alongside reactive motion generation.
"""

import os
import re
import random
from os.path import join as pjoin
from typing import Dict, Any, List, Optional

import numpy as np
from torch.utils.data import Dataset

from reactmotion.dataset.reactmotionnet_dataset import resolve_vq_path


# ============================================================
# Caption reformulation
# ============================================================

# Subjects to strip from captions
_SUBJECT_PATTERNS = [
    r"^a\s+person\s+",
    r"^the\s+person\s+",
    r"^a\s+man\s+",
    r"^the\s+man\s+",
    r"^a\s+woman\s+",
    r"^the\s+woman\s+",
    r"^a\s+guy\s+",
    r"^the\s+guy\s+",
    r"^a\s+figure\s+",
    r"^the\s+figure\s+",
    r"^someone\s+",
    r"^somebody\s+",
    r"^a\s+human\s+",
    r"^the\s+human\s+",
    r"^a\s+child\s+",
    r"^the\s+child\s+",
    r"^a\s+boy\s+",
    r"^the\s+boy\s+",
    r"^a\s+girl\s+",
    r"^the\s+girl\s+",
    r"^he\s+",
    r"^she\s+",
    r"^they\s+",
    r"^stick\s+figure\s+",
]

# "is walking" → "walk", "is waving" → "wave"
_IS_GERUND = re.compile(r"^is\s+(\w+ing)\b")

# "walks" → "walk", "jumps" → "jump" (simple s-removal for 3rd person)
_THIRD_PERSON = re.compile(r"^(\w+?)(s|es)\b")

DIALOGUE_TEMPLATES = [
    "Can you {action}?",
    "Could you show me how to {action}?",
    "Please {action}.",
    "I'd like you to {action}.",
    "Try to {action}.",
    "Show me {action}.",
    "Do {action} for me.",
    "Perform {action}.",
]


def _strip_subject(text: str) -> str:
    """Remove leading subject phrases like 'a person', 'the man', etc."""
    for pat in _SUBJECT_PATTERNS:
        text = re.sub(pat, "", text, count=1, flags=re.IGNORECASE)
    return text.strip()


def _normalize_verb(text: str) -> str:
    """Convert 'is walking' → 'walking', 'walks' → 'walk'."""
    # Handle "is + gerund"
    m = _IS_GERUND.match(text)
    if m:
        return text[m.start(1):]

    # Handle third-person singular: "walks forward" → "walk forward"
    words = text.split(None, 1)
    if words:
        verb = words[0]
        rest = words[1] if len(words) > 1 else ""
        # Don't de-conjugate short words or words that aren't verbs
        if len(verb) > 3:
            m2 = _THIRD_PERSON.match(verb)
            if m2:
                base = m2.group(1)
                # "dances" → "dance", "touches" → "touch"
                if verb.endswith("ches") or verb.endswith("shes") or verb.endswith("sses"):
                    base = verb[:-2]
                elif verb.endswith("ies"):
                    base = verb[:-3] + "y"
                elif verb.endswith("es"):
                    base = verb[:-2]
                elif verb.endswith("s") and not verb.endswith("ss"):
                    base = verb[:-1]
                return (base + " " + rest).strip() if rest else base
    return text


def reformulate_caption(caption: str) -> str:
    """
    Transform a HumanML3D caption into a speaker dialogue prompt.

    Examples:
        "a person walks forward" → "Can you walk forward?"
        "the man is waving his hands" → "Show me waving his hands."
        "someone jumps and lands" → "Please jump and land."
    """
    text = caption.strip().lower()

    # Strip trailing period
    if text.endswith("."):
        text = text[:-1].strip()

    # Remove subject
    action = _strip_subject(text)
    if not action:
        action = text  # fallback: use original if stripping removed everything

    # Normalize verb form
    action = _normalize_verb(action)

    if not action:
        action = text

    # Pick random dialogue template
    template = random.choice(DIALOGUE_TEMPLATES)
    return template.format(action=action)


def load_caption(text_path: str) -> Optional[str]:
    """Load first caption from a HumanML3D text file (text before first #)."""
    if not os.path.isfile(text_path):
        return None
    with open(text_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Take text before first #
            caption = line.split("#")[0].strip()
            if caption:
                return caption
    return None


# ============================================================
# Dataset
# ============================================================

class HumanML3DDataset(Dataset):
    """
    HumanML3D text-to-motion dataset for co-training.

    Returns items in the same format as ReactMotionNet so the collator
    can handle both datasets uniformly.
    """

    def __init__(
        self,
        split: str,
        dataset_dir: str,
        humanml3d_dir: Optional[str] = None,
        debug_print_k: int = 3,
    ):
        assert split in ["train", "val", "test"]

        self.split = split
        self.dataset_dir = dataset_dir

        hml_dir = humanml3d_dir or pjoin(dataset_dir, "HumanML3D")
        self.texts_dir = pjoin(hml_dir, "texts")
        self.vqvae_dir = pjoin(hml_dir, "VQVAE")

        # Load split file
        split_file = pjoin(hml_dir, f"{split}.txt")
        if not os.path.isfile(split_file):
            raise RuntimeError(f"Missing split file: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            motion_ids = [line.strip() for line in f if line.strip()]

        # Build items
        self.items: List[Dict[str, Any]] = []
        skipped_vq = 0
        skipped_caption = 0

        for mid in motion_ids:
            # Resolve VQ path
            vq_path = resolve_vq_path(self.vqvae_dir, mid)
            if vq_path is None:
                skipped_vq += 1
                continue

            # Load caption
            # Try both original ID and zero-padded
            text_path = pjoin(self.texts_dir, f"{mid}.txt")
            if not os.path.isfile(text_path):
                padded = str(mid).zfill(6)
                text_path = pjoin(self.texts_dir, f"{padded}.txt")

            caption = load_caption(text_path)
            if not caption:
                skipped_caption += 1
                continue

            self.items.append(dict(
                motion_id=mid,
                caption=caption,
                vq_path=vq_path,
            ))

        print(
            f"[HumanML3D] split={split} items={len(self.items)} "
            f"skipped_vq={skipped_vq} skipped_caption={skipped_caption}"
        )

        # Debug print
        for i, it in enumerate(self.items[:debug_print_k]):
            dialogue = reformulate_caption(it["caption"])
            print(f"  [{i}] {it['motion_id']}: \"{it['caption']}\" → \"{dialogue}\"")

        if len(self.items) == 0:
            raise RuntimeError(
                f"[HumanML3D] 0 items after filtering for split={split}.\n"
                f"  texts_dir={self.texts_dir}\n"
                f"  vqvae_dir={self.vqvae_dir}\n"
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]

        # Reformulate caption as dialogue prompt each time (random template)
        dialogue = reformulate_caption(it["caption"])

        return dict(
            key=f"t2m_{it['motion_id']}",
            transcription=dialogue,
            emotion="",
            gold_vq_paths=[it["vq_path"]],
            silver_vq_paths=[it["vq_path"]],  # dummy
            neg_vq_paths=[it["vq_path"]],      # dummy
            group_w=1.0,
            is_t2m=True,
        )
