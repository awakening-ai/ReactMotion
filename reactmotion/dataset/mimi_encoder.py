import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import pandas as pd
import os

import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
from moshi.models import loaders  # pip install moshiko-pytorch
import soundfile as sf          # needed for sf.read
import tempfile                 # TemporaryDirectory
import subprocess               # subprocess.run
from pathlib import Path        # Path(td) / "tmp.wav"
from typing import Optional, Tuple  # Optional[int], Tuple[Tensor,int]

import math, tempfile, subprocess
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["NO_CUDA_GRAPH"] = "1"  # disable Mimi CUDA graph capture
torch.backends.cudnn.benchmark = True  # allow auto-selection of faster algorithms
torch.set_float32_matmul_precision("high")  # helps some ops, negligible side effects

class MimiStreamingEncoder:
    """
    Wrap Mimi for streaming encode.
    - encode_codes(): return code indexes [K, T_frames] (Long, CPU)
    - encode_codes_and_latent(): also return quantized latent [D, T_frames] (Float, CPU)
    """

    def __init__(self, device="cuda", codebooks: int = 8,
                 repo: Optional[str] = None, weight_name: Optional[str] = None):
        repo = repo or loaders.DEFAULT_REPO
        weight_name = weight_name or loaders.MIMI_NAME
        ckpt = hf_hub_download(repo, weight_name)
        mimi = loaders.get_mimi(ckpt, device=device)
        mimi.set_num_codebooks(codebooks)
        mimi.eval()
        self.mimi = mimi
        self.device = device
        self.sample_rate = int(mimi.sample_rate)  # 24000
        self.frame_rate = float(mimi.frame_rate)  # 12.5
        self.frame_size = int(mimi.frame_size)  # 1920
        self.cardinality = int(mimi.cardinality)  # e.g., 2048
        self.num_codebooks = int(mimi.num_codebooks)

    @torch.no_grad()
    def encode_many_concat(self, wav_list: List[torch.Tensor], chunk_frames: int = 32, return_latent: bool = True):
        """
        Batch concatenation for speedup: pad each [1,T_i] to a multiple of chunk_len,
        concatenate into a single [1, T_cat], encode in one streaming loop, then split
        back into per-sample results by frame count.
        Returns:
          codes_list: List[[K,T_i]]
          z_list:     List[[D,T_i]] or None
        """
        assert len(wav_list) > 0
        frame_size = self.frame_size
        chunk_len = frame_size * max(1, int(chunk_frames))

        # 1) pad each sample to a multiple of chunk_len and record frame counts
        padded = []
        T_frames = []
        for w in wav_list:
            w = w.to(self.device, non_blocking=True)  # [1,T]
            T = w.shape[-1]
            # align to frame boundary
            n_frames = math.ceil(T / chunk_len)
            T_pad = n_frames * chunk_len
            if T_pad > T:
                w = torch.cat([w, torch.zeros(1, T_pad - T, device=w.device, dtype=w.dtype)], dim=-1)
            padded.append(w)
            T_frames.append(n_frames * chunk_frames)  # corresponding frame count (each chunk = chunk_frames frames)

        # 2) concatenate along time dimension -> [1, T_cat]
        cat = torch.cat(padded, dim=-1)  # [1, T_cat]
        x = cat.unsqueeze(0)  # [1,1,T_cat]

        # 3) single streaming loop
        codes_all = []
        lat_all = [] if return_latent else None
        with self.mimi.streaming(batch_size=1):
            for s in range(0, x.shape[-1], chunk_len):
                chunk = x[:, :, s:s + chunk_len]  # [1,1,chunk_len]
                codes = self.mimi.encode(chunk)  # [1,K,chunk_frames]
                codes_all.append(codes)
                if return_latent:
                    z = self.mimi.decode_latent(codes)  # [1,D,chunk_frames]
                    lat_all.append(z)

        codes_cat = torch.cat(codes_all, dim=-1).squeeze(0).long().cpu()  # [K, sum_Tf]
        z_cat = torch.cat(lat_all, dim=-1).squeeze(0).float().cpu() if return_latent else None  # [D, sum_Tf]

        # 4) split back by recorded frame counts
        codes_list, z_list = [], []
        start = 0
        for Tf in T_frames:
            end = start + Tf
            codes_list.append(codes_cat[:, start:end].contiguous())
            if return_latent:
                z_list.append(z_cat[:, start:end].contiguous())
            start = end

        return codes_list, (z_list if return_latent else None)

    @torch.no_grad()
    def _encode_chunked(self, wav_1t: torch.Tensor, chunk_frames: int = 32, return_latent: bool = False):
        """
        wav_1t: [1, T] float32 on self.device
        returns:
          codes [K, Tf] (long, cpu)
          z     [D, Tf] (float32, cpu) if return_latent else None
        """
        wav_1ct = wav_1t.unsqueeze(0)  # [1,1,T] on device
        frame_size = self.frame_size
        chunk_len = frame_size * max(1, int(chunk_frames))
        Ttot = wav_1ct.shape[-1]
        rem = Ttot % chunk_len
        if rem:
            pad = chunk_len - rem
            wav_1ct = torch.cat([wav_1ct, torch.zeros((1, 1, pad), dtype=wav_1ct.dtype, device=wav_1ct.device)], dim=-1)

        codes_list = []
        lat_list = [] if return_latent else None
        with self.mimi.streaming(batch_size=1):
            for s in range(0, wav_1ct.shape[-1], chunk_len):
                chunk = wav_1ct[:, :, s:s + chunk_len]  # [1,1,chunk_len]
                codes = self.mimi.encode(chunk)  # [1,K,chunk_frames]
                codes_list.append(codes)
                if return_latent:
                    z = self.mimi.decode_latent(codes)  # [1,D,chunk_frames]
                    lat_list.append(z)
        codes_cat = torch.cat(codes_list, dim=-1).squeeze(0).long().cpu()  # [K,Tf]
        if return_latent:
            z_cat = torch.cat(lat_list, dim=-1).squeeze(0).float().cpu()  # [D,Tf]
        else:
            z_cat = None
        return codes_cat, z_cat

    @torch.no_grad()
    def encode_codes(self, wav_1t: torch.Tensor, chunk_frames: int = 32):
        wav_1t = wav_1t.to(self.device)
        codes, _ = self._encode_chunked(wav_1t, chunk_frames, return_latent=False)
        return codes

    @torch.no_grad()
    def encode_codes_and_latent(self, wav_1t: torch.Tensor, chunk_frames: int = 32):
        wav_1t = wav_1t.to(self.device)
        codes, z = self._encode_chunked(wav_1t, chunk_frames, return_latent=True)
        return codes, z