# reactmotion.utils.evaluate
import os
import re
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from typing import List, Optional
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import LogitsProcessor, LogitsProcessorList


import csv, math
from collections import Counter

import json, hashlib, time
from dataclasses import dataclass

def _stable_key_from_prompt(prompt: str) -> str:
    # as long as the prompt is the same, the key is the same; avoids reliance on filename/ID
    h = hashlib.md5(prompt.encode("utf-8")).hexdigest()
    return f"promptmd5:{h}"

class JsonlWriter:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "a", encoding="utf-8")  # append, supports multi-round accumulation across repeat_time
        self.n = 0

    def write(self, obj: dict):
        self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.n += 1
        if self.n % 200 == 0:
            self.f.flush()

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()


def longest_run(seq):
    if not seq: return 0
    best = 1; run = 1; prev = seq[0]
    for x in seq[1:]:
        if x == prev:
            run += 1
            best = max(best, run)
        else:
            prev = x
            run = 1
    return best

def unique_ratio(seq):
    return (len(set(seq)) / float(len(seq))) if seq else 0.0

def token_entropy(seq):
    # Shannon entropy in bits
    if not seq: return 0.0
    c = Counter(seq)
    n = float(len(seq))
    ent = 0.0
    for _, v in c.items():
        p = v / n
        ent -= p * math.log(p, 2)
    return ent

class GenTokenLogger:
    """
    Records generated token sequences and statistics per sample -> CSV
    """
    def __init__(self, out_csv: str):
        self.out_csv = out_csv
        self.f = open(out_csv, "w", newline="", encoding="utf-8")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "idx", "L", "max_token", "unique_ratio", "longest_run", "entropy_bits",
            "has_end_tag", "ended_by_eos", "parsed_empty",
        ])
        self.n = 0

    def add(self, codes, text: str, eos_id: int, output_ids):
        L = len(codes)
        mx = max(codes) if L > 0 else -1
        has_end = ("</Motion Tokens>" in text)
        # ended_by_eos: last token id is eos (note: outputs includes prompt+gen)
        ended_by_eos = (int(output_ids[-1]) == int(eos_id)) if eos_id is not None and len(output_ids) > 0 else False

        self.w.writerow([
            self.n, L, mx,
            unique_ratio(codes), longest_run(codes), token_entropy(codes),
            int(has_end), int(ended_by_eos), int(L == 0),
        ])
        self.n += 1

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def summarize_csv(path: str):
    import pandas as pd
    df = pd.read_csv(path)
    def stats(col):
        s = df[col].to_numpy()
        return dict(
            min=float(s.min()),
            p50=float(np.median(s)),
            mean=float(s.mean()),
            p90=float(np.quantile(s, 0.9)),
            max=float(s.max()),
        )
    out = {
        "processed": int(len(df)),
        "L": stats("L"),
        "max_token": stats("max_token"),
        "unique_ratio": stats("unique_ratio"),
        "longest_run": stats("longest_run"),
        "entropy(bits)": stats("entropy_bits"),
        "has_end_tag_rate": float(df["has_end_tag"].mean()),
        "ended_by_eos_rate": float(df["ended_by_eos"].mean()),
        "parsed_empty_rate": float(df["parsed_empty"].mean()),
    }
    return out


class StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_ids: torch.LongTensor):
        super().__init__()
        self.stop_ids = stop_ids  # shape [L]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids: [bs, cur_len]
        L = self.stop_ids.numel()
        if input_ids.size(1) < L:
            return False
        return torch.all(input_ids[0, -L:] == self.stop_ids).item()

def build_stop_ids(tokenizer, text: str, device: str):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) == 0:
        raise RuntimeError(f"Stop text `{text}` encodes to empty ids.")
    return torch.tensor(ids, dtype=torch.long, device=device)


_MOTION_SPAN_RE = re.compile(r"<Motion Tokens>(.*?)</Motion Tokens>", re.DOTALL)
_MOTION_TOKEN_RE = re.compile(r"<Motion Token\s+(\d+)>")


def parse_motion_tokens(text: str, max_len: Optional[int] = None, codebook_size: int = 512) -> List[int]:
    """
    Only parses <Motion Token i>, preferring content within <Motion Tokens>...</Motion Tokens>.
    """
    m = _MOTION_SPAN_RE.search(text)
    span = m.group(1) if m else text

    codes = [int(x) for x in _MOTION_TOKEN_RE.findall(span)]
    # filter out invalid codes
    codes = [c for c in codes if 0 <= c < codebook_size]

    if max_len is not None:
        codes = codes[:max_len]
    return codes

@torch.no_grad()
def evaluation_1PosePerTime(val_loader, net, model, logger, tokenizer, eval_wrapper,
                            max_new_tokens=200, top_k=200, do_sample=True, temperature=0.8, cal_multimodality=False):
    model.eval()
    gen_csv = "gen_vq_tokens_eval.csv"
    gen_logger = GenTokenLogger(gen_csv)

    gen_jsonl = f"gen_tokens_{int(time.time())}.jsonl"  # optionally include checkpoint/seed
    gen_writer = JsonlWriter(gen_jsonl)  # JsonlWriter should support mode

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    device = torch.device("cuda")
    stop_ids = build_stop_ids(tokenizer, "</Motion Tokens>", device=str(device))
    stopping = StoppingCriteriaList([StopOnSubsequence(stop_ids)])


    nb_sample = 0
    for batch in tqdm(val_loader):
        motion_caption_word_embeddings, motion_caption_pos_one_hots, input_text, motion_caption_sent_len, pose, m_length, sample_id = batch

        bs, seq = pose.shape[:2]

        # ---- DEBUG stats (per batch)
        dbg_lens = []
        dbg_empty = 0
        dbg_has_motion = 0
        dbg_has_audio = 0
        dbg_eos_early = 0
        dbg_print_limit = 2  # how many raw outputs to print per batch
        dbg_printed = 0


        motion_multimodality_batch = []
        if cal_multimodality:
            i_range_num = 30
        else:
            i_range_num = 1

        for i in range(i_range_num):

            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                prompt = input_text[k]

                device = torch.device("cuda")
                sid = str(sample_id[k])
                motion_id = sid.split("_", 1)[0].zfill(6)

                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                )
                enc = {k: v.to(device) for k, v in enc.items()}  # compatible with all transformers versions

                outputs = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_k=top_k,
                    stopping_criteria=stopping,
                    eos_token_id=tokenizer.eos_token_id,  # keep
                    pad_token_id=tokenizer.pad_token_id,  # recommended to set explicitly
                    repetition_penalty=1.1,

                )
                text = tokenizer.decode(outputs[0], skip_special_tokens=False)
                print("has </Motion Tokens>:", "</Motion Tokens>" in text)


                # ---- DEBUG flags
                if "<Motion Token" in text:
                    dbg_has_motion += 1
                if "<Audio Level" in text:
                    dbg_has_audio += 1
                if "</s>" in text and text.strip().endswith("</s>"):
                    dbg_eos_early += 1

                codes = parse_motion_tokens(text, max_len=seq, codebook_size=512)
                out_ids = outputs[0].detach().cpu().tolist()
                ended_by_eos = (int(outputs[0][-1]) == int(tokenizer.eos_token_id))

                gen_writer.write({
                    "key": sid,
                    "motion_id": motion_id,
                    "tokens": codes,
                    "len": len(codes),
                    "has_end_tag": ("</Motion Tokens>" in text),
                    "ended_by_eos": ended_by_eos,
                })

                dbg_lens.append(len(codes))
                if len(codes) == 0:
                    dbg_empty += 1

                gen_logger.add(
                    codes=codes,
                    text=text,
                    eos_id=tokenizer.eos_token_id,
                    output_ids=outputs[0].detach().cpu().tolist()
                )

                # sample-print raw outputs to inspect what was generated
                if dbg_printed < dbg_print_limit:
                    logger.info(f"[DBG] prompt(head)={prompt[:120]}...")
                    logger.info(f"[DBG] decoded(head)={text[:500]}")
                    logger.info(f"[DBG] parsed_len={len(codes)} first10={codes[:10]}")
                    dbg_printed += 1

                # ---- fallback
                if len(codes) == 0:
                    codes = [1] * seq

                index_motion = torch.tensor(codes, device="cuda", dtype=torch.long).unsqueeze(0)
                pred_pose = net.forward_decoder(index_motion)

                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k + 1, :cur_len] = pred_pose[:, :seq]

            et_pred, em_pred = eval_wrapper.get_co_embeddings(motion_caption_word_embeddings, motion_caption_pos_one_hots, motion_caption_sent_len, pred_pose_eval,
                                                              pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            # ---- batch summary
            if len(dbg_lens) > 0:
                mean_len = sum(dbg_lens) / len(dbg_lens)
                logger.info(
                    f"[DBG] batch token_len: mean={mean_len:.1f} min={min(dbg_lens)} max={max(dbg_lens)} "
                    f"empty={dbg_empty}/{len(dbg_lens)} has_motion={dbg_has_motion}/{len(dbg_lens)} "
                    f"has_audio={dbg_has_audio}/{len(dbg_lens)} eos_end={dbg_eos_early}/{len(dbg_lens)}"
                )


            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(motion_caption_word_embeddings, motion_caption_pos_one_hots, motion_caption_sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    if cal_multimodality:
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = (f"FID. {fid:.4f}, "
           f"Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, "
           f"R_precision_real. {R_precision_real}, R_precision. {R_precision}, "
           f"matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, "
           f"multimodality. {multimodality:.4f}")
    logger.info(msg)

    model.train()

    gen_logger.close()
    logger.info(f"[GEN] saved -> {gen_csv}")

    gen_writer.close()
    logger.info(f"[GEN] saved jsonl -> {gen_jsonl}")

    # optional: summarize directly in eval (requires pandas)
    try:
        s = summarize_csv(gen_csv)
        logger.info(f"[GEN] token stats: {s}")
    except Exception as e:
        logger.info(f"[GEN] summarize failed (ok): {e}")


    return fid, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, logger



#######################################################################################

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov



def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat



def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score



def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    n = activation.shape[0]
    if n < 2:
        return 0.0  # or np.nan

    diversity_times = int(min(diversity_times, n - 1))
    # safety check: avoid diversity_times=0
    diversity_times = max(diversity_times, 1)

    first_indices = np.random.choice(n, diversity_times, replace=False)
    second_indices = np.random.choice(n, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()

