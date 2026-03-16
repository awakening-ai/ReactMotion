# ReactMotion

<p align="center">
<h1 align="center">ReactMotion: Generating Reactive Listener Motions<br>from Speaker Utterance</h1>
<p align="center">
    <a href="">Cheng Luo</a><sup>1*</sup>
    &middot;
    <a href="">Bizhu Wu</a><sup>2,4,5*</sup>
    &middot;
    <a href="">Bing Li</a><sup>1&dagger;</sup>
    &middot;
    <a href="">Jianfeng Ren</a><sup>4</sup>
    &middot;
    <a href="">Ruibin Bai</a><sup>4</sup>
    &middot;
    <a href="">Rong Qu</a><sup>5</sup>
    &middot;
    <a href="">Linlin Shen</a><sup>2,3&dagger;</sup>
    &middot;
    <a href="">Bernard Ghanem</a><sup>1</sup>
    <br>
    <sup>1</sup>King Abdullah University of Science and Technology
    <sup>2</sup>School of Artificial Intelligence, Shenzhen University<br>
    <sup>3</sup>Guangdong Provincial Key Laboratory of Intelligent Information Processing, Shenzhen University<br>
    <sup>4</sup>School of Computer Science, University of Nottingham Ningbo China
    <sup>5</sup>School of Computer Science, University of Nottingham, UK<br>
    <sup>*</sup>Equal contribution &nbsp; <sup>&dagger;</sup>Corresponding author
</p>
  <h2 align="center"><a href="https://reactmotion.github.io">Project Page</a> | <a href="https://www.youtube.com/watch?v=48jq_G1uU5s">Video</a></h2>
</p>

[![Watch the video](https://img.youtube.com/vi/48jq_G1uU5s/maxresdefault.jpg)](https://www.youtube.com/watch?v=48jq_G1uU5s)

*We introduce **Reactive Listener Motion Generation from Speaker Utterance** — a new task that generates naturalistic listener body motions appropriately responding to a speaker's utterance. Our unified framework **ReactMotion** jointly models text, audio, emotion, and motion with preference-based objectives, producing natural, diverse, and appropriate listener responses.*

## Updates

- \[2026.xx.xx\] Code released

## TLDR

Modeling nonverbal listener behavior is challenging due to the inherently **non-deterministic** nature of human reactions — the same speaker utterance can elicit many appropriate listener responses.

We present:

- **ReactMotionNet** — A large-scale dataset pairing speaker utterances with multiple candidate listener motions annotated with varying degrees of appropriateness (gold/silver/negative), explicitly capturing the **one-to-many** nature of listener behavior
- **ReactMotion** — A unified generative framework built on T5 that jointly models **text**, **audio**, **emotion**, and **motion**, trained with preference-based ranking objectives to encourage both appropriate and diverse listener responses
- **JudgeNetwork** — A multi-modal contrastive scorer that ranks generated motion candidates via InfoNCE loss for **best-of-K selection**
- **Preference-oriented evaluation protocols** tailored to assess reactive appropriateness, where conventional motion metrics fall short

ReactMotion outperforms retrieval baselines and cascaded LLM-based pipelines, generating more natural, diverse, and appropriate listener motions.

## Architecture

| Component | Backbone | Input | Output |
|---|---|---|---|
| **ReactMotion** | T5-base | Text + Audio + Emotion | Motion token sequence |
| **JudgeNetwork** | T5 text enc + Mimi audio enc | Multi-modal conditions + motion | Ranking score |

**Conditioning modes** — flexibly combine modalities:

| Mode | Description |
|------|-------------|
| `t` | Text (transcription) only |
| `a` | Audio only |
| `t+e` | Text + Emotion |
| `a+e` | Audio + Emotion |
| `t+a` | Text + Audio |
| `t+a+e` | Text + Audio + Emotion (full) |

## Installation

```bash
conda create -n reactmotion python=3.11 -y
conda activate reactmotion
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

We use [wandb](https://wandb.ai) to log and visualize the training process:

```bash
wandb login
```

## Data Preparation

Prepare the following data before training:

**1. Motion VQ codes**

Place motion VQ-VAE codes as `.npy` files under:
```
{DATASET_DIR}/HumanML3D/VQVAE/
├── 000000.npy
├── 000001.npy
└── ...
```

**2. Audio codes**

Pre-encode audio with [Mimi](https://github.com/kyutai-labs/moshi) and place under:
```
{AUDIO_CODE_DIR}/
├── audio_001.npz
├── audio_002.npz
└── ...
```

**3. CSV splits**

Prepare `train.csv`, `val.csv`, `test.csv` with the following columns:

| Column | Description |
|---|---|
| `group_id` | Unique group identifier |
| `label` | Sample quality: `gold` / `silver` / `neg` |
| `sayings` | Speaker transcription |
| `emotion` | Speaker emotion label |
| `file_name` | Motion file ID prefix |
| `generated_wav_name` | Audio file stem |
| `item_w` *(optional)* | Per-item weight |
| `group_w` *(optional)* | Per-group weight |

## Training

### Train ReactMotion (Generator)

```bash
bash scripts/train_reactmotion.sh [cond_mode] [loss_type]
# Example:
bash scripts/train_reactmotion.sh t+a+e multi_ce_rank
```

Or run directly:

```bash
python -m reactmotion.train.train_reactmotion \
  --model_name google-t5/t5-base \
  --dataset_dir /path/to/dataset \
  --pairs_csv /path/to/data \
  --output_dir /path/to/output \
  --cond_mode t+a+e \
  --audio_mode code \
  --audio_code_dir /path/to/audio_codes \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --max_steps 100000
```

<details>
<summary>Key training hyperparameters</summary>

| Parameter | Default |
|---|---|
| Batch size | 8 |
| Gradient accumulation | 2 |
| Learning rate | 5e-5 |
| Max steps | 100,000 |
| K gold samples | 2 |
| Modality dropout | 0.30 |
| Rank loss margin | 0.5 |
| Loss weights (w_rank / w_gn) | 0.25 / 0.25 |

</details>

### Train JudgeNetwork (Scorer)

```bash
bash scripts/train_judge.sh
```

Or run directly:

```bash
python -m reactmotion.train.train_judge \
  --dataset_dir /path/to/dataset \
  --pairs_csv /path/to/data \
  --audio_code_dir /path/to/audio_codes \
  --save_dir /path/to/output \
  --batch_size 16 \
  --epochs 50
```

<details>
<summary>Key scorer hyperparameters</summary>

| Parameter | Default |
|---|---|
| Batch size | 16 |
| Learning rate | 5e-5 |
| Weight decay | 0.01 |
| Epochs | 50 |
| Force single-modality ratio | 0.10 |
| Ordering margin (gold-silver / silver-neg) | 0.20 / 0.20 |

</details>

## Evaluation

### Generate Motions

```bash
bash scripts/eval_reactmotion.sh [cond_mode]
# Example:
bash scripts/eval_reactmotion.sh t+a+e
```

Or run directly:

```bash
python -m reactmotion.eval.eval_reactmotion \
  --gen_ckpt /path/to/checkpoint \
  --pairs_csv /path/to/test.csv \
  --dataset_dir /path/to/dataset \
  --cond_mode t+a+e \
  --num_gen 3 \
  --out_dir /path/to/output
```

### Generate + Rank with JudgeNetwork

```bash
python -m reactmotion.eval.eval_reactmotion_with_judge \
  --gen_ckpt /path/to/generator/checkpoint \
  --judge_ckpt /path/to/judge/best.pt \
  --pairs_csv /path/to/test.csv \
  --dataset_dir /path/to/dataset \
  --cond_mode t+a+e \
  --num_gen 3 \
  --out_dir /path/to/output
```

### Unified Evaluation (FID + Diversity + Win-Rate)

```bash
bash scripts/evaluate.sh [cond_mode] [pipeline]
# pipeline: all | winrate | fid
# Example:
bash scripts/evaluate.sh t+a+e all
```

### FID & Diversity Metrics

```bash
python -m reactmotion.eval.eval_fid_diversity \
  --gen_root /path/to/generated \
  --dataset_dir /path/to/dataset \
  --pairs_csv /path/to/test.csv
```

## Project Structure

```
reactmotion/
├── reactmotion/                    # Main package
│   ├── models/
│   │   └── judge_network.py           # JudgeNetwork scorer
│   ├── dataset/
│   │   ├── reactmotionnet_dataset.py  # Dataset class
│   │   ├── collator.py                # Ranking-aware collator
│   │   ├── prompt_builder.py          # Multi-modal prompt construction
│   │   ├── mimi_encoder.py            # Mimi streaming audio encoder
│   │   └── audio_aug.py               # Audio augmentation
│   ├── train/
│   │   ├── train_reactmotion.py       # Train generator (seq2seq)
│   │   ├── train_judge.py             # Train scorer (contrastive)
│   │   └── trainer_reactmotion.py     # Custom Seq2SeqTrainer
│   ├── eval/
│   │   ├── eval_reactmotion.py        # Generate motions
│   │   ├── eval_judge.py              # Evaluate scorer ranking
│   │   ├── eval_reactmotion_with_judge.py  # Generate + rank
│   │   └── eval_fid_diversity.py      # FID & diversity metrics
│   ├── visualization/
│   │   └── plot_3d_global.py          # 3D skeleton visualization
│   ├── utils/                         # Rotation, skeleton, losses
│   ├── baselines/                     # Baseline methods
│   └── data/                          # CSV data splits
├── scripts/                           # Training & evaluation scripts
├── requirements.txt
└── setup.py
```

## Acknowledgements

This project builds upon the following open-source projects:
[HumanML3D](https://github.com/EricGuo5513/HumanML3D),
[Moshi](https://github.com/kyutai-labs/moshi),
[Hugging Face Transformers](https://github.com/huggingface/transformers),
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT).

## Citation

```bibtex
```

## License

This project is released under the MIT License.
