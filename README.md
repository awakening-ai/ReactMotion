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
  <h2 align="center"><a href="https://arxiv.org/pdf/2603.15083">Paper</a> | <a href="https://reactmotion.github.io">Project Page</a> | <a href="https://www.youtube.com/watch?v=48jq_G1uU5s">Video</a> | <a href="https://huggingface.co/awakening-ai/ReactMotion1.0">Hugging Face</a></h2>
</p>

[![Watch the video](https://img.youtube.com/vi/48jq_G1uU5s/maxresdefault.jpg)](https://www.youtube.com/watch?v=48jq_G1uU5s)

*We introduce **Reactive Listener Motion Generation from Speaker Utterance** — a new task that generates naturalistic listener body motions appropriately responding to a speaker's utterance. Our unified framework **ReactMotion** jointly models text, audio, emotion, and motion with preference-based objectives, producing natural, diverse, and appropriate listener responses.*

## 📢 Updates

- \[2026.03.17\] 🎮 **Inference Demo & Gradio UI** released
- \[2026.03.16\] 🎯 **Full Training, Evaluation Code** released

## 🚀 TLDR

Modeling nonverbal listener behavior is challenging due to the inherently **non-deterministic** nature of human reactions — the same speaker utterance can elicit many appropriate listener responses.

🔥 **ReactMotion** generates naturalistic listener body motions from speaker utterance (**text + audio + emotion**), trained with **preference-based ranking** on our **ReactMotionNet** dataset that captures the one-to-many nature of listener behavior.

We present:

- **ReactMotionNet** — A large-scale dataset pairing speaker utterances with multiple candidate listener motions annotated with varying degrees of appropriateness (gold/silver/negative)
- **ReactMotion** — A unified generative framework built on T5 that jointly models **text**, **audio**, **emotion**, and **motion**, trained with preference-based ranking objectives
- **JudgeNetwork** — A multi-modal contrastive scorer for **best-of-K selection**
- **Preference-oriented evaluation protocols** tailored to assess reactive appropriateness

## 🏗️ Architecture

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

## 🛠️ Installation

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

## 📥 Pretrained Models & Evaluators

Download the pretrained Motion VQ-VAE and evaluation models:

```bash
bash prepare/download_vqvae.sh
bash prepare/download_evaluators.sh
```

Once downloaded, your `external/` directory should look like:

```
external/
├── pretrained_vqvae/
│   └── t2m.pth                           # Motion VQ-VAE checkpoint
└── t2m/
    ├── Comp_v6_KLD005/                   # T2M evaluation model
    ├── text_mot_match/                   # Text-motion matching model
    └── VQVAEV3_CB1024_CMT_H1024_NRES3/
        └── meta/
            ├── mean.npy                  # Per-dim mean for normalization
            └── std.npy                   # Per-dim std for normalization
```

## 📦 Data Preparation

### HumanML3D Dataset

We use the [HumanML3D](https://github.com/EricGuo5513/HumanML3D) 3D human motion-language dataset. Please follow the [HumanML3D instructions](https://github.com/EricGuo5513/HumanML3D) to download and prepare the dataset, then place it under the `dataset/` directory:

```
dataset/HumanML3D/
├── new_joint_vecs/      # Joint feature vectors (263-dim)
├── texts/               # Motion captions
├── Mean.npy             # Per-dim mean for normalization
├── Std.npy              # Per-dim std for normalization
├── train.txt
├── val.txt
├── test.txt
├── train_val.txt
└── all.txt
```

### Motion VQ-VAE Codes

Pre-encode the HumanML3D motions with the Motion VQ-VAE and place the codes as `.npy` files:
```
dataset/HumanML3D/VQVAE/
├── 000000.npy
├── 000001.npy
├── M000000.npy
└── ...
```

Each `.npy` file contains a 1D integer array of VQ codebook indices (codebook size = 512).

### Speaker Audio

We provide both pre-extracted Mimi code indices and raw wav files:

| Resource | Description | Link |
|---|---|---|
| **Audio Codes** | Mimi encoder code indices (`.npz`) | [Google Drive](https://drive.google.com/drive/folders/1FVbEp13IQp95L6W0--mbPAx4os7TQnBA?usp=sharing) |
| **Audio Raw** | Raw speaker wav files (`.wav`) | [Google Drive](https://drive.google.com/drive/folders/1u9TT1mbeQyoWvhmgO3BNmlHMSwVriplr?usp=sharing) |

Download and place them under your dataset directory:

```
{DATASET_DIR}/
├── audio_code/   # Mimi code indices (for audio_mode=code)
│   ├── 001193_1_reaction_fearful_4.npz
│   └── ...
└── audio_wav/    # Raw speaker wav (for audio_mode=wav)
    ├── 001193_1_reaction_fearful_4.wav
    └── ...
```

The audio codes are pre-extracted using [Mimi](https://github.com/kyutai-labs/moshi) (from the Moshi project). Mimi weights are **automatically downloaded** from HuggingFace on first use if you want to re-encode from raw wav.

### CSV Splits

Prepare `train.csv`, `val.csv`, `test.csv` with the following columns:

| Column | Description |
|---|---|
| `group_id` | Unique group identifier |
| `item_id` | Unique item identifier |
| `tier_label` | Sample quality tier: `gold` / `silver` / `neg` |
| `speaker_transcript` | Speaker transcription text |
| `speaker_emotion` | Speaker emotion label |
| `listener_motion_caption` | Text description of the listener motion |
| `motion_id` | Motion file ID (6-digit zero-padded, e.g. `000267`) |
| `speaker_audio_wav` | Audio file stem (maps to audio code/wav files) |
| `group_w` *(optional)* | Per-group weight for weighted training |

## 🤗 Model Card

Our pretrained models are available on Hugging Face:

| Model | Backbone | Description | Link |
|---|---|---|---|
| **ReactMotion 1.0** | T5-base | Generator (Text + Audio + Emotion → Motion) | [awakening-ai/ReactMotion1.0](https://huggingface.co/awakening-ai/ReactMotion1.0) |
| **ReactMotion-Judge** | T5 text enc + Mimi audio enc | Multi-modal judge network for best-of-K selection | [awakening-ai/ReactMotion-Judge](https://huggingface.co/awakening-ai/ReactMotion-Judge) |

Download via CLI:
```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download the generator
huggingface-cli download awakening-ai/ReactMotion1.0 --local-dir models/ReactMotion1.0

# Download the judge network
huggingface-cli download awakening-ai/ReactMotion-Judge --local-dir models/ReactMotion-Judge
```

Or in Python:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="awakening-ai/ReactMotion1.0", local_dir="models/ReactMotion1.0")
snapshot_download(repo_id="awakening-ai/ReactMotion-Judge", local_dir="models/ReactMotion-Judge")
```

## ⚡ Quick Demo

### Inference Demo

Download the pretrained model from [Hugging Face](https://huggingface.co/awakening-ai/ReactMotion1.0) and make sure you have run the [prepare scripts](#-pretrained-models--evaluators) to download the VQ-VAE checkpoint and normalization files. Then run:

```bash
python demo_inference.py \
  --gen_ckpt   models/ReactMotion1.0 \
  --vqvae_ckpt external/pretrained_vqvae/t2m.pth \
  --mean_path  external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
  --std_path   external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy \
  --text "It is really nice to meet you!" \
  --emotion "excited" \
  --cond_mode t+e \
  --num_gen 3 \
  --out_path output/demo_text_meet.mp4
```

The generated videos will be saved in `output/`, example shown below:

https://github.com/user-attachments/assets/e0096715-f8b9-400c-8dd8-434d1e10c8d4

<!-- TODO: add demo video/gif here -->
<!-- <p align="center">
  <img src="images/demo_meet.gif" alt="Demo: It is really nice to meet you!" style="max-height:320px; width:auto;">
</p> -->

<details>
<summary>More demo examples</summary>

**Audio + Text input:**
```bash
python demo_inference.py \
  --gen_ckpt   models/ReactMotion1.0 \
  --vqvae_ckpt external/pretrained_vqvae/t2m.pth \
  --mean_path  external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
  --std_path   external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy \
  --text "Let's celebrate!" \
  --audio samples/speaker.wav \
  --cond_mode t+a+e --emotion "happy" \
  --out_path output/demo_fused.mp4
```

**Audio-only input:**
```bash
python demo_inference.py \
  --gen_ckpt   models/ReactMotion1.0 \
  --vqvae_ckpt external/pretrained_vqvae/t2m.pth \
  --mean_path  external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
  --std_path   external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy \
  --audio samples/speaker.wav \
  --cond_mode a+e --emotion "happy" \
  --out_path output/demo_audio.mp4
```

**Batch demo with shell script:**
```bash
bash scripts/demo.sh            # 3 text-only examples
bash scripts/demo.sh --audio    # text + audio example
```

</details>
<br>

### 🖥️ Gradio Web UI

Launch an interactive web demo:

```bash
python demo_gradio.py \
  --gen_ckpt   models/ReactMotion1.0 \
  --vqvae_ckpt external/pretrained_vqvae/t2m.pth \
  --mean_path  external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy \
  --std_path   external/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy \
  --port 7860 --share
```

Or via the shell script:

```bash
bash scripts/demo.sh --gradio
```

Features:
- Text input, audio upload, emotion selection
- Adjustable generation parameters (temperature, top-p, top-k)
- Side-by-side comparison of multiple generated motions

## 🎯 Training

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
  --audio_code_dir /path/to/audio_code \
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
  --audio_code_dir /path/to/audio_code \
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
<br>

## 📊 Evaluation

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

## 📁 Project Structure

```
reactmotion/
├── reactmotion/                       # Main package
│   ├── models/
│   │   ├── vqvae.py                      # Motion VQ-VAE (HumanVQVAE)
│   │   ├── encdec.py                     # 1D Conv Encoder/Decoder
│   │   ├── quantize_cnn.py               # VQ codebook (EMA reset)
│   │   ├── resnet.py                     # Resnet1D blocks
│   │   └── judge_network.py              # JudgeNetwork scorer
│   ├── dataset/
│   │   ├── reactmotionnet_dataset.py     # ReactMotionNet dataset
│   │   ├── humanml3d_dataset.py          # HumanML3D T2M dataset
│   │   ├── joint_dataset.py              # Joint training dataset
│   │   ├── collator.py                   # Ranking-aware collator
│   │   ├── joint_collator.py             # Mixed-task collator
│   │   ├── prompt_builder.py             # Multi-modal prompt construction
│   │   ├── mimi_encoder.py               # Mimi streaming audio encoder
│   │   └── audio_aug.py                  # Audio augmentation
│   ├── train/
│   │   ├── train_reactmotion.py          # Train generator (seq2seq + ranking)
│   │   ├── train_judge.py                # Train scorer (contrastive)
│   │   └── trainer_reactmotion.py        # Custom Seq2SeqTrainer
│   ├── eval/
│   │   ├── eval_reactmotion.py           # Generate motions
│   │   ├── eval_reactmotion_with_judge.py # Generate + rank
│   │   ├── eval_judge.py                 # Evaluate scorer ranking
│   │   └── eval_fid_diversity.py         # FID & diversity metrics
│   ├── visualization/
│   │   └── plot_3d_global.py             # 3D skeleton visualization
│   ├── utils/                            # Rotation, skeleton, losses
│   └── baselines/                        # Baseline methods
├── demo_inference.py                  # CLI inference demo
├── demo_gradio.py                     # Gradio web UI demo
├── scripts/                           # Training & evaluation scripts
│   ├── train_reactmotion.sh
│   ├── train_judge.sh
│   ├── eval_reactmotion.sh
│   ├── evaluate.sh
│   └── demo.sh
├── requirements.txt
└── setup.py
```

## 💡 Acknowledgements

This project builds upon the following amazing open-source projects:
[HumanML3D](https://github.com/EricGuo5513/HumanML3D),
[T2M-GPT](https://github.com/Mael-zys/T2M-GPT),
[Moshi/Mimi](https://github.com/kyutai-labs/moshi),
[Hugging Face Transformers](https://github.com/huggingface/transformers),
[wandb](https://github.com/wandb/wandb).

## 📄 Citation

```bibtex
@article{luo2026reactmotion,
  title={ReactMotion: Generating Reactive Listener Motions from Speaker Utterance},
  author={Luo, Cheng and Wu, Bizhu and Li, Bing and Ren, Jianfeng and Bai, Ruibin and Qu, Rong and Shen, Linlin and Ghanem, Bernard},
  journal={arXiv preprint arXiv:2603.15083},
  year={2026}
}
```

## License

This project is released under the MIT License.
