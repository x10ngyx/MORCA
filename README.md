# MORCA: Offline-to-Online Reinforcement Learning for Adaptive Cache Reuse in Video Diffusion Acceleration

<p align="center">
  <a href="https://arxiv.org/abs/2610.10457">
    <img src="https://img.shields.io/badge/arXiv-2610.10457-b31b1b.svg?logo=arXiv" alt="arXiv: 2610.10457" />
  </a>
  <a href="https://x10ngyx.github.io/MORCA/">
    <img src="https://img.shields.io/badge/Project-%3CWebsite%3E-blue.svg" alt="Project Website" />
  </a>
</p>

## Supported Models

| Model | Task |
|---|---|
| [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | Text-to-Video |
| [Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) | Text-to-Video |

## Installation

```bash
conda create -n morca python=3.10 -y
conda activate morca
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

git clone https://github.com/Wan-Video/Wan2.1.git third_party/Wan2.1
git -C third_party/Wan2.1 checkout 65386b2e03c490796eede31b0325a6a595cc684e
git clone https://github.com/Wan-Video/Wan2.2.git third_party/Wan2.2
git -C third_party/Wan2.2 checkout 42bf4cfaa384bc21833865abc2f9e6c0e67233dc
```

## Model Download

Download the base models and choose an output directory:

```bash
export MODEL_ROOT="$PWD/models"
export HF_HOME="$MODEL_ROOT/huggingface"
export TORCH_HOME="$MODEL_ROOT/torch"
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$MODEL_ROOT/Wan2.1-T2V-1.3B"
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir "$MODEL_ROOT/Wan2.2-T2V-A14B"
export OUTPUT_ROOT=/path/to/experiments
export CUDA_VISIBLE_DEVICES=0
```

Run the following commands from the repository root. Model and evaluation weights are stored under `models/`.

## Inference

### Configurations & Checkpoints

**Base Model Configurations**

| Base Model | Configuration |
|---|---|
| Wan2.1-T2V-1.3B | [config/wan21/model.json](config/wan21/model.json) |
| Wan2.2-T2V-A14B | [config/wan22/model.json](config/wan22/model.json) |

**Method Configurations**

| Method | Wan2.1 | Wan2.2 |
|---|---|---|
| MORCA | [morca.json](config/wan21/morca.json) | [morca.json](config/wan22/morca.json) |
| Original | [original.json](config/wan21/original.json) | [original.json](config/wan22/original.json) |
| SeaCache | [seacache.json](config/wan21/seacache.json) | [seacache.json](config/wan22/seacache.json) |
| TeaCache | [teacache.json](config/wan21/teacache.json) | [teacache.json](config/wan22/teacache.json) |
| MagCache | [magcache.json](config/wan21/magcache.json) | [magcache.json](config/wan22/magcache.json) |
| DiCache | [dicache.json](config/wan21/dicache.json) | [dicache.json](config/wan22/dicache.json) |

**MORCA Checkpoints**

| Base Model | MORCA Checkpoint | Size |
|---|---|---|
| Wan2.1-T2V-1.3B | Coming soon | — |
| Wan2.2-T2V-A14B | [wan22.pt](models/MORCA/wan22.pt) | 18 MB |

### Single Prompt

```bash
python main.py infer --model wan21 --method morca \
  --policy models/MORCA/wan21.pt \
  --prompt "A cat walking through a garden." --output "$OUTPUT_ROOT/single/morca"
```

### VBench200

```bash
python main.py infer --model wan21 --method morca \
  --policy models/MORCA/wan21.pt \
  --prompts data/vbench200.jsonl --output "$OUTPUT_ROOT/vbench200/morca"
```

## Evaluation

```bash
python main.py evaluate --reference "$OUTPUT_ROOT/vbench200/original" \
  --candidate "$OUTPUT_ROOT/vbench200/morca" \
  --output "$OUTPUT_ROOT/vbench200/metrics"
```

This reports RGB PSNR, SSIM and LPIPS. Generation records total latency and T5/DiT/VAE times. Add `--profile` for a separate TFLOPs pass; PyTorch counts supported operations only, excluding unsupported fused kernels. Use the unprofiled pass for latency.

## Training

Prepare calibration and training prompts as text (one prompt per line) or JSONL with `sample_id` and `prompt`. Set `split` to `train` or `val` in JSONL, or use the default 80/20 split by prompt. Keep training prompts separate from VBench200.

### Calibration

Measure latency at different reuse budgets:

```bash
python main.py calibrate --model wan21 --prompts calibration_prompts.txt \
  --output "$OUTPUT_ROOT/calibration"
```

### Data Collection

Collect trajectories with random reuse budgets and terminal RGB PSNR rewards:

```bash
python main.py collect --model wan21 --prompts train_prompts.jsonl \
  --calibration "$OUTPUT_ROOT/calibration/calibration.json" \
  --trajectories 3 --speed-range 1.5 3.5 --output "$OUTPUT_ROOT/offline"
```

### Offline Training

Train the CNN+G1 scheduler with IQL:

```bash
python main.py train --model wan21 --data "$OUTPUT_ROOT/offline" \
  --weights models/MORCA/offline --output "$OUTPUT_ROOT/training"
```

Both models train for 200 epochs with batch size 256, τ=.7, β=1.5 and weight cap 30. `selected.pt` is chosen by validation actor agreement and Q stability (Wan2.1: epochs 171–199; Wan2.2: 180–200). Per-epoch checkpoints are also saved. Use `--epochs` to change the training duration.

### Online Fine-tuning

Collect 100 trajectories per round with 90% actor sampling and 10% uniform exploration, then update the scheduler on cumulative replay. Use a separate training-only prompt pool with `split: "train"`.

<details>
<summary>Online fine-tuning commands</summary>

```bash
POLICY=models/MORCA/offline/selected.pt
REPLAY=("$OUTPUT_ROOT/offline")
RESTORE=()
for ROUND in $(seq 1 8); do
  python main.py collect --model wan21 --prompts online_prompts.jsonl \
    --calibration "$OUTPUT_ROOT/calibration/calibration.json" \
    --policy "$POLICY" --samples 100 --seed "$((42 + ROUND))" \
    --output "$OUTPUT_ROOT/online/round_$ROUND"
  REPLAY+=("$OUTPUT_ROOT/online/round_$ROUND")
  python main.py train --model wan21 --online --resume "$POLICY" \
    --data "${REPLAY[@]}" "${RESTORE[@]}" \
    --weights "models/MORCA/online/round_$ROUND" \
    --output "$OUTPUT_ROOT/online/training_$ROUND"
  POLICY="models/MORCA/online/round_$ROUND/selected.pt"
  RESTORE=(--restore-optimizer)
done
```

</details>

Each round uses 5 critic-only epochs and 20 joint epochs, τ=.7, β=1, cap 20, and actor/critic learning rates 4e-5/1e-4. The normalizer stays fixed. Checkpoint selection uses actor agreement and probability drift over the last 10 joint epochs on fixed training states. Use the selected checkpoint for inference.

## Acknowledgements

This repository builds on [Wan2.1](https://github.com/Wan-Video/Wan2.1), [Wan2.2](https://github.com/Wan-Video/Wan2.2), [SeaCache](https://github.com/jiwoogit/SeaCache), [TeaCache](https://github.com/ali-vilab/TeaCache), [MagCache](https://github.com/Zehong-Ma/MagCache), [DiCache](https://github.com/Bujiazi/DiCache), and [VBench](https://github.com/Vchitect/VBench). We thank their authors for sharing their work. See `licenses/` for attribution and license notices.

## Citation

```bibtex
@article{xiong2026morca,
  title={MORCA: Offline-to-Online Reinforcement Learning for Adaptive Cache Reuse in Video Diffusion Acceleration},
  author={Xiong, Yuxiang and Wang, Ruiyan and Wang, Wenqiang and Hu, Teng and Shen, Songhang and Feng, Bohao and Deng, Hongqian and Yi, Ran},
  journal={arXiv preprint arXiv:2610.10457},
  year={2026},
  url={https://arxiv.org/abs/2610.10457}
}
```
