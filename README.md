# sFWI: Stochastic Full-Waveform Inversion

This repository contains the **accepted-code snapshot** of the sFWI project, including:
- core `sFWI` modules and experiment scripts,
- a vendored `DAPS` dependency,
- a vendored `score_sde_pytorch` dependency.

The previous remote codebase is intended to be kept as a **legacy branch**.

## What Is sFWI

sFWI is a stochastic inversion framework for seismic velocity reconstruction under limited illumination settings (for example, single-shot regimes).  
Instead of outputting one deterministic map, sFWI models a posterior family of solutions and supports downstream selection strategies.

## Why This Repo Is Different

- Physics-consistent inversion workflows are preserved (forward operator + misfit-aware sampling).
- Includes both random sampling (`RSS`) and group-guided search (`GSS`) pipelines.
- Keeps baseline comparison support (AE / UNet / VAE / DiffAE / InversionNet adapters).
- Consolidates training + evaluation scripts used for the accepted manuscript revision cycle.

## Repository Layout

```text
.
├── sFWI/
│   ├── config.py
│   ├── data/
│   ├── models/
│   ├── operators/
│   ├── training/
│   ├── experiments/
│   └── evaluation/
├── DAPS/
├── score_sde_pytorch/
├── baselines.json
└── baselines_inversionnet.json
```

## Environment Setup

Recommended: Python 3.10+ with CUDA-enabled PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r DAPS/requirements.txt
pip install -r score_sde_pytorch/requirements.txt
pip install deepwave segyio pot omegaconf ml-collections scikit-learn tensorboard piq ninja
```

If your CUDA extensions fail to build, clear torch extension cache and retry:

```bash
rm -rf ~/.cache/torch_extensions
```

## Data / Checkpoint Path Convention

`sFWI/config.py` resolves paths relative to the **parent of this repository directory**.

Expected external layout:

```text
<workspace_parent>/
├── sFWI/                      # this repository
├── checkpoints/               # generated model checkpoints
├── outputs/                   # experiment outputs
├── SEAM_I_2D_Model/
│   └── SEAM_I_2D_Model/SEAM_Vp_Elastic_N23900.sgy
└── solving_inverse_in_SGM/
    └── dataset/seismic_dataset.pkl
```

You can also override default paths via CLI arguments in training/experiment scripts.

## Quick Run (Minimal)

1) Marmousi pretrain

```bash
python sFWI/training/marmousi_train.py \
  --data_path /path/to/seismic_dataset.pkl \
  --workdir /path/to/workdir \
  --batch_size 64 \
  --n_iters 5001
```

2) SEAM fine-tune

```bash
python sFWI/training/seam_finetune.py \
  --workdir /path/to/workdir \
  --pretrain_ckpt /path/to/workdir/checkpoints/marmousi_checkpoint_5.pth \
  --batch_size 64 \
  --n_iters 5001
```

3) Build GSS assets

```bash
python sFWI/experiments/rss_gss.py \
  --model_tag seam_finetune \
  --ckpt_dir /path/to/workdir/checkpoints \
  --ckpt_file seam_finetune_checkpoint_5.pth \
  --asset_dir /path/to/workdir/gss_assets \
  --cluster_max_samples 1000 \
  --k 100 \
  --n_noise_seeds 1500 \
  --master_seed 8 \
  --skip_rss \
  --skip_cluster_vis
```

4) OOD comparison

```bash
python sFWI/experiments/ood_generalization.py \
  --sampling_method gss_cached \
  --seam_model_tag seam_finetune \
  --seam_ckpt_dir /path/to/workdir/checkpoints \
  --seam_ckpt_file seam_finetune_checkpoint_5.pth \
  --marmousi_ckpt_dir /path/to/workdir/checkpoints \
  --marmousi_ckpt_file marmousi_checkpoint_5.pth \
  --sm_path_seam /path/to/workdir/gss_assets/sm_dataset-seam_model-seam_finetune_k100_j1500_seed8.pt \
  --sm_path_marmousi /path/to/workdir/gss_assets/sm_dataset-marmousi_model-marmousi_k100_j1500_seed8.pt \
  --n_samples 50 \
  --gss_top_k 50
```

## Notes

- This repository intentionally excludes large datasets, trained weights, figures, and temporary outputs.
- For full replacement publishing workflow, keep the old remote state on a `legacy-*` branch before force-updating default branch.
