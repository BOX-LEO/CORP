# CORP — Activation-based structured pruning for transformers

One-shot structured pruning with closed-form compensation. A single generic
pipeline (`pruning/`) drives four model families (DeiT/ViT, DINOv2, OPT) via
a task-registry architecture and YAML configs.

## Install

```bash
pip install -r requirements.txt
```

PyTorch is expected to be installed with matching CUDA; install it first if
needed (see https://pytorch.org).

## Layout

```
pruning/             # Core algorithm: collect → analyze → rank → compensate → apply
config/              # Schemas (enums + dataclasses) + YAML loader
model/               # Model loaders keyed by `source` (timm | torch_hub_dinov2 | hf_opt)
dataset/             # Dataset loaders keyed by `task`
evaluation/          # Evaluation functions keyed by `task`
model_prune/         # Orchestration: runner + experiment driver + baseline cache
configs/             # Example YAML configs (one per demo run)
run_prune.py         # `python run_prune.py --config <yaml>`
run_experiments.py   # `python run_experiments.py --config <yaml>`
```

Each of `model/`, `dataset/`, `evaluation/` has a single dispatch function
keyed by a string from the YAML. To add a new model or task: add one file
+ one registry entry; no other code changes.

## Run

Point each config's dataset paths at your local data, then:

```bash
# DeiT-tiny MLP pruning
python run_prune.py --config configs/deit_tiny_mlp.yaml

# Override any field on the command line
python run_prune.py --config configs/deit_tiny_mlp.yaml --set pruning.sparsity=0.4

# Per-target sparsities (target=both): set `mlp_sparsity` + `attn_sparsity`
# in place of `sparsity` — see configs/vit_base_both.yaml.

# ViT-base with separate sparsities
python run_prune.py --config configs/vit_base_both.yaml

# OPT-125m perplexity
python run_prune.py --config configs/opt_125m.yaml

# DINOv2-base (requires head .pth files, see below)
python run_prune.py --config configs/dinov2_base.yaml

# Sweep
python run_experiments.py --config configs/experiments/deit_sparsity_sweep.yaml
```

## Data

All paths are relative. Either drop your data under `data/` or override the
path in the YAML:

| Task | Expected layout |
| --- | --- |
| `imagenet_classification` | `data/imagenet-mini/val/<class>/*.JPEG` (ImageFolder) |
| `language_modeling` | Streamed from HuggingFace Datasets (C4 + WikiText-2); no local data. |
| `dinov2_vision` | `data/nyu_v2/nyu2_train|nyu2_test/...` and `data/ade20k/ADEChallengeData2016/...` |

For DINOv2 downstream evaluation (depth/seg), drop the pretrained head
weights under `model/heads/`:

- `dinov2_vitb14_nyu_dpt_head.pth`
- `dinov2_vitb14_ade20k_linear_head.pth`

(swap `vitb14` for `vits14`/`vitl14`/`vitg14` for other sizes).

## Caches

Written under `cache/` (relative to the YAML unless absolute). Safe to delete.

- `cache/baselines/baseline.json` — model baseline metrics keyed by `model|val_path|task`.
- `cache/stats/*.pkl` — activation statistics; re-collection is deterministic.
- `cache/experiments/*.json` — sweep results keyed by override signature.

## YAML reference

See `configs/deit_tiny_mlp.yaml` for a minimal example. Supported blocks:

- `model`: `source`, `name`, `checkpoint?`, `pretrained?`, `heads_dir?`.
- `task`: one of `imagenet_classification`, `language_modeling`, `dinov2_vision`.
- `dataset`: per-task — see `dataset/` module for each task's keys.
- `cache`: `baseline_dir`, `stats_dir`, `experiments_dir`, `force_recollect`.
- `pruning`: `target`, `schedule`, `sparsity` (or `mlp_sparsity`+`attn_sparsity`), `ranker`, `lambda_reg`, `min_channels`, `min_heads` (OPT attn only), `min_qk_dim`, `qk_sparsity`. Generic attention pruning is always dim-logit; OPT uses head pruning.
- `collector`: `covariance_mode`, `sketch_dim`, `store_raw`, `subsample_tokens`.
- `runner`: `device`, `dtype`, `seed`, `output_dir`, `save_pruned_path`, `calib_samples`.
- `experiment`: sweep + `result_cache` (only for `run_experiments.py`).

Paths in the YAML resolve relative to the YAML file's directory.
