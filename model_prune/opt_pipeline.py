"""OPT-specific pruning pipeline.

The generic ``PruneRunner`` in ``pruning/runner.py`` expects ``model.blocks[i]``
with fused QKV, which doesn't match HuggingFace OPT (``model.model.decoder.layers[i]``
with separate q/k/v/out_proj). This file keeps the OPT-specific orchestration
behind the same ``RunResult``-like contract so the unified runner can dispatch
on ``meta['source'] == 'hf_opt'``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import copy
import logging

import torch
import torch.nn as nn

from pruning.apply_masks import MaskApplier
from pruning.collect import ActivationCollector, LayerActivationStats
from pruning.compensate import AffineCompensator, QKDimCompensator
from pruning.ranking import QKDimRanker, RankingPolicy, StructureRanker
from pruning.stats import RedundancyAnalyzer, RedundancyReport

from config.schemas import AttnMode, FullConfig, PruneTarget, RankerType, ScheduleType
from model.opt import (
    prune_attention_heads as prune_opt_attention_heads,
    prune_attention_qk_dims as prune_opt_attention_qk_dims,
)

logger = logging.getLogger(__name__)


def _make_opt_attn_head_hook(collector: ActivationCollector, layer_name: str,
                             num_heads: int, head_dim: int):
    """Per-head activation hook for OPT self-attn head pruning.

    OPT's `out_proj` receives a `(B, T, num_heads * head_dim)` tensor; this
    hook reshapes it to per-head L2 norms and feeds them into the collector as
    a `num_heads`-dimensional "feature" vector. The generic ViT path uses
    dim-logit attention instead, so this helper lives here rather than in
    pruning/collect.py.
    """
    def hook(module, inputs, output):
        if not collector._collecting:
            return
        x = inputs[0] if isinstance(inputs, tuple) else inputs
        x = x.detach()
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == num_heads * head_dim, \
            f"Expected embed_dim={num_heads}*{head_dim}={num_heads*head_dim}, got {embed_dim}"
        x = x.view(batch_size, seq_len, num_heads, head_dim)
        x_head_norm = (x ** 2).sum(dim=-1).reshape(-1, num_heads)
        collector._process_activation_raw(layer_name, x_head_norm, is_head_stat=True)
    return hook


def _make_opt_qk_dim_hook(collector: ActivationCollector, layer_name: str,
                          self_attn: nn.Module):
    """Per-head per-dim Q/K activation hook for OPT dim-logit pruning.

    Hooks the self_attn module forward, recomputes `q_proj(hidden_states)` and
    `k_proj(hidden_states)` (cheap during calibration), and feeds the per-head
    Q, K, and joint q²·k² stats into the collector under keys matching the
    generic dim-logit path so `RedundancyAnalyzer.generate_all_qk_dim_reports`
    works unchanged. Reports come back keyed `{layer_name}.attn.head_{h}`.

    Note: must be registered with `with_kwargs=True` because OPTDecoderLayer
    calls `self_attn(hidden_states=..., ...)` — all kwargs, no positional args.
    """
    num_heads = self_attn.num_heads
    head_dim = self_attn.head_dim
    q_proj = self_attn.q_proj
    k_proj = self_attn.k_proj
    subsample = collector.config.subsample_tokens

    def hook(module, args, kwargs, output):
        if not collector._collecting:
            return
        x = kwargs.get("hidden_states")
        if x is None and args:
            x = args[0]
        if x is None:
            return
        x = x.detach()
        with torch.no_grad():
            q = q_proj(x).reshape(x.shape[0], x.shape[1], num_heads, head_dim)
            k = k_proj(x).reshape(x.shape[0], x.shape[1], num_heads, head_dim)

        if subsample is not None and q.shape[1] > subsample:
            indices = torch.randperm(q.shape[1], device=q.device)[:subsample]
            q = q[:, indices, :, :]
            k = k[:, indices, :, :]

        for h in range(num_heads):
            q_h = q[:, :, h, :].reshape(-1, head_dim)
            k_h = k[:, :, h, :].reshape(-1, head_dim)
            collector._process_activation_raw(f"{layer_name}.q.head_{h}", q_h, is_head_stat=False)
            collector._process_activation_raw(f"{layer_name}.k.head_{h}", k_h, is_head_stat=False)
            q2k2 = (q_h ** 2) * (k_h ** 2)
            collector._accumulate_q2k2(f"{layer_name}.qk.head_{h}", q2k2, head_dim)
    return hook


@dataclass
class OPTRunResult:
    success: bool
    original_params: int
    pruned_params: int
    compression_ratio: float
    step_results: List[dict] = field(default_factory=list)
    pruned_model: Optional[nn.Module] = None
    error_message: Optional[str] = None
    output_dir: Optional[Any] = None


def _register_attn_hook(collector, layer, layer_idx: int, attn_mode: str) -> None:
    """Register the right attention hook for `attn_mode`. Returns None."""
    if attn_mode == "dim_logit":
        # Hook self_attn itself so the hook can access hidden_states (kwargs)
        # and recompute Q, K per-head. Report keys: {name}.attn.head_{h}.
        # `with_kwargs=True` is required because OPTDecoderLayer calls
        # self_attn(hidden_states=...) with everything as kwargs.
        name = f"model.decoder.layers.{layer_idx}.self_attn"
        hook = _make_opt_qk_dim_hook(collector, name, layer.self_attn)
        collector._hooks.append(
            layer.self_attn.register_forward_hook(hook, with_kwargs=True)
        )
    else:
        # Head-pruning: hook out_proj input, store per-head L2 norms.
        name = f"model.decoder.layers.{layer_idx}.self_attn.out_proj"
        hook = _make_opt_attn_head_hook(
            collector, name, layer.self_attn.num_heads, layer.self_attn.head_dim,
        )
        collector._hooks.append(layer.self_attn.out_proj.register_forward_hook(hook))


def _collect_layer_stats(
    model, layer_idx, calib_loader, collector_cfg, device, target: str,
    attn_mode: str = "head",
) -> Dict[str, LayerActivationStats]:
    layer = model.model.decoder.layers[layer_idx]
    collector = ActivationCollector(model, collector_cfg, device)

    if target in ("mlp", "both"):
        name = f"model.decoder.layers.{layer_idx}.fc2"
        hook = collector._create_input_hook(name)
        collector._hooks.append(layer.fc2.register_forward_hook(hook))

    if target in ("attn", "both"):
        _register_attn_hook(collector, layer, layer_idx, attn_mode)

    with collector.collect():
        for batch in calib_loader:
            with torch.no_grad():
                model(input_ids=batch[0].to(device))
    collector.clear_hooks()
    return collector.get_all_stats()


def _collect_all_stats(
    model, calib_loader, collector_cfg, num_layers, device, target: str,
    attn_mode: str = "head",
) -> Dict[str, LayerActivationStats]:
    collector = ActivationCollector(model, collector_cfg, device)
    for i in range(num_layers):
        layer = model.model.decoder.layers[i]
        if target in ("mlp", "both"):
            name = f"model.decoder.layers.{i}.fc2"
            hook = collector._create_input_hook(name)
            collector._hooks.append(layer.fc2.register_forward_hook(hook))
        if target in ("attn", "both"):
            _register_attn_hook(collector, layer, i, attn_mode)
    with collector.collect():
        for batch in calib_loader:
            with torch.no_grad():
                model(input_ids=batch[0].to(device))
    collector.clear_hooks()
    return collector.get_all_stats()


def _apply_mlp(model, layer_idx, stats, report, ranker, compensator, mask_applier,
               sparsity, skip_compensation) -> dict:
    layer = model.model.decoder.layers[layer_idx]
    prune_idx, surv_idx = ranker.select_for_sparsity(
        report, sparsity, module=layer, module_type="mlp",
    )
    n_prune = len(prune_idx)
    n_total = report.feature_dim
    if n_prune == 0:
        return {"layer": layer_idx, "type": "mlp", "pruned": 0, "total": n_total}
    compensation = None
    if not skip_compensation and stats.covariance is not None:
        compensation = compensator.fit(stats, prune_idx, surv_idx)
    mask_applier.prune_ffn_intermediate(layer, prune_idx, compensation)
    return {"layer": layer_idx, "type": "mlp", "pruned": n_prune, "total": n_total}


def _apply_attn_head(model, layer_idx, report, ranker, sparsity, num_heads, min_heads) -> dict:
    """Whole-head pruning (existing path)."""
    layer = model.model.decoder.layers[layer_idx]
    target_prune = max(0, int(num_heads * sparsity))
    target_prune = min(target_prune, num_heads - min_heads)
    if target_prune <= 0:
        return {"layer": layer_idx, "type": "attn", "pruned": 0, "total": num_heads}

    if ranker.policy in (
        RankingPolicy.WEIGHT_MAGNITUDE,
        RankingPolicy.ENERGY_WEIGHTMAGNITUDE,
        RankingPolicy.ACTIVE_WEIGHTMAGNITUDE,
        RankingPolicy.ENSEMBLE,
    ):
        prune_idx, _ = ranker.select_for_sparsity(
            report, sparsity, module=layer.self_attn, module_type="attn",
        )
    else:
        ranked = ranker.rank(report)
        prune_idx = ranked[:target_prune]

    prune_opt_attention_heads(layer.self_attn, prune_idx)
    return {"layer": layer_idx, "type": "attn", "pruned": len(prune_idx), "total": num_heads}


def _apply_attn_qk_dim(
    model, layer_idx, qk_reports, qk_ranker, qk_compensator,
    qk_sparsity, skip_compensation,
) -> dict:
    """Per-head Q/K dim-logit pruning. Mirrors generic ViT path."""
    layer = model.model.decoder.layers[layer_idx]
    self_attn = layer.self_attn
    num_heads = self_attn.num_heads
    head_dim = self_attn.head_dim
    layer_name = f"model.decoder.layers.{layer_idx}.self_attn"

    compensation_results = []
    total_pruned = 0
    for h in range(num_heads):
        key = f"{layer_name}.attn.head_{h}"
        if key not in qk_reports:
            logger.warning(f"Missing Q/K report for {key}, skipping head")
            continue
        report = qk_reports[key]
        prune_idx, surv_idx = qk_ranker.select_for_sparsity(report, qk_sparsity)
        result = qk_compensator.fit_dim_logit(
            report.q_stats, report.k_stats, prune_idx, surv_idx, layer_name, h,
        )
        compensation_results.append(result)
        total_pruned += len(prune_idx)

    if compensation_results:
        prune_opt_attention_qk_dims(
            self_attn, compensation_results,
            qk_compensator=None if skip_compensation else qk_compensator,
        )

    return {
        "layer": layer_idx, "type": "attn_qk_dim",
        "pruned": total_pruned, "total": num_heads * head_dim,
    }


def run_opt_prune(
    model: nn.Module,
    calib_loader,
    full_cfg: FullConfig,
    num_layers: int,
    skip_compensation: bool = False,
    mlp_sparsity: Optional[float] = None,
    attn_sparsity: Optional[float] = None,
) -> OPTRunResult:
    """OPT-specific pruning loop. Returns an OPTRunResult with a pruned_model."""
    target_value = full_cfg.pruning.target.value
    prune_mlp = target_value in ("mlp", "both")
    prune_attn = target_value in ("attn", "both")
    schedule = full_cfg.pruning.schedule.value
    attn_mode = full_cfg.pruning.attn_mode.value
    device = full_cfg.runner.device

    original_params = sum(p.numel() for p in model.parameters())
    pruned_model = copy.deepcopy(model).to(device)
    model.cpu()
    torch.cuda.empty_cache()

    analyzer = RedundancyAnalyzer()
    ranker = StructureRanker.from_config(
        full_cfg.pruning.ranker,
        min_channels=full_cfg.pruning.min_channels,
        keep_topk_outliers=full_cfg.pruning.keep_topk_outliers,
    )
    compensator = AffineCompensator(
        lambda_reg=full_cfg.pruning.lambda_reg,
        auto_shrinkage=full_cfg.pruning.auto_shrinkage,
    )
    qk_ranker = QKDimRanker(min_qk_dim=full_cfg.pruning.min_qk_dim)
    qk_compensator = QKDimCompensator(
        lambda_reg=full_cfg.pruning.lambda_reg,
        auto_shrinkage=full_cfg.pruning.auto_shrinkage,
    )
    mask_applier = MaskApplier(validate_shapes=True)

    from config.schemas import CollectorConfig, CovarianceMode
    collector_cfg = CollectorConfig(
        target=full_cfg.pruning.target,
        covariance_mode=CovarianceMode.EXACT,
        store_raw=False,
        subsample_tokens=full_cfg.collector.subsample_tokens,
    )

    mlp_s = mlp_sparsity if mlp_sparsity is not None else full_cfg.pruning.sparsity
    attn_s = attn_sparsity if attn_sparsity is not None else full_cfg.pruning.sparsity
    qk_s = full_cfg.pruning.qk_sparsity
    results: List[dict] = []

    def _do_attn(stats_dict, reports_dict, qk_reports, layer_idx):
        if attn_mode == "dim_logit":
            results.append(_apply_attn_qk_dim(
                pruned_model, layer_idx, qk_reports, qk_ranker, qk_compensator,
                qk_s, skip_compensation,
            ))
        else:
            key = f"model.decoder.layers.{layer_idx}.self_attn.out_proj"
            if key in stats_dict:
                heads = pruned_model.model.decoder.layers[layer_idx].self_attn.num_heads
                results.append(_apply_attn_head(
                    pruned_model, layer_idx, reports_dict[key], ranker, attn_s, heads,
                    full_cfg.pruning.min_heads,
                ))

    try:
        if schedule == "global":
            all_stats = _collect_all_stats(
                pruned_model, calib_loader, collector_cfg, num_layers, device, target_value,
                attn_mode=attn_mode,
            )
            all_reports = analyzer.reports_from_all_stats(all_stats)
            qk_reports = (
                analyzer.generate_all_qk_dim_reports(all_stats)
                if prune_attn and attn_mode == "dim_logit" else {}
            )
            for i in range(num_layers):
                if prune_mlp:
                    key = f"model.decoder.layers.{i}.fc2"
                    if key in all_stats:
                        results.append(_apply_mlp(
                            pruned_model, i, all_stats[key], all_reports[key],
                            ranker, compensator, mask_applier, mlp_s, skip_compensation,
                        ))
                if prune_attn:
                    _do_attn(all_stats, all_reports, qk_reports, i)
        else:  # layerwise
            for i in range(num_layers):
                stats = _collect_layer_stats(
                    pruned_model, i, calib_loader, collector_cfg, device, target_value,
                    attn_mode=attn_mode,
                )
                reports = analyzer.reports_from_all_stats(stats)
                qk_reports = (
                    analyzer.generate_all_qk_dim_reports(stats)
                    if prune_attn and attn_mode == "dim_logit" else {}
                )
                if prune_mlp:
                    key = f"model.decoder.layers.{i}.fc2"
                    if key in stats:
                        results.append(_apply_mlp(
                            pruned_model, i, stats[key], reports[key],
                            ranker, compensator, mask_applier, mlp_s, skip_compensation,
                        ))
                if prune_attn:
                    _do_attn(stats, reports, qk_reports, i)
                del stats, reports, qk_reports
                torch.cuda.empty_cache()

        pruned_params = sum(p.numel() for p in pruned_model.parameters())
        return OPTRunResult(
            success=True,
            original_params=original_params,
            pruned_params=pruned_params,
            compression_ratio=original_params / pruned_params,
            step_results=results,
            pruned_model=pruned_model,
        )
    except Exception as e:
        logger.exception("OPT pruning failed")
        return OPTRunResult(
            success=False,
            original_params=original_params,
            pruned_params=original_params,
            compression_ratio=1.0,
            step_results=results,
            pruned_model=None,
            error_message=str(e),
        )
