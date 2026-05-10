"""Command line entry points for training-base initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from cfie_training.training_base.capacity_planning import (
    capacity_report_from_manifest,
    estimate_qwen35_moe_capacity,
)
from cfie_training.training_base.checkpoint_import import (
    Qwen35MoeCheckpointImportConfig,
    import_qwen35_moe_checkpoint,
    initialize_fp32_store_from_import_plan,
    qwen35_moe_checkpoint_key_filter,
)
from cfie_training.training_base.checkpoint_io import iter_checkpoint_tensors
from cfie_training.training_base.checkpoint_io import CheckpointTensorLoadStats
from cfie_training.training_base.manifest_builder import (
    ManifestShardConfig,
    Qwen35MoeManifestConfig,
    TrainingBaseManifestBuilder,
)


# ────── CLI 入口：cfie-training-base ──────
def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = args.func(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cfie-training-base",
        description="Prepare CFIE large-model training-base stores.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate = subparsers.add_parser(
        "estimate-qwen35-moe",
        help="Dry-run capacity planning from Qwen3.5 MoE dimensions.",
    )
    _add_qwen_manifest_args(estimate)
    _add_manifest_shard_args(estimate)
    estimate.set_defaults(func=_run_estimate_qwen35_moe)

    init = subparsers.add_parser(
        "init-qwen35-moe",
        help="Import Qwen3.5 MoE checkpoint tensors into training stores.",
    )
    init.add_argument("--checkpoint", required=True, type=Path)
    init.add_argument("--root", required=True, type=Path)
    init.add_argument("--dry-run", action="store_true")
    init.add_argument("--progress-every-tensors", type=int, default=0)
    _add_import_args(init)
    _add_manifest_shard_args(init)
    init.set_defaults(func=_run_init_qwen35_moe)

    train = subparsers.add_parser(
        "train",
        help="Run training loop on imported stores.",
    )
    train.add_argument("--root", required=True, type=Path)
    train.add_argument("--checkpoint", required=True, type=Path)
    train.add_argument("--steps", type=int, default=50)
    train.add_argument("--window-steps", type=int, default=50)
    train.add_argument("--lr", type=float, default=0.01)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--seq-len", type=int, default=8)
    train.add_argument("--num-layers", type=int, default=2)
    train.add_argument("--num-experts", type=int, default=4)
    train.add_argument("--hidden-size", type=int, default=3072)
    train.add_argument("--intermediate-size", type=int, default=1024)
    train.add_argument("--generation", type=int, default=0)
    train.add_argument("--grad-bucket-count", type=int, default=4)
    train.add_argument("--grad-bucket-size-mib", type=int, default=512)
    _add_manifest_shard_args(train)
    train.set_defaults(func=_run_train)

    return parser


def _run_estimate_qwen35_moe(args: argparse.Namespace) -> dict[str, Any]:
    qwen_config = _qwen_manifest_config_from_args(args)
    shard_config = _manifest_shard_config_from_args(args)
    report = estimate_qwen35_moe_capacity(qwen_config, shard_config)
    return {
        "command": "estimate-qwen35-moe",
        "dry_run": True,
        "qwen_config": _dataclass_dict(qwen_config),
        "shard_config": _dataclass_dict(shard_config),
        "capacity": report.to_dict(),
    }


def _run_init_qwen35_moe(args: argparse.Namespace) -> dict[str, Any]:
    total_start = time.perf_counter()
    import_config = _import_config_from_args(args)
    shard_config = _manifest_shard_config_from_args(args)
    read_stats = CheckpointTensorLoadStats()
    checkpoint_tensors = iter_checkpoint_tensors(
        args.checkpoint,
        key_filter=qwen35_moe_checkpoint_key_filter(import_config),
        stats=read_stats,
    )
    if args.progress_every_tensors > 0:
        checkpoint_tensors = _iter_with_progress(
            checkpoint_tensors,
            read_stats,
            args.progress_every_tensors,
        )

    import_start = time.perf_counter()
    plan = import_qwen35_moe_checkpoint(
        checkpoint_tensors,
        config=import_config,
    )
    import_seconds = time.perf_counter() - import_start

    manifest_start = time.perf_counter()
    manifest = TrainingBaseManifestBuilder(shard_config).build(plan.specs)
    report = capacity_report_from_manifest(manifest)
    manifest_seconds = time.perf_counter() - manifest_start
    phase_seconds = {
        "checkpoint_import": import_seconds,
        "manifest_build": manifest_seconds,
        "store_write": 0.0,
        "total": time.perf_counter() - total_start,
    }

    payload: dict[str, Any] = {
        "command": "init-qwen35-moe",
        "dry_run": bool(args.dry_run),
        "checkpoint": str(args.checkpoint),
        "root": str(args.root),
        "imported_param_count": len(plan.imported_params),
        "skipped_key_count": len(plan.skipped_keys),
        "prefilter_enabled": True,
        "checkpoint_read": read_stats.to_dict(),
        "phase_seconds": phase_seconds,
        "capacity": report.to_dict(),
    }
    if args.dry_run:
        phase_seconds["total"] = time.perf_counter() - total_start
        return payload

    write_start = time.perf_counter()
    result = initialize_fp32_store_from_import_plan(
        plan=plan,
        root=args.root,
        manifest_config=shard_config,
        generation=args.generation,
    )
    phase_seconds["store_write"] = time.perf_counter() - write_start
    phase_seconds["total"] = time.perf_counter() - total_start
    payload.update(
        {
            "fp32_manifest": str(result.fp32_store.manifest_path),
            "adam_manifest": str(result.adam_store.manifest_path),
            "gptq_manifest": str(result.gptq_store.manifest_path),
            "generation": args.generation,
        }
    )
    return payload


def _iter_with_progress(
    tensors,
    stats: CheckpointTensorLoadStats,
    every: int,
):
    for name, tensor in tensors:
        yield name, tensor
        if stats.yielded_tensor_count % every:
            continue
        print(
            json.dumps(
                {
                    "event": "checkpoint_read_progress",
                    "yielded_tensor_count": stats.yielded_tensor_count,
                    "yielded_tensor_bytes": stats.yielded_tensor_bytes,
                    "last_key": name,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )


def _add_qwen_manifest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-layers", required=True, type=int)
    parser.add_argument("--num-experts", required=True, type=int)
    parser.add_argument("--hidden-size", required=True, type=int)
    parser.add_argument("--intermediate-size", required=True, type=int)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-prefix", default="layers")
    parser.add_argument("--local-expert-ids", default="")
    parser.add_argument("--no-trainable", action="store_true")
    parser.add_argument("--no-gptq-cache", action="store_true")


def _add_import_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-layer-prefix", default="layers")
    parser.add_argument("--checkpoint-mlp-name", default="mlp")
    parser.add_argument("--checkpoint-experts-name", default="experts")
    parser.add_argument("--internal-layer-prefix", default="layers")
    parser.add_argument("--gate-proj-name", default="gate_proj")
    parser.add_argument("--up-proj-name", default="up_proj")
    parser.add_argument("--down-proj-name", default="down_proj")
    parser.add_argument("--weight-name", default="weight")
    parser.add_argument("--qweight-name", default="qweight")
    parser.add_argument("--scales-name", default="scales")
    parser.add_argument("--qzeros-name", default="qzeros")
    parser.add_argument("--g-idx-name", default="g_idx")
    parser.add_argument("--gptq-decoded-layout", choices=("k_n", "n_k"),
                        default="n_k")
    parser.add_argument("--known-root-prefixes", default="model.,")
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end-exclusive", type=int)
    parser.add_argument("--local-expert-ids", default="")
    parser.add_argument("--no-trainable", action="store_true")
    parser.add_argument("--no-gptq-cache", action="store_true")
    parser.add_argument("--generation", type=int, default=0)


def _add_manifest_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fp32-shard-bytes", type=int, default=1 << 30)
    parser.add_argument("--adam-shard-bytes", type=int, default=1 << 30)
    parser.add_argument("--gptq-shard-bytes", type=int, default=1 << 30)
    parser.add_argument("--adam-block-size", type=int, default=128)
    parser.add_argument("--gptq-group-size", type=int, default=128)
    parser.add_argument("--fp32-shard-prefix", default="fp32")
    parser.add_argument("--adam-shard-prefix", default="adam")
    parser.add_argument("--gptq-shard-prefix", default="gptq")


def _qwen_manifest_config_from_args(
    args: argparse.Namespace,
) -> Qwen35MoeManifestConfig:
    return Qwen35MoeManifestConfig(
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        tp_size=args.tp_size,
        layer_start=args.layer_start,
        layer_prefix=args.layer_prefix,
        trainable=not args.no_trainable,
        include_gptq_cache=not args.no_gptq_cache,
        local_expert_ids=_parse_int_tuple(args.local_expert_ids),
    )


def _import_config_from_args(
    args: argparse.Namespace,
) -> Qwen35MoeCheckpointImportConfig:
    return Qwen35MoeCheckpointImportConfig(
        checkpoint_layer_prefix=args.checkpoint_layer_prefix,
        checkpoint_mlp_name=args.checkpoint_mlp_name,
        checkpoint_experts_name=args.checkpoint_experts_name,
        internal_layer_prefix=args.internal_layer_prefix,
        gate_proj_name=args.gate_proj_name,
        up_proj_name=args.up_proj_name,
        down_proj_name=args.down_proj_name,
        weight_name=args.weight_name,
        qweight_name=args.qweight_name,
        scales_name=args.scales_name,
        qzeros_name=args.qzeros_name,
        g_idx_name=args.g_idx_name,
        gptq_group_size=args.gptq_group_size,
        gptq_decoded_layout=args.gptq_decoded_layout,
        known_root_prefixes=_parse_string_tuple(args.known_root_prefixes),
        include_gptq_cache=not args.no_gptq_cache,
        trainable=not args.no_trainable,
        layer_start=args.layer_start,
        layer_end_exclusive=args.layer_end_exclusive,
        local_expert_ids=_parse_int_tuple(args.local_expert_ids),
    )


def _manifest_shard_config_from_args(
    args: argparse.Namespace,
) -> ManifestShardConfig:
    return ManifestShardConfig(
        fp32_shard_bytes=args.fp32_shard_bytes,
        adam_shard_bytes=args.adam_shard_bytes,
        gptq_shard_bytes=args.gptq_shard_bytes,
        adam_block_size=args.adam_block_size,
        gptq_group_size=args.gptq_group_size,
        fp32_shard_prefix=args.fp32_shard_prefix,
        adam_shard_prefix=args.adam_shard_prefix,
        gptq_shard_prefix=args.gptq_shard_prefix,
    )


def _parse_int_tuple(value: str) -> tuple[int, ...] | None:
    if value.strip() == "":
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_string_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(","))


def _dataclass_dict(value: Any) -> dict[str, Any]:
    return {
        field: getattr(value, field)
        for field in value.__dataclass_fields__
    }


def _run_train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from cfie_training.training_base.adam_update import AdamWConfig
    from cfie_training.training_base.model_loader import Qwen35RealImporter
    from cfie_training.training_base.training_model import Qwen35ForTraining
    from cfie_training.training_base.training_loop import (
        TrainingLoop,
        TrainingLoopConfig,
    )

    importer = Qwen35RealImporter(
        checkpoint_dir=args.checkpoint,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
    )
    fp32_store, adam_store, gptq_store, manifest, progress = (
        importer.import_to_stores(
            args.root,
            manifest_config=_manifest_shard_config_from_args(args),
            layers=tuple(range(args.num_layers)),
            experts=tuple(range(args.num_experts)),
        )
    )

    hot_param_ids: list[str] = []
    for layer_id in range(args.num_layers):
        for expert_id in range(args.num_experts):
            hot_param_ids.append(
                f"layers.{layer_id}.experts.{expert_id}.w13_weight"
            )
            hot_param_ids.append(
                f"layers.{layer_id}.experts.{expert_id}.w2_weight"
            )
    hot_param_ids_tuple = tuple(hot_param_ids)

    config = TrainingLoopConfig(
        adam_config=AdamWConfig(lr=args.lr),
        shadow_dtype=torch.float32,
        shadow_device="cpu",
        bucket_capacity_bytes=args.grad_bucket_size_mib << 20,
        max_sealed_buckets=args.grad_bucket_count,
        enable_peak_monitor=True,
    )

    loop = TrainingLoop.from_stores(
        fp32_store=fp32_store,
        adam_store=adam_store,
        gptq_store=gptq_store,
        manifest=manifest,
        progress_writer=progress,
        hot_param_ids=hot_param_ids_tuple,
        config=config,
    )

    model = Qwen35ForTraining(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=8,
        vocab_size=248320,
        dtype=torch.float32,
        device="cpu",
    )
    model.setup_hot_params(loop.hot_window.shadow_store, hot_param_ids_tuple)
    loop.attach_model(model)

    def _data_iter(steps, batch, seq):
        for s in range(steps):
            from cfie_training.training_base.training_loop import TrainingDataBatchInput
            yield TrainingDataBatchInput(
                input_ids=torch.randint(0, 1000, (batch, seq)),
                global_step=s, epoch=0, dataset_cursor=str(s),
                consumed_samples=batch, consumed_tokens=batch * seq,
            )

    loop.attach_dataloader(_data_iter(args.steps, args.batch_size, args.seq_len))
    results = loop.run(num_steps=args.steps)

    return {
        "command": "train",
        "total_steps": len(results),
        "final_step": results[-1].global_step if results else 0,
        "window_committed": results[-1].window_committed if results else False,
        "final_loss": results[-1].loss if results else 0.0,
        "num_results": len(results),
    }


if __name__ == "__main__":
    raise SystemExit(main())
