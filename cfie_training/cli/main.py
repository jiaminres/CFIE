"""Command line entrypoint for the standalone CFIE training package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from cfie_training.config import TrainingProjectConfig
from cfie_training.predictor import PredictorTraceDataset, PredictorTrainer
from cfie_training.profiles import (
    DEFAULT_TRAINING_PROFILE,
    SUPPORTED_TRAINING_PROFILES,
    build_profile_config,
)
from cfie_training.runtime.project import TrainingProject
from cfie_training.runtime.memory import TrainingStartupEstimator
from cfie_training.runtime.types import (
    BatchPlannerCheckpoint,
    BatchShape,
    TrainingRuntimeSnapshot,
    TrainingSessionCheckpoint,
)


def build_parser() -> argparse.ArgumentParser:
    # 创建训练侧 CLI 顶层解析器。
    parser = argparse.ArgumentParser(
        prog="cfie-train",
        description="Standalone training scaffold for CFIE.",
    )
    # 为各子命令准备统一的 subparser 容器。
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -----------------
    # 注册 blueprint / 规划类命令。
    # 注册默认训练蓝图渲染命令。
    plan_parser = subparsers.add_parser(
        "plan",
        help="Render the default resource-first training blueprint.",
    )
    plan_parser.add_argument("--config", type=Path, help="Optional JSON config file.")
    plan_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )
    plan_parser.add_argument("--model", help="Override the development model label.")
    plan_parser.add_argument(
        "--target-model",
        help="Override the target model label.",
    )
    plan_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the plan as JSON.",
    )

    # 注册 predictor-routed MoE 蓝图渲染命令。
    predictor_parser = subparsers.add_parser(
        "predictor-plan",
        help="Render the predictor-routed MoE design blueprint.",
    )
    predictor_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    predictor_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )
    predictor_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor plan as JSON.",
    )

    # -----------------
    # 注册 predictor trace / train / inspect / eval / export 命令。
    # 注册 predictor teacher trace 采样命令。
    predictor_trace_parser = subparsers.add_parser(
        "predictor-trace",
        help="Build teacher-trace samples for predictor training.",
    )
    predictor_trace_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    predictor_trace_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )
    predictor_trace_parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of planning steps used to build trace examples.",
    )
    predictor_trace_parser.add_argument(
        "--examples-per-step",
        type=int,
        default=None,
        help="Optional override for the number of trace examples per step.",
    )
    predictor_trace_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Base samples per trace-capture step.",
    )
    predictor_trace_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per sample used during trace capture.",
    )
    predictor_trace_parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional text or JSONL dataset file used to build tokenized trace batches.",
    )
    predictor_trace_parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer path. Defaults to the training profile model path.",
    )
    predictor_trace_parser.add_argument(
        "--dataset-format",
        choices=("auto", "text", "jsonl"),
        default="auto",
        help="Dataset file format. Defaults to auto-detect from extension.",
    )
    predictor_trace_parser.add_argument(
        "--dataset-text-key",
        default="text",
        help="JSONL text field to read when --dataset-format jsonl is used.",
    )
    predictor_trace_parser.add_argument(
        "--output",
        type=Path,
        help="Optional path that receives the captured predictor trace dataset as JSON.",
    )
    predictor_trace_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor trace dataset as JSON.",
    )

    # 注册 predictor 训练命令。
    predictor_train_parser = subparsers.add_parser(
        "predictor-train",
        help="Train the bounded predictor model on teacher traces.",
    )
    predictor_train_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    predictor_train_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )
    predictor_train_parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of planning steps used to build teacher traces.",
    )
    predictor_train_parser.add_argument(
        "--examples-per-step",
        type=int,
        default=None,
        help="Optional override for the number of trace examples per step.",
    )
    predictor_train_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Base samples per trace-capture step.",
    )
    predictor_train_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per sample used during trace capture.",
    )
    predictor_train_parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional text or JSONL dataset file used to build tokenized trace batches.",
    )
    predictor_train_parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer path. Defaults to the training profile model path.",
    )
    predictor_train_parser.add_argument(
        "--dataset-format",
        choices=("auto", "text", "jsonl"),
        default="auto",
        help="Dataset file format. Defaults to auto-detect from extension.",
    )
    predictor_train_parser.add_argument(
        "--dataset-text-key",
        default="text",
        help="JSONL text field to read when --dataset-format jsonl is used.",
    )
    predictor_train_parser.add_argument(
        "--trace-input",
        type=Path,
        help="Optional predictor trace dataset JSON used instead of recapturing traces.",
    )
    predictor_train_parser.add_argument(
        "--init-from-checkpoint",
        "--resume-checkpoint",
        dest="init_from_checkpoint",
        type=Path,
        help="Optional predictor checkpoint used to initialize model weights before training.",
    )
    predictor_train_parser.add_argument(
        "--checkpoint-output",
        type=Path,
        help="Optional output path that receives the trained predictor checkpoint.",
    )
    predictor_train_parser.add_argument(
        "--schema-output",
        type=Path,
        help="Optional output path that receives the predictor runtime schema JSON.",
    )
    predictor_train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training epochs.",
    )
    predictor_train_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor training run trace as JSON.",
    )

    # 注册 predictor checkpoint 元信息检查命令。
    predictor_inspect_parser = subparsers.add_parser(
        "predictor-inspect",
        help="Inspect predictor checkpoint metadata.",
    )
    predictor_inspect_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Predictor checkpoint to inspect.",
    )
    predictor_inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor checkpoint metadata as JSON.",
    )

    # 注册 predictor 评估命令。
    predictor_eval_parser = subparsers.add_parser(
        "predictor-eval",
        help="Evaluate a predictor checkpoint on teacher traces.",
    )
    predictor_eval_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    predictor_eval_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )
    predictor_eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Predictor checkpoint to evaluate.",
    )
    predictor_eval_parser.add_argument(
        "--trace-input",
        type=Path,
        help="Optional predictor trace dataset JSON used instead of recapturing traces.",
    )
    predictor_eval_parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of planning steps used to build evaluation traces.",
    )
    predictor_eval_parser.add_argument(
        "--examples-per-step",
        type=int,
        default=None,
        help="Optional override for the number of evaluation trace examples per step.",
    )
    predictor_eval_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Base samples per evaluation trace-capture step.",
    )
    predictor_eval_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per sample used during evaluation trace capture.",
    )
    predictor_eval_parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional text or JSONL dataset file used to build tokenized evaluation batches.",
    )
    predictor_eval_parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer path. Defaults to the training profile model path.",
    )
    predictor_eval_parser.add_argument(
        "--dataset-format",
        choices=("auto", "text", "jsonl"),
        default="auto",
        help="Dataset file format. Defaults to auto-detect from extension.",
    )
    predictor_eval_parser.add_argument(
        "--dataset-text-key",
        default="text",
        help="JSONL text field to read when --dataset-format jsonl is used.",
    )
    predictor_eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor evaluation trace as JSON.",
    )

    # 注册 predictor bundle 导出命令。
    predictor_export_parser = subparsers.add_parser(
        "predictor-export",
        help="Export a predictor checkpoint as a deployment bundle.",
    )
    predictor_export_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Predictor checkpoint to export.",
    )
    predictor_export_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory that receives the exported deployment bundle.",
    )
    predictor_export_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor deployment manifest as JSON.",
    )

    # -----------------
    # 注册训练基座校验 / 模拟 / 启动估算 / 训练命令。
    # 注册配置校验命令。
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate the training project configuration.",
    )
    validate_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file. Defaults to the built-in scaffold config.",
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validated config as JSON.",
    )

    # 注册单步模拟命令。
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run the first-version training engine and emit step traces.",
    )
    simulate_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    simulate_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to simulate.",
    )
    simulate_parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of sequential training steps to simulate.",
    )
    simulate_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Samples per simulated step.",
    )
    simulate_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=512,
        help="Tokens per sample in the simulated batch.",
    )
    simulate_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the simulated run trace as JSON.",
    )
    simulate_parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional runtime snapshot JSON used to resume a simulated run.",
    )
    simulate_parser.add_argument(
        "--save-snapshot",
        type=Path,
        help="Optional path that receives the runtime snapshot after simulation.",
    )

    # 注册启动参数估算命令。
    estimate_startup_parser = subparsers.add_parser(
        "estimate-startup",
        help="Estimate startup parameters for different GPU hot budgets.",
    )
    estimate_startup_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    estimate_startup_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to estimate when --config is not provided.",
    )
    estimate_startup_parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Samples per startup-estimate batch.",
    )
    estimate_startup_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per sample in the estimated startup batch.",
    )
    estimate_startup_parser.add_argument(
        "--gpu-budgets-gb",
        default="",
        help="Comma-separated GPU hot budget candidates in GiB.",
    )
    estimate_startup_parser.add_argument(
        "--active-expert-candidates",
        default="",
        help="Comma-separated active_experts_per_step candidates.",
    )
    estimate_startup_parser.add_argument(
        "--max-live-bucket-candidates",
        default="",
        help="Comma-separated max_live_buckets candidates.",
    )
    estimate_startup_parser.add_argument(
        "--prefetch-bucket-candidates",
        default="",
        help="Comma-separated prefetch_buckets candidates.",
    )
    estimate_startup_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the startup estimates as JSON.",
    )

    # 注册多步训练会话命令。
    train_parser = subparsers.add_parser(
        "train",
        help="Run a multi-step training session with synthetic or dataset-backed batches.",
    )
    train_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    train_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to run.",
    )
    train_parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of training steps to execute.",
    )
    train_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Base samples per synthetic batch.",
    )
    train_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per training sample.",
    )
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional runtime snapshot JSON used to resume training.",
    )
    train_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Optional directory for periodic runtime snapshots.",
    )
    train_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save a runtime snapshot every N steps. Disabled at 0.",
    )
    train_parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional text or JSONL dataset file used to build tokenizer-backed batches.",
    )
    train_parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer path. Defaults to the training profile model path.",
    )
    train_parser.add_argument(
        "--dataset-format",
        choices=("auto", "text", "jsonl"),
        default="auto",
        help="Dataset file format. Defaults to auto-detect from extension.",
    )
    train_parser.add_argument(
        "--dataset-text-key",
        default="text",
        help="JSONL text field to read when --dataset-format jsonl is used.",
    )
    train_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the session trace as JSON.",
    )

    # 注册固定 Qwen3.5 profile 的快捷蓝图命令。
    qwen35_parser = subparsers.add_parser(
        "qwen35-plan",
        help="Render the dedicated Qwen3.5-35B-A3B training blueprint.",
    )
    qwen35_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the Qwen3.5 blueprint as JSON.",
    )
    # 返回构造完成的 CLI 解析器。
    return parser


def _load_config(args: argparse.Namespace) -> TrainingProjectConfig:
    # 优先从显式 JSON 配置加载训练配置。
    if getattr(args, "config", None):
        config = TrainingProjectConfig.from_json_file(args.config)
    else:
        # 未显式给配置时，退回到内置 profile 配置。
        profile_name = getattr(args, "profile", DEFAULT_TRAINING_PROFILE)
        config = build_profile_config(profile_name)
    # 若命令行显式覆盖 development model，则写回配置。
    if getattr(args, "model", None):
        config.model_targets.development_model = args.model
    # 若命令行显式覆盖 target model，则写回配置。
    if getattr(args, "target_model", None):
        config.model_targets.target_model = args.target_model
    # 最后做统一校验并返回。
    return config.validate()


def _parse_csv_numbers(
    raw: str,
    *,
    cast,
) -> tuple:
    # 空字符串直接返回空元组。
    if not raw.strip():
        return ()
    # 用列表累积解析后的数值。
    values = []
    for chunk in raw.split(","):
        # 逐段裁掉空白。
        item = chunk.strip()
        # 连续逗号留下的空段直接跳过。
        if not item:
            continue
        # 对每个有效片段执行外部给定的类型转换。
        values.append(cast(item))
    # 统一返回不可变元组。
    return tuple(values)


def _default_active_expert_candidates(
    config: TrainingProjectConfig,
) -> tuple[int, ...]:
    # 读取模型和当前 expert rotation 配置。
    model = config.model_spec
    current = config.expert_rotation.active_experts_per_step
    # 候选步长至少为 1，并优先复用 num_experts_per_tok。
    step = max(1, model.num_experts_per_tok)
    # 默认搜索窗口向上放宽到 4 个 step，并额外加入更激进的倍增档位。
    upper = min(model.num_experts, max(current + (4 * step), current * 4))
    # 合并基础值、当前值、逐级放大值和更激进的倍增值作为候选集。
    values = {
        model.num_experts_per_tok,
        current,
        min(model.num_experts, current + step),
        min(model.num_experts, current + (2 * step)),
        min(model.num_experts, current + (3 * step)),
        upper,
        min(model.num_experts, max(current * 2, model.num_experts_per_tok * 2)),
        min(model.num_experts, max(current * 4, model.num_experts // 4)),
    }
    # 过滤掉小于最小可执行 expert 数的非法项后排序返回。
    return tuple(sorted(value for value in values if value >= model.num_experts_per_tok))


def _default_max_live_bucket_candidates(
    config: TrainingProjectConfig,
) -> tuple[int, ...]:
    # 候选上界至少保留 4，给高显存设备留出更激进的并发搜索空间。
    upper = max(4, config.bucket_schedule.max_live_buckets)
    # 构造从 1 到上界的连续候选区间。
    return tuple(range(1, upper + 1))


def _default_prefetch_bucket_candidates(
    config: TrainingProjectConfig,
) -> tuple[int, ...]:
    # 预取 bucket 上界至少保留 2，避免默认搜索过于保守。
    upper = max(2, config.bucket_schedule.prefetch_buckets)
    # prefetch 候选允许从 0 开始，表示完全不预取。
    return tuple(range(0, upper + 1))


def main(argv: Sequence[str] | None = None) -> int:
    # 先构建解析器并解析命令行参数。
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    # qwen35-plan 是不依赖通用配置装载的快捷路径。
    if args.command == "qwen35-plan":
        # 固定加载 Qwen3.5-35B-A3B profile。
        config = build_profile_config("qwen35-35b-a3b")
        # 基于该 profile 构造训练项目和蓝图。
        project = TrainingProject(config)
        blueprint = project.build_blueprint()
        # 按用户要求选择 JSON 或文本输出。
        if args.json:
            print(blueprint.to_json())
        else:
            print(blueprint.render_text())
        # 命令成功返回 0。
        return 0

    # predictor-inspect 也是独立于通用配置加载的 checkpoint 检查路径。
    if args.command == "predictor-inspect":
        # 直接读取 predictor checkpoint 元信息。
        metadata = PredictorTrainer.read_checkpoint_metadata(args.checkpoint)
        # 按用户要求选择 JSON 或摘要文本输出。
        if args.json:
            print(json.dumps(metadata.to_dict(), indent=2, sort_keys=True))
        else:
            print(
                f"checkpoint_kind={metadata.checkpoint_kind} "
                f"profile={metadata.profile_name} "
                f"teacher_source={metadata.teacher_source} "
                f"summary_source={metadata.summary_source}"
            )
            print(
                f"input_summary_dim={metadata.input_summary_dim} "
                f"hidden_dim={metadata.hidden_dim} "
                f"window={metadata.window_layers} "
                f"stride={metadata.stride_layers} "
                f"num_experts={metadata.num_experts}"
            )
            print(
                f"candidate={metadata.candidate_experts_per_layer} "
                f"executed={metadata.executed_experts_per_layer} "
                f"epochs={metadata.epochs} "
                f"final_loss={metadata.final_mean_loss:.6f} "
                f"recall@candidate={metadata.final_recall_at_candidate_budget:.4f}"
            )
        # 命令成功返回 0。
        return 0

    # predictor-export 同样先于通用配置加载执行。
    if args.command == "predictor-export":
        # 导出 predictor checkpoint 对应的部署 bundle。
        manifest = PredictorTrainer.export_checkpoint_bundle(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )
        # 按用户要求选择 JSON 或摘要文本输出。
        if args.json:
            print(manifest.to_json())
        else:
            print(
                f"Exported predictor deployment bundle for profile "
                f"{manifest.profile_name}."
            )
            print(
                f"teacher_source={manifest.teacher_source} "
                f"summary_source={manifest.summary_source} "
                f"weights={manifest.weights_file} "
                f"schema={manifest.schema_file} "
                f"metrics={manifest.metrics_file}"
            )
            print(f"bundle_dir={args.output_dir}")
        # 命令成功返回 0。
        return 0

    # 其余命令统一走标准训练配置装载。
    config = _load_config(args)

    # -----------------
    # 先处理 blueprint / predictor blueprint / trace 相关命令。
    if args.command == "plan":
        # 基于配置构造训练项目并生成默认蓝图。
        project = TrainingProject(config)
        blueprint = project.build_blueprint()
        # 按用户要求选择 JSON 或文本输出。
        if args.json:
            print(blueprint.to_json())
        else:
            print(blueprint.render_text())
        return 0

    if args.command == "predictor-plan":
        # 基于配置构造 predictor 蓝图。
        project = TrainingProject(config)
        blueprint = project.build_predictor_blueprint()
        # 按用户要求选择 JSON 或文本输出。
        if args.json:
            print(blueprint.to_json())
        else:
            print(blueprint.render_text())
        return 0

    if args.command == "predictor-trace":
        # 构造 predictor 训练器并生成 trace dataset。
        trainer = PredictorTrainer(config)
        dataset = trainer.build_trace_dataset(
            steps=args.steps,
            examples_per_step=args.examples_per_step,
            samples=args.samples,
            tokens_per_sample=args.tokens_per_sample,
            dataset_path=None if args.dataset is None else str(args.dataset),
            tokenizer_path=None if args.tokenizer is None else str(args.tokenizer),
            dataset_format=args.dataset_format,
            dataset_text_key=args.dataset_text_key,
        )
        # 若给了输出路径，则先把 trace dataset 落盘。
        if args.output is not None:
            dataset.write_json(args.output)
        # 按用户要求选择 JSON 或摘要文本输出。
        if args.json:
            print(dataset.to_json())
        else:
            print(
                f"Built {dataset.example_count} predictor trace example(s) for profile "
                f"{dataset.profile_name}."
            )
            print(
                f"teacher_source={dataset.teacher_source} "
                f"summary_source={dataset.summary_source} "
                f"window={dataset.window_layers} "
                f"candidate={dataset.candidate_experts_per_layer} "
                f"executed={dataset.executed_experts_per_layer}"
            )
            if args.output is not None:
                print(f"saved_trace={args.output}")
        return 0

    # -----------------
    # 处理 predictor 训练与评估命令。
    if args.command == "predictor-train":
        # 先构造 predictor 训练器。
        trainer = PredictorTrainer(config)
        # 预留可选的 checkpoint 初始化模型。
        model = None
        if args.init_from_checkpoint is not None:
            # 若显式提供 checkpoint，则先加载其模型权重。
            model, _ = trainer.load_checkpoint(args.init_from_checkpoint)
        if args.trace_input is not None:
            # 有现成 trace 输入时直接读取并训练。
            dataset = PredictorTraceDataset.from_json_file(args.trace_input)
            model, run_trace = trainer.fit_dataset(
                dataset,
                epochs=args.epochs,
                model=model,
            )
        else:
            # 否则先现场构造 trace dataset，再执行训练。
            dataset = trainer.build_trace_dataset(
                steps=args.steps,
                examples_per_step=args.examples_per_step,
                samples=args.samples,
                tokens_per_sample=args.tokens_per_sample,
                dataset_path=None if args.dataset is None else str(args.dataset),
                tokenizer_path=None if args.tokenizer is None else str(args.tokenizer),
                dataset_format=args.dataset_format,
                dataset_text_key=args.dataset_text_key,
            )
            model, run_trace = trainer.fit_dataset(
                dataset,
                epochs=args.epochs,
                model=model,
            )
        # 如有要求，则保存训练后的 checkpoint。
        if args.checkpoint_output is not None:
            trainer.save_checkpoint(
                model=model,
                run_trace=run_trace,
                path=args.checkpoint_output,
            )
        # 如有要求，则同步导出 runtime schema。
        if args.schema_output is not None:
            trainer.build_runtime_schema(
                teacher_source=run_trace.teacher_source,
                summary_source=run_trace.summary_source,
            ).write_json(args.schema_output)
        # 按用户要求选择 JSON 或摘要文本输出。
        if args.json:
            print(run_trace.to_json())
        else:
            print(
                f"Trained predictor for profile {run_trace.profile_name} on "
                f"{run_trace.example_count} trace example(s)."
            )
            print(
                f"teacher_source={run_trace.teacher_source} "
                f"summary_source={run_trace.summary_source} "
                f"epochs={run_trace.epochs} "
                f"final_loss={run_trace.final_mean_loss:.6f} "
                f"recall@candidate={run_trace.final_recall_at_candidate_budget:.4f}"
            )
            if args.checkpoint_output is not None:
                print(f"saved_checkpoint={args.checkpoint_output}")
            if args.schema_output is not None:
                print(f"saved_schema={args.schema_output}")
        return 0

    if args.command == "predictor-eval":
        # 先构造 predictor 训练器。
        trainer = PredictorTrainer(config)
        if args.trace_input is not None:
            # 有现成 trace 输入时直接读取。
            dataset = PredictorTraceDataset.from_json_file(args.trace_input)
        else:
            # 否则先构造评估用 trace dataset。
            dataset = trainer.build_trace_dataset(
                steps=args.steps,
                examples_per_step=args.examples_per_step,
                samples=args.samples,
                tokens_per_sample=args.tokens_per_sample,
                dataset_path=None if args.dataset is None else str(args.dataset),
                tokenizer_path=None if args.tokenizer is None else str(args.tokenizer),
                dataset_format=args.dataset_format,
                dataset_text_key=args.dataset_text_key,
            )
        # 使用指定 checkpoint 对 trace dataset 做评估。
        evaluation = trainer.evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            dataset=dataset,
        )
        # 按用户要求选择 JSON 或摘要文本输出。
        if args.json:
            print(evaluation.to_json())
        else:
            print(
                f"Evaluated predictor for profile {evaluation.profile_name} on "
                f"{evaluation.example_count} trace example(s)."
            )
            print(
                f"teacher_source={evaluation.teacher_source} "
                f"summary_source={evaluation.summary_source} "
                f"loss={evaluation.mean_loss:.6f} "
                f"recall@candidate={evaluation.recall_at_candidate_budget:.4f} "
                f"recall@executed={evaluation.recall_at_executed_budget:.4f}"
            )
        return 0

    # validate 只需确认配置可通过校验。
    if args.command == "validate":
        # 按用户要求输出完整配置 JSON 或简短确认信息。
        if args.json:
            print(config.to_json())
        else:
            print("Training project configuration is valid.")
        return 0

    # -----------------
    # 嵌套 helper：统一解析 resume checkpoint / runtime snapshot。
    def _load_checkpoint(
        path: Path | None,
    ) -> tuple[TrainingRuntimeSnapshot | None, BatchPlannerCheckpoint | None]:
        # 未提供路径时直接返回空快照。
        if path is None:
            return None, None
        # 读取快照 JSON 文本并解析。
        snapshot_payload = json.loads(path.read_text(encoding="utf-8"))
        # 顶层必须解析成对象。
        if not isinstance(snapshot_payload, dict):
            raise ValueError("runtime snapshot JSON must decode to an object")
        # 若是完整训练会话 checkpoint，则同时恢复 runtime_snapshot 和 planner。
        if snapshot_payload.get("checkpoint_kind") == "training_session_checkpoint":
            checkpoint = TrainingSessionCheckpoint.from_dict(snapshot_payload)
            return checkpoint.runtime_snapshot, checkpoint.planner
        # 否则按裸 runtime snapshot 处理。
        return TrainingRuntimeSnapshot.from_dict(snapshot_payload), None

    # -----------------
    # 处理 simulate / estimate-startup / train 三条训练基座主链命令。
    if args.command == "simulate":
        # 构造训练项目，并按需要恢复运行时快照。
        project = TrainingProject(config)
        snapshot, _ = _load_checkpoint(args.resume_from)
        # 基于快照构造训练引擎。
        engine = project.build_engine(snapshot)
        # 执行指定步数的代表性模拟。
        run_trace = engine.simulate(
            steps=args.steps,
            batch=BatchShape(
                samples=args.samples,
                tokens_per_sample=args.tokens_per_sample,
            ),
        )
        # 如有要求，则把模拟后的快照写回磁盘。
        if args.save_snapshot is not None:
            args.save_snapshot.write_text(
                engine.snapshot_state().to_json(),
                encoding="utf-8",
            )
        # 按用户要求选择 JSON 或摘要文本输出。
        if args.json:
            print(json.dumps(run_trace.to_dict(), indent=2, sort_keys=True))
        else:
            # 先抽取资源规划摘要，便于打印统一头部信息。
            resource = run_trace.resource_plan
            print(
                f"Simulated {run_trace.step_count} step(s) for profile "
                f"{run_trace.profile_name} with compute_device="
                f"{config.execution.compute_device} and batch "
                f"{run_trace.batch.samples}x{run_trace.batch.tokens_per_sample}."
            )
            print(
                "resource "
                f"gpu_hot={resource['gpu_hot']['resident_gib']:.2f}GiB/"
                f"{resource['gpu_hot']['available_gib']:.2f}GiB "
                f"cpu_hot={resource['cpu_hot']['resident_gib']:.2f}GiB/"
                f"{resource['cpu_hot']['available_gib']:.2f}GiB "
                f"nvme_cold={resource['nvme_cold']['resident_gib']:.2f}GiB/"
                f"{resource['nvme_cold']['available_gib']:.2f}GiB "
                f"fit={resource['all_tiers_within_budget']}"
            )
            for step in run_trace.steps:
                # warm_bucket_count 统计预取前已经有 CPU-hot shard 的 bucket 数。
                warm_bucket_count = sum(
                    trace.cpu_hot_shards_before_prefetch > 0
                    for trace in step.bucket_stream_traces
                )
                print(
                    f"step={step.step_index} active={list(step.active_expert_ids)} "
                    f"prefetch={list(step.prefetched_expert_ids)} "
                    f"buckets={len(step.layer_buckets)} actions={len(step.actions)} "
                    f"residency={len(step.residency_transitions)} "
                    f"warehouse={0 if step.warehouse_summary is None else step.warehouse_summary.total_shards} "
                    f"store={0 if step.parameter_store_summary is None else step.parameter_store_summary.tracked_shards} "
                    f"store_manifest={0 if step.parameter_store_summary is None else step.parameter_store_summary.manifest_backed_shards} "
                    f"dirty={0 if step.warehouse_summary is None else step.warehouse_summary.dirty_shards} "
                    f"transport={0 if step.transport_summary is None else step.transport_summary.matched_shards}/"
                    f"{0 if step.transport_summary is None else step.transport_summary.file_count}f "
                    f"io={0 if step.transport_execution_summary is None else step.transport_execution_summary.staged_file_count}s/"
                    f"{0 if step.transport_execution_summary is None else step.transport_execution_summary.reused_file_count}r/"
                    f"{0 if step.transport_execution_summary is None else step.transport_execution_summary.evicted_file_count}e "
                    f"warm={warm_bucket_count}/{len(step.bucket_stream_traces)} "
                    f"sources={0 if step.parameter_source_summary is None else step.parameter_source_summary.manifest_backed_shards}/"
                    f"{0 if step.parameter_source_summary is None else step.parameter_source_summary.synthetic_seeded_shards} "
                    f"src_cache={0 if step.parameter_source_summary is None else step.parameter_source_summary.transport_backed_shards} "
                    f"micro={len(step.scheduled_micro_batches)} "
                    f"prefetch={0 if step.parameter_prefetch_summary is None else step.parameter_prefetch_summary.transport_cache_prefetches}t/"
                    f"{0 if step.parameter_prefetch_summary is None else step.parameter_prefetch_summary.direct_manifest_prefetches}m/"
                    f"{0 if step.parameter_prefetch_summary is None else step.parameter_prefetch_summary.buffer_reuses}b/"
                    f"{0 if step.parameter_prefetch_summary is None else step.parameter_prefetch_summary.cpu_hot_reuses}h "
                    f"loads={0 if step.parameter_load_summary is None else step.parameter_load_summary.transport_cache_loads}t/"
                    f"{0 if step.parameter_load_summary is None else step.parameter_load_summary.direct_manifest_loads}m/"
                    f"{0 if step.parameter_load_summary is None else step.parameter_load_summary.buffer_reuses}b/"
                    f"{0 if step.parameter_load_summary is None else step.parameter_load_summary.cpu_hot_reuses}h "
                    f"overlap={0.0 if step.stream_overlap_summary is None else step.stream_overlap_summary.overlap_ratio:.3f} "
                    f"lag={0 if step.stream_overlap_summary is None else step.stream_overlap_summary.max_update_lag_us}us "
                    f"updates={len(step.optimizer_updates)} "
                    f"optimizer={0 if step.optimizer_summary is None else step.optimizer_summary.tracked_shards} "
                    f"loss={0.0 if step.execution_summary is None else step.execution_summary.total_loss:.6f}"
                )
        return 0

    if args.command == "estimate-startup":
        # 构造启动估算用 batch 形状。
        batch = BatchShape(
            samples=args.samples,
            tokens_per_sample=args.tokens_per_sample,
        )
        # 解析 GPU 热预算候选列表。
        gpu_budget_candidates = _parse_csv_numbers(
            args.gpu_budgets_gb,
            cast=float,
        )
        # 未显式给候选时，退回当前配置中的默认预算。
        if not gpu_budget_candidates:
            gpu_budget_candidates = (config.memory_budget.gpu_hot_budget_gb,)
        # 解析 active expert 候选列表。
        active_expert_candidates = _parse_csv_numbers(
            args.active_expert_candidates,
            cast=int,
        )
        # 未显式给候选时，使用自动构造的默认候选。
        if not active_expert_candidates:
            active_expert_candidates = _default_active_expert_candidates(config)
        # 解析 max live bucket 候选列表。
        max_live_bucket_candidates = _parse_csv_numbers(
            args.max_live_bucket_candidates,
            cast=int,
        )
        # 未显式给候选时，使用自动构造的默认候选。
        if not max_live_bucket_candidates:
            max_live_bucket_candidates = _default_max_live_bucket_candidates(config)
        # 解析 prefetch bucket 候选列表。
        prefetch_bucket_candidates = _parse_csv_numbers(
            args.prefetch_bucket_candidates,
            cast=int,
        )
        # 未显式给候选时，使用自动构造的默认候选。
        if not prefetch_bucket_candidates:
            prefetch_bucket_candidates = _default_prefetch_bucket_candidates(config)

        # 调用启动估算器，对所有候选组合做搜索。
        estimates = TrainingStartupEstimator(config).estimate(
            batch=batch,
            gpu_hot_budget_candidates_gb=tuple(sorted(set(gpu_budget_candidates))),
            active_expert_candidates=tuple(sorted(set(active_expert_candidates))),
            max_live_bucket_candidates=tuple(sorted(set(max_live_bucket_candidates))),
            prefetch_bucket_candidates=tuple(sorted(set(prefetch_bucket_candidates))),
        )
        # 组装统一输出载荷。
        payload = {
            "profile_name": config.profile_name,
            "batch": batch.to_dict(),
            "gpu_budget_candidates_gb": list(gpu_budget_candidates),
            "active_expert_candidates": list(active_expert_candidates),
            "max_live_bucket_candidates": list(max_live_bucket_candidates),
            "prefetch_bucket_candidates": list(prefetch_bucket_candidates),
            "estimates": [estimate.to_dict() for estimate in estimates],
        }
        # 按用户要求选择 JSON 或摘要文本输出。
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                f"Estimated startup parameters for profile {config.profile_name} "
                f"and batch {batch.samples}x{batch.tokens_per_sample}."
            )
            for estimate in estimates:
                print(
                    "gpu_budget="
                    f"{estimate.gpu_hot_budget_gb:.2f}GiB "
                    f"available={estimate.gpu_hot_available_gib:.2f}GiB "
                    f"active_experts={estimate.active_experts_per_step} "
                    f"max_live_buckets={estimate.max_live_buckets} "
                    f"prefetch_buckets={estimate.prefetch_buckets} "
                    f"gpu_hot={estimate.planned_gpu_hot_gib:.2f}GiB "
                    f"fill={estimate.gpu_fill_ratio:.3f} "
                    f"fit={estimate.fits_within_budget}"
                )
        return 0

    if args.command == "train":
        # 构造训练项目，并按需要恢复运行时快照 / planner checkpoint。
        project = TrainingProject(config)
        snapshot, planner_checkpoint = _load_checkpoint(args.resume_from)
        # 执行多步训练会话。
        session_trace = project.train(
            steps=args.steps,
            samples=args.samples,
            tokens_per_sample=args.tokens_per_sample,
            snapshot=snapshot,
            checkpoint_dir=None if args.checkpoint_dir is None else str(args.checkpoint_dir),
            checkpoint_interval=args.checkpoint_interval,
            dataset_path=None if args.dataset is None else str(args.dataset),
            tokenizer_path=None if args.tokenizer is None else str(args.tokenizer),
            dataset_format=args.dataset_format,
            dataset_text_key=args.dataset_text_key,
            planner_checkpoint=planner_checkpoint,
            retain_step_traces=args.json,
        )
        # 按用户要求选择 JSON 或摘要文本输出。
        if args.json:
            print(json.dumps(session_trace.to_dict(), indent=2, sort_keys=True))
        else:
            # 根据是否传入数据集，标记当前 batch 来源。
            batch_source = "tokenizer-backed" if args.dataset is not None else "synthetic"
            print(
                f"Trained {session_trace.total_steps} {batch_source} step(s) for profile "
                f"{session_trace.profile_name} with compute_device="
                f"{config.execution.compute_device}."
            )
            print(
                f"loss avg={session_trace.average_loss:.6f} "
                f"max={session_trace.max_loss:.6f} "
                f"peak_activation_bytes={session_trace.peak_activation_bytes}"
            )
            if session_trace.checkpoint_paths:
                print(
                    "checkpoints "
                    + ", ".join(session_trace.checkpoint_paths)
                )
        return 0

    # 理论上不该走到这里；若进入则让 argparse 给出标准错误。
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    # 作为脚本执行时，直接把 main() 返回码抛给系统。
    raise SystemExit(main())
