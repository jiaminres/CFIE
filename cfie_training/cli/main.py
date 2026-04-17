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
    # ------------------------------- 创建 CLI 顶层解析器与子命令容器 -------------------------------
    # 创建训练项目的命令行参数解析器，并设置程序名与整体说明信息。
    parser = argparse.ArgumentParser(
        prog="cfie-train",
        description="Standalone training scaffold for CFIE.",
    )
    # 创建子命令解析器容器，用于挂载不同功能模块对应的命令。
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------- 注册 predictor trace、训练、检查、评估与导出命令 -------------------------------
    # 注册 predictor 教师轨迹采样命令，用于构造 predictor 训练所需的教师轨迹样本。
    predictor_trace_parser = subparsers.add_parser(
        "predictor-trace",
        help="Build teacher-trace samples for predictor training.",
    )
    # 添加配置文件路径参数，用于从外部 JSON 文件加载 predictor trace 配置。
    predictor_trace_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    # 添加训练档位参数，在未提供配置文件时选择内置训练档位。
    predictor_trace_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )
    # 添加规划步数参数，用于指定构造轨迹样本时使用多少个 planning step。
    predictor_trace_parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of planning steps used to build trace examples.",
    )
    # 添加每步样本数覆盖参数，用于显式指定每个 step 采集多少条轨迹样本。
    predictor_trace_parser.add_argument(
        "--examples-per-step",
        type=int,
        default=None,
        help="Optional override for the number of trace examples per step.",
    )
    # 添加每个轨迹采集步骤的基础样本数参数。
    predictor_trace_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Base samples per trace-capture step.",
    )
    # 添加每个样本的 token 数参数，用于控制轨迹采集时的样本长度。
    predictor_trace_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per sample used during trace capture.",
    )
    # 添加数据集路径参数，用于从文本文件或 JSONL 文件构造 token 化后的轨迹批次。
    predictor_trace_parser.add_argument(
        "--dataset",
        type=Path,
        help="Text or JSONL dataset file used to build tokenized trace batches.",
    )
    # 添加 tokenizer 路径参数，未指定时默认使用训练档位对应的模型路径。
    predictor_trace_parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer path. Defaults to the training profile model path.",
    )
    # 添加数据集格式参数，用于声明数据集文件格式或启用自动识别。
    predictor_trace_parser.add_argument(
        "--dataset-format",
        choices=("auto", "text", "jsonl"),
        default="auto",
        help="Dataset file format. Defaults to auto-detect from extension.",
    )
    # 添加 JSONL 文本字段名参数，用于指定从 JSONL 的哪个字段中读取文本。
    predictor_trace_parser.add_argument(
        "--dataset-text-key",
        default="text",
        help="JSONL text field to read when --dataset-format jsonl is used.",
    )
    # 添加输出路径参数，用于将采集到的 predictor trace 数据集写出为 JSON 文件。
    predictor_trace_parser.add_argument(
        "--output",
        type=Path,
        help="Optional path that receives the captured predictor trace dataset as JSON.",
    )
    # 添加 JSON 输出开关，用于控制是否以 JSON 形式打印 predictor trace 数据集。
    predictor_trace_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor trace dataset as JSON.",
    )

    # 注册 predictor 训练命令，用于基于教师轨迹训练受限 predictor 模型。
    predictor_train_parser = subparsers.add_parser(
        "predictor-train",
        help="Train the bounded predictor model on teacher traces.",
    )
    # 添加配置文件路径参数，用于从外部 JSON 文件加载 predictor 训练配置。
    predictor_train_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    # 添加训练档位参数，在未提供配置文件时选择内置训练档位。
    predictor_train_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )
    # 添加规划步数参数，用于指定构造教师轨迹时使用的 planning step 数量。
    predictor_train_parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of planning steps used to build teacher traces.",
    )
    # 添加每步样本数覆盖参数，用于显式指定每个 step 采集多少条轨迹样本。
    predictor_train_parser.add_argument(
        "--examples-per-step",
        type=int,
        default=None,
        help="Optional override for the number of trace examples per step.",
    )
    # 添加每个轨迹采集步骤的基础样本数参数。
    predictor_train_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Base samples per trace-capture step.",
    )
    # 添加每个样本的 token 数参数，用于控制轨迹采集长度。
    predictor_train_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per sample used during trace capture.",
    )
    # 添加数据集路径参数，用于从文本文件或 JSONL 文件构造 token 化训练批次。
    predictor_train_parser.add_argument(
        "--dataset",
        type=Path,
        help="Text or JSONL dataset file used to build tokenized trace batches.",
    )
    # 添加 tokenizer 路径参数，未指定时默认使用训练档位对应的模型路径。
    predictor_train_parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer path. Defaults to the training profile model path.",
    )
    # 添加数据集格式参数，用于声明数据集格式或启用自动识别。
    predictor_train_parser.add_argument(
        "--dataset-format",
        choices=("auto", "text", "jsonl"),
        default="auto",
        help="Dataset file format. Defaults to auto-detect from extension.",
    )
    # 添加 JSONL 文本字段名参数，用于指定从 JSONL 的哪个字段中读取训练文本。
    predictor_train_parser.add_argument(
        "--dataset-text-key",
        default="text",
        help="JSONL text field to read when --dataset-format jsonl is used.",
    )
    # 添加轨迹输入文件参数，用于直接复用已有 predictor trace 数据，而不是重新采集。
    predictor_train_parser.add_argument(
        "--trace-input",
        type=Path,
        help="Optional predictor trace dataset JSON used instead of recapturing traces.",
    )
    # 创建互斥参数组，保证初始化 checkpoint 与恢复训练 checkpoint 只能二选一。
    predictor_train_checkpoint_group = (
        predictor_train_parser.add_mutually_exclusive_group()
    )
    # 添加初始化 checkpoint 参数，仅用于加载模型初始权重，不恢复优化器状态。
    predictor_train_checkpoint_group.add_argument(
        "--init-from-checkpoint",
        type=Path,
        help="Optional predictor checkpoint used only to initialize model weights before training.",
    )
    # 添加恢复训练 checkpoint 参数，用于同时恢复模型状态与优化器状态以继续训练。
    predictor_train_checkpoint_group.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Optional predictor checkpoint used to resume model and optimizer state for continued training.",
    )
    # 添加训练后 checkpoint 输出路径参数，用于保存训练完成后的 predictor checkpoint。
    predictor_train_parser.add_argument(
        "--checkpoint-output",
        type=Path,
        help="Optional output path that receives the trained predictor checkpoint.",
    )
    # 添加运行时 schema 输出路径参数，用于保存 predictor 运行时结构定义 JSON。
    predictor_train_parser.add_argument(
        "--schema-output",
        type=Path,
        help="Optional output path that receives the predictor runtime schema JSON.",
    )
    # 添加训练轮数覆盖参数，用于覆盖默认训练 epoch 数。
    predictor_train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for training epochs.",
    )
    # 添加 JSON 输出开关，用于控制是否输出 predictor 训练过程轨迹。
    predictor_train_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor training run trace as JSON.",
    )

    # 注册 predictor checkpoint 元信息检查命令，用于查看 checkpoint 中保存的基础元数据。
    predictor_inspect_parser = subparsers.add_parser(
        "predictor-inspect",
        help="Inspect predictor checkpoint metadata.",
    )
    # 添加 checkpoint 路径参数，该参数必填，用于指定需要检查的 predictor checkpoint。
    predictor_inspect_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Predictor checkpoint to inspect.",
    )
    # 添加 JSON 输出开关，用于控制是否以 JSON 形式输出 checkpoint 元数据。
    predictor_inspect_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor checkpoint metadata as JSON.",
    )

    # 注册 predictor 评估命令，用于在教师轨迹上评估指定 predictor checkpoint。
    predictor_eval_parser = subparsers.add_parser(
        "predictor-eval",
        help="Evaluate a predictor checkpoint on teacher traces.",
    )
    # 添加配置文件路径参数，用于从外部 JSON 文件加载评估配置。
    predictor_eval_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    # 添加训练档位参数，在未提供配置文件时选择内置训练档位。
    predictor_eval_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )
    # 添加 checkpoint 路径参数，该参数必填，用于指定待评估的 predictor checkpoint。
    predictor_eval_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Predictor checkpoint to evaluate.",
    )
    # 添加轨迹输入文件参数，用于直接复用已有 predictor trace 数据进行评估。
    predictor_eval_parser.add_argument(
        "--trace-input",
        type=Path,
        help="Optional predictor trace dataset JSON used instead of recapturing traces.",
    )
    # 添加规划步数参数，用于指定构造评估轨迹时使用的 planning step 数量。
    predictor_eval_parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of planning steps used to build evaluation traces.",
    )
    # 添加每步样本数覆盖参数，用于显式指定每个评估 step 生成多少条轨迹样本。
    predictor_eval_parser.add_argument(
        "--examples-per-step",
        type=int,
        default=None,
        help="Optional override for the number of evaluation trace examples per step.",
    )
    # 添加每个评估轨迹采集步骤的基础样本数参数。
    predictor_eval_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Base samples per evaluation trace-capture step.",
    )
    # 添加每个评估样本的 token 数参数，用于控制评估轨迹长度。
    predictor_eval_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per sample used during evaluation trace capture.",
    )
    # 添加数据集路径参数，用于从文本文件或 JSONL 文件构造 token 化评估批次。
    predictor_eval_parser.add_argument(
        "--dataset",
        type=Path,
        help="Text or JSONL dataset file used to build tokenized evaluation batches.",
    )
    # 添加 tokenizer 路径参数，未指定时默认使用训练档位对应的模型路径。
    predictor_eval_parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer path. Defaults to the training profile model path.",
    )
    # 添加数据集格式参数，用于声明数据集格式或启用自动识别。
    predictor_eval_parser.add_argument(
        "--dataset-format",
        choices=("auto", "text", "jsonl"),
        default="auto",
        help="Dataset file format. Defaults to auto-detect from extension.",
    )
    # 添加 JSONL 文本字段名参数，用于指定从 JSONL 的哪个字段中读取评估文本。
    predictor_eval_parser.add_argument(
        "--dataset-text-key",
        default="text",
        help="JSONL text field to read when --dataset-format jsonl is used.",
    )
    # 添加 JSON 输出开关，用于控制是否输出 predictor 评估轨迹。
    predictor_eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor evaluation trace as JSON.",
    )

    # 注册 predictor bundle 导出命令，用于将 predictor checkpoint 导出为部署包。
    predictor_export_parser = subparsers.add_parser(
        "predictor-export",
        help="Export a predictor checkpoint as a deployment bundle.",
    )
    # 添加 checkpoint 路径参数，该参数必填，用于指定待导出的 predictor checkpoint。
    predictor_export_parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Predictor checkpoint to export.",
    )
    # 添加输出目录参数，该参数必填，用于指定导出部署包的目标目录。
    predictor_export_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory that receives the exported deployment bundle.",
    )
    # 添加 JSON 输出开关，用于控制是否输出 predictor 部署清单。
    predictor_export_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the predictor deployment manifest as JSON.",
    )

    # ------------------------------- 注册训练基座校验、模拟、启动估算与训练命令 -------------------------------
    # 注册配置校验命令，用于检查训练项目配置是否合法。
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate the training project configuration.",
    )
    # 添加训练档位参数，在未提供配置文件时使用内置训练档位进行校验。
    validate_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to use when --config is not provided.",
    )
    # 添加配置文件路径参数，用于指定待校验的 JSON 配置文件。
    validate_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file. Defaults to the built-in scaffold config.",
    )
    # 添加 JSON 输出开关，用于控制是否以 JSON 形式输出校验后的配置。
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the validated config as JSON.",
    )

    # 注册单步模拟命令，用于运行第一版训练引擎并输出 step 级跟踪信息。
    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Run the first-version training engine and emit step traces.",
    )
    # 添加配置文件路径参数，用于从外部 JSON 文件加载模拟配置。
    simulate_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    # 添加训练档位参数，用于指定模拟运行采用的训练档位。
    simulate_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to simulate.",
    )
    # 添加模拟步数参数，用于指定连续模拟多少个训练 step。
    simulate_parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of sequential training steps to simulate.",
    )
    # 添加每步样本数参数，用于指定模拟 batch 中的样本数量。
    simulate_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Samples per simulated step.",
    )
    # 添加每个样本的 token 数参数，用于控制模拟 batch 的序列长度。
    simulate_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=512,
        help="Tokens per sample in the simulated batch.",
    )
    # 添加 JSON 输出开关，用于控制是否以 JSON 形式输出模拟运行轨迹。
    simulate_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the simulated run trace as JSON.",
    )
    # 添加恢复快照路径参数，用于从已有运行时快照中恢复模拟执行。
    simulate_parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional runtime snapshot JSON used to resume a simulated run.",
    )
    # 添加快照保存路径参数，用于在模拟结束后输出新的运行时快照。
    simulate_parser.add_argument(
        "--save-snapshot",
        type=Path,
        help="Optional path that receives the runtime snapshot after simulation.",
    )

    # 注册启动参数估算命令，用于估算不同 GPU 热预算下的启动参数组合。
    estimate_startup_parser = subparsers.add_parser(
        "estimate-startup",
        help="Estimate startup parameters for different GPU hot budgets.",
    )
    # 添加配置文件路径参数，用于从外部 JSON 文件加载启动估算配置。
    estimate_startup_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    # 添加训练档位参数，在未提供配置文件时使用内置训练档位执行估算。
    estimate_startup_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to estimate when --config is not provided.",
    )
    # 添加样本数参数，用于指定启动估算 batch 的样本数量。
    estimate_startup_parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Samples per startup-estimate batch.",
    )
    # 添加每个样本的 token 数参数，用于控制估算 batch 的长度。
    estimate_startup_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per sample in the estimated startup batch.",
    )
    # 添加 GPU 热预算候选参数，用逗号分隔多个 GiB 候选值。
    estimate_startup_parser.add_argument(
        "--gpu-budgets-gb",
        default="",
        help="Comma-separated GPU hot budget candidates in GiB.",
    )
    # 添加每步激活专家数候选参数，用逗号分隔多个 active_experts_per_step 候选值。
    estimate_startup_parser.add_argument(
        "--active-expert-candidates",
        default="",
        help="Comma-separated active_experts_per_step candidates.",
    )
    # 添加最大存活 bucket 数候选参数，用逗号分隔多个 max_live_buckets 候选值。
    estimate_startup_parser.add_argument(
        "--max-live-bucket-candidates",
        default="",
        help="Comma-separated max_live_buckets candidates.",
    )
    # 添加预取 bucket 数候选参数，用逗号分隔多个 prefetch_buckets 候选值。
    estimate_startup_parser.add_argument(
        "--prefetch-bucket-candidates",
        default="",
        help="Comma-separated prefetch_buckets candidates.",
    )
    # 添加 JSON 输出开关，用于控制是否以 JSON 形式输出启动估算结果。
    estimate_startup_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the startup estimates as JSON.",
    )

    # 注册多步训练会话命令，用于执行完整训练过程，可使用合成数据或真实数据集。
    train_parser = subparsers.add_parser(
        "train",
        help="Run a multi-step training session with dataset-backed batches.",
    )
    # 添加配置文件路径参数，用于从外部 JSON 文件加载训练配置。
    train_parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON config file.",
    )
    # 添加训练档位参数，用于指定训练运行时采用的训练档位。
    train_parser.add_argument(
        "--profile",
        choices=SUPPORTED_TRAINING_PROFILES,
        default=DEFAULT_TRAINING_PROFILE,
        help="Training profile to run.",
    )
    # 添加训练步数参数，用于指定本次训练会话实际执行多少个 step。
    train_parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of training steps to execute.",
    )
    # 添加基础样本数参数，用于合成 batch 时控制每个 batch 的样本数。
    train_parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Base samples per training batch.",
    )
    # 添加每个训练样本的 token 数参数，用于控制序列长度。
    train_parser.add_argument(
        "--tokens-per-sample",
        type=int,
        default=256,
        help="Tokens per training sample.",
    )
    # 添加恢复快照路径参数，用于从已有运行时快照恢复训练。
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help="Optional runtime snapshot JSON used to resume training.",
    )
    # 添加 checkpoint 目录参数，用于保存周期性的运行时快照。
    train_parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Optional directory for periodic runtime snapshots.",
    )
    # 添加 checkpoint 间隔参数，用于指定每隔多少个 step 保存一次运行时快照。
    train_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save a runtime snapshot every N steps. Disabled at 0.",
    )
    # 添加数据集路径参数，用于从文本文件或 JSONL 文件构造基于 tokenizer 的训练 batch。
    train_parser.add_argument(
        "--dataset",
        type=Path,
        help="Text or JSONL dataset file used to build tokenizer-backed batches.",
    )
    # 添加 tokenizer 路径参数，未指定时默认使用训练档位对应的模型路径。
    train_parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional tokenizer path. Defaults to the training profile model path.",
    )
    # 添加数据集格式参数，用于声明训练数据文件格式或启用自动识别。
    train_parser.add_argument(
        "--dataset-format",
        choices=("auto", "text", "jsonl"),
        default="auto",
        help="Dataset file format. Defaults to auto-detect from extension.",
    )
    # 添加 JSONL 文本字段名参数，用于指定从 JSONL 的哪个字段中读取训练文本。
    train_parser.add_argument(
        "--dataset-text-key",
        default="text",
        help="JSONL text field to read when --dataset-format jsonl is used.",
    )
    # 添加 JSON 输出开关，用于控制是否输出训练会话轨迹。
    train_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the session trace as JSON.",
    )

    # ------------------------------- 返回构造完成的命令行解析器 -------------------------------
    # 返回已经注册好全部子命令与参数的 CLI 解析器对象。
    return parser


def _load_config(args: argparse.Namespace) -> TrainingProjectConfig:
    # ------------------------------- 按优先级加载训练项目配置 -------------------------------
    # 当命令行参数中显式提供配置文件路径时，优先从该 JSON 文件加载训练配置。
    if getattr(args, "config", None):
        # 从外部 JSON 配置文件中反序列化训练项目配置对象。
        config = TrainingProjectConfig.from_json_file(args.config)
    else:
        # 当未显式提供配置文件时，从命令行参数中读取训练档位名称。
        profile_name = getattr(args, "profile", DEFAULT_TRAINING_PROFILE)
        # 基于训练档位名称构造内置训练配置对象。
        config = build_profile_config(profile_name)

    # ------------------------------- 处理命令行对 development model 的显式覆盖 -------------------------------
    # 当命令行参数中显式提供开发阶段模型名称时，用该值覆盖配置中的 development_model。
    if getattr(args, "model", None):
        # 将命令行指定的开发阶段模型名称写回配置对象。
        config.model_targets.development_model = args.model

    # ------------------------------- 处理命令行对 target model 的显式覆盖 -------------------------------
    # 当命令行参数中显式提供目标模型名称时，用该值覆盖配置中的 target_model。
    if getattr(args, "target_model", None):
        # 将命令行指定的目标模型名称写回配置对象。
        config.model_targets.target_model = args.target_model

    # ------------------------------- 对最终配置执行统一校验并返回 -------------------------------
    # 对组合完成的训练配置执行合法性校验，并返回校验后的配置对象。
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
    # ------------------------------- 构建命令行解析器并解析输入参数 -------------------------------
    # 构建当前程序使用的命令行参数解析器。
    parser = build_parser()
    # 解析外部传入的命令行参数；当 argv 为空时，默认直接读取进程命令行参数。
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "predictor-trace" and args.dataset is None:
        parser.error("predictor-trace requires --dataset")
    if (
        args.command in {"predictor-train", "predictor-eval"}
        and args.trace_input is None
        and args.dataset is None
    ):
        parser.error(f"{args.command} requires --trace-input or --dataset")
    if (
        args.command == "train"
        and args.dataset is None
        and getattr(args, "resume_from", None) is None
    ):
        parser.error(
            "train requires --dataset unless --resume-from points to a "
            "dataset-backed session checkpoint"
        )

    # ------------------------------- 处理独立于通用配置加载的 predictor checkpoint 检查命令 -------------------------------
    # 当命令为 predictor-inspect 时，直接读取 checkpoint 元信息，而不走通用配置加载流程。
    if args.command == "predictor-inspect":
        # 读取指定 predictor checkpoint 的元信息。
        metadata = PredictorTrainer.read_checkpoint_metadata(args.checkpoint)

        # 根据用户是否指定 --json，选择 JSON 或摘要文本方式输出 checkpoint 元信息。
        if args.json:
            # 将 checkpoint 元信息转换为字典后，以格式化 JSON 方式输出。
            print(json.dumps(metadata.to_dict(), indent=2, sort_keys=True))
        else:
            # 输出 checkpoint 的基础身份信息与来源信息。
            print(
                f"checkpoint_kind={metadata.checkpoint_kind} "
                f"profile={metadata.profile_name} "
                f"teacher_source={metadata.teacher_source} "
                f"summary_source={metadata.summary_source}"
            )
            # 输出模型结构相关元信息。
            print(
                f"input_summary_dim={metadata.input_summary_dim} "
                f"hidden_dim={metadata.hidden_dim} "
                f"window={metadata.window_layers} "
                f"stride={metadata.stride_layers} "
                f"num_experts={metadata.num_experts}"
            )
            # 输出训练轮数、最终损失与召回指标等结果信息。
            print(
                f"candidate={metadata.candidate_experts_per_layer} "
                f"executed={metadata.executed_experts_per_layer} "
                f"epochs={metadata.epochs} "
                f"final_loss={metadata.final_mean_loss:.6f} "
                f"recall@candidate={metadata.final_recall_at_candidate_budget:.4f}"
            )

        # 当前命令执行成功，返回 0。
        return 0

    # ------------------------------- 处理独立于通用配置加载的 predictor 导出命令 -------------------------------
    # 当命令为 predictor-export 时，直接执行 checkpoint 导出逻辑，而不走通用配置加载流程。
    if args.command == "predictor-export":
        # 将指定 checkpoint 导出为部署 bundle，并返回导出清单。
        manifest = PredictorTrainer.export_checkpoint_bundle(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )

        # 根据用户是否指定 --json，选择 JSON 或摘要文本方式输出导出结果。
        if args.json:
            # 输出部署清单的 JSON 表示。
            print(manifest.to_json())
        else:
            # 输出导出成功信息与对应 profile 名称。
            print(
                f"Exported predictor deployment bundle for profile "
                f"{manifest.profile_name}."
            )
            # 输出 bundle 中各关键文件的路径与来源信息。
            print(
                f"teacher_source={manifest.teacher_source} "
                f"summary_source={manifest.summary_source} "
                f"weights={manifest.weights_file} "
                f"schema={manifest.schema_file} "
                f"metrics={manifest.metrics_file}"
            )
            # 输出最终导出的 bundle 目录。
            print(f"bundle_dir={args.output_dir}")

        # 当前命令执行成功，返回 0。
        return 0

    # ------------------------------- 加载其余命令共享的通用训练配置 -------------------------------
    # 对除独立命令外的其余命令，统一按标准方式加载训练配置。
    config = _load_config(args)

    # ------------------------------- 处理 predictor trace 数据集构造命令 -------------------------------
    # 当命令为 predictor-trace 时，构造教师轨迹数据集。
    if args.command == "predictor-trace":
        # 基于当前配置创建 predictor 训练器。
        trainer = PredictorTrainer(config)
        # 构造 predictor trace 数据集。
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

        # ------------------------------- 按需保存 predictor trace 数据集到磁盘 -------------------------------
        # 当用户显式指定输出路径时，将生成的数据集写入目标文件。
        if args.output is not None:
            # 将 predictor trace 数据集保存为 JSON 文件。
            dataset.write_json(args.output)

        # ------------------------------- 输出 predictor trace 数据集构造结果 -------------------------------
        # 根据用户是否指定 --json，选择 JSON 或摘要文本方式输出结果。
        if args.json:
            # 输出完整 predictor trace 数据集的 JSON 表示。
            print(dataset.to_json())
        else:
            # 输出已构造样本数量与 profile 信息。
            print(
                f"Built {dataset.example_count} predictor trace example(s) for profile "
                f"{dataset.profile_name}."
            )
            # 输出教师来源、摘要来源以及专家窗口等关键信息。
            print(
                f"teacher_source={dataset.teacher_source} "
                f"summary_source={dataset.summary_source} "
                f"window={dataset.window_layers} "
                f"candidate={dataset.candidate_experts_per_layer} "
                f"executed={dataset.executed_experts_per_layer}"
            )
            # 当存在输出文件路径时，额外打印保存位置。
            if args.output is not None:
                # 输出 trace 数据集保存路径。
                print(f"saved_trace={args.output}")

        # 当前命令执行成功，返回 0。
        return 0

    # ------------------------------- 处理 predictor 训练命令 -------------------------------
    # 当命令为 predictor-train 时，执行 predictor 模型训练流程。
    if args.command == "predictor-train":
        # 基于当前配置创建 predictor 训练器。
        trainer = PredictorTrainer(config)

        # ------------------------------- 初始化可选的模型状态、优化器状态与历史运行轨迹 -------------------------------
        # 默认先不加载任何已有模型。
        model = None
        # 默认先不加载任何已有优化器状态。
        optimizer_state_dict = None
        # 默认先不加载任何历史训练轨迹。
        initial_run_trace = None

        # ------------------------------- 按需加载初始化 checkpoint 或续训 checkpoint -------------------------------
        # 当用户提供初始化 checkpoint 时，只加载模型权重作为训练起点。
        if args.init_from_checkpoint is not None:
            # 从初始化 checkpoint 中恢复模型权重。
            model, _ = trainer.load_checkpoint(args.init_from_checkpoint)

        # 当用户提供续训 checkpoint 时，同时恢复模型、优化器与历史运行轨迹。
        if args.resume_checkpoint is not None:
            # 从续训 checkpoint 中恢复模型状态、训练轨迹与优化器状态。
            model, _, initial_run_trace, optimizer_state_dict = (
                trainer.load_training_checkpoint(args.resume_checkpoint)
            )

        # ------------------------------- 构造或加载 predictor 训练所需的 trace 数据集 -------------------------------
        # 当用户提供现成的 trace 输入文件时，直接从 JSON 文件读取。
        if args.trace_input is not None:
            # 从已有 JSON 文件中恢复 predictor trace 数据集。
            dataset = PredictorTraceDataset.from_json_file(args.trace_input)
            # 基于该数据集执行训练。
            model, run_trace, optimizer_state_dict = trainer.fit_dataset(
                dataset,
                epochs=args.epochs,
                model=model,
                optimizer_state_dict=optimizer_state_dict,
                initial_run_trace=initial_run_trace,
            )
        else:
            # 未提供现成 trace 输入时，现场构造训练所需的 predictor trace 数据集。
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
            # 基于新构造的数据集执行训练。
            model, run_trace, optimizer_state_dict = trainer.fit_dataset(
                dataset,
                epochs=args.epochs,
                model=model,
                optimizer_state_dict=optimizer_state_dict,
                initial_run_trace=initial_run_trace,
            )

        # ------------------------------- 按需导出训练后的 checkpoint 与 runtime schema -------------------------------
        # 当用户指定 checkpoint 输出路径时，保存训练后的模型与状态。
        if args.checkpoint_output is not None:
            # 将训练后的模型、轨迹与优化器状态保存为 checkpoint。
            trainer.save_checkpoint(
                model=model,
                run_trace=run_trace,
                path=args.checkpoint_output,
                optimizer_state_dict=optimizer_state_dict,
            )

        # 当用户指定 schema 输出路径时，导出运行时 schema。
        if args.schema_output is not None:
            # 基于训练轨迹中的来源信息构造运行时 schema，并写入目标文件。
            trainer.build_runtime_schema(
                teacher_source=run_trace.teacher_source,
                summary_source=run_trace.summary_source,
            ).write_json(args.schema_output)

        # ------------------------------- 输出 predictor 训练结果 -------------------------------
        # 根据用户是否指定 --json，选择 JSON 或摘要文本方式输出训练结果。
        if args.json:
            # 输出完整训练轨迹的 JSON 表示。
            print(run_trace.to_json())
        else:
            # 输出训练完成信息与样本规模。
            print(
                f"Trained predictor for profile {run_trace.profile_name} on "
                f"{run_trace.example_count} trace example(s)."
            )
            # 输出关键训练指标，如 epochs、最终损失与召回率。
            print(
                f"teacher_source={run_trace.teacher_source} "
                f"summary_source={run_trace.summary_source} "
                f"epochs={run_trace.epochs} "
                f"final_loss={run_trace.final_mean_loss:.6f} "
                f"recall@candidate={run_trace.final_recall_at_candidate_budget:.4f}"
            )
            # 当保存了 checkpoint 时，额外输出 checkpoint 路径。
            if args.checkpoint_output is not None:
                # 输出保存的 checkpoint 路径。
                print(f"saved_checkpoint={args.checkpoint_output}")
            # 当导出了 schema 时，额外输出 schema 路径。
            if args.schema_output is not None:
                # 输出保存的 schema 路径。
                print(f"saved_schema={args.schema_output}")

        # 当前命令执行成功，返回 0。
        return 0

    # ------------------------------- 处理 predictor 评估命令 -------------------------------
    # 当命令为 predictor-eval 时，执行 checkpoint 评估流程。
    if args.command == "predictor-eval":
        # 基于当前配置创建 predictor 训练器。
        trainer = PredictorTrainer(config)

        # ------------------------------- 构造或加载 predictor 评估所需的 trace 数据集 -------------------------------
        # 当用户提供现成的 trace 输入文件时，直接读取该数据集。
        if args.trace_input is not None:
            # 从已有 JSON 文件中恢复 predictor trace 数据集。
            dataset = PredictorTraceDataset.from_json_file(args.trace_input)
        else:
            # 未提供现成 trace 输入时，现场构造评估所需的 predictor trace 数据集。
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

        # ------------------------------- 使用指定 checkpoint 对 trace 数据集执行评估 -------------------------------
        # 基于给定 checkpoint 与评估数据集执行评估。
        evaluation = trainer.evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            dataset=dataset,
        )

        # ------------------------------- 输出 predictor 评估结果 -------------------------------
        # 根据用户是否指定 --json，选择 JSON 或摘要文本方式输出评估结果。
        if args.json:
            # 输出完整评估结果的 JSON 表示。
            print(evaluation.to_json())
        else:
            # 输出评估完成信息与样本规模。
            print(
                f"Evaluated predictor for profile {evaluation.profile_name} on "
                f"{evaluation.example_count} trace example(s)."
            )
            # 输出平均损失与不同预算下的召回率指标。
            print(
                f"teacher_source={evaluation.teacher_source} "
                f"summary_source={evaluation.summary_source} "
                f"loss={evaluation.mean_loss:.6f} "
                f"recall@candidate={evaluation.recall_at_candidate_budget:.4f} "
                f"recall@executed={evaluation.recall_at_executed_budget:.4f}"
            )

        # 当前命令执行成功，返回 0。
        return 0

    # ------------------------------- 处理训练配置校验命令 -------------------------------
    # 当命令为 validate 时，只需确认配置能够通过校验。
    if args.command == "validate":
        # 根据用户是否指定 --json，选择输出完整配置或简短确认信息。
        if args.json:
            # 输出校验通过后的配置 JSON。
            print(config.to_json())
        else:
            # 输出简短的配置合法提示。
            print("Training project configuration is valid.")

        # 当前命令执行成功，返回 0。
        return 0

    # ------------------------------- 定义内部辅助函数以统一加载恢复快照与会话 checkpoint -------------------------------
    # 定义内部函数，用于从磁盘路径加载运行时快照或完整训练会话 checkpoint。
    def _load_checkpoint(
        path: Path | None,
    ) -> tuple[TrainingRuntimeSnapshot | None, BatchPlannerCheckpoint | None]:
        # ------------------------------- 处理未提供恢复路径的情况 -------------------------------
        # 当未提供恢复路径时，直接返回空运行时快照与空 planner checkpoint。
        if path is None:
            # 返回空恢复状态。
            return None, None

        # ------------------------------- 读取并解析 checkpoint JSON 载荷 -------------------------------
        # 读取目标路径中的 JSON 文本并反序列化为 Python 对象。
        snapshot_payload = json.loads(path.read_text(encoding="utf-8"))

        # ------------------------------- 校验 checkpoint 顶层结构是否合法 -------------------------------
        # 要求解析后的顶层对象必须为字典，否则视为非法 checkpoint 格式。
        if not isinstance(snapshot_payload, dict):
            # 抛出格式错误，提示调用者当前 JSON 结构不符合预期。
            raise ValueError("runtime snapshot JSON must decode to an object")

        # ------------------------------- 识别并恢复完整训练会话 checkpoint -------------------------------
        # 当 checkpoint 标识为 training_session_checkpoint 时，同时恢复运行时快照与 planner 状态。
        if snapshot_payload.get("checkpoint_kind") == "training_session_checkpoint":
            # 从字典恢复完整训练会话 checkpoint 对象。
            checkpoint = TrainingSessionCheckpoint.from_dict(snapshot_payload)
            # 返回其中的运行时快照与 planner checkpoint。
            return checkpoint.runtime_snapshot, checkpoint.planner

        # ------------------------------- 识别并恢复裸 runtime snapshot -------------------------------
        # 若不是完整训练会话 checkpoint，则按裸运行时快照处理。
        return TrainingRuntimeSnapshot.from_dict(snapshot_payload), None

    # ------------------------------- 处理 simulate 命令 -------------------------------
    # 当命令为 simulate 时，执行第一版训练引擎的多步模拟。
    if args.command == "simulate":
        # 基于当前配置创建训练项目对象。
        project = TrainingProject(config)
        # 根据用户提供的恢复路径加载运行时快照。
        snapshot, _ = _load_checkpoint(args.resume_from)
        # 基于恢复后的快照创建训练引擎。
        engine = project.build_engine(snapshot)

        # ------------------------------- 构造模拟 batch 形状并执行模拟 -------------------------------
        # 按命令行参数构造模拟使用的 batch 形状。
        run_trace = engine.simulate(
            steps=args.steps,
            batch=BatchShape(
                samples=args.samples,
                tokens_per_sample=args.tokens_per_sample,
            ),
        )

        # ------------------------------- 按需保存模拟完成后的运行时快照 -------------------------------
        # 当用户指定保存路径时，将模拟结束后的快照写回磁盘。
        if args.save_snapshot is not None:
            # 将当前引擎状态序列化为 JSON 并写入指定文件。
            args.save_snapshot.write_text(
                engine.snapshot_state().to_json(),
                encoding="utf-8",
            )

        # ------------------------------- 输出 simulate 命令结果 -------------------------------
        # 根据用户是否指定 --json，选择 JSON 或摘要文本方式输出模拟结果。
        if args.json:
            # 输出完整模拟轨迹的 JSON 表示。
            print(json.dumps(run_trace.to_dict(), indent=2, sort_keys=True))
        else:
            # 从模拟轨迹中提取资源规划摘要，便于后续统一打印。
            resource = run_trace.resource_plan
            # 输出模拟整体概览信息，包括步数、profile、设备与 batch 形状。
            print(
                f"Simulated {run_trace.step_count} step(s) for profile "
                f"{run_trace.profile_name} with compute_device="
                f"{config.execution.compute_device} and batch "
                f"{run_trace.batch.samples}x{run_trace.batch.tokens_per_sample}."
            )
            # 输出各级存储资源占用与预算匹配情况。
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

            # ------------------------------- 逐步输出每个模拟 step 的关键执行摘要 -------------------------------
            # 遍历模拟轨迹中的每个 step，打印 step 级统计信息。
            for step in run_trace.steps:
                # 统计在预取发生之前，已有 CPU-hot shard 的 bucket 数量。
                warm_bucket_count = sum(
                    trace.cpu_hot_shards_before_prefetch > 0
                    for trace in step.bucket_stream_traces
                )
                # 输出当前 step 的激活专家、预取专家、bucket 数、传输、更新与损失等关键信息。
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

        # 当前命令执行成功，返回 0。
        return 0

    # ------------------------------- 处理 estimate-startup 命令 -------------------------------
    # 当命令为 estimate-startup 时，对不同候选组合进行启动参数估算。
    if args.command == "estimate-startup":
        # 按命令行参数构造启动估算使用的 batch 形状。
        batch = BatchShape(
            samples=args.samples,
            tokens_per_sample=args.tokens_per_sample,
        )

        # ------------------------------- 解析 GPU 热预算候选列表 -------------------------------
        # 将逗号分隔的 GPU 热预算字符串解析为浮点数元组。
        gpu_budget_candidates = _parse_csv_numbers(
            args.gpu_budgets_gb,
            cast=float,
        )
        # 当用户未显式提供 GPU 热预算候选时，回退到当前配置中的默认预算值。
        if not gpu_budget_candidates:
            # 使用配置中的 GPU 热预算作为唯一候选。
            gpu_budget_candidates = (config.memory_budget.gpu_hot_budget_gb,)

        # ------------------------------- 解析 active expert 候选列表 -------------------------------
        # 将逗号分隔的 active expert 候选字符串解析为整数元组。
        active_expert_candidates = _parse_csv_numbers(
            args.active_expert_candidates,
            cast=int,
        )
        # 当用户未显式提供 active expert 候选时，自动构造默认候选集合。
        if not active_expert_candidates:
            # 生成默认的 active expert 候选列表。
            active_expert_candidates = _default_active_expert_candidates(config)

        # ------------------------------- 解析 max live bucket 候选列表 -------------------------------
        # 将逗号分隔的 max live bucket 候选字符串解析为整数元组。
        max_live_bucket_candidates = _parse_csv_numbers(
            args.max_live_bucket_candidates,
            cast=int,
        )
        # 当用户未显式提供 max live bucket 候选时，自动构造默认候选集合。
        if not max_live_bucket_candidates:
            # 生成默认的 max live bucket 候选列表。
            max_live_bucket_candidates = _default_max_live_bucket_candidates(config)

        # ------------------------------- 解析 prefetch bucket 候选列表 -------------------------------
        # 将逗号分隔的 prefetch bucket 候选字符串解析为整数元组。
        prefetch_bucket_candidates = _parse_csv_numbers(
            args.prefetch_bucket_candidates,
            cast=int,
        )
        # 当用户未显式提供 prefetch bucket 候选时，自动构造默认候选集合。
        if not prefetch_bucket_candidates:
            # 生成默认的 prefetch bucket 候选列表。
            prefetch_bucket_candidates = _default_prefetch_bucket_candidates(config)

        # ------------------------------- 调用启动估算器搜索所有候选组合 -------------------------------
        # 对去重并排序后的候选组合执行启动参数估算。
        estimates = TrainingStartupEstimator(config).estimate(
            batch=batch,
            gpu_hot_budget_candidates_gb=tuple(sorted(set(gpu_budget_candidates))),
            active_expert_candidates=tuple(sorted(set(active_expert_candidates))),
            max_live_bucket_candidates=tuple(sorted(set(max_live_bucket_candidates))),
            prefetch_bucket_candidates=tuple(sorted(set(prefetch_bucket_candidates))),
        )

        # ------------------------------- 组装统一输出载荷 -------------------------------
        # 将估算结果与输入候选集合整理成统一的输出字典。
        payload = {
            "profile_name": config.profile_name,
            "batch": batch.to_dict(),
            "gpu_budget_candidates_gb": list(gpu_budget_candidates),
            "active_expert_candidates": list(active_expert_candidates),
            "max_live_bucket_candidates": list(max_live_bucket_candidates),
            "prefetch_bucket_candidates": list(prefetch_bucket_candidates),
            "estimates": [estimate.to_dict() for estimate in estimates],
        }

        # ------------------------------- 输出 estimate-startup 命令结果 -------------------------------
        # 根据用户是否指定 --json，选择 JSON 或摘要文本方式输出估算结果。
        if args.json:
            # 输出完整估算结果载荷的 JSON 表示。
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            # 输出估算任务的整体概览信息。
            print(
                f"Estimated startup parameters for profile {config.profile_name} "
                f"and batch {batch.samples}x{batch.tokens_per_sample}."
            )
            # 遍历每个估算结果，输出其关键组合与资源匹配情况。
            for estimate in estimates:
                # 输出单个候选组合的预算、填充率与是否满足预算等信息。
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

        # 当前命令执行成功，返回 0。
        return 0

    # ------------------------------- 处理 train 命令 -------------------------------
    # 当命令为 train 时，执行多步训练会话主流程。
    if args.command == "train":
        # 基于当前配置创建训练项目对象。
        project = TrainingProject(config)
        # 根据恢复路径加载运行时快照与 planner checkpoint。
        snapshot, planner_checkpoint = _load_checkpoint(args.resume_from)

        # ------------------------------- 调用训练项目执行多步训练会话 -------------------------------
        # 按命令行参数驱动训练项目执行训练，并返回训练会话轨迹。
        session_trace = project.train(
            steps=args.steps,
            samples=args.samples,
            tokens_per_sample=args.tokens_per_sample,
            checkpoint_dir=None if args.checkpoint_dir is None else str(args.checkpoint_dir),
            checkpoint_interval=args.checkpoint_interval,
            dataset_path=None if args.dataset is None else str(args.dataset),
            tokenizer_path=None if args.tokenizer is None else str(args.tokenizer),
            dataset_format=args.dataset_format,
            dataset_text_key=args.dataset_text_key,
            retain_step_traces=args.json,

            planner_checkpoint=planner_checkpoint,
            snapshot=snapshot,
        )

        # ------------------------------- 输出 train 命令结果 -------------------------------
        # 根据用户是否提供数据集参数，标记当前训练批次来源。
        if args.json:
            # 输出完整训练会话轨迹的 JSON 表示。
            print(json.dumps(session_trace.to_dict(), indent=2, sort_keys=True))
        else:
            # 当提供数据集路径时，说明当前 batch 为基于 tokenizer 的真实数据批次。
            batch_source = "dataset-backed"
            # 输出训练完成信息、步数、profile 与执行设备。
            print(
                f"Trained {session_trace.total_steps} {batch_source} step(s) for profile "
                f"{session_trace.profile_name} with compute_device="
                f"{config.execution.compute_device}."
            )
            # 输出平均损失、最大损失与峰值激活显存等核心训练指标。
            print(
                f"loss avg={session_trace.average_loss:.6f} "
                f"max={session_trace.max_loss:.6f} "
                f"peak_activation_bytes={session_trace.peak_activation_bytes}"
            )
            # 当训练过程中产生了 checkpoint 文件时，输出全部 checkpoint 路径。
            if session_trace.checkpoint_paths:
                # 输出训练过程中保存的 checkpoint 路径列表。
                print(
                    "checkpoints "
                    + ", ".join(session_trace.checkpoint_paths)
                )

        # 当前命令执行成功，返回 0。
        return 0

    # ------------------------------- 处理理论上不应到达的未知命令分支 -------------------------------
    # 若命令未被前面任何分支处理，则让 argparse 输出标准错误信息。
    parser.error(f"unsupported command: {args.command}")
    # 返回非零状态码，表示命令执行失败。
    return 2


if __name__ == "__main__":
    # 作为脚本执行时，直接把 main() 返回码抛给系统。
    raise SystemExit(main())
