"""CFIE training-base 初始化与训练命令行入口。"""

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


def main(argv: Sequence[str] | None = None) -> int:
    """解析命令行参数，执行对应子命令，并把结果以 JSON 形式输出。"""

    # ------------------------------- 构造并解析命令行参数 -------------------------------
    # 构建统一的 CLI 参数解析器，使所有子命令共享同一个入口解析流程。
    parser = build_parser()
    # 解析外部传入的 argv；为 None 时 argparse 会自动读取 sys.argv。
    args = parser.parse_args(argv)

    # ------------------------------- 调度子命令并输出结果 -------------------------------
    # 调用子命令注册到 args.func 的执行函数，得到可序列化的结果负载。
    payload = args.func(args)
    # 将执行结果打印为稳定排序的 UTF-8 JSON，便于脚本读取和人工排查。
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    # 返回 0 表示 CLI 命令正常完成。
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造 cfie-training-base 的顶层解析器与所有子命令解析器。"""

    # ------------------------------- 初始化顶层解析器 -------------------------------
    # 创建顶层 argparse 解析器，用于承载所有 training-base 子命令。
    parser = argparse.ArgumentParser(
        prog="cfie-training-base",
        description="Prepare CFIE large-model training-base stores.",
    )
    # 创建必选子命令集合，确保调用方必须显式选择一个执行路径。
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------- 注册容量估算子命令 -------------------------------
    # 注册 Qwen3.5 MoE 容量估算命令，仅根据模型维度和分片配置生成 dry-run 报告。
    estimate = subparsers.add_parser(
        "estimate-qwen35-moe",
        help="Dry-run capacity planning from Qwen3.5 MoE dimensions.",
    )

    # 为容量估算命令补充 Qwen3.5 MoE manifest 所需的模型维度参数。
    _add_qwen_manifest_args(estimate)

    # 为容量估算命令补充分片大小、块大小和分片名前缀参数。
    _add_manifest_shard_args(estimate)

    # 将该子命令绑定到容量估算执行函数，供 main 统一调度。
    estimate.set_defaults(func=_run_estimate_qwen35_moe)

    # ------------------------------- 注册 checkpoint 初始化子命令 -------------------------------
    # 注册 Qwen3.5 MoE checkpoint 导入命令，用于把 checkpoint 张量转换为训练存储。
    init = subparsers.add_parser(
        "init-qwen35-moe",
        help="Import Qwen3.5 MoE checkpoint tensors into training stores.",
    )

    # 指定输入 checkpoint 路径，后续会从该目录或文件读取模型张量。
    init.add_argument("--checkpoint", required=True, type=Path)

    # 指定训练存储根目录，后续 FP32、Adam 和 GPTQ manifest 都会写到该根目录下。
    init.add_argument("--root", required=True, type=Path)

    # 允许只生成导入计划和容量报告，不实际写入训练存储。
    init.add_argument("--dry-run", action="store_true")

    # 控制每读取多少个 checkpoint 张量打印一次进度；0 表示关闭进度日志。
    init.add_argument("--progress-every-tensors", type=int, default=0)

    # 为初始化命令补充 checkpoint key 映射、GPTQ 布局和专家过滤参数。
    _add_import_args(init)

    # 为初始化命令补充分片配置，保证 manifest 构建和存储写入使用同一组分片规则。
    _add_manifest_shard_args(init)

    # 将该子命令绑定到 checkpoint 初始化执行函数，供 main 统一调度。
    init.set_defaults(func=_run_init_qwen35_moe)

    # ------------------------------- 注册训练子命令 -------------------------------
    # 注册训练命令，用于在已经导入或可导入的训练存储上运行简化训练循环。
    train = subparsers.add_parser(
        "train",
        help="Run training loop on imported stores.",
    )
    # 指定训练存储根目录，训练循环会从该位置读取或写入参数状态。
    train.add_argument("--root", required=True, type=Path)

    # 指定源 checkpoint 路径，真实导入器会以此构造训练存储。
    train.add_argument("--checkpoint", required=True, type=Path)

    # 指定训练总步数，决定 dataloader 生成多少个 batch。
    train.add_argument("--steps", type=int, default=50)

    # 保留热窗口步数配置入口，供后续窗口提交策略扩展使用。
    train.add_argument("--window-steps", type=int, default=50)

    # 指定 AdamW 学习率，影响每步参数更新幅度。
    train.add_argument("--lr", type=float, default=0.01)

    # 指定训练 batch 大小 B，后续 input_ids 形状为 [B, T]。
    train.add_argument("--batch-size", type=int, default=2)

    # 指定序列长度 T，后续 input_ids 形状为 [B, T]。
    train.add_argument("--seq-len", type=int, default=8)

    # 指定参与训练的 MoE 层数，用于导入参数和构造训练模型。
    train.add_argument("--num-layers", type=int, default=2)

    # 指定每层专家数，用于导入专家参数和构造 hot 参数列表。
    train.add_argument("--num-experts", type=int, default=4)

    # 指定 hidden 维度 H，模型内部激活和权重会按该维度构造。
    train.add_argument("--hidden-size", type=int, default=3072)

    # 指定专家中间维度 I，MoE MLP 的 w13/w2 权重形状会依赖该值。
    train.add_argument("--intermediate-size", type=int, default=1024)

    # 指定当前训练存储 generation，用于区分不同版本的参数状态。
    train.add_argument("--generation", type=int, default=0)

    # 指定最多保留的梯度桶数量，限制训练循环中的待提交梯度缓存规模。
    train.add_argument("--grad-bucket-count", type=int, default=4)

    # 指定单个梯度桶容量 MiB，后续会换算为字节传入训练循环配置。
    train.add_argument("--grad-bucket-size-mib", type=int, default=512)

    # 为训练命令补充分片配置，使导入器和训练循环共享相同存储布局。
    _add_manifest_shard_args(train)

    # 将该子命令绑定到训练执行函数，供 main 统一调度。
    train.set_defaults(func=_run_train)

    # 返回完整解析器，交由 main 执行实际参数解析。
    return parser


def _run_estimate_qwen35_moe(args: argparse.Namespace) -> dict[str, Any]:
    """根据命令行参数估算 Qwen3.5 MoE 训练存储容量。"""

    # ------------------------------- 从命令行参数构造配置 -------------------------------
    # 将模型维度、层范围和专家过滤参数转换为 manifest 构建配置。
    qwen_config = _qwen_manifest_config_from_args(args)
    # 将分片大小、块大小和分片名前缀转换为统一分片配置。
    shard_config = _manifest_shard_config_from_args(args)

    # ------------------------------- 执行容量估算并组织返回结果 -------------------------------
    # 基于 Qwen3.5 MoE 维度和分片规则估算 FP32、Adam、GPTQ 等存储容量。
    report = estimate_qwen35_moe_capacity(qwen_config, shard_config)
    # 返回标准化 JSON 负载，便于 CLI 调用方保存和对比 dry-run 结果。
    return {
        "command": "estimate-qwen35-moe",
        "dry_run": True,
        "qwen_config": _dataclass_dict(qwen_config),
        "shard_config": _dataclass_dict(shard_config),
        "capacity": report.to_dict(),
    }


def _run_init_qwen35_moe(args: argparse.Namespace) -> dict[str, Any]:
    """导入 Qwen3.5 MoE checkpoint，并可选择写入训练基础存储。"""

    # ------------------------------- 初始化计时与导入配置 -------------------------------
    # 记录整个初始化流程起点，用于最终统计 total 耗时。
    total_start = time.perf_counter()
    # 从 CLI 参数生成 checkpoint key 映射、层过滤、专家过滤和 GPTQ 布局配置。
    import_config = _import_config_from_args(args)
    # 从 CLI 参数生成训练存储分片配置，后续 manifest 构建和写入都复用该配置。
    shard_config = _manifest_shard_config_from_args(args)
    # 创建 checkpoint 读取统计对象，用于记录读取和过滤后的张量数量与字节数。
    read_stats = CheckpointTensorLoadStats()
    # 创建 checkpoint 张量迭代器，并在读取阶段按 import_config 预过滤无关 key。
    checkpoint_tensors = iter_checkpoint_tensors(
        args.checkpoint,
        key_filter=qwen35_moe_checkpoint_key_filter(import_config),
        stats=read_stats,
    )
    # 当用户开启进度间隔时，用包装迭代器在 stderr 输出 checkpoint 读取进度。
    if args.progress_every_tensors > 0:
        # 包装原始张量迭代器，保持导入逻辑不变，仅额外输出进度事件。
        checkpoint_tensors = _iter_with_progress(
            checkpoint_tensors,
            read_stats,
            args.progress_every_tensors,
        )

    # ------------------------------- 导入 checkpoint 并构造 manifest -------------------------------
    # 记录 checkpoint 到导入计划转换的起点，用于单独统计导入耗时。
    import_start = time.perf_counter()
    # 将 checkpoint 张量解析为内部参数规格和导入计划，不在此处写入实际 store。
    plan = import_qwen35_moe_checkpoint(
        checkpoint_tensors,
        config=import_config,
    )
    # 计算 checkpoint 导入阶段耗时，便于定位慢点来自读取还是解析。
    import_seconds = time.perf_counter() - import_start

    # 记录 manifest 构建起点，用于单独统计容量布局规划耗时。
    manifest_start = time.perf_counter()
    # 根据导入计划中的参数规格构造训练基础 manifest，确定各参数的分片位置。
    manifest = TrainingBaseManifestBuilder(shard_config).build(plan.specs)
    # 根据 manifest 汇总容量报告，使 dry-run 和真实写入都能返回容量信息。
    report = capacity_report_from_manifest(manifest)
    # 计算 manifest 构建和容量统计阶段耗时。
    manifest_seconds = time.perf_counter() - manifest_start
    # 初始化各阶段耗时字典；store_write 会在真实写入后被更新。
    phase_seconds = {
        "checkpoint_import": import_seconds,
        "manifest_build": manifest_seconds,
        "store_write": 0.0,
        "total": time.perf_counter() - total_start,
    }

    # ------------------------------- 构造基础返回负载 -------------------------------
    # 汇总导入数量、跳过数量、读取统计、耗时和容量报告，作为 dry-run 与真实写入的共同返回结构。
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
    # dry-run 模式只返回导入计划和容量信息，不创建任何训练存储文件。
    if args.dry_run:
        # 在提前返回前刷新 total 耗时，保证 dry-run 统计覆盖完整流程。
        phase_seconds["total"] = time.perf_counter() - total_start
        # 返回 dry-run 结果，调用方可据此确认容量和导入范围。
        return payload

    # ------------------------------- 写入训练基础存储 -------------------------------
    # 记录 store 写入起点，用于单独统计磁盘写入阶段耗时。
    write_start = time.perf_counter()
    # 按导入计划初始化 FP32、Adam 和 GPTQ 训练存储，并写出对应 manifest。
    result = initialize_fp32_store_from_import_plan(
        plan=plan,
        root=args.root,
        manifest_config=shard_config,
        generation=args.generation,
    )
    # 更新 store 写入耗时，帮助区分 IO 写入成本和导入解析成本。
    phase_seconds["store_write"] = time.perf_counter() - write_start
    # 写入完成后刷新 total 耗时，确保总耗时覆盖全部真实初始化流程。
    phase_seconds["total"] = time.perf_counter() - total_start
    # 将实际产物路径和 generation 写入返回负载，便于后续训练命令定位 manifest。
    payload.update(
        {
            "fp32_manifest": str(result.fp32_store.manifest_path),
            "adam_manifest": str(result.adam_store.manifest_path),
            "gptq_manifest": str(result.gptq_store.manifest_path),
            "generation": args.generation,
        }
    )
    # 返回真实初始化结果，调用方可读取 manifest 路径继续训练或校验。
    return payload


def _iter_with_progress(
        tensors,
        stats: CheckpointTensorLoadStats,
        every: int,
):
    """包装 checkpoint 张量迭代器，并按固定张量间隔输出读取进度。"""

    # ------------------------------- 逐个转发张量并按间隔输出进度 -------------------------------
    # 遍历底层 checkpoint 张量流，保持 name 和 tensor 的原始顺序不变。
    for name, tensor in tensors:
        # 先把当前张量交给下游导入逻辑，避免进度打印改变消费语义。
        yield name, tensor
        # 非进度边界时跳过打印，降低 stderr 日志量和 JSON 序列化开销。
        if stats.yielded_tensor_count % every:
            # 继续消费下一个张量，直到累计数量到达 every 的整数倍。
            continue
        # 在进度边界输出结构化事件，便于外部日志系统解析 checkpoint 读取状态。
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
    """向子命令解析器追加 Qwen3.5 MoE manifest 维度参数。"""

    # ------------------------------- 注册模型维度与范围参数 -------------------------------
    # 指定模型层数 L，用于生成每层专家参数规格。
    parser.add_argument("--num-layers", required=True, type=int)
    # 指定每层专家数量 E，用于生成每层每个专家的参数规格。
    parser.add_argument("--num-experts", required=True, type=int)
    # 指定 hidden 维度 H，用于推导专家输入输出权重形状。
    parser.add_argument("--hidden-size", required=True, type=int)
    # 指定专家中间维度 I，用于推导 w13 和 w2 的权重形状。
    parser.add_argument("--intermediate-size", required=True, type=int)
    # 指定张量并行规模 TP，用于后续按 TP 规则估算参数切分容量。
    parser.add_argument("--tp-size", type=int, default=1)
    # 指定内部层编号起点，便于只构建某个层范围之后的 manifest。
    parser.add_argument("--layer-start", type=int, default=0)
    # 指定内部层名前缀，保证生成的参数 ID 与训练存储命名约定一致。
    parser.add_argument("--layer-prefix", default="layers")
    # 指定本地专家 ID 列表；空字符串表示不做专家过滤。
    parser.add_argument("--local-expert-ids", default="")
    # 允许把导入参数标记为不可训练，用于只读缓存或容量验证场景。
    parser.add_argument("--no-trainable", action="store_true")
    # 允许跳过 GPTQ cache 规格生成，用于只关心 FP32/Adam 存储的场景。
    parser.add_argument("--no-gptq-cache", action="store_true")


def _add_import_args(parser: argparse.ArgumentParser) -> None:
    """向初始化子命令追加 checkpoint 导入与命名映射参数。"""

    # ------------------------------- 注册 checkpoint 命名映射参数 -------------------------------
    # 指定 checkpoint 中层路径前缀，用于从原始 key 中识别层编号。
    parser.add_argument("--checkpoint-layer-prefix", default="layers")
    # 指定 checkpoint 中 MLP 模块名，用于定位专家权重所属路径段。
    parser.add_argument("--checkpoint-mlp-name", default="mlp")
    # 指定 checkpoint 中 experts 模块名，用于定位专家 ID 所属路径段。
    parser.add_argument("--checkpoint-experts-name", default="experts")
    # 指定内部训练存储层路径前缀，用于生成导入后的参数 ID。
    parser.add_argument("--internal-layer-prefix", default="layers")
    # 指定 gate_proj 名称，用于识别和重命名专家门控投影权重。
    parser.add_argument("--gate-proj-name", default="gate_proj")
    # 指定 up_proj 名称，用于识别和重命名专家上投影权重。
    parser.add_argument("--up-proj-name", default="up_proj")
    # 指定 down_proj 名称，用于识别和重命名专家下投影权重。
    parser.add_argument("--down-proj-name", default="down_proj")
    # 指定普通权重字段名，用于识别 FP32 或解码后的 dense 权重张量。
    parser.add_argument("--weight-name", default="weight")
    # 指定 GPTQ qweight 字段名，用于识别量化权重张量。
    parser.add_argument("--qweight-name", default="qweight")
    # 指定 GPTQ scales 字段名，用于识别量化缩放因子张量。
    parser.add_argument("--scales-name", default="scales")
    # 指定 GPTQ qzeros 字段名，用于识别量化零点张量。
    parser.add_argument("--qzeros-name", default="qzeros")
    # 指定 GPTQ g_idx 字段名，用于识别 group 到输入通道的映射张量。
    parser.add_argument("--g-idx-name", default="g_idx")
    # 指定 GPTQ 解码后权重布局，决定内部按 [K, N] 还是 [N, K] 解释权重。
    parser.add_argument(
        "--gptq-decoded-layout",
        choices=("k_n", "n_k"),
        default="n_k",
    )
    # 指定可剥离的 checkpoint 根前缀列表，用于兼容 model.xxx 这类保存格式。
    parser.add_argument("--known-root-prefixes", default="model.,")
    # 指定导入层编号起点，用于过滤起点之前的 checkpoint 层。
    parser.add_argument("--layer-start", type=int, default=0)
    # 指定导入层编号右开边界；None 表示不限制结束层。
    parser.add_argument("--layer-end-exclusive", type=int)
    # 指定本地专家 ID 列表；空字符串表示导入所有专家。
    parser.add_argument("--local-expert-ids", default="")
    # 允许把导入参数标记为不可训练，用于冻结或只读存储初始化。
    parser.add_argument("--no-trainable", action="store_true")
    # 允许跳过 GPTQ cache 导入，用于只初始化 FP32/Adam 训练状态。
    parser.add_argument("--no-gptq-cache", action="store_true")
    # 指定写入存储的 generation，后续训练可据此区分参数版本。
    parser.add_argument("--generation", type=int, default=0)


def _add_manifest_shard_args(parser: argparse.ArgumentParser) -> None:
    """向子命令解析器追加训练存储分片参数。"""

    # ------------------------------- 注册分片容量与命名参数 -------------------------------
    # 指定单个 FP32 参数分片最大字节数，用于控制主参数文件大小。
    parser.add_argument("--fp32-shard-bytes", type=int, default=1 << 30)

    # 指定单个 Adam 状态分片最大字节数，用于控制优化器状态文件大小。
    parser.add_argument("--adam-shard-bytes", type=int, default=1 << 30)

    # 指定单个 GPTQ cache 分片最大字节数，用于控制量化缓存文件大小。
    parser.add_argument("--gptq-shard-bytes", type=int, default=1 << 30)

    # 指定 Adam 状态块大小，用于按块组织优化器状态和更新粒度。
    parser.add_argument("--adam-block-size", type=int, default=128)

    # 指定 GPTQ group size，用于描述量化分组尺度和 cache 布局。
    parser.add_argument("--gptq-group-size", type=int, default=128)

    # 指定 FP32 分片文件名前缀，用于生成稳定的主参数分片路径。
    parser.add_argument("--fp32-shard-prefix", default="fp32")

    # 指定 Adam 分片文件名前缀，用于生成稳定的优化器状态分片路径。
    parser.add_argument("--adam-shard-prefix", default="adam")

    # 指定 GPTQ 分片文件名前缀，用于生成稳定的量化缓存分片路径。
    parser.add_argument("--gptq-shard-prefix", default="gptq")


def _qwen_manifest_config_from_args(
        args: argparse.Namespace,
) -> Qwen35MoeManifestConfig:
    """把 CLI 参数转换为 Qwen3.5 MoE manifest 配置对象。"""

    # ------------------------------- 组装模型 manifest 配置 -------------------------------
    # 返回 dataclass 配置对象，使后续容量估算和 manifest 构建不直接依赖 argparse。
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
    """把 CLI 参数转换为 Qwen3.5 MoE checkpoint 导入配置对象。"""

    # ------------------------------- 组装 checkpoint 导入配置 -------------------------------
    # 返回导入配置对象，把 checkpoint 命名、内部命名、量化布局和过滤条件集中传递给导入器。
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
    """把 CLI 参数转换为训练存储分片配置对象。"""

    # ------------------------------- 组装 manifest 分片配置 -------------------------------
    # 返回统一分片配置对象，保证容量估算、manifest 构建和实际写入使用同一组规则。
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
    """把逗号分隔的整数字符串解析为整数元组；空字符串表示不过滤。"""

    # ------------------------------- 解析可选整数列表 -------------------------------
    # 空白输入表示调用方没有指定过滤范围，返回 None 让下游使用默认全集语义。
    if value.strip() == "":
        # 返回 None 而不是空元组，避免被下游误解为显式选择 0 个元素。
        return None
    # 按逗号拆分并丢弃空项，得到稳定的整数过滤元组。
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_string_tuple(value: str) -> tuple[str, ...]:
    """把逗号分隔的字符串解析为字符串元组。"""

    # ------------------------------- 解析字符串列表 -------------------------------
    # 按逗号拆分并去除两侧空白，供 known_root_prefixes 等命名规则直接使用。
    return tuple(item.strip() for item in value.split(","))


def _dataclass_dict(value: Any) -> dict[str, Any]:
    """把 dataclass 对象转换为普通字典，便于 JSON 输出。"""

    # ------------------------------- 导出 dataclass 字段 -------------------------------
    # 遍历 dataclass 字段名并读取对应值，保持返回内容和配置对象字段一致。
    return {
        field: getattr(value, field)
        for field in value.__dataclass_fields__
    }


def _run_train(args: argparse.Namespace) -> dict[str, Any]:
    """导入真实 Qwen3.5 MoE 参数存储，并运行简化训练循环。"""

    # ------------------------------- 延迟导入训练依赖 -------------------------------
    # 延迟导入 torch，避免只执行容量估算或初始化命令时强制加载训练运行时。
    import torch
    # 延迟导入 AdamW 配置，只有 train 子命令需要构造优化器更新规则。
    from cfie_training.training_base.adam_update import AdamWConfig
    # 延迟导入真实 checkpoint 导入器，只有 train 子命令需要构造训练存储。
    from cfie_training.training_base.model_loader import Qwen35RealImporter
    # 延迟导入训练模型，避免非训练命令加载模型定义和 torch 依赖。
    from cfie_training.training_base.training_model import Qwen35ForTraining
    # 延迟导入训练循环及配置，保持 CLI 其他子命令的启动成本更低。
    from cfie_training.training_base.training_loop import (
        TrainingLoop,
        TrainingLoopConfig,
    )

    # ------------------------------- 导入参数并准备训练存储 -------------------------------
    # 构造真实 Qwen3.5 MoE 导入器，用 checkpoint 路径和模型结构参数定位专家权重。
    importer = Qwen35RealImporter(
        checkpoint_dir=args.checkpoint,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
    )

    # 将指定层和专家导入 FP32、Adam、GPTQ 存储，并返回 manifest 与进度写入器。
    fp32_store, adam_store, gptq_store, manifest, progress = (
        importer.import_to_stores(
            args.root,
            manifest_config=_manifest_shard_config_from_args(args),
            layers=tuple(range(args.num_layers)),
            experts=tuple(range(args.num_experts)),
        )
    )

    # ------------------------------- 构造热参数窗口 ID 列表 -------------------------------
    # 初始化热参数 ID 列表，后续只把这些专家权重加载到训练热窗口。
    hot_param_ids: list[str] = []

    # 遍历层编号 L，确保每层专家权重都纳入训练热参数集合。
    for layer_id in range(args.num_layers):
        # 遍历专家编号 E，按层和专家组合生成 MoE 专家参数 ID。
        for expert_id in range(args.num_experts):
            # 注册当前专家的 w13 权重 ID，通常对应 gate/up 融合权重。
            hot_param_ids.append(
                f"layers.{layer_id}.experts.{expert_id}.w13_weight"
            )
            # 注册当前专家的 w2 权重 ID，通常对应 down projection 权重。
            hot_param_ids.append(
                f"layers.{layer_id}.experts.{expert_id}.w2_weight"
            )
    # 将热参数列表冻结为 tuple，避免后续训练流程中意外修改参数窗口范围。
    hot_param_ids_tuple = tuple(hot_param_ids)

    # ------------------------------- 构造训练循环配置 -------------------------------
    # 构造训练循环配置，集中指定优化器、影子参数位置、梯度桶容量和峰值监控开关。
    config = TrainingLoopConfig(
        adam_config=AdamWConfig(lr=args.lr),
        shadow_dtype=torch.float32,
        shadow_device="cpu",
        bucket_capacity_bytes=args.grad_bucket_size_mib << 20,
        max_sealed_buckets=args.grad_bucket_count,
        enable_peak_monitor=True,
    )

    # ------------------------------- 创建训练循环并挂载模型 -------------------------------
    # 从底层训练存储构造训练循环，使 loop 负责参数窗口、梯度桶和状态提交。
    loop = TrainingLoop.from_stores(
        fp32_store=fp32_store,
        adam_store=adam_store,
        gptq_store=gptq_store,
        manifest=manifest,
        progress_writer=progress,
        hot_param_ids=hot_param_ids_tuple,
        config=config,
    )

    # 构造 CPU float32 训练模型；输入 token 形状后续为 [B, T]。
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
    # 将模型热参数绑定到 loop 的 shadow_store，使模型前反向使用当前窗口内参数。
    model.setup_hot_params(loop.hot_window.shadow_store, hot_param_ids_tuple)
    # 把模型挂载到训练循环，后续 run 会通过该模型执行 forward/backward/update。
    loop.attach_model(model)

    # ------------------------------- 构造随机训练数据迭代器 -------------------------------
    # 定义最小训练数据迭代器，用随机 token 驱动训练循环功能验证。
    def _data_iter(steps, batch, seq):
        # 按训练步数生成 batch，保证 dataloader 与 loop.run 的步数语义一致。
        for s in range(steps):
            # 延迟导入 batch 输入结构，避免外层非训练路径加载训练循环细节类型。
            from cfie_training.training_base.training_loop import TrainingDataBatchInput
            # 生成一批随机 token，input_ids 形状为 [B, T]，其中 B=batch、T=seq。
            yield TrainingDataBatchInput(
                input_ids=torch.randint(0, 1000, (batch, seq)),
                global_step=s,
                epoch=0,
                dataset_cursor=str(s),
                consumed_samples=batch,
                consumed_tokens=batch * seq,
            )

    # 把随机数据迭代器挂载到训练循环，后续 run 会逐步消费 [B, T] token batch。
    loop.attach_dataloader(_data_iter(args.steps, args.batch_size, args.seq_len))
    # 执行指定步数训练，返回每步的 loss、global_step 和窗口提交状态。
    results = loop.run(num_steps=args.steps)

    # ------------------------------- 汇总训练结果 -------------------------------
    # 返回训练命令摘要，便于 CLI 调用方快速判断训练是否完成以及最终 loss 状态。
    return {
        "command": "train",
        "total_steps": len(results),
        "final_step": results[-1].global_step if results else 0,
        "window_committed": results[-1].window_committed if results else False,
        "final_loss": results[-1].loss if results else 0.0,
        "num_results": len(results),
    }


# 当该文件作为脚本直接执行时，进入 CLI 主流程并用返回码退出进程。
if __name__ == "__main__":
    # 把 main 的整数返回值交给 SystemExit，使 shell 能接收到正确退出码。
    raise SystemExit(main())
