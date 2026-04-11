"""Dedicated training profile for Qwen3.5-35B-A3B."""

from __future__ import annotations

from cfie_training.config import (
    BucketScheduleConfig,
    ExecutionConfig,
    ExpertRotationConfig,
    MemoryBudgetConfig,
    ModelSpecConfig,
    ModelSourceConfig,
    ModelTargets,
    OptimizerConfig,
    PredictorTrainerConfig,
    ResourcePolicyConfig,
    RuntimeQuantizationConfig,
    StateBytesConfig,
    TransportConfig,
    TrainingProjectConfig,
)

QWEN35_35B_A3B_PROFILE = "qwen35-35b-a3b"


# 构造 Qwen3.5-35B-A3B 的默认训练基座配置。
def build_qwen35_35b_a3b_config() -> TrainingProjectConfig:
    # ------------------------------- 构造 Qwen3.5-35B-A3B 专用训练档位配置对象 -------------------------------
    # 返回面向 Qwen3.5-35B-A3B 训练方案的完整训练项目配置。
    return TrainingProjectConfig(
        # 当前训练档位名称，用于标识该配置属于 Qwen3.5-35B-A3B 专用档位。
        profile_name=QWEN35_35B_A3B_PROFILE,

        # ------------------------------- 配置训练目标模型与模型族信息 -------------------------------
        # 配置开发阶段模型、目标模型、模型家族与当前阶段标识。
        model_targets=ModelTargets(
            # 当前开发与验证阶段实际使用的模型名称。
            development_model="Qwen3.5-35B-A3B",
            # 训练方案最终对齐或服务的目标模型名称。
            target_model="Qwen3.5-122B-class-MoE",
            # 当前模型所属的模型家族标识。
            family="qwen3.5_moe",
            # 当前配置所处的阶段，这里标记为开发阶段。
            stage="development",
        ),

        # ------------------------------- 配置模型结构与量化规格 -------------------------------
        # 配置模型架构、层数、注意力结构、专家结构与静态量化参数。
        model_spec=ModelSpecConfig(
            # 模型在运行时或配置层面使用的架构名称。
            architecture="Qwen3_5MoeForConditionalGeneration",
            # 文本主干模型类型标识。
            text_model_type="qwen3_5_moe_text",
            # 模型隐藏维度大小。
            hidden_size=2048,
            # 模型总隐藏层数。
            num_hidden_layers=40,
            # 注意力头数量。
            num_attention_heads=16,
            # Key/Value 头数量。
            num_key_value_heads=2,
            # 总专家数量。
            num_experts=256,
            # 每个 token 实际路由到的专家数。
            num_experts_per_tok=8,
            # routed expert 的中间层维度。
            moe_intermediate_size=512,
            # shared expert 的中间层维度。
            shared_expert_intermediate_size=512,
            # full attention 的周期性插入间隔。
            full_attention_interval=4,
            # 模型支持的最大位置长度。
            max_position_embeddings=262144,
            # 多 token prediction 相关隐藏层数。
            mtp_num_hidden_layers=1,
            # 按层循环使用的注意力模式模板。
            attention_pattern=(
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ),
            # 模型静态量化方法。
            quantization="gptq",
            # 量化位宽。
            quant_bits=4,
            # 量化分组大小。
            quant_group_size=128,
            # 是否采用对称量化。
            quant_sym=True,
            # 指定在动态量化排除列表中的模块模式。
            quant_dynamic_exclusions=(
                ".*attn.*",
                ".*shared_expert.*",
                ".*mtp.*",
                ".*visual.*",
            ),
            # 模型总参数量，单位为十亿参数。
            total_params_billion=35.0,
        ),

        # ------------------------------- 配置模型权重来源与索引文件信息 -------------------------------
        # 配置模型在本地文件系统中的来源路径与权重索引方式。
        model_source=ModelSourceConfig(
            # 本地 HuggingFace 缓存中的模型快照路径。
            model_path=(
                "/home/gaojiamin/.cache/huggingface/hub/"
                "models--Qwen--Qwen3.5-35B-A3B/snapshots/"
                "ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
            ),
            # 权重索引文件名称。
            index_filename="model.safetensors.index.json",
            # 是否优先使用本地权重清单文件。
            use_local_weight_manifest=True,
        ),

        # ------------------------------- 配置专家轮转训练策略 -------------------------------
        # 配置 MoE 专家的轮转训练方式与选择策略。
        expert_rotation=ExpertRotationConfig(
            # 是否启用专家轮转机制。
            enabled=True,
            # 每一步训练中激活的专家数量。
            active_experts_per_step=8,
            # 每隔多少步执行一次专家轮转。
            rotate_every_steps=1,
            # 是否在每一步都训练共享专家。
            train_shared_expert_every_step=True,
            # 专家选择策略，这里按 router 热度选择。
            selection_strategy="router_hotness",
        ),

        # ------------------------------- 配置按桶执行的调度策略 -------------------------------
        # 配置训练按 expert 粒度分桶以及反向、更新、释放的执行策略。
        bucket_schedule=BucketScheduleConfig(
            # 分桶单位，这里按 expert 进行分桶。
            unit="expert",
            # 设备侧允许同时存活的最大 bucket 数量。
            max_live_buckets=1,
            # 提前预取的 bucket 数量。
            prefetch_buckets=1,
            # host 侧梯度缓冲区的作用域，仅保留当前 bucket。
            host_gradient_buffer_scope="current_bucket_only",
            # 是否在使用完梯度后立即释放梯度。
            release_gradients_immediately=True,
            # 是否在反向结束后立即执行参数更新。
            update_immediately_after_backward=True,
        ),

        # ------------------------------- 配置训练执行路径与流并行策略 -------------------------------
        # 配置优化器、梯度、激活策略以及计算流与传输流名称。
        execution=ExecutionConfig(
            # 优化器计算所在设备，这里放在 CPU。
            optimizer_device="cpu",
            # 梯度缓冲所在设备，这里放在 CPU。
            gradient_device="cpu",
            # 激活保存策略，这里使用重计算策略。
            activation_policy="recompute",
            # 是否开启反向传播与参数更新的重叠执行。
            overlap_backward_and_update=True,
            # 负责反向计算的流名称。
            compute_stream_name="backward_compute",
            # 负责 CPU 更新与释放动作的流名称。
            transfer_stream_name="cpu_update_release",
            # 样本级并行度配置。
            sample_parallelism=2,
        ),

        # ------------------------------- 配置优化器算法与状态存储格式 -------------------------------
        # 配置 AdamW 优化器超参数以及 CPU 侧状态压缩策略。
        optimizer=OptimizerConfig(
            # 优化器算法名称。
            algorithm="adamw",
            # 学习率。
            learning_rate=1e-5,
            # AdamW 的一阶动量系数。
            beta1=0.9,
            # AdamW 的二阶动量系数。
            beta2=0.95,
            # 数值稳定项 epsilon。
            epsilon=1e-8,
            # 权重衰减系数。
            weight_decay=0.0,
            # CPU 侧优化器状态的存储数据类型。
            cpu_state_storage_dtype="fp8_e4m3fn",
            # 梯度缓冲区的存储数据类型。
            gradient_buffer_storage_dtype="fp8_e4m3fn",
            # 参数更新后是否将优化器状态重新卸载。
            offload_state_after_update=True,
        ),

        # ------------------------------- 配置资源优先级与权重卸载策略 -------------------------------
        # 配置系统对 GPU、CPU 与 NVMe 资源的使用倾向。
        resource_policy=ResourcePolicyConfig(
            # 是否将 GPU 视为最稀缺资源。
            gpu_is_scarcest_resource=True,
            # 是否优先节省内存而非追求吞吐。
            prioritize_memory_over_throughput=True,
            # 是否允许 CPU 参与训练路径中的关键工作。
            allow_cpu_participation=True,
            # 权重卸载后端类型，这里使用 CPU 加 NVMe 组合。
            weight_offload_backend="cpu+nvme",
        ),

        # ------------------------------- 配置权重传输缓存与淘汰策略 -------------------------------
        # 配置 staged 文件缓存大小、跨步复用策略与缓存淘汰算法。
        transport=TransportConfig(
            # 最大 staged 文件缓存容量，单位为 GiB。
            max_staged_file_cache_gb=8.0,
            # 是否在多个 step 之间复用 staged 文件。
            reuse_staged_files_across_steps=True,
            # 缓存淘汰策略，这里采用 LRU。
            eviction_policy="lru",
        ),

        # ------------------------------- 配置运行时量化策略 -------------------------------
        # 配置运行时权重视图的量化方法、位宽与持久化策略。
        runtime_quantization=RuntimeQuantizationConfig(
            # 是否启用运行时量化。
            enabled=True,
            # 运行时量化方法。
            method="gptq",
            # 运行时量化位宽。
            bits=4,
            # 运行时量化分组大小。
            group_size=128,
            # 是否采用对称量化。
            sym=True,
            # 运行时计算视图使用的数据类型。
            compute_view_dtype="fp32",
            # 是否将 FP32 形式持久化到 NVMe。
            persist_fp32_to_nvme=True,
        ),

        # ------------------------------- 配置 predictor 训练器超参数 -------------------------------
        # 配置 predictor 训练时使用的输入维度、隐藏维度、批大小与训练轮数等参数。
        predictor_trainer=PredictorTrainerConfig(
            # 输入摘要向量维度。
            input_summary_dim=64,
            # predictor 隐藏层维度。
            hidden_dim=128,
            # predictor 训练批大小。
            batch_size=8,
            # predictor 默认训练轮数。
            epochs=4,
            # predictor 训练学习率。
            learning_rate=1e-3,
            # predictor 权重衰减系数。
            weight_decay=1e-4,
            # 每一步默认构造的样本数。
            examples_per_step=4,
            # 合成轨迹噪声幅度。
            synthetic_trace_noise_scale=0.05,
            # 随机种子。
            seed=0,
        ),

        # ------------------------------- 配置多级存储预算 -------------------------------
        # 配置 GPU、CPU 与 NVMe 三层存储预算以及安全余量。
        memory_budget=MemoryBudgetConfig(
            # GPU 热数据预算，单位为 GiB。
            gpu_hot_budget_gb=6.0,
            # CPU 热数据预算，单位为 GiB。
            cpu_hot_budget_gb=24.0,
            # NVMe 冷数据预算，单位为 GiB。
            nvme_cold_budget_gb=512.0,
            # GPU 预算安全余量，单位为 GiB。
            gpu_safety_margin_gb=1.0,
            # CPU 预算安全余量，单位为 GiB。
            cpu_safety_margin_gb=2.0,
            # NVMe 预算安全余量，单位为 GiB。
            nvme_safety_margin_gb=16.0,
        ),

        # ------------------------------- 配置参数、梯度与激活的字节开销估计 -------------------------------
        # 配置训练过程中各类状态按参数或按元素计的字节成本估算。
        state_bytes=StateBytesConfig(
            # 设备侧权重每个参数占用的字节数。
            device_weight_bytes_per_param=2,
            # 梯度每个参数占用的字节数。
            gradient_bytes_per_param=2,
            # 主权重每个参数占用的字节数。
            master_weight_bytes_per_param=4,
            # 优化器状态每个参数占用的字节数。
            optimizer_state_bytes_per_param=8,
            # 激活中每个元素占用的字节数。
            activation_bytes_per_element=2,
            # 激活驻留倍数估计，用于放大激活占用。
            activation_residency_multiplier=2.0,
        ),

        # ------------------------------- 配置当前训练档位的补充说明信息 -------------------------------
        # 记录该训练档位的设计假设、资源策略与实现约束说明。
        notes=(
            "Dedicated training profile for Qwen3.5-35B-A3B.",
            "Model geometry is confirmed from the local HuggingFace config cache.",
            "Treat the 256 routed experts as a rotating pool; never keep them all trainable on device.",
            "Use per-bucket backward/update/release with CPU-side optimizer math as the default path.",
            "Compress CPU-side AdamW state and bucket gradient buffers with FP8 storage where the training path remains stable.",
            "Reserve host gradient memory only for the current bucket-sized ingress/update window.",
            "Align layer buckets with the Qwen3.5 4-layer attention cadence: 3 linear-attention layers plus 1 full-attention layer.",
        ),
    )
