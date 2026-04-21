"""Bounded predictor trainer for candidate-routed MoE experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from cfie_training.config import TrainingProjectConfig
from cfie_training.runtime.data import TokenizedDatasetBatchPlanner
from cfie_training.runtime.types import BatchShape

if TYPE_CHECKING:
    from cfie.v1.engine.llm_engine import LLMEngine


class PredictorBatchPlanner(Protocol):
    # 返回指定 step 使用的 batch 形状。
    def batch_for_step(self, step_index: int) -> BatchShape:
        ...


@dataclass(slots=True, frozen=True)
class CapturedForwardBatch:
    layer_hidden_states: tuple[torch.Tensor, ...]
    layer_teacher_topk_ids: tuple[torch.Tensor, ...]


@dataclass(slots=True, frozen=True)
class CapturedHiddenStatePayload:
    """trainer 内部使用的标准 hidden-state 捕获载荷。

    worker 侧必须返回结构化字典协议；trainer 侧收到后立即转换成该对象。
    这样合并逻辑只面对具名字段，不再传递难读的多元素 tuple，也不会再
    根据返回位置猜测层号或 rank 语义。
    """

    request_id: str
    layer_ids: tuple[int, ...]
    hidden_states: tuple[torch.Tensor, ...]
    pp_rank: int
    tp_rank: int


class PredictorTeacherModelBackend(Protocol):
    def capture_batch(self, batch: BatchShape) -> CapturedForwardBatch:
        ...


class EngineRouterTeacherModelBackend:
    """复用推理引擎执行真实前向的 predictor teacher 后端。

    该后端不直接走 `AutoModelForCausalLM.from_pretrained`，而是复用 CFIE
    推理主链创建 `LLMEngine`，把真实 token prompt 送入引擎执行一次最小生成步。
    这样训练侧拿到的 hidden state、路由专家结果、KV/卸载路径与线上推理更一致，
    不会因为单独造一套 teacher 前向实现而偏离真实部署行为。
    """

    # 把运行时属性显式声明在类体上，便于 IDE 和静态检查器理解该对象的内部状态。
    _config: TrainingProjectConfig
    _model_path: str
    _capture_layer_ids: tuple[int, ...]
    _engine_quantization: str | None
    _gpu_memory_utilization: float
    _engine: LLMEngine | None
    _engine_capacity: tuple[int, int, int] | None
    _request_serial: int
    _capture_fragments_by_request: dict[str, dict[int, torch.Tensor]]

    # ------------------------------- 初始化引擎复用型 teacher 后端 -------------------------------
    def __init__(self, config: TrainingProjectConfig) -> None:
        # `__init__` 只保留一个很薄的壳，真正初始化逻辑统一放在 `create` 中；
        # 这样既方便后续扩展工厂式构造，也避免初始化流程散落在多个入口。
        initialized = self.create(config)
        # 把工厂中准备好的内部状态整体灌回当前实例，
        # 让外部仍可按普通类实例化方式使用该后端。
        self.__dict__.update(initialized.__dict__)

    @classmethod
    def create(
            cls,
            config: TrainingProjectConfig,
    ) -> "EngineRouterTeacherModelBackend":
        # ------------------------------- 构造未初始化实例并固化训练配置 -------------------------------
        # 这里直接调用 `object.__new__` 生成空实例，
        # 避免再触发 `__init__` 自己造成递归初始化。
        self = object.__new__(cls)

        # 先缓存训练项目配置，后续 teacher 模型路径、预算与引擎参数都从这里派生。
        self._config = config

        # ------------------------------- 解析 teacher 模型来源与捕获范围 -------------------------------
        # teacher 必须直接复用当前训练档位指定的模型路径，
        # 这样量化模型、离线路径与实际部署模型都能保持一致。
        self._model_path = self._resolve_model_path()
        if not self._model_path:
            raise ValueError(
                "model_source.model_path is required for forward-capture traces"
            )

        # predictor 训练要监督每一层的后续路由，因此这里一次性固定全层捕获列表，
        # 后面启用 worker 侧 capture 时直接把这组层号下发即可。
        self._capture_layer_ids = tuple(
            range(self._config.model_spec.num_hidden_layers)
        )

        # ------------------------------- 推导 teacher 引擎的部署口径 -------------------------------
        # 若当前模型是 GPTQ，则这里优先沿用量化推理路径，
        # 避免训练侧 teacher 偷偷退回非量化模型而失去工程一致性。
        self._engine_quantization = self._resolve_engine_quantization()

        # 依据训练配置中的 GPU hot budget 估算 teacher 引擎可用显存比例，
        # 让 trace 任务默认遵守训练侧预算约束，而不是吃满整卡显存。
        self._gpu_memory_utilization = self._resolve_gpu_memory_utilization()

        # ------------------------------- 初始化懒加载运行时状态 -------------------------------
        # teacher 引擎按 batch 尺寸懒创建；
        # 这样可以用真实 prompt 容量启动，而不是在构造阶段拍脑袋分配。
        self._engine = None
        # 记录当前引擎已承载的容量上限，后续遇到更小 batch 时即可直接复用。
        self._engine_capacity: tuple[int, int, int] | None = None
        # 请求序号用于构造稳定且不冲突的 request_id，便于后续按请求收回捕获结果。
        self._request_serial = 0
        # PP 场景下同一请求的不同层段可能来自不同 worker；
        # 这里按 request_id 暂存尚未凑齐的层结果，避免某次 RPC 只返回局部层时数据被覆盖或丢弃。
        self._capture_fragments_by_request = {}
        return self

    # ------------------------------- 解析 teacher 模型路径 -------------------------------
    def _resolve_model_path(self) -> str:
        # 训练侧 teacher 必须严格使用配置里的模型路径；
        # 这里显式禁止旧方案里“量化模型自动回退到 base 模型”的隐式行为。
        return str(self._config.model_source.model_path).strip()

    # ------------------------------- 推导 teacher 引擎量化方式 -------------------------------
    def _resolve_engine_quantization(self) -> str | None:
        # 非 GPTQ 模型不需要额外声明引擎量化方式，保持原生加载路径即可。
        if self._config.model_spec.quantization != "gptq":
            return None
        # 训练档位里的 GPTQ 模型统一按 Marlin 口径启动，
        # 使 teacher 捕获与平时 chat / generate 的实际执行路线保持一致。
        return "gptq_marlin"

    # ------------------------------- 推导 teacher 引擎显存占用比例 -------------------------------
    def _resolve_gpu_memory_utilization(self) -> float:
        # 若当前环境没有 CUDA，则返回一个保守默认值；
        # 这样即使在离线检查或 CPU 环境下也能顺利构造对象。
        if not torch.cuda.is_available():
            return 0.6

        # 读取首张卡的总显存容量，用来把绝对预算换算成引擎接受的比例值。
        total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_gib <= 0:
            # 理论上这里不应出现非正显存；若驱动返回异常值，则回退到保守默认值。
            return 0.6

        # 用训练侧 GPU hot budget 对整卡显存做归一化，得到目标占用比例。
        ratio = self._config.memory_budget.gpu_hot_budget_gb / total_gib
        # 最终再做区间夹紧，避免比例过低导致引擎几乎无法工作，
        # 也避免比例过高把训练进程的其他预算吃掉。
        return max(0.35, min(0.85, ratio))

    # ------------------------------- 推导 teacher 引擎 CPU 卸载预算 -------------------------------
    def _resolve_cpu_offload_gb(self) -> float:
        # teacher 引擎只能消费“预算减去安全余量”后的 CPU 热存空间，
        # 否则训练侧很容易把预留给系统与其他组件的缓冲区也占满。
        available_cpu_budget = (
                self._config.memory_budget.cpu_hot_budget_gb
                - self._config.memory_budget.cpu_safety_margin_gb
        )
        # 即使配置不合理导致预算为负，也在这里钳成 0，避免把非法值传进引擎。
        return max(0.0, available_cpu_budget)

    # ------------------------------- 将训练侧卸载策略映射成引擎后端 -------------------------------
    def _resolve_engine_offload_backend(self) -> str:
        # 训练配置里的 `weight_offload_backend` 是项目层面的资源策略，
        # 但推理引擎只接受自己定义的一组后端枚举，因此这里要做一次语义映射。
        backend = self._config.resource_policy.weight_offload_backend

        # 纯 CPU 卸载最接近引擎里的 UVA 路线；
        # 直接映射成 `uva` 可以让 teacher 路径明确走主存访问逻辑。
        if backend == "cpu":
            return "uva"

        # 对 `nvme` 与 `cpu+nvme` 则统一交给 `auto`，
        # 让引擎依据当前 `cpu_offload_gb` 和其他预算自行选择细节实现，
        # 避免训练侧自定义枚举直接穿透到底层后触发配置校验失败。
        return "auto"

    # ------------------------------- 构造训练 teacher 专用 additional_config -------------------------------
    def _resolve_engine_additional_config(self) -> dict[str, Any]:
        # predictor teacher 的职责是“尽快启动真实前向并采集监督信号”，
        # 不是完整复刻推理侧 MoE tiered cache 的冷启动链路。
        #
        # 若这里不显式关闭，CfieConfig 在量化 MoE 场景下可能自动展开
        # 一整套 tiered cache 规划，并进一步拉起较重的 CPU expert pool。
        # 这对训练 trace 来说收益很小，却会显著拖慢首批采样启动速度。
        #
        # 因此此处专门给训练 teacher 注入一份 disabled 配置：
        # - 仅影响 predictor 训练专用 teacher 引擎
        # - 不影响普通 chat / native-generate 主链
        # - 也不改变后续 predictor 真正挂载到推理侧时的配置口径
        return {
            "moe_tiered_cache": {
                "enabled": False,
                "reason": "predictor_teacher_engine_disabled",
            }
        }

    # ------------------------------- 按真实有效长度提取 prompt 行 -------------------------------
    @staticmethod
    def _effective_prompt_rows(
            batch: BatchShape,
    ) -> tuple[tuple[int, ...], ...]:
        # teacher 前向必须拿到显式 token rows；
        # 若数据规划阶段没有写入 token_rows，就无法构造真实请求。
        if not batch.has_token_rows:
            raise ValueError(
                "forward-capture traces require BatchShape.token_rows to be populated"
            )

        # ------------------------------- 按 attention mask 剥离尾部 padding -------------------------------
        # 数据规划阶段的 `token_rows` 往往为了批处理对齐而做成等长；
        # 但 teacher 前向只能看到真实 prompt，不能把补齐 token 误当作上下文送入模型。
        if not batch.attention_mask_rows:
            # 若没有单独提供 mask，则默认整行都是有效 prompt。
            return batch.token_rows

        prompt_rows: list[tuple[int, ...]] = []
        for row_index, (token_row, mask_row) in enumerate(
                zip(batch.token_rows, batch.attention_mask_rows, strict=True)
        ):
            # `BatchShape` 已保证 mask 为“前缀有效、尾部 padding”形式，
            # 因而这里直接求和就能得到真实 prompt 长度。
            valid_length = sum(int(value) for value in mask_row)

            # teacher 请求至少要有一个有效 token；
            # 若整行都被 padding 掉，说明上游样本构造已经不满足真实前向约束。
            if valid_length < 1:
                raise ValueError(
                    "forward-capture traces require at least one valid token "
                    f"per row; row {row_index} is fully padded"
                )

            # 只截取前缀有效 token，确保后续路由监督完全来自真实 prompt。
            prompt_rows.append(token_row[:valid_length])
        return tuple(prompt_rows)

    # ------------------------------- 根据真实 prompt 估算引擎容量 -------------------------------
    def _capacity_for_batch(self, batch: BatchShape) -> tuple[int, int, int]:
        # 容量估算必须基于真实 prompt 长度，
        # 否则按 padded 长度启动会把 teacher 引擎无谓放大。
        prompt_rows = self._effective_prompt_rows(batch)

        # `max_model_len` 决定单请求最长上下文，因此取所有 prompt 中的最大长度。
        max_prompt_len = max(len(prompt_row) for prompt_row in prompt_rows)

        # `max_num_seqs` 直接对应同批并发请求数，也就是本批 prompt 行数。
        max_num_seqs = max(1, len(prompt_rows))

        # `max_num_batched_tokens` 取本批真实 token 总数，
        # 让调度器按实际吞吐而不是 padding 后体积规划空间。
        max_num_batched_tokens = max(
            1,
            sum(len(prompt_row) for prompt_row in prompt_rows),
        )

        # teacher 请求统一用 `max_tokens=1` 做最小生成，
        # 因此上下文容量还要额外预留 1 个生成位。
        max_model_len = max_prompt_len + 1
        return max_model_len, max_num_seqs, max_num_batched_tokens

    # ------------------------------- 构造或扩容 teacher 引擎 -------------------------------
    def _ensure_engine(self, batch: BatchShape) -> LLMEngine:
        # 先基于当前批次真实 prompt 估算最小可用容量。
        required_capacity = self._capacity_for_batch(batch)
        if (
                self._engine is not None
                and self._engine_capacity is not None
                and self._engine_capacity[0] >= required_capacity[0]
                and self._engine_capacity[1] >= required_capacity[1]
                and self._engine_capacity[2] >= required_capacity[2]
        ):
            # 若现有引擎已经覆盖本批需求，则直接复用，
            # 避免重复冷启动引擎和重复装载权重。
            return self._engine

        # 若容量不足或引擎为空，则先彻底关闭旧实例，避免多套引擎并存占资源。
        self._shutdown_engine()

        # 延迟导入推理引擎相关模块，避免训练侧纯配置路径也触发沉重依赖初始化。
        from cfie.engine.arg_utils import EngineArgs
        from cfie.v1.engine.llm_engine import LLMEngine

        # ------------------------------- 按当前 batch 需求组装引擎参数 -------------------------------
        max_model_len, max_num_seqs, max_num_batched_tokens = required_capacity
        # 先把本次启动会复用的实例字段摊平成局部变量，
        # 这样 IDE 更容易跟踪类型，也能减少参数列表里重复访问对象内部状态。
        model_path = self._model_path
        engine_quantization = self._engine_quantization
        engine_dtype = (
            "float16" if engine_quantization == "gptq_marlin" else "auto"
        )
        gpu_memory_utilization = self._gpu_memory_utilization
        offload_backend = self._resolve_engine_offload_backend()
        memory_budget = self._config.memory_budget
        cpu_offload_gb = self._resolve_cpu_offload_gb()
        additional_config = self._resolve_engine_additional_config()

        engine_args = EngineArgs(
            # teacher 模型与 tokenizer 都直接指向当前训练档位模型目录，
            # 确保 token 化与权重来源保持一致。
            model=model_path,
            tokenizer=model_path,
            # 训练 teacher 使用本地受控模型目录，不走远端自定义代码加载。
            trust_remote_code=False,
            # GPTQ Marlin 路线下显式给出 `float16`，避免量化推理路径的 dtype 推断偏移。
            dtype=engine_dtype,
            # 把上面解析出的量化方式显式传给引擎，保证与实际部署口径一致。
            quantization=engine_quantization,
            # 这三个容量参数共同约束调度器、KV 与请求并发上限。
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            # 显存占用比例直接遵守训练侧预算推导结果。
            gpu_memory_utilization=gpu_memory_utilization,
            # 把训练项目层面的卸载策略翻译成引擎能识别的后端枚举。
            offload_backend=offload_backend,
            # CPU 相关预算继续沿用训练配置，避免 teacher 启动绕开整体资源规划。
            moe_cpu_budget_gb=memory_budget.cpu_hot_budget_gb,
            moe_cpu_min_free_gb=memory_budget.cpu_safety_margin_gb,
            cpu_offload_gb=cpu_offload_gb,
            # predictor 训练必须把最终 routed experts 带回来，作为 teacher top-k 标签。
            enable_return_routed_experts=True,
            # teacher 路径强制 eager，减少 cudagraph / 编译态对抓取链路的额外干扰。
            enforce_eager=True,
            # 采样训练 trace 时不需要常规统计日志，避免噪声过多。
            disable_log_stats=True,
            # 注入训练 teacher 的额外配置覆盖项。
            additional_config=additional_config,
        )

        # ------------------------------- 创建引擎并开启 hidden-state 捕获 -------------------------------
        self._engine = LLMEngine.from_engine_args(
            engine_args,
            # 训练 teacher 也显式走多进程路径，
            # 这样 Windows 下可直接复用 chat / generate 常用通道进行调试，
            # 避免 teacher 单独停留在 inproc 特殊分支，导致训练侧与日常推理侧行为脱节。
            enable_multiprocessing=True,
        )
        # 通知所有 worker 开启 predictor hidden-state 捕获，并固定捕获层集合。
        self._engine.collective_rpc(
            "enable_predictor_capture",
            args=(self._capture_layer_ids,),
        )
        # 记录当前引擎已满足的容量，下次可先判断是否直接复用。
        self._engine_capacity = required_capacity
        return self._engine

    # ------------------------------- 关闭 teacher 引擎 -------------------------------
    def _shutdown_engine(self) -> None:
        # 没有已启动引擎时无需收尾，直接返回即可。
        if self._engine is None:
            return

        # 先尝试通知 worker 关闭 predictor capture，
        # 避免遗留的捕获状态影响后续重建或析构。
        try:
            self._engine.collective_rpc("disable_predictor_capture")
        except Exception:
            # 析构链路以“尽最大努力回收”为主，收尾失败不再二次抛异常。
            pass

        # 再关闭 engine core，释放底层调度器、worker 与相关资源。
        try:
            self._engine.engine_core.shutdown()
        except Exception:
            pass

        # 最后把本地句柄与容量记录清空，表示当前没有可复用的 teacher 引擎。
        self._engine = None
        self._engine_capacity = None
        self._capture_fragments_by_request.clear()

    # ------------------------------- 归一化 worker 返回的捕获载荷 -------------------------------
    @staticmethod
    def _ensure_hidden_state_tensor(value: Any) -> torch.Tensor:
        # 多进程 RPC 在部分路径下会把 CPU tensor 还原成 Python list；
        # 训练侧统一转回 float32 tensor，保证后续按层 torch.cat 时拿到稳定张量类型。
        if isinstance(value, torch.Tensor):
            return value
        return torch.as_tensor(value, dtype=torch.float32)

    def _normalize_captured_hidden_state_payload(
            self,
            request_id: str,
            payload: dict[str, Any],
    ) -> CapturedHiddenStatePayload:
        # ------------------------------- 校验 worker 返回协议 -------------------------------
        # predictor capture 已进入工程化落地阶段，worker 必须返回结构化字典；
        # 如果这里继续接受 tuple/list 这类非标准载荷，会掩盖 worker 侧没有按协议上报的问题。
        if not isinstance(payload, dict):
            raise ValueError(
                "captured hidden-state payload must be a dict with "
                "layer_ids, hidden_states, pp_rank and tp_rank"
            )

        # `layer_ids` 和 `hidden_states` 决定每个 tensor 属于哪一层；
        # `pp_rank` 与 `tp_rank` 决定跨 PP 分片和 TP 重复副本的稳定合并顺序。
        required_keys = ("layer_ids", "hidden_states", "pp_rank", "tp_rank")
        missing_keys = tuple(key for key in required_keys if key not in payload)
        if missing_keys:
            raise ValueError(
                "captured hidden-state payload missing required keys: "
                + ", ".join(missing_keys)
            )

        # ------------------------------- 解析层号与 hidden-state 张量 -------------------------------
        # worker 通过 `layer_ids` 明确说明当前 payload 覆盖的层集合；
        # trainer 不再根据返回位置猜测层号，避免 PP 场景下误把分片当完整模型。
        layer_ids = tuple(int(layer_id) for layer_id in payload["layer_ids"])

        # 多进程 RPC 可能把 CPU tensor 序列化成 Python 数组；
        # 这里统一转回 tensor，后续合并和 torch.cat 都只处理一种张量类型。
        hidden_states = tuple(
            self._ensure_hidden_state_tensor(hidden_state)
            for hidden_state in payload["hidden_states"]
        )

        # ------------------------------- 校验层号和 hidden-state 数量一致性 -------------------------------
        # 层号和 hidden-state 必须一一对应；
        # 如果长度不同，后续合并时无法判断某个 tensor 属于哪一层，因此这里直接失败。
        if len(layer_ids) != len(hidden_states):
            raise ValueError(
                "captured hidden-state payload layer_ids and hidden_states "
                "must have the same length"
            )

        # ------------------------------- 读取并行 rank 元数据 -------------------------------
        # PP rank 用来让后续合并逻辑按层段稳定补齐；
        # TP rank 用来固定重复副本的处理顺序，后续同层副本会按“先到先用”自然去重。
        pp_rank = int(payload["pp_rank"])
        tp_rank = int(payload["tp_rank"])

        # 这里返回具名载荷对象，让合并逻辑使用字段表达语义；
        # 不再依赖 tuple 下标，避免后续维护时读错 request、rank 或层号的位置。
        return CapturedHiddenStatePayload(
            request_id=str(request_id),
            layer_ids=layer_ids,
            hidden_states=hidden_states,
            pp_rank=pp_rank,
            tp_rank=tp_rank,
        )

    # ------------------------------- 汇总多 worker 的 hidden-state 捕获结果 -------------------------------
    def _merge_captured_hidden_states(
            self,
            worker_results: list[dict[str, dict[str, Any]]],
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        """
        合并多 worker 返回的 hidden-state 捕获结果。

        `worker_results` 中的每一项都表示一个 worker 本轮返回的
        `request_id -> 捕获载荷` 映射。由于同一个 request 的目标层可能被
        PP 切分到不同 rank 返回，同时同一层也可能被多个 TP rank 返回重复副本，
        因此这里需要先把载荷归一化，再按稳定顺序逐请求合并。

        函数只返回“已经收齐全部目标层”的 request 结果；对于还未收齐的 request，
        已到达的层分片会继续保存在 `_capture_fragments_by_request` 中，
        等待下一轮 RPC 返回剩余层后再补齐。
        """
        # ------------------------------- 归一化多 worker 返回的捕获载荷 -------------------------------
        # 这里先把“按 worker 分组”的返回结果摊平成统一条目列表，
        # 后续合并时就可以只关注 request、层号和 rank 信息，而不再关心它来自哪个 worker。
        normalized_entries: list[CapturedHiddenStatePayload] = []
        for worker_result in worker_results:
            # 当前 `worker_result` 表示单个 worker 本轮返回的全部 request 捕获结果；
            # 这里继续展开到单个 request 粒度，方便后续统一排序和逐请求拼装。
            for request_id, payload in worker_result.items():
                normalized_entries.append(
                    self._normalize_captured_hidden_state_payload(
                        str(request_id),
                        payload,
                    )
                )

        # ------------------------------- 按稳定顺序整理待合并条目 -------------------------------
        # 这里先按 request_id 分组，再按 PP rank、TP rank 排序，。
        normalized_entries.sort(
            key=lambda entry: (entry.request_id, entry.pp_rank, entry.tp_rank)
        )

        # ------------------------------- 初始化目标层集合与本轮返回容器 -------------------------------
        # `merged` 只保存本轮已经收齐全部目标层的 request；
        # 没有收齐的 request 不会提前返回，而是继续留在暂存桶中等待补齐。
        merged: dict[str, tuple[torch.Tensor, ...]] = {}

        # 这里取出调用方要求捕获的目标层顺序；
        # 最终返回结果必须严格按这个顺序组织 tuple，不能依赖 worker 的返回顺序。
        expected_layer_ids = self._capture_layer_ids

        # 这里额外构造集合版本，后面做层号合法性校验和“是否收齐”判断时会更直接。
        expected_layer_id_set = set(expected_layer_ids)

        # ------------------------------- 逐请求拼装层分片并处理 TP 重复副本 -------------------------------
        for entry in normalized_entries:
            # 如果当前 request 已经在本轮完成合并，
            # 那么后续再次命中的条目通常只是 TP 重复副本，这里可以直接跳过。
            request_id = entry.request_id
            if request_id in merged:
                continue

            # 这里为当前 request 取出对应的暂存桶；
            # 暂存桶以 layer_id 为键，用来逐层积累当前 request 已经收集到的 hidden state。
            request_fragments = self._capture_fragments_by_request.setdefault(
                request_id,
                {},
            )

            for layer_id, hidden_state in zip(entry.layer_ids, entry.hidden_states):
                # 返回层号必须属于本轮请求的目标层集合；
                # 一旦出现集合外层号，说明 worker 返回结果和调用方请求不一致，需要立刻报错。
                if layer_id not in expected_layer_id_set:
                    raise ValueError(
                        f"captured hidden-state layer {layer_id} is outside "
                        "the requested capture layer set"
                    )

                # 对同一 request 的同一层只保留第一份结果；
                # 后续命中通常来自其他 TP rank 的重复副本，这里不再覆盖先到结果。
                if layer_id not in request_fragments:
                    request_fragments[layer_id] = hidden_state

            # ------------------------------- 检查当前请求是否已经收齐全部目标层 -------------------------------
            # 只有当暂存桶已经覆盖全部目标层时，当前 request 才算真正合并完成；
            # 否则说明仍有 PP 分段尚未返回，需要继续把已到达分片保留到下一轮 RPC。
            if expected_layer_id_set.issubset(request_fragments.keys()):
                # 这里严格按 `_capture_layer_ids` 的既定顺序构造最终结果，
                # 保证调用方拿到的各层 hidden state 顺序稳定且与请求顺序一致。
                merged[request_id] = tuple(
                    request_fragments[layer_id] for layer_id in expected_layer_ids
                )
                # 当前 request 已经完成合并后，就不再需要保留它的暂存分片；
                # 这里立即清理对应状态，避免后续轮次重复占用缓存。
                self._capture_fragments_by_request.pop(request_id, None)

        # ------------------------------- 返回本轮已经完成合并的请求结果 -------------------------------
        # 返回值只包含本轮已经收齐全部目标层的 request；
        # 还未收齐的 request 会继续保留在 `_capture_fragments_by_request` 中等待补齐。
        return merged

    def _take_captured_hidden_states(
            self,
            engine,
            request_ids: list[str],
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        """
        从引擎收割一次已完成的 hidden-state 捕获结果
        """
        # 通过 collective RPC 主动把当前已完成请求的 hidden state 从 worker 侧取回，
        # 避免结果只停留在远端缓存，直到请求结束后再被动清理掉。
        return self._merge_captured_hidden_states(
            engine.collective_rpc(
                "take_predictor_hidden_states",
                args=(request_ids,),
            )
        )

    # ------------------------------- 执行单批 teacher 前向捕获 -------------------------------
    def capture_batch(self, batch: BatchShape) -> CapturedForwardBatch:
        # predictor teacher 训练必须显式提供 token rows；
        # 否则既无法提交真实 prompt，也无法和 teacher top-k 对齐。
        if not batch.has_token_rows:
            raise ValueError(
                "forward-capture traces require BatchShape.token_rows to be populated"
            )

        # 延迟导入推理输入与采样参数，避免仅构造对象时就拉起整套推理依赖。
        from cfie.inputs import TokensPrompt
        from cfie.sampling_params import RequestOutputKind, SamplingParams

        # ------------------------------- 懒创建引擎 ----------------------------------------
        # 先确保当前批次对应的 teacher 引擎已经存在且容量足够。
        engine = self._ensure_engine(batch)

        # teacher capture 只提交真实 token 前缀，尾部 padding 不进入真实前向。
        prompt_rows = self._effective_prompt_rows(batch)

        # ------------------------------- 提交逐行 prompt 请求 -------------------------------
        # 每一行 prompt 都拆成独立 request，便于后续按 request_id 精确回收
        request_ids: list[str] = []
        for row_index, prompt_row in enumerate(prompt_rows):
            # 用递增序号加行号构造 request_id，保证同一次训练进程内不会冲突。
            request_id = f"predictor-trace-{self._request_serial}-{row_index}"

            self._request_serial += 1

            request_ids.append(request_id)

            engine.add_request(
                request_id,
                # teacher 输入直接使用已分词 token ids，避免再次受文本模板影响。
                TokensPrompt(prompt_token_ids=list(prompt_row)),
                SamplingParams(
                    # teacher 只需要确定性最小生成，不需要任何采样随机性。
                    temperature=0.0,
                    top_p=1.0,
                    # 只多生成 1 个 token，用于驱动完整收尾并拿到最终 routed experts。
                    max_tokens=1,
                    # 训练侧只关心最终结果对象，不需要流式中间输出。
                    output_kind=RequestOutputKind.FINAL_ONLY,
                ),
            )

        # `request_outputs` 保存最终输出对象，后面从中读取 routed experts。
        request_outputs: dict[str, Any] = {}

        # `hidden_states_by_request` 保存每个请求按层捕获的 hidden states。
        hidden_states_by_request: dict[str, tuple[torch.Tensor, ...]] = {}
        try:
            # ------------------------------- 驱动真实引擎 step 并持续收割捕获结果 -------------------------------
            # 与真实推理链路保持一致
            while engine.has_unfinished_requests():
                for output in engine.step():
                    # 每个 step 可能返回多个请求的阶段性或最终结果；
                    # 这里只保留属于本批请求的输出对象。
                    request_id = getattr(output, "request_id", None)
                    if request_id in request_ids:
                        request_outputs[str(request_id)] = output

                # 有些请求的 prompt capture 会在中途 step 就先写入 worker 缓存；
                # 这里每轮都尝试取走一次，避免全部结束后部分结果已被后续流程清理。
                hidden_states_by_request.update(
                    self._take_captured_hidden_states(engine, request_ids)
                )

            # 全部请求结束后再补取一次，确保最后一批已完成捕获也被收回来。
            hidden_states_by_request.update(
                self._take_captured_hidden_states(engine, request_ids)
            )
        except Exception:
            # 一旦 teacher 前向中途失败，直接销毁当前引擎，
            # 避免残留半失效 capture 状态影响下一批训练。
            self._shutdown_engine()
            raise

        # ------------------------------- 按层汇总 hidden state 与 teacher top-k -------------------------------
        num_layers = self._config.model_spec.num_hidden_layers

        # 先为每一层准备一个收集桶，后面逐请求追加，再按层拼接。
        layer_hidden_states: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
        layer_teacher_topk_ids: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

        for row_index, request_id in enumerate(request_ids):
            # 先校验当前请求确实拿到了按层 hidden state。
            hidden_states = hidden_states_by_request.get(request_id)
            if hidden_states is None:
                raise ValueError(
                    f"predictor capture missing hidden states for request {request_id}"
                )
            if len(hidden_states) != num_layers:
                raise ValueError(
                    "captured hidden-state layer count does not match "
                    "model_spec.num_hidden_layers"
                )

            # 再校验最终输出对象存在，以便从中读取 routed experts 标签。
            request_output = request_outputs.get(request_id)
            if request_output is None or not getattr(request_output, "outputs", None):
                raise ValueError(
                    f"predictor capture missing final output for request {request_id}"
                )

            # 当前训练 teacher 只取第一条 completion；
            # 其 `routed_experts` 就是每个 token、每层对应的 teacher top-k 专家标签。
            completion = request_output.outputs[0]
            routed_experts = getattr(completion, "routed_experts", None)
            if routed_experts is None:
                raise ValueError(
                    f"predictor capture missing routed experts for request {request_id}"
                )

            # 统一转成 long tensor，便于后续按层切片并拼接到训练监督张量中。
            routed_experts_tensor = torch.as_tensor(
                routed_experts,
                dtype=torch.long,
            )
            if routed_experts_tensor.ndim != 3:
                raise ValueError(
                    "captured routed experts must have shape "
                    "[num_tokens, num_layers, topk]"  # 期望采样形状
                )
            if routed_experts_tensor.shape[1] != num_layers:
                raise ValueError(
                    "captured routed-expert layer count does not match "
                    "model_spec.num_hidden_layers"
                )

            # teacher 标签的 token 维必须与真实 prompt 长度严格一致，
            # 否则说明 capture 结果和提交请求已经失配。
            expected_num_tokens = len(prompt_rows[row_index])
            if routed_experts_tensor.shape[0] != expected_num_tokens:
                raise ValueError(
                    "captured routed experts token count does not match prompt length"
                )

            # 把当前请求的每层 hidden state 与 top-k 标签分别归档到对应层桶里，
            # 最终形成“按层训练”的监督布局。
            for layer_index in range(num_layers):
                layer_hidden_states[layer_index].append(hidden_states[layer_index])
                layer_teacher_topk_ids[layer_index].append(
                    routed_experts_tensor[:, layer_index, :]
                )

        # ------------------------------- 拼接层级监督结果并返回 -------------------------------
        # 每层把来自不同请求的 token 维结果顺序拼接，
        # 输出给后续 predictor 数据集构造与训练流程直接消费。
        return CapturedForwardBatch(
            layer_hidden_states=tuple(
                torch.cat(layer_hidden_states[layer_index], dim=0)
                for layer_index in range(num_layers)
            ),
            layer_teacher_topk_ids=tuple(
                torch.cat(layer_teacher_topk_ids[layer_index], dim=0)
                for layer_index in range(num_layers)
            ),
        )

    # ------------------------------- 析构时兜底回收 teacher 引擎 -------------------------------
    def __del__(self) -> None:
        # 即使调用方忘记显式关闭，这里也尽力把捕获状态与引擎资源回收掉。
        self._shutdown_engine()


@dataclass(slots=True, frozen=True)
class PredictorTraceExample:
    example_index: int
    step_index: int
    token_index: int
    insertion_layer_index: int
    future_layer_indices: tuple[int, ...]
    hidden_state: tuple[float, ...]
    future_teacher_topk_ids: tuple[tuple[int, ...], ...]

    # 将单条 predictor trace 样本序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出样本编号、插入层、hidden_state 和未来 teacher top-k 标签。
        return {
            "example_index": self.example_index,
            "step_index": self.step_index,
            "token_index": self.token_index,
            "insertion_layer_index": self.insertion_layer_index,
            "future_layer_indices": list(self.future_layer_indices),
            "hidden_state": list(self.hidden_state),
            "future_teacher_topk_ids": [
                list(expert_ids) for expert_ids in self.future_teacher_topk_ids
            ],
        }

    @classmethod
    # 从字典恢复单条 predictor trace 样本。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorTraceExample":
        # 逐字段恢复样本索引、未来层索引、hidden_state 和 top-k 标签；
        # 同时兼容旧版数据里的 hidden_summary 字段。
        return cls(
            example_index=int(payload["example_index"]),
            step_index=int(payload["step_index"]),
            token_index=int(payload.get("token_index", 0)),
            insertion_layer_index=int(payload["insertion_layer_index"]),
            future_layer_indices=tuple(
                int(layer_index) for layer_index in payload["future_layer_indices"]
            ),
            hidden_state=tuple(
                float(value)
                for value in payload.get(
                    "hidden_state",
                    payload.get("hidden_summary", ()),
                )
            ),
            future_teacher_topk_ids=tuple(
                tuple(int(expert_id) for expert_id in expert_ids)
                for expert_ids in payload["future_teacher_topk_ids"]
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorTraceDataset:
    profile_name: str
    example_count: int
    window_layers: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    examples: tuple[PredictorTraceExample, ...]

    # 将 predictor trace 数据集序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出数据集来源、窗口配置和所有样本。
        return {
            "profile_name": self.profile_name,
            "example_count": self.example_count,
            "window_layers": self.window_layers,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "examples": [example.to_dict() for example in self.examples],
        }

    # 将 predictor trace 数据集导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将 predictor trace 数据集写入 JSON 文件。
    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        # 直接把 JSON 文本写到目标路径。
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    # 从字典恢复 predictor trace 数据集。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorTraceDataset":
        # 先逐条恢复 examples 列表。
        examples = tuple(
            PredictorTraceExample.from_dict(example)
            for example in payload.get("examples", [])
        )
        # 再恢复数据集元信息和 examples 元组。
        return cls(
            profile_name=str(payload["profile_name"]),
            example_count=int(payload.get("example_count", len(examples))),
            window_layers=int(payload["window_layers"]),
            candidate_experts_per_layer=int(payload["candidate_experts_per_layer"]),
            executed_experts_per_layer=int(payload["executed_experts_per_layer"]),
            examples=examples,
        )

    @classmethod
    # 从 JSON 文件恢复 predictor trace 数据集。
    def from_json_file(cls, path: str | Path) -> "PredictorTraceDataset":
        # 读取并解析 JSON 文本。
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # 顶层必须解析成对象。
        if not isinstance(payload, dict):
            raise ValueError("predictor trace dataset JSON must decode to an object")
        # 继续按字典格式恢复数据集。
        return cls.from_dict(payload)


@dataclass(slots=True, frozen=True)
class PredictorEpochSummary:
    epoch_index: int
    mean_loss: float
    recall_at_candidate_budget: float
    recall_at_executed_budget: float

    # 将单个 epoch 汇总序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 loss 和两档 recall 指标。
        return {
            "epoch_index": self.epoch_index,
            "mean_loss": self.mean_loss,
            "recall_at_candidate_budget": self.recall_at_candidate_budget,
            "recall_at_executed_budget": self.recall_at_executed_budget,
        }

    @classmethod
    # 从字典恢复单个 epoch 汇总。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorEpochSummary":
        # 逐字段恢复 epoch 编号、loss 与两档 recall。
        return cls(
            epoch_index=int(payload["epoch_index"]),
            mean_loss=float(payload["mean_loss"]),
            recall_at_candidate_budget=float(
                payload["recall_at_candidate_budget"]
            ),
            recall_at_executed_budget=float(
                payload["recall_at_executed_budget"]
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorTrainingRunTrace:
    profile_name: str
    example_count: int
    epochs: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    epoch_summaries: tuple[PredictorEpochSummary, ...]

    @property
    # 返回最后一个 epoch 的平均损失。
    def final_mean_loss(self) -> float:
        # 没有 epoch 时退回 0。
        return self.epoch_summaries[-1].mean_loss if self.epoch_summaries else 0.0

    @property
    # 返回最后一个 epoch 的 candidate budget recall。
    def final_recall_at_candidate_budget(self) -> float:
        # 没有 epoch 时退回 0。
        if not self.epoch_summaries:
            return 0.0
        # 否则取最后一个 epoch 的 recall。
        return self.epoch_summaries[-1].recall_at_candidate_budget

    @property
    # 返回最后一个 epoch 的 executed budget recall。
    def final_recall_at_executed_budget(self) -> float:
        # 没有 epoch 时退回 0。
        if not self.epoch_summaries:
            return 0.0
        # 否则取最后一个 epoch 的 recall。
        return self.epoch_summaries[-1].recall_at_executed_budget

    # 将整个 predictor 训练 run 序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出训练来源、epoch 数、最终指标和逐 epoch 汇总。
        return {
            "profile_name": self.profile_name,
            "example_count": self.example_count,
            "epochs": self.epochs,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "final_mean_loss": self.final_mean_loss,
            "final_recall_at_candidate_budget": (
                self.final_recall_at_candidate_budget
            ),
            "final_recall_at_executed_budget": (
                self.final_recall_at_executed_budget
            ),
            "epoch_summaries": [
                summary.to_dict() for summary in self.epoch_summaries
            ],
        }

    # 将 predictor 训练 run 导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    # 从字典恢复 predictor 训练 run。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorTrainingRunTrace":
        # 先恢复逐 epoch 汇总列表。
        epoch_summaries = tuple(
            PredictorEpochSummary.from_dict(summary)
            for summary in payload.get("epoch_summaries", [])
        )
        # 再恢复训练来源、预算和总 epoch 数。
        return cls(
            profile_name=str(payload["profile_name"]),
            example_count=int(payload["example_count"]),
            epochs=int(payload["epochs"]),
            candidate_experts_per_layer=int(
                payload["candidate_experts_per_layer"]
            ),
            executed_experts_per_layer=int(
                payload["executed_experts_per_layer"]
            ),
            epoch_summaries=epoch_summaries,
        )


@dataclass(slots=True, frozen=True)
class PredictorCheckpointMetadata:
    checkpoint_kind: str
    profile_name: str
    input_summary_dim: int
    hidden_dim: int
    window_layers: int
    stride_layers: int
    num_experts: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    selection_mode: str
    online_expert_source: str
    allow_candidate_mismatch: bool
    example_count: int
    epochs: int
    final_mean_loss: float
    final_recall_at_candidate_budget: float
    final_recall_at_executed_budget: float

    # 将 predictor checkpoint 元信息序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 checkpoint 对运行时兼容性和最终指标有影响的全部字段。
        return {
            "checkpoint_kind": self.checkpoint_kind,
            "profile_name": self.profile_name,
            "input_summary_dim": self.input_summary_dim,
            "hidden_dim": self.hidden_dim,
            "window_layers": self.window_layers,
            "stride_layers": self.stride_layers,
            "num_experts": self.num_experts,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "selection_mode": self.selection_mode,
            "online_expert_source": self.online_expert_source,
            "allow_candidate_mismatch": self.allow_candidate_mismatch,
            "example_count": self.example_count,
            "epochs": self.epochs,
            "final_mean_loss": self.final_mean_loss,
            "final_recall_at_candidate_budget": (
                self.final_recall_at_candidate_budget
            ),
            "final_recall_at_executed_budget": (
                self.final_recall_at_executed_budget
            ),
        }

    @classmethod
    # 从字典恢复 predictor checkpoint 元信息。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorCheckpointMetadata":
        # 逐字段恢复 checkpoint 兼容性约束和最终指标。
        return cls(
            checkpoint_kind=str(
                payload.get("checkpoint_kind", "cfie_predictor_checkpoint")
            ),
            profile_name=str(payload["profile_name"]),
            input_summary_dim=int(payload["input_summary_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            window_layers=int(payload["window_layers"]),
            stride_layers=int(payload["stride_layers"]),
            num_experts=int(payload["num_experts"]),
            candidate_experts_per_layer=int(
                payload["candidate_experts_per_layer"]
            ),
            executed_experts_per_layer=int(
                payload["executed_experts_per_layer"]
            ),
            selection_mode=str(payload["selection_mode"]),
            online_expert_source=str(payload["online_expert_source"]),
            allow_candidate_mismatch=bool(
                payload.get("allow_candidate_mismatch", True)
            ),
            example_count=int(payload["example_count"]),
            epochs=int(payload["epochs"]),
            final_mean_loss=float(payload["final_mean_loss"]),
            final_recall_at_candidate_budget=float(
                payload["final_recall_at_candidate_budget"]
            ),
            final_recall_at_executed_budget=float(
                payload.get("final_recall_at_executed_budget", 0.0)
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorRuntimeSchema:
    schema_kind: str
    profile_name: str
    input_summary_dim: int
    predictor_hidden_dim: int
    window_layers: int
    stride_layers: int
    num_experts: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    selection_mode: str
    online_expert_source: str
    allow_candidate_mismatch: bool

    # 将 predictor runtime schema 序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出运行时推理侧需要的 predictor 结构与路由约束。
        return {
            "schema_kind": self.schema_kind,
            "profile_name": self.profile_name,
            "input_summary_dim": self.input_summary_dim,
            "predictor_hidden_dim": self.predictor_hidden_dim,
            "window_layers": self.window_layers,
            "stride_layers": self.stride_layers,
            "num_experts": self.num_experts,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "selection_mode": self.selection_mode,
            "online_expert_source": self.online_expert_source,
            "allow_candidate_mismatch": self.allow_candidate_mismatch,
        }

    # 将 predictor runtime schema 导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将 predictor runtime schema 写入 JSON 文件。
    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        # 直接把 JSON 文本写到目标路径。
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    # 基于 checkpoint 元信息构造运行时 schema。
    def from_checkpoint_metadata(
            cls,
            metadata: PredictorCheckpointMetadata,
    ) -> "PredictorRuntimeSchema":
        # 把 checkpoint 中的兼容性字段映射到 runtime schema。
        return cls(
            schema_kind="cfie_predictor_runtime_schema",
            profile_name=metadata.profile_name,
            input_summary_dim=metadata.input_summary_dim,
            predictor_hidden_dim=metadata.hidden_dim,
            window_layers=metadata.window_layers,
            stride_layers=metadata.stride_layers,
            num_experts=metadata.num_experts,
            candidate_experts_per_layer=metadata.candidate_experts_per_layer,
            executed_experts_per_layer=metadata.executed_experts_per_layer,
            selection_mode=metadata.selection_mode,
            online_expert_source=metadata.online_expert_source,
            allow_candidate_mismatch=metadata.allow_candidate_mismatch,
        )

    @classmethod
    # 从字典恢复 predictor runtime schema。
    def from_dict(cls, payload: dict[str, Any]) -> "PredictorRuntimeSchema":
        # 逐字段恢复运行时 schema 内容。
        return cls(
            schema_kind=str(
                payload.get("schema_kind", "cfie_predictor_runtime_schema")
            ),
            profile_name=str(payload["profile_name"]),
            input_summary_dim=int(payload["input_summary_dim"]),
            predictor_hidden_dim=int(payload["predictor_hidden_dim"]),
            window_layers=int(payload["window_layers"]),
            stride_layers=int(payload["stride_layers"]),
            num_experts=int(payload["num_experts"]),
            candidate_experts_per_layer=int(
                payload["candidate_experts_per_layer"]
            ),
            executed_experts_per_layer=int(
                payload["executed_experts_per_layer"]
            ),
            selection_mode=str(payload["selection_mode"]),
            online_expert_source=str(payload["online_expert_source"]),
            allow_candidate_mismatch=bool(
                payload.get("allow_candidate_mismatch", True)
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorEvaluationTrace:
    profile_name: str
    example_count: int
    candidate_experts_per_layer: int
    executed_experts_per_layer: int
    mean_loss: float
    recall_at_candidate_budget: float
    recall_at_executed_budget: float
    checkpoint_metadata: PredictorCheckpointMetadata | None = None

    # 将 predictor 评估结果序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出评估来源、loss、recall，以及可选的 checkpoint 元信息。
        return {
            "profile_name": self.profile_name,
            "example_count": self.example_count,
            "candidate_experts_per_layer": self.candidate_experts_per_layer,
            "executed_experts_per_layer": self.executed_experts_per_layer,
            "mean_loss": self.mean_loss,
            "recall_at_candidate_budget": self.recall_at_candidate_budget,
            "recall_at_executed_budget": self.recall_at_executed_budget,
            "checkpoint_metadata": (
                None
                if self.checkpoint_metadata is None
                else self.checkpoint_metadata.to_dict()
            ),
        }

    # 将 predictor 评估结果导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(slots=True, frozen=True)
class PredictorMetricsSummary:
    metrics_kind: str
    profile_name: str
    example_count: int
    epochs: int
    final_mean_loss: float
    final_recall_at_candidate_budget: float
    final_recall_at_executed_budget: float

    # 将 predictor 指标摘要序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出最终 loss 与 recall 指标摘要。
        return {
            "metrics_kind": self.metrics_kind,
            "profile_name": self.profile_name,
            "example_count": self.example_count,
            "epochs": self.epochs,
            "final_mean_loss": self.final_mean_loss,
            "final_recall_at_candidate_budget": (
                self.final_recall_at_candidate_budget
            ),
            "final_recall_at_executed_budget": (
                self.final_recall_at_executed_budget
            ),
        }

    # 将 predictor 指标摘要导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将 predictor 指标摘要写入 JSON 文件。
    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        # 直接把 JSON 文本写到目标路径。
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    @classmethod
    # 根据 checkpoint 元信息构造指标摘要。
    def from_checkpoint_metadata(
            cls,
            metadata: PredictorCheckpointMetadata,
    ) -> "PredictorMetricsSummary":
        # 直接把 checkpoint 中的最终指标映射成 metrics summary。
        return cls(
            metrics_kind="cfie_predictor_metrics_summary",
            profile_name=metadata.profile_name,
            example_count=metadata.example_count,
            epochs=metadata.epochs,
            final_mean_loss=metadata.final_mean_loss,
            final_recall_at_candidate_budget=(
                metadata.final_recall_at_candidate_budget
            ),
            final_recall_at_executed_budget=(
                metadata.final_recall_at_executed_budget
            ),
        )


@dataclass(slots=True, frozen=True)
class PredictorDeploymentManifest:
    bundle_kind: str
    profile_name: str
    source_checkpoint: str
    weights_kind: str
    weights_format: str
    weights_file: str
    schema_kind: str
    schema_file: str
    metrics_kind: str
    metrics_file: str

    # 将 predictor 部署清单序列化为字典。
    def to_dict(self) -> dict[str, Any]:
        # 输出 bundle 中各个文件及其语义类型。
        return {
            "bundle_kind": self.bundle_kind,
            "profile_name": self.profile_name,
            "source_checkpoint": self.source_checkpoint,
            "weights_kind": self.weights_kind,
            "weights_format": self.weights_format,
            "weights_file": self.weights_file,
            "schema_kind": self.schema_kind,
            "schema_file": self.schema_file,
            "metrics_kind": self.metrics_kind,
            "metrics_file": self.metrics_file,
        }

    # 将 predictor 部署清单导出为 JSON。
    def to_json(self, *, indent: int = 2) -> str:
        # 使用稳定排序键和指定缩进导出 JSON 文本。
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    # 将 predictor 部署清单写入 JSON 文件。
    def write_json(self, path: str | Path, *, indent: int = 2) -> None:
        # 直接把 JSON 文本写到目标路径。
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")


class FutureExpertPredictor(nn.Module):
    def __init__(
            self,
            *,
            input_dim: int,
            hidden_dim: int,
            window_layers: int,
            num_experts: int,
    ) -> None:
        super().__init__()
        self.window_layers = window_layers
        self.num_experts = num_experts
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layer_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, window_layers * num_experts),
        )

    @staticmethod
    def _normalize_hidden_state(hidden_state: torch.Tensor) -> torch.Tensor:
        if hidden_state.ndim == 1:
            return hidden_state.unsqueeze(0)
        if hidden_state.ndim != 2:
            raise ValueError("hidden_state must be rank-1 or rank-2")
        return hidden_state

    @staticmethod
    def _normalize_layer_index(
            layer_index: int | torch.Tensor,
            *,
            batch_size: int,
            device: torch.device,
    ) -> torch.Tensor:
        if isinstance(layer_index, int):
            return torch.full(
                (batch_size,),
                int(layer_index),
                dtype=torch.float32,
                device=device,
            )
        if layer_index.ndim == 0:
            return torch.full(
                (batch_size,),
                int(layer_index.item()),
                dtype=torch.float32,
                device=device,
            )
        if layer_index.ndim != 1 or int(layer_index.shape[0]) != batch_size:
            raise ValueError("layer_index must be scalar or match batch size")
        return layer_index.to(device=device, dtype=torch.float32)

    @staticmethod
    def _layer_features(layer_index: torch.Tensor) -> torch.Tensor:
        layer_index = layer_index.unsqueeze(-1)
        return torch.cat(
            (
                layer_index,
                layer_index.square(),
                torch.sin(layer_index * 0.1),
                torch.cos(layer_index * 0.1),
                torch.sin(layer_index * 0.01),
                torch.cos(layer_index * 0.01),
            ),
            dim=-1,
        )

    def forward(
            self,
            hidden_state: torch.Tensor,
            layer_index: int | torch.Tensor,
    ) -> torch.Tensor:
        hidden_state = self._normalize_hidden_state(hidden_state)
        layer_index = self._normalize_layer_index(
            layer_index,
            batch_size=int(hidden_state.shape[0]),
            device=hidden_state.device,
        )
        fused_hidden = self.input_proj(hidden_state) + self.layer_proj(
            self._layer_features(layer_index)
        )
        logits = self.net(fused_hidden)
        return logits.view(-1, self.window_layers, self.num_experts)


HiddenStateFutureExpertPredictor = FutureExpertPredictor


class PredictorTraceBuilderBase:
    # ------------------------------- 初始化 trace 构建公共上下文 -------------------------------
    def __init__(self, config: TrainingProjectConfig) -> None:
        # 先校验并缓存训练配置，确保后续字段读取都基于合法配置。
        self.config = config.validate()
        # predictor trace 生成必须依赖明确的模型几何信息。
        if not self.config.model_spec.is_defined():
            raise ValueError(
                "predictor trace generation requires a defined model_spec"
            )
        # 缓存 hidden 维度，后续会用于校验捕获到的 hidden_state 形状。
        self._hidden_dim = self.config.model_spec.hidden_size
        # 缓存模型层数，后续用于校验按层捕获结果是否完整。
        self._num_layers = self.config.model_spec.num_hidden_layers
        # 未来窗口层数决定每个样本需要收集多少个 future layer 标签。
        self._window_layers = self.config.predictor_routing.window_layers
        # 执行预算决定每个未来层取多少个 teacher top-k experts。
        self._executed_experts = (
            self.config.predictor_routing.executed_experts_per_layer
        )
        # stride 控制可作为插入层的采样步长。
        self._stride_layers = self.config.predictor_routing.stride_layers
        # 预先生成可选插入层索引集合，避免每步重复计算。
        self._insertion_layer_indices = tuple(
            range(
                0,
                max(self._num_layers - self._window_layers, 0),
                self._stride_layers,
            )
        )

    # ------------------------------- 解析每步样本数量 -------------------------------
    def _resolve_examples_per_step(
            self,
            examples_per_step: int | None,
    ) -> int:
        # 未显式传入时回退到配置默认值；传入时统一转成 int。
        resolved = (
            self.config.predictor_trainer.examples_per_step
            if examples_per_step is None
            else int(examples_per_step)
        )
        # 每步样本数量至少为 1，防止生成空监督集。
        if resolved < 1:
            raise ValueError("examples_per_step must be >= 1")
        return resolved

    # ------------------------------- 生成当前 step 的样本选点规格 -------------------------------
    def _selected_example_specs(
            self,
            *,
            step_index: int,
            examples_per_step: int,
            token_count: int,
    ) -> tuple[tuple[int, int, int], ...]:
        # 若没有任何可用插入层，则无法构造监督样本。
        if not self._insertion_layer_indices:
            raise ValueError(
                "predictor trace generation requires at least one eligible insertion layer"
            )
        # token 数量至少按 1 处理，避免异常输入导致除零或空槽位。
        token_count = max(int(token_count), 1)
        # 总槽位 = 可用插入层数 × token 行数。
        total_slots = len(self._insertion_layer_indices) * token_count
        # 实际样本数不超过总槽位，避免越界选点。
        example_count = min(examples_per_step, total_slots)
        # 通过 step 偏移实现跨 step 的轮转采样，提升覆盖率。
        start = (step_index * example_count) % total_slots
        # 每条规格格式为 (example_offset, insertion_layer_index, token_index)。
        specs: list[tuple[int, int, int]] = []
        for example_offset in range(example_count):
            # 先在线性槽位空间中定位当前位置，再映射回 (token, layer) 二维索引。
            flat_index = (start + example_offset) % total_slots
            token_index = flat_index // len(self._insertion_layer_indices)
            insertion_index = flat_index % len(self._insertion_layer_indices)
            specs.append(
                (
                    example_offset,
                    self._insertion_layer_indices[insertion_index],
                    token_index,
                )
            )
        return tuple(specs)

    # ------------------------------- 计算插入层对应的 future 窗口层 -------------------------------
    def _future_layer_indices(
            self,
            insertion_layer_index: int,
    ) -> tuple[int, ...]:
        # 从插入层下一层开始，连续取 window_layers 个 future 层索引。
        return tuple(
            insertion_layer_index + offset + 1
            for offset in range(self._window_layers)
        )


class ForwardCaptureTraceBuilder(PredictorTraceBuilderBase):
    # ------------------------------- 初始化 forward-capture 轨迹构建器 -------------------------------
    def __init__(
            self,
            config: TrainingProjectConfig,
            teacher_model_backend: PredictorTeacherModelBackend,
    ) -> None:
        # 先复用基类初始化，完成层数、窗口、预算等公共元信息的准备。
        super().__init__(config)
        # 缓存 teacher backend，用于按 batch 执行真实前向并抓取中间信号。
        self._teacher_model_backend = teacher_model_backend

    # ------------------------------- 构造 predictor 监督样本 -------------------------------
    def build_examples(
            self,
            *,
            steps: int,
            examples_per_step: int | None = None,
            batch_planner: PredictorBatchPlanner | None = None,
    ) -> tuple[PredictorTraceExample, ...]:
        # 至少需要执行 1 个 step，0 step 无法产出任何监督样本。
        if steps < 1:
            raise ValueError("steps must be >= 1")

        # forward-capture 路径必须依赖 batch planner 提供真实 token 批次。
        if batch_planner is None:
            raise ValueError(
                "forward-capture predictor traces require a batch planner"
            )

        # 将可选的 examples_per_step 解析为最终生效的每步样本数量。
        resolved_examples_per_step = self._resolve_examples_per_step(
            examples_per_step
        )

        # 累积所有 step 生成的 predictor 监督样本。
        examples: list[PredictorTraceExample] = []

        # ------------------------------- 按 step 抓取 teacher 前向并生成样本 -------------------------------
        for step_index in range(steps):
            # 从 batch planner 获取当前 step 对应的输入批次。
            batch = batch_planner.batch_for_step(step_index)

            # 执行 teacher 前向捕获，得到各层 hidden_state 与 teacher top-k experts。
            captured = self._teacher_model_backend.capture_batch(batch)

            # hidden_state 层数必须与配置层数一致，避免样本与模型形状错配。
            if len(captured.layer_hidden_states) != self._num_layers:
                raise ValueError(
                    "captured hidden-state layer count does not match model_spec.num_hidden_layers"
                )

            # teacher top-k 层数同样必须与配置层数一致。
            if len(captured.layer_teacher_topk_ids) != self._num_layers:
                raise ValueError(
                    "captured teacher-topk layer count does not match "
                    "model_spec.num_hidden_layers"
                )

            # token_count 由第 0 层 hidden_state 的 token 维给出。
            token_count = int(captured.layer_hidden_states[0].shape[0])

            # ------------------------------- 选择插入点并提取单条样本 -------------------------------
            # _selected_example_specs 返回 (example_offset, insertion_layer_index, token_index)。
            # 这里 example_offset 不参与最终样本字段，因此用 "_" 丢弃。
            for _, insertion_layer_index, token_index in self._selected_example_specs(
                    step_index=step_index,
                    examples_per_step=resolved_examples_per_step,
                    token_count=token_count,
            ):
                # 读取插入层在指定 token 上的 hidden_state，作为 predictor 输入特征。
                hidden_state = captured.layer_hidden_states[insertion_layer_index][
                    token_index
                ]
                # 输入特征维度必须与配置 hidden_size 一致。
                if hidden_state.numel() != self._hidden_dim:
                    raise ValueError(
                        "captured hidden_state size does not match model_spec.hidden_size"
                    )
                # 根据插入层位置推导未来窗口层索引集合。
                future_layer_indices = self._future_layer_indices(
                    insertion_layer_index
                )
                # 收集未来每层 teacher top-k experts，作为多标签监督目标。
                future_teacher_topk_ids = []
                for future_layer_index in future_layer_indices:
                    # teacher 后端已经返回真实前向里的执行专家 top-k，
                    # 这里直接读取即可，避免再从全量 logits 二次推导。
                    teacher_topk = captured.layer_teacher_topk_ids[
                        future_layer_index
                    ][token_index]
                    future_teacher_topk_ids.append(
                        tuple(int(expert_id) for expert_id in teacher_topk.tolist())
                    )

                # ------------------------------- 组装并写入样本 -------------------------------
                # example_index 使用当前累积长度，保证全局连续递增。
                examples.append(
                    PredictorTraceExample(
                        example_index=len(examples),
                        step_index=step_index,
                        token_index=token_index,
                        insertion_layer_index=insertion_layer_index,
                        future_layer_indices=future_layer_indices,
                        # hidden_state 统一转为 float 元组，保持序列化与训练读回一致。
                        hidden_state=tuple(
                            float(value) for value in hidden_state.tolist()
                        ),
                        # future top-k 监督信号转为不可变元组结构，避免后续被意外修改。
                        future_teacher_topk_ids=tuple(future_teacher_topk_ids),
                    )
                )
        # 返回不可变样本元组，供训练/评估流程直接消费。
        return tuple(examples)


class PredictorTrainer:
    def __init__(
            self,
            config: TrainingProjectConfig,
            *,
            teacher_model_backend: PredictorTeacherModelBackend | None = None,
    ) -> None:
        # -------------------- 校验并缓存训练配置 --------------------
        # 训练器构造完成后会立即依赖 model_spec、routing 与 trainer 超参，
        # 因此这里先做一次全量配置校验，避免带着不完整配置进入后续流程。
        self.config = config.validate()

        # predictor 训练需要明确知道层数、expert 数和 hidden 维度；
        # 若 model_spec 未定义，就无法构造样本与模型结构。
        if not self.config.model_spec.is_defined():
            raise ValueError("predictor training requires a defined model_spec")

        self._teacher_model_backend = teacher_model_backend
        self._trace_builder: ForwardCaptureTraceBuilder | None = None

    def _resolve_teacher_model_backend(
            self,
    ) -> PredictorTeacherModelBackend:
        # ------------------------------- 懒加载 teacher backend -------------------------------
        # 若调用方未注入 backend，这里按默认实现创建一次并缓存复用。
        if self._teacher_model_backend is None:
            self._teacher_model_backend = EngineRouterTeacherModelBackend.create(
                self.config
            )
        # 返回当前训练器绑定的 teacher backend 实例。
        return self._teacher_model_backend

    def _resolve_trace_builder(self) -> ForwardCaptureTraceBuilder:
        # ------------------------------- 懒加载 trace builder -------------------------------
        # trace builder 依赖 teacher backend，因此首次访问时统一在这里串联初始化。
        if self._trace_builder is None:
            self._trace_builder = ForwardCaptureTraceBuilder(
                self.config,
                self._resolve_teacher_model_backend(),
            )
        # 返回当前训练器绑定的 trace builder 实例。
        return self._trace_builder

    def _build_batch_planner(
            self,
            *,
            samples: int,
            tokens_per_sample: int,
            dataset_path: str | None = None,
            tokenizer_path: str | None = None,
            dataset_format: str = "auto",
            dataset_text_key: str = "text",
    ) -> PredictorBatchPlanner:
        # ------------------------------- 构造 predictor trace 采样批规划器 -------------------------------
        # predictor 轨迹采样依赖真实数据集驱动的 token 批次，
        # 因此这里强制要求提供 dataset_path，避免进入无数据源的空规划路径。
        if dataset_path is None:
            raise ValueError(
                "predictor trace generation requires a dataset-backed batch planner; "
                "pass dataset_path/--dataset"
            )

        # 统一封装为 TokenizedDatasetBatchPlanner，供后续 trace builder 连续取批。
        return TokenizedDatasetBatchPlanner(
            config=self.config,
            dataset_path=dataset_path,
            base_samples=samples,
            tokens_per_sample=tokens_per_sample,
            tokenizer_path=tokenizer_path,
            dataset_format=dataset_format,
            dataset_text_key=dataset_text_key,
        )

    def build_model(self) -> FutureExpertPredictor:
        # -------------------- 读取决定 predictor 形状的关键配置 --------------------
        # trainer 配置决定隐藏层宽度；
        # 输入维度当前直接取 model hidden_size。
        trainer_cfg = self.config.predictor_trainer
        # routing 配置决定未来窗口层数，也就决定输出张量的第二维大小。
        routing_cfg = self.config.predictor_routing

        # model_spec 里的 num_experts 决定每个未来层要预测多少个 expert logit。
        # 这里三类配置共同决定 predictor 的完整张量形状。
        return FutureExpertPredictor(
            input_dim=self.config.model_spec.hidden_size,
            hidden_dim=trainer_cfg.hidden_dim,
            window_layers=routing_cfg.window_layers,
            num_experts=self.config.model_spec.num_experts,
        )

    def build_runtime_schema(self) -> PredictorRuntimeSchema:
        # 读取 routing 与 trainer 配置。
        routing_cfg = self.config.predictor_routing
        trainer_cfg = self.config.predictor_trainer
        # 组装当前配置对应的 runtime schema。
        return PredictorRuntimeSchema(
            schema_kind="cfie_predictor_runtime_schema",
            profile_name=self.config.profile_name,
            input_summary_dim=self.config.model_spec.hidden_size,
            predictor_hidden_dim=trainer_cfg.hidden_dim,
            window_layers=routing_cfg.window_layers,
            stride_layers=routing_cfg.stride_layers,
            num_experts=self.config.model_spec.num_experts,
            candidate_experts_per_layer=routing_cfg.candidate_experts_per_layer,
            executed_experts_per_layer=routing_cfg.executed_experts_per_layer,
            selection_mode=routing_cfg.selection_mode,
            online_expert_source=routing_cfg.online_expert_source,
            allow_candidate_mismatch=routing_cfg.allow_candidate_mismatch,
        )

    @staticmethod
    # 从 checkpoint 文件读取原始载荷。
    def _read_checkpoint_payload(path: str | Path) -> dict[str, Any]:
        # 使用 torch.load 在 CPU 上读取 checkpoint。
        payload = torch.load(Path(path), map_location="cpu")
        # 顶层必须解码成字典。
        if not isinstance(payload, dict):
            raise ValueError("predictor checkpoint must decode to a dictionary")
        # 返回原始 payload。
        return payload

    @classmethod
    # 只读取 predictor checkpoint 的 metadata 部分。
    def read_checkpoint_metadata(
            cls,
            path: str | Path,
    ) -> PredictorCheckpointMetadata:
        # 先读取 checkpoint 原始载荷。
        payload = cls._read_checkpoint_payload(path)
        # 取出 metadata 部分。
        metadata_payload = payload.get("metadata")
        # metadata 必须存在且为字典。
        if not isinstance(metadata_payload, dict):
            raise ValueError("predictor checkpoint is missing metadata")
        # 继续恢复为 PredictorCheckpointMetadata。
        return PredictorCheckpointMetadata.from_dict(metadata_payload)

    @classmethod
    # 只读取 predictor checkpoint 的 run_trace 部分。
    def read_checkpoint_run_trace(
            cls,
            path: str | Path,
    ) -> PredictorTrainingRunTrace | None:
        # 先读取 checkpoint 原始载荷。
        payload = cls._read_checkpoint_payload(path)
        # 取出可选的 run_trace 部分。
        run_trace_payload = payload.get("run_trace")
        # 旧 checkpoint 没有 run_trace 时直接返回空。
        if run_trace_payload is None:
            return None
        # run_trace 若存在，则必须是字典。
        if not isinstance(run_trace_payload, dict):
            raise ValueError("predictor checkpoint run_trace must be a dictionary")
        # 继续恢复为 PredictorTrainingRunTrace。
        return PredictorTrainingRunTrace.from_dict(run_trace_payload)

    def _checkpoint_metadata(
            self,
            run_trace: PredictorTrainingRunTrace,
    ) -> PredictorCheckpointMetadata:
        # 读取 routing 与 trainer 配置。
        routing_cfg = self.config.predictor_routing
        trainer_cfg = self.config.predictor_trainer
        # 根据 run_trace 和当前配置组装 checkpoint 元信息。
        return PredictorCheckpointMetadata(
            checkpoint_kind="cfie_predictor_checkpoint",
            profile_name=run_trace.profile_name,
            input_summary_dim=self.config.model_spec.hidden_size,
            hidden_dim=trainer_cfg.hidden_dim,
            window_layers=routing_cfg.window_layers,
            stride_layers=routing_cfg.stride_layers,
            num_experts=self.config.model_spec.num_experts,
            candidate_experts_per_layer=routing_cfg.candidate_experts_per_layer,
            executed_experts_per_layer=routing_cfg.executed_experts_per_layer,
            selection_mode=routing_cfg.selection_mode,
            online_expert_source=routing_cfg.online_expert_source,
            allow_candidate_mismatch=routing_cfg.allow_candidate_mismatch,
            example_count=run_trace.example_count,
            epochs=run_trace.epochs,
            final_mean_loss=run_trace.final_mean_loss,
            final_recall_at_candidate_budget=(
                run_trace.final_recall_at_candidate_budget
            ),
            final_recall_at_executed_budget=(
                run_trace.final_recall_at_executed_budget
            ),
        )

    def _validate_checkpoint_metadata(
            self,
            metadata: PredictorCheckpointMetadata,
    ) -> None:
        # 读取当前 trainer 与 routing 配置。
        trainer_cfg = self.config.predictor_trainer
        routing_cfg = self.config.predictor_routing
        # 用列表累积所有不兼容字段。
        mismatches = []
        if metadata.input_summary_dim != self.config.model_spec.hidden_size:
            mismatches.append("input_summary_dim")
        if metadata.hidden_dim != trainer_cfg.hidden_dim:
            mismatches.append("hidden_dim")
        if metadata.window_layers != routing_cfg.window_layers:
            mismatches.append("window_layers")
        if metadata.stride_layers != routing_cfg.stride_layers:
            mismatches.append("stride_layers")
        if metadata.num_experts != self.config.model_spec.num_experts:
            mismatches.append("num_experts")
        if (
                metadata.candidate_experts_per_layer
                != routing_cfg.candidate_experts_per_layer
        ):
            mismatches.append("candidate_experts_per_layer")
        if (
                metadata.executed_experts_per_layer
                != routing_cfg.executed_experts_per_layer
        ):
            mismatches.append("executed_experts_per_layer")
        # 只要存在任一关键字段不匹配，就拒绝加载 checkpoint。
        if mismatches:
            raise ValueError(
                "predictor checkpoint is incompatible with current config: "
                + ", ".join(mismatches)
            )

    def _validate_resume_run_trace(
            self,
            run_trace: PredictorTrainingRunTrace,
    ) -> None:
        # 先校验 profile/source 与当前训练器一致。
        mismatches = []
        if run_trace.profile_name != self.config.profile_name:
            mismatches.append("profile_name")
        if run_trace.candidate_experts_per_layer != (
                self.config.predictor_routing.candidate_experts_per_layer
        ):
            mismatches.append("candidate_experts_per_layer")
        if run_trace.executed_experts_per_layer != (
                self.config.predictor_routing.executed_experts_per_layer
        ):
            mismatches.append("executed_experts_per_layer")
        # 若 run_trace 自身与当前配置不兼容，则拒绝作为续训起点。
        if mismatches:
            raise ValueError(
                "predictor checkpoint run_trace is incompatible with current config: "
                + ", ".join(mismatches)
            )

    def _validate_resume_dataset(
            self,
            dataset: PredictorTraceDataset,
            run_trace: PredictorTrainingRunTrace,
    ) -> None:
        # 续训时要求数据集来源与历史 run_trace 一致。
        mismatches = []
        if dataset.profile_name != run_trace.profile_name:
            mismatches.append("profile_name")
        if dataset.example_count != run_trace.example_count:
            mismatches.append("example_count")
        if dataset.candidate_experts_per_layer != (
                run_trace.candidate_experts_per_layer
        ):
            mismatches.append("candidate_experts_per_layer")
        if dataset.executed_experts_per_layer != (
                run_trace.executed_experts_per_layer
        ):
            mismatches.append("executed_experts_per_layer")
        # 数据集与续训起点不一致时直接报错。
        if mismatches:
            raise ValueError(
                "predictor resume dataset is incompatible with checkpoint run_trace: "
                + ", ".join(mismatches)
            )

    def save_checkpoint(
            self,
            *,
            model: FutureExpertPredictor,
            run_trace: PredictorTrainingRunTrace,
            path: str | Path,
            optimizer_state_dict: dict[str, Any] | None = None,
    ) -> PredictorCheckpointMetadata:
        # 规范化 checkpoint 路径。
        checkpoint_path = Path(path)
        # 预先创建父目录。
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        # 基于 run_trace 生成 metadata。
        metadata = self._checkpoint_metadata(run_trace)
        # 以 torch.save 写入 metadata 和 model_state_dict。
        payload = {
            "checkpoint_kind": metadata.checkpoint_kind,
            "metadata": metadata.to_dict(),
            "model_state_dict": model.state_dict(),
            "run_trace": run_trace.to_dict(),
        }
        # 若调用方提供了优化器状态，则一并写入 checkpoint。
        if optimizer_state_dict is not None:
            payload["optimizer_state_dict"] = optimizer_state_dict
        torch.save(payload, checkpoint_path)
        # 返回写入的 metadata。
        return metadata

    def load_checkpoint(
            self,
            path: str | Path,
    ) -> tuple[FutureExpertPredictor, PredictorCheckpointMetadata]:
        # 先读取 checkpoint 原始载荷。
        payload = self._read_checkpoint_payload(path)
        # 提取模型参数字典。
        state_dict = payload.get("model_state_dict")
        # checkpoint 必须包含 model_state_dict。
        if not isinstance(state_dict, dict):
            raise ValueError("predictor checkpoint is missing model_state_dict")
        # 再读取并校验 metadata。
        metadata = self.read_checkpoint_metadata(path)
        self._validate_checkpoint_metadata(metadata)
        # 基于当前配置重建模型实例。
        model = self.build_model()
        # 加载 checkpoint 权重。
        model.load_state_dict(state_dict)
        # 返回模型和 metadata。
        return model, metadata

    def load_training_checkpoint(
            self,
            path: str | Path,
    ) -> tuple[
        FutureExpertPredictor,
        PredictorCheckpointMetadata,
        PredictorTrainingRunTrace | None,
        dict[str, Any] | None,
    ]:
        # 先读取 checkpoint 原始载荷。
        payload = self._read_checkpoint_payload(path)
        # 复用现有模型权重与 metadata 加载逻辑。
        model, metadata = self.load_checkpoint(path)
        # 再读取可选的 run_trace。
        run_trace = self.read_checkpoint_run_trace(path)
        if run_trace is not None:
            # 若存在 run_trace，则校验它与当前配置兼容。
            self._validate_resume_run_trace(run_trace)
        # 读取可选的 optimizer 状态。
        optimizer_state_dict = payload.get("optimizer_state_dict")
        if optimizer_state_dict is not None and not isinstance(
                optimizer_state_dict,
                dict,
        ):
            raise ValueError(
                "predictor checkpoint optimizer_state_dict must be a dictionary"
            )
        # 返回完整训练续训所需的全部状态。
        return model, metadata, run_trace, optimizer_state_dict

    @classmethod
    def export_checkpoint_bundle(
            cls,
            *,
            checkpoint_path: str | Path,
            output_dir: str | Path,
    ) -> PredictorDeploymentManifest:
        # 先读取 checkpoint 原始载荷。
        payload = cls._read_checkpoint_payload(checkpoint_path)
        # 提取模型参数字典。
        state_dict = payload.get("model_state_dict")
        # checkpoint 必须包含 model_state_dict。
        if not isinstance(state_dict, dict):
            raise ValueError("predictor checkpoint is missing model_state_dict")

        # 再恢复 checkpoint metadata、runtime schema 和指标摘要。
        metadata = cls.read_checkpoint_metadata(checkpoint_path)
        schema = PredictorRuntimeSchema.from_checkpoint_metadata(metadata)
        metrics = PredictorMetricsSummary.from_checkpoint_metadata(metadata)

        # 规范化 bundle 输出目录并预先创建。
        bundle_dir = Path(output_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # 约定 bundle 内各文件名。
        weights_path = bundle_dir / "predictor_weights.pt"
        schema_path = bundle_dir / "predictor_schema.json"
        metrics_path = bundle_dir / "predictor_metrics.json"
        manifest_path = bundle_dir / "predictor_bundle.json"

        # 写出权重文件。
        torch.save(
            {
                "weights_kind": "cfie_predictor_weights",
                "profile_name": metadata.profile_name,
                "model_state_dict": state_dict,
            },
            weights_path,
        )
        # 写出 schema 和 metrics 文件。
        schema.write_json(schema_path)
        metrics.write_json(metrics_path)

        # 组装 bundle manifest。
        manifest = PredictorDeploymentManifest(
            bundle_kind="cfie_predictor_deployment_bundle",
            profile_name=metadata.profile_name,
            source_checkpoint=Path(checkpoint_path).name,
            weights_kind="cfie_predictor_weights",
            weights_format="torch_state_dict",
            weights_file=weights_path.name,
            schema_kind=schema.schema_kind,
            schema_file=schema_path.name,
            metrics_kind=metrics.metrics_kind,
            metrics_file=metrics_path.name,
        )
        # 写出 manifest 并返回。
        manifest.write_json(manifest_path)
        return manifest

    def build_trace_dataset(
            self,
            *,
            steps: int,
            examples_per_step: int | None = None,
            samples: int = 2,
            tokens_per_sample: int = 256,
            dataset_path: str | None = None,
            tokenizer_path: str | None = None,
            dataset_format: str = "auto",
            dataset_text_key: str = "text",
    ) -> PredictorTraceDataset:
        # -------------------- 构造样本批规划器 --------------------
        # batch planner 负责给每个训练 step 产出带真实 token rows 的抽象 batch 形状；
        # forward-capture trace 会消费这些 token rows 直接运行 teacher forward。
        batch_planner = self._build_batch_planner(
            samples=samples,
            tokens_per_sample=tokens_per_sample,
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            dataset_format=dataset_format,
            dataset_text_key=dataset_text_key,
        )

        # -------------------- 生成 predictor 监督样本 --------------------
        # trace builder 会产出每条样本的 hidden_state，
        # 以及对应 future window 的 teacher top-k experts。
        trace_builder = self._resolve_trace_builder()
        examples = trace_builder.build_examples(
            steps=steps,
            examples_per_step=examples_per_step,
            batch_planner=batch_planner,
        )

        # -------------------- 打包为数据集对象 --------------------
        # 除了样本本身，还把 teacher/source、窗口长度和预算元信息一起固化下来，
        # 便于后续训练、评估和 checkpoint 兼容性校验。
        return PredictorTraceDataset(
            profile_name=self.config.profile_name,
            example_count=len(examples),
            window_layers=self.config.predictor_routing.window_layers,
            candidate_experts_per_layer=(
                self.config.predictor_routing.candidate_experts_per_layer
            ),
            executed_experts_per_layer=(
                self.config.predictor_routing.executed_experts_per_layer
            ),
            examples=examples,
        )

    def _dataset_tensors(
            self,
            dataset: PredictorTraceDataset,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # -------------------- 物化特征张量 --------------------
        # 每条样本里的 hidden_state 都是定长 tuple；
        # 这里把它们堆成 [example_count, hidden_size] 的 CPU 浮点张量。
        features = torch.tensor(
            [example.hidden_state for example in dataset.examples],
            dtype=torch.float32,
            device="cpu",
        )
        layer_indices = torch.tensor(
            [example.insertion_layer_index for example in dataset.examples],
            dtype=torch.float32,
            device="cpu",
        )

        # -------------------- 物化多标签目标张量 --------------------
        # predictor 要同时预测“未来多层 x 每层多个 expert”，
        # 因此目标不是单类别标签，而是 [example, future_layer, expert] 的多热张量。
        targets = torch.zeros(
            (
                dataset.example_count,
                dataset.window_layers,
                self.config.model_spec.num_experts,
            ),
            dtype=torch.float32,
            device="cpu",
        )
        for example_index, example in enumerate(dataset.examples):
            for future_index, teacher_topk_ids in enumerate(
                    example.future_teacher_topk_ids
            ):
                # teacher 选中的 top-k experts 位置置为 1；
                # 这样后续 BCEWithLogitsLoss 就能把它当作多标签监督来训练。
                targets[example_index, future_index, list(teacher_topk_ids)] = 1.0

        # 返回训练和评估都会复用的特征/标签张量对。
        return features, layer_indices, targets

    def _mean_recall_at_budget(
            self,
            *,
            logits: torch.Tensor,
            targets: torch.Tensor,
            budget: int,
    ) -> float:
        # 先按预算取每个样本/未来层的 top-k experts。
        topk = torch.topk(logits, k=budget, dim=-1).indices
        # 取出 top-k 位置上的命中标签。
        matches = torch.gather(targets, dim=-1, index=topk)
        # 分母等于 teacher 正样本数，至少夹到 1。
        denom = targets.sum(dim=-1).clamp_min(1.0)
        # recall 等于 top-k 命中数除以 teacher 正样本数。
        recall = matches.sum(dim=-1) / denom
        # 返回全数据集平均 recall。
        return float(recall.mean().item())

    def _validate_dataset_compatibility(
            self,
            dataset: PredictorTraceDataset,
    ) -> None:
        # 数据集至少要有 1 条样本。
        if dataset.example_count < 1:
            raise ValueError("predictor trace dataset must contain at least 1 example")
        # 取第一条样本检查 hidden_state 维度。
        first_example = dataset.examples[0]
        if (
                len(first_example.hidden_state)
                != self.config.model_spec.hidden_size
        ):
            raise ValueError(
                "predictor trace dataset hidden_state size does not match "
                "model_spec.hidden_size"
            )
        if dataset.window_layers != self.config.predictor_routing.window_layers:
            raise ValueError(
                "predictor trace dataset window_layers does not match "
                "predictor_routing.window_layers"
            )

    def evaluate_dataset(
            self,
            dataset: PredictorTraceDataset,
            *,
            model: FutureExpertPredictor | None = None,
            checkpoint_metadata: PredictorCheckpointMetadata | None = None,
    ) -> PredictorEvaluationTrace:
        # -------------------- 校验输入并准备评估对象 --------------------
        # 评估前先确认数据集维度与当前 trainer 配置兼容，避免 silent mismatch。
        self._validate_dataset_compatibility(dataset)
        # routing 配置里的候选预算和执行预算会直接决定 recall 指标的取值口径。
        routing_cfg = self.config.predictor_routing
        # 若调用方未传入模型，就按当前配置构造一个新 predictor 实例。
        model = self.build_model() if model is None else model
        # 把 trace 数据集转换成可直接前向的张量形式。
        features, layer_indices, targets = self._dataset_tensors(dataset)

        # -------------------- 关闭梯度并执行前向评估 --------------------
        with torch.no_grad():
            # 评估阶段不需要 dropout/batchnorm 等训练时行为，因此切到 eval 模式。
            model.eval()
            # 前向得到每个 future layer、每个 expert 的原始 logits。
            logits = model(features, layer_indices)
            # 标签是多热而不是单类别，因此这里使用 BCEWithLogitsLoss 而不是 softmax CE。
            mean_loss = float(
                F.binary_cross_entropy_with_logits(logits, targets).item()
            )
            # 按候选预算统计 recall，用来度量“给推理侧预热多少 experts”时的覆盖能力。
            candidate_recall = self._mean_recall_at_budget(
                logits=logits,
                targets=targets,
                budget=routing_cfg.candidate_experts_per_layer,
            )
            # 按执行预算统计 recall，用来度量“最终真正会执行的 experts”覆盖得如何。
            executed_recall = self._mean_recall_at_budget(
                logits=logits,
                targets=targets,
                budget=routing_cfg.executed_experts_per_layer,
            )

        # -------------------- 打包评估结果 --------------------
        # 把本次评估的 loss、recall 和可选 checkpoint 元信息固化为 trace，供日志和导出使用。
        return PredictorEvaluationTrace(
            profile_name=self.config.profile_name,
            example_count=dataset.example_count,
            candidate_experts_per_layer=routing_cfg.candidate_experts_per_layer,
            executed_experts_per_layer=routing_cfg.executed_experts_per_layer,
            mean_loss=mean_loss,
            recall_at_candidate_budget=candidate_recall,
            recall_at_executed_budget=executed_recall,
            checkpoint_metadata=checkpoint_metadata,
        )

    def evaluate_checkpoint(
            self,
            *,
            checkpoint_path: str | Path,
            dataset: PredictorTraceDataset,
    ) -> PredictorEvaluationTrace:
        # 先加载 checkpoint 对应的模型和元信息。
        model, metadata = self.load_checkpoint(checkpoint_path)
        # 再对目标数据集执行评估。
        return self.evaluate_dataset(
            dataset,
            model=model,
            checkpoint_metadata=metadata,
        )

    def train_dataset(
            self,
            dataset: PredictorTraceDataset,
            *,
            epochs: int | None = None,
    ) -> PredictorTrainingRunTrace:
        # train_dataset 是 fit_dataset 的轻包装：
        # 调用方若只关心训练轨迹而不关心模型对象和优化器状态，就走这里。
        _, run_trace, _ = self.fit_dataset(
            dataset,
            epochs=epochs,
        )
        return run_trace

    def fit_dataset(
            self,
            dataset: PredictorTraceDataset,
            *,
            epochs: int | None = None,
            model: FutureExpertPredictor | None = None,
            optimizer_state_dict: dict[str, Any] | None = None,
            initial_run_trace: PredictorTrainingRunTrace | None = None,
    ) -> tuple[
        FutureExpertPredictor,
        PredictorTrainingRunTrace,
        dict[str, Any],
    ]:
        # -------------------- 校验训练输入并解析超参数 --------------------
        # 训练前先做数据集维度校验，避免把不匹配的 summary 或窗口长度送进模型。
        self._validate_dataset_compatibility(dataset)
        # 若当前是续训，还要确认历史 run_trace 与新数据集来源一致，避免接错 checkpoint。
        if initial_run_trace is not None:
            self._validate_resume_dataset(dataset, initial_run_trace)
        # trainer 配置负责优化超参，routing 配置负责 recall 预算口径。
        trainer_cfg = self.config.predictor_trainer
        routing_cfg = self.config.predictor_routing
        # 若调用方未显式给 epochs，就沿用配置默认值。
        epochs = trainer_cfg.epochs if epochs is None else int(epochs)
        # 训练轮数至少为 1；0 轮训练没有意义，也无法产出新的 run_trace。
        if epochs < 1:
            raise ValueError("epochs must be >= 1")

        # -------------------- 物化张量并初始化训练对象 --------------------
        # 先把 trace 数据集物化成 CPU 张量，后续训练循环直接复用，避免重复构造。
        features, layer_indices, targets = self._dataset_tensors(dataset)
        # 固定随机种子，保证模型初始化和 epoch 内打乱顺序可复现。
        torch.manual_seed(trainer_cfg.seed)
        # 若调用方未传入模型，就按当前 trainer 配置构造一个新 predictor。
        model = self.build_model() if model is None else model
        # 训练目前使用 AdamW；这里只优化 predictor 自身参数，不涉及主模型参数。
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=trainer_cfg.learning_rate,
            weight_decay=trainer_cfg.weight_decay,
        )
        # 若本次是从 checkpoint 续训，就把优化器状态一并恢复，保持动量连续。
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        # batch_size 不能超过样本总数，避免最后构造空 batch。
        batch_size = min(trainer_cfg.batch_size, dataset.example_count)
        # epoch_summaries 既承载本轮训练结果，也在续训场景下承接历史轨迹。
        epoch_summaries: list[PredictorEpochSummary] = (
            []
            if initial_run_trace is None
            else list(initial_run_trace.epoch_summaries)
        )
        # 续训时，新的 epoch 编号要接在历史已完成 epoch 之后。
        completed_epochs = 0 if initial_run_trace is None else initial_run_trace.epochs

        # -------------------- 执行逐 epoch 的训练循环 --------------------
        for epoch_offset in range(epochs):
            # 当前 epoch 的逻辑编号需要考虑续训场景下的历史偏移量。
            epoch_index = completed_epochs + epoch_offset
            # 每个 epoch 使用独立 generator 打乱顺序；
            # 这样既保证可复现，又能让不同 epoch 的样本顺序不同。
            generator = torch.Generator(device="cpu")
            generator.manual_seed(trainer_cfg.seed + epoch_index)
            order = torch.randperm(dataset.example_count, generator=generator)
            # 累积当前 epoch 的样本加权损失，后面再除以总样本数得到 mean_loss。
            total_loss = 0.0
            total_examples = 0

            # -------------------- 执行逐 mini-batch 的前向、反向与更新 --------------------
            for start in range(0, dataset.example_count, batch_size):
                # 根据打乱后的顺序切出当前 mini-batch 的样本索引。
                batch_indices = order[start: start + batch_size]
                # 从全量张量中抽取当前 batch 的特征和标签。
                batch_features = features.index_select(0, batch_indices)
                batch_layer_indices = layer_indices.index_select(0, batch_indices)
                batch_targets = targets.index_select(0, batch_indices)
                # 前向得到每个 future layer / expert 的 logits。
                logits = model(batch_features, batch_layer_indices)
                # 目标是多标签预测，因此这里按多热标签计算 BCEWithLogitsLoss。
                loss = F.binary_cross_entropy_with_logits(logits, batch_targets)
                # 标准训练三步：清梯度、反向传播、参数更新。
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                # 用样本数加权累计 batch loss，避免最后一个不满 batch 时均值失真。
                batch_count = int(batch_features.shape[0])
                total_loss += float(loss.item()) * batch_count
                total_examples += batch_count

            # -------------------- 在全量数据上计算 epoch 级指标 --------------------
            with torch.no_grad():
                # 训练完一个 epoch 后，在全量数据上做一次统一评估，记录可比较的指标。
                model.eval()
                all_logits = model(features, layer_indices)
                # candidate recall 衡量“给推理侧准备的候选 experts”覆盖了多少 teacher experts。
                candidate_recall = self._mean_recall_at_budget(
                    logits=all_logits,
                    targets=targets,
                    budget=routing_cfg.candidate_experts_per_layer,
                )
                # executed recall 衡量“最终真正允许执行的 experts”能覆盖多少 teacher experts。
                executed_recall = self._mean_recall_at_budget(
                    logits=all_logits,
                    targets=targets,
                    budget=routing_cfg.executed_experts_per_layer,
                )
                # 评估完成后切回 train 模式，保证下一个 epoch 继续按训练模式运行。
                model.train()

            # 把本轮 epoch 的损失和 recall 汇总成一条稳定记录，供日志、导出与续训使用。
            epoch_summaries.append(
                PredictorEpochSummary(
                    epoch_index=epoch_index,
                    mean_loss=total_loss / max(total_examples, 1),
                    recall_at_candidate_budget=candidate_recall,
                    recall_at_executed_budget=executed_recall,
                )
            )

        # -------------------- 返回训练产物 --------------------
        # fit_dataset 会同时返回：
        # 1) 训练后的 predictor 模型
        # 2) 完整训练轨迹 run_trace
        # 3) 优化器状态，便于后续继续续训
        return (
            model,
            PredictorTrainingRunTrace(
                profile_name=self.config.profile_name,
                example_count=dataset.example_count,
                epochs=completed_epochs + epochs,
                candidate_experts_per_layer=(
                    routing_cfg.candidate_experts_per_layer
                ),
                executed_experts_per_layer=(
                    routing_cfg.executed_experts_per_layer
                ),
                epoch_summaries=tuple(epoch_summaries),
            ),
            optimizer.state_dict(),
        )

    def train(
            self,
            *,
            steps: int,
            examples_per_step: int | None = None,
            epochs: int | None = None,
            samples: int = 2,
            tokens_per_sample: int = 256,
            dataset_path: str | None = None,
            tokenizer_path: str | None = None,
            dataset_format: str = "auto",
            dataset_text_key: str = "text",
    ) -> PredictorTrainingRunTrace:
        # -------------------- 先构造训练数据集 --------------------
        # train() 是最上层入口：它先根据 step 数和数据源参数生成 predictor trace 数据集。
        dataset = self.build_trace_dataset(
            steps=steps,
            examples_per_step=examples_per_step,
            samples=samples,
            tokens_per_sample=tokens_per_sample,
            dataset_path=dataset_path,
            tokenizer_path=tokenizer_path,
            dataset_format=dataset_format,
            dataset_text_key=dataset_text_key,
        )

        # -------------------- 再委托给数据集训练主链 --------------------
        # 数据集构造和参数更新解耦后，调用方既可以走 train() 一步到位，
        # 也可以先单独生成 dataset，再调用 train_dataset()/fit_dataset()。
        return self.train_dataset(
            dataset,
            epochs=epochs,
        )
