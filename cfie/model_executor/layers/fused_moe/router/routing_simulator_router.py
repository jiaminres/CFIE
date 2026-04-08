# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch

import cfie.envs as envs
from cfie.distributed.eplb.eplb_state import EplbLayerState
from cfie.logger import init_logger
from cfie.model_executor.layers.fused_moe.config import RoutingMethodType
from cfie.model_executor.layers.fused_moe.router.base_router import BaseRouter

logger = init_logger(__name__)


class RoutingStrategy(ABC):
    # token -> expert 路由策略的抽象基类。

    @abstractmethod
    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按当前策略把 token 路由到 experts，返回 (topk_weights, topk_ids)。
        pass


class DistributionBasedRouting(RoutingStrategy):
    # 基于概率分布的随机路由策略。
    # 当前主要支持 uniform 和 normal 两类分布，用于测试不同 routing 模式。

    def __init__(self, distribution: str = "uniform", **distribution_params: Any):
        # distribution 指定采样分布类型。
        # 目前支持：
        # - uniform：均匀分布
        # - normal：高斯分布
        # 其余额外参数保存在 distribution_params 中。
        self.distribution = distribution.lower()
        self.distribution_params = distribution_params

        # 初始化时先校验分布类型与参数是否合法。
        self._validate_distribution_params()

    def _validate_distribution_params(self):
        # 校验分布类型是否合法，并补齐默认参数。
        valid_distributions = ["uniform", "normal"]

        if self.distribution not in valid_distributions:
            raise ValueError(
                f"Unsupported distribution: {self.distribution}. "
                f"Supported distributions: {valid_distributions}"
            )

        # normal 分布未显式给出参数时，补上默认 mean / std。
        if self.distribution == "normal":
            self.distribution_params.setdefault("mean", 0.0)
            self.distribution_params.setdefault("std", 1.0)

    def route_tokens(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 按当前分布为每个 token 随机采样 experts，并生成对应权重。
        num_tokens = hidden_states.shape[0]
        num_experts = router_logits.shape[-1]

        if indices_type is None:
            indices_type = torch.long

        # 先按分布采样 expert id。
        topk_ids = self._sample_expert_ids(
            num_tokens, num_experts, top_k, hidden_states.device, indices_type
        )

        # 再按同一分布生成 routing 权重。
        topk_weights = self._generate_weights(num_tokens, top_k, hidden_states.device)

        return topk_weights, topk_ids

    def _sample_expert_ids(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        indices_type: torch.dtype,
    ) -> torch.Tensor:
        # 按当前分布采样 expert id。

        if self.distribution == "uniform":
            # uniform 直接做离散均匀随机采样。
            return torch.randint(
                low=0,
                high=num_experts,
                size=(num_tokens, top_k),
                dtype=indices_type,
                device=device,
            )

        elif self.distribution == "normal":
            # normal 先采连续值，再映射到离散 expert id。
            continuous_samples = self._sample_continuous_distribution(
                num_tokens, top_k, device
            )

            # 先归一化到 [0, 1]，再放缩到 [0, num_experts)。
            normalized_samples = self._normalize_samples(continuous_samples)
            expert_ids = (normalized_samples * num_experts).long()
            expert_ids = torch.clamp(expert_ids, 0, num_experts - 1)

            return expert_ids.to(dtype=indices_type)

        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

    def _sample_continuous_distribution(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> torch.Tensor:
        # 从连续分布中采样原始值。
        shape = (num_tokens, top_k)

        if self.distribution == "normal":
            mean = self.distribution_params["mean"]
            std = self.distribution_params["std"]
            return torch.normal(mean, std, size=shape, device=device)

        else:
            raise ValueError(
                f"Unsupported continuous distribution: {self.distribution}"
            )

    def _normalize_samples(self, samples: torch.Tensor) -> torch.Tensor:
        # 把连续采样结果归一化到 [0, 1] 区间。
        if self.distribution == "normal":
            # normal 分布通过 sigmoid 压到 [0, 1]。
            return torch.sigmoid(samples)

        else:
            raise ValueError(
                f"Unsupported distribution for normalization: {self.distribution}"
            )

    def _generate_weights(
        self, num_tokens: int, top_k: int, device: torch.device
    ) -> torch.Tensor:
        # 按当前分布生成 routing 权重。
        if self.distribution == "uniform":
            # uniform 路径直接返回全 1 权重。
            return torch.ones(
                (num_tokens, top_k),
                dtype=torch.float32,
                device=device,
            )

        elif self.distribution == "normal":
            # normal 路径复用同一连续分布采样权重。
            continuous_weights = self._sample_continuous_distribution(
                num_tokens, top_k, device
            )
            # 取绝对值转正，并在 top-k 维上归一化到和为 1。
            weights = torch.abs(continuous_weights)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            return weights

        else:
            raise ValueError(
                f"Unsupported distribution for weight generation: {self.distribution}"
            )

    def get_distribution_info(self) -> dict:
        # 返回当前分布配置，便于调试或打印。
        return {
            "distribution": self.distribution,
            "parameters": self.distribution_params.copy(),
        }


class RoutingSimulator:
    # token-to-expert 路由模拟器。
    # 用于测试不同 routing 策略的行为，不参与真实模型推理语义。

    # 类级别的路由策略注册表。
    _routing_strategies: dict[str, RoutingStrategy] = {
        # 内置的基础策略。
        "uniform_random": DistributionBasedRouting(
            distribution="uniform", mean=0.0, std=1.0
        ),
        "normal_routing": DistributionBasedRouting(
            distribution="normal", mean=0.0, std=1.0
        ),
    }

    @classmethod
    def register_strategy(cls, name: str, strategy: RoutingStrategy):
        # 注册一个自定义 routing 策略。
        cls._routing_strategies[name] = strategy

    @classmethod
    def get_available_strategies(cls) -> list[str]:
        # 返回当前已注册的 routing 策略名字列表。
        return list(cls._routing_strategies.keys())

    @staticmethod
    def simulate_routing(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        strategy_name: str,
        top_k: int,
        indices_type: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 用指定策略模拟 token->expert 的 routing 结果。
        if strategy_name not in RoutingSimulator._routing_strategies:
            raise ValueError(
                f"Unknown routing strategy: {strategy_name}. "
                f"Available strategies: "
                f"{list(RoutingSimulator._routing_strategies.keys())}"
            )
        logger.warning_once(
            "Simulating MoE routing using a %s strategy. "
            "This should only be used for performance testing. "
            "Model outputs will not be valid.",
            strategy_name,
        )

        strategy = RoutingSimulator._routing_strategies[strategy_name]
        return strategy.route_tokens(
            hidden_states=hidden_states,
            router_logits=router_logits,
            top_k=top_k,
            indices_type=indices_type,
        )


class RoutingSimulatorRouter(BaseRouter):
    # 供测试 / 调试使用的 router，内部直接走 routing simulator。

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        eplb_state: EplbLayerState,
        enable_eplb: bool = False,
        indices_type_getter: Callable[[], torch.dtype | None] | None = None,
    ):
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            eplb_state=eplb_state,
            enable_eplb=enable_eplb,
            indices_type_getter=indices_type_getter,
        )

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Simulated

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 使用环境变量指定的模拟策略计算 routing 结果。
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        topk_weights, topk_ids = RoutingSimulator.simulate_routing(
            hidden_states=hidden_states,
            router_logits=router_logits,
            strategy_name=routing_strategy,
            top_k=self.top_k,
            indices_type=indices_type,
        )
        return topk_weights, topk_ids
