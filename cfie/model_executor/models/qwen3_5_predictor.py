# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Predictor-isolated Qwen3.5-MoE runtime classes."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import torch
from torch import nn

from cfie.config.cfie import CfieConfig
from cfie.distributed import get_pp_group
from cfie.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from cfie.model_executor.layers.quantization import QuantizationConfig
from cfie.multimodal import MULTIMODAL_REGISTRY
from cfie.predictor import (
    CandidateLayerPlan,
    LoadedPredictorBundle,
    PredictorCandidatePlan,
    PredictorCandidatePlanner,
    PredictorOnlineExpertState,
    PredictorObservedRoutingTracker,
    PredictorParameterBucket,
    PredictorRuntimeSchema,
    bucketize_module_parameters,
    load_predictor_model,
)
from cfie.sequence import IntermediateTensors
from cfie.transformers_utils.configs.qwen3_5 import Qwen3_5TextConfig
from cfie.transformers_utils.configs.qwen3_5_moe import Qwen3_5MoeTextConfig
from cfie.transformers_utils.configs.qwen3_5_moe_predictor import (
    Qwen3_5MoePredictorConfig,
    Qwen3_5MoePredictorTextConfig,
)

from .qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from .qwen3_next import Qwen3NextSparseMoeBlock
from .utils import AutoWeightsLoader
from .utils import extract_layer_index
from .utils import WeightsMapper

PREDICTOR_PENDING_LAYER_PLANS_KEY = "__cfie_predictor_pending_layer_plans__"
SUPPORTED_PREDICTOR_SELECTION_MODES = frozenset({"masked_candidate_topk"})
SUPPORTED_PREDICTOR_ONLINE_EXPERT_SOURCES = frozenset(
    {"cpu_hot_only", "cpu_or_nvme"}
)


@dataclass(slots=True)
class PredictorRuntimeState:
    bundle_path: Path
    bundle: LoadedPredictorBundle
    planner: PredictorCandidatePlanner
    parameter_bucket: PredictorParameterBucket | None = None
    online_expert_states: dict[int, PredictorOnlineExpertState] = field(
        default_factory=dict
    )
    observed_online_expert_states: dict[int, PredictorOnlineExpertState] = field(
        default_factory=dict
    )
    observed_routing_tracker: PredictorObservedRoutingTracker = field(
        default_factory=PredictorObservedRoutingTracker
    )

    @property
    def schema(self) -> PredictorRuntimeSchema:
        return self.bundle.schema

    def validate_runtime_support(self) -> "PredictorRuntimeState":
        if self.schema.selection_mode not in SUPPORTED_PREDICTOR_SELECTION_MODES:
            raise ValueError(
                "unsupported predictor selection_mode for inference runtime: "
                f"{self.schema.selection_mode!r}"
            )
        if (
                self.schema.online_expert_source
                not in SUPPORTED_PREDICTOR_ONLINE_EXPERT_SOURCES
        ):
            raise ValueError(
                "unsupported predictor online_expert_source for inference runtime: "
                f"{self.schema.online_expert_source!r}"
            )
        return self

    def bind_online_expert_state(
            self,
            *,
            layer_index: int,
            online_expert_state: PredictorOnlineExpertState | None,
    ) -> None:
        if online_expert_state is None:
            self.online_expert_states.pop(int(layer_index), None)
            return
        self.online_expert_states[int(layer_index)] = online_expert_state

    def bind_online_expert_states(
            self,
            online_expert_states: dict[int, PredictorOnlineExpertState | None],
    ) -> None:
        self.online_expert_states.clear()
        for layer_index, online_expert_state in online_expert_states.items():
            self.bind_online_expert_state(
                layer_index=layer_index,
                online_expert_state=online_expert_state,
            )

    def get_online_expert_state(
            self,
            layer_index: int,
    ) -> PredictorOnlineExpertState | None:
        return self.online_expert_states.get(
            int(layer_index)
        ) or self.observed_online_expert_states.get(int(layer_index))

    def get_explicit_online_expert_state(
            self,
            layer_index: int,
    ) -> PredictorOnlineExpertState | None:
        return self.online_expert_states.get(int(layer_index))

    def get_observed_online_expert_state(
            self,
            layer_index: int,
    ) -> PredictorOnlineExpertState | None:
        return self.observed_online_expert_states.get(int(layer_index))

    def clear_online_expert_states(self) -> None:
        self.online_expert_states.clear()

    def clear_observed_online_expert_states(self) -> None:
        self.observed_online_expert_states.clear()

    def reset_observed_online_expert_step(self) -> None:
        self.observed_routing_tracker.start_step()

    def observe_routed_experts(
            self,
            *,
            layer_index: int,
            expert_ids: torch.Tensor,
    ) -> None:
        self.observed_routing_tracker.observe(
            layer_index=layer_index,
            expert_ids=expert_ids,
        )

    def commit_observed_online_expert_states(self) -> None:
        self.observed_online_expert_states = (
            self.observed_routing_tracker.finalize_step(
                hot_budget=self.schema.candidate_experts_per_layer,
            )
        )


@dataclass(slots=True)
class PredictorLayerRoutingState:
    insertion_layer_index: int
    layer_plan: CandidateLayerPlan


@dataclass(slots=True)
class PredictorForwardState:
    pending_layer_plans: dict[int, PredictorLayerRoutingState] = field(
        default_factory=dict
    )
    emitted_window_plans: list[PredictorCandidatePlan] = field(default_factory=list)

    def reset(self) -> None:
        self.pending_layer_plans.clear()
        self.emitted_window_plans.clear()

    def upsert_pending_layer_plan(
            self,
            routing_state: PredictorLayerRoutingState,
    ) -> None:
        future_layer_index = int(routing_state.layer_plan.future_layer_index)
        existing_state = self.pending_layer_plans.get(future_layer_index)
        if (
                existing_state is None
                or routing_state.insertion_layer_index
                >= existing_state.insertion_layer_index
        ):
            self.pending_layer_plans[future_layer_index] = routing_state

    def restore_pending_layer_plans(
            self,
            routing_states: tuple[PredictorLayerRoutingState, ...]
            | list[PredictorLayerRoutingState],
    ) -> None:
        for routing_state in routing_states:
            self.upsert_pending_layer_plan(routing_state)

    def export_pending_layer_plans(
            self,
            *,
            min_future_layer_index: int | None = None,
    ) -> tuple[PredictorLayerRoutingState, ...]:
        return tuple(
            self.pending_layer_plans[future_layer_index]
            for future_layer_index in sorted(self.pending_layer_plans)
            if min_future_layer_index is None
            or future_layer_index >= min_future_layer_index
        )


class Qwen3_5MoePredictorProcessingInfo(Qwen3_5MoeProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5MoePredictorConfig)


class Qwen3_5PredictorSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """主线二 predictor-aware MoE block 的隔离落点。"""

    def __init__(self, cfie_config: CfieConfig, prefix: str = ""):
        super().__init__(cfie_config=cfie_config, prefix=prefix)
        self.layer_idx = extract_layer_index(prefix)
        self.predictor_runtime: PredictorRuntimeState | None = None
        self.active_layer_plan: PredictorLayerRoutingState | None = None
        self.active_online_expert_state: PredictorOnlineExpertState | None = None
        self.last_layer_plan: PredictorLayerRoutingState | None = None
        self._install_predictor_router()

    @staticmethod
    def _default_topk_routing(
            gating_output: torch.Tensor,
            *,
            topk: int,
            renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_logit_vals, topk_idx = torch.topk(
            gating_output,
            k=topk,
            dim=-1,
            sorted=False,
        )
        if renormalize:
            topk_vals = torch.softmax(topk_logit_vals, dim=-1)
        else:
            logz = torch.logsumexp(gating_output, dim=-1, keepdim=True)
            topk_vals = (topk_logit_vals - logz).exp()
        return topk_vals.to(torch.float32), topk_idx.to(torch.int32)

    def _install_predictor_router(self) -> None:
        self.experts.custom_routing_function = self._predictor_custom_routing_function
        self.experts.router = create_fused_moe_router(
            top_k=self.experts.top_k,
            global_num_experts=self.experts.global_num_experts,
            eplb_state=self.experts.eplb_state,
            renormalize=self.experts.renormalize,
            use_grouped_topk=self.experts.use_grouped_topk,
            num_expert_group=self.experts.num_expert_group,
            topk_group=self.experts.topk_group,
            custom_routing_function=self._predictor_custom_routing_function,
            scoring_func=self.experts.scoring_func,
            routed_scaling_factor=self.experts.routed_scaling_factor,
            e_score_correction_bias=self.experts.e_score_correction_bias,
            num_fused_shared_experts=self.experts.num_fused_shared_experts,
            enable_eplb=self.experts.enable_eplb,
            indices_type_getter=lambda: self.experts.quant_method.topk_indices_dtype,
        )
        self.experts.routing_method_type = self.experts.router.routing_method_type
        self.experts.runner = self.experts._init_runner()

    def _record_observed_routing(
            self,
            *,
            routed_expert_ids: torch.Tensor,
    ) -> None:
        runtime_state = self.predictor_runtime
        if runtime_state is None:
            return
        runtime_state.observe_routed_experts(
            layer_index=self.layer_idx,
            expert_ids=routed_expert_ids,
        )

    def _predictor_custom_routing_function(
            self,
            hidden_states: torch.Tensor,
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden_states

        active_state = self.active_layer_plan
        runtime_state = self.predictor_runtime
        if active_state is None or runtime_state is None:
            topk_weights, topk_ids = self._default_topk_routing(
                gating_output,
                topk=topk,
                renormalize=renormalize,
            )
            self._record_observed_routing(routed_expert_ids=topk_ids)
            return topk_weights, topk_ids

        candidate_ids = tuple(
            expert_id
            for expert_id in active_state.layer_plan.candidate_expert_ids
            if 0 <= expert_id < gating_output.shape[-1]
        )
        if runtime_state.schema.online_expert_source == "cpu_hot_only":
            active_online_state = self.active_online_expert_state
            if active_online_state is not None:
                cpu_hot_expert_ids = {
                    expert_id
                    for expert_id in active_online_state.cpu_hot_expert_ids
                    if 0 <= expert_id < gating_output.shape[-1]
                }
                candidate_ids = tuple(
                    expert_id
                    for expert_id in candidate_ids
                    if expert_id in cpu_hot_expert_ids
                )
        candidate_index = torch.tensor(
            candidate_ids,
            dtype=torch.long,
            device=gating_output.device,
        )

        if candidate_index.numel() < topk:
            if runtime_state.schema.allow_candidate_mismatch:
                topk_weights, topk_ids = self._default_topk_routing(
                    gating_output,
                    topk=topk,
                    renormalize=renormalize,
                )
                self._record_observed_routing(routed_expert_ids=topk_ids)
                return topk_weights, topk_ids
            raise RuntimeError(
                "predictor candidate pool is smaller than model top-k "
                f"for layer {active_state.layer_plan.future_layer_index}: "
                f"{candidate_index.numel()} < {topk}"
            )

        masked_logits = torch.full_like(gating_output, float("-inf"))
        masked_logits[:, candidate_index] = gating_output[:, candidate_index]

        topk_weights, topk_ids = self._default_topk_routing(
            masked_logits,
            topk=topk,
            renormalize=renormalize,
        )
        self._record_observed_routing(routed_expert_ids=topk_ids)
        return topk_weights, topk_ids

    def bind_predictor_runtime(
            self,
            runtime_state: PredictorRuntimeState | None,
    ) -> None:
        self.predictor_runtime = runtime_state

    def bind_active_layer_plan(
            self,
            layer_plan: PredictorLayerRoutingState | None,
    ) -> None:
        self.active_layer_plan = layer_plan
        self.last_layer_plan = layer_plan

    def bind_active_online_expert_state(
            self,
            online_expert_state: PredictorOnlineExpertState | None,
    ) -> None:
        self.active_online_expert_state = online_expert_state


class Qwen3_5PredictorDecoderLayer(Qwen3_5DecoderLayer):
    def build_mlp(
            self,
            *,
            cfie_config: CfieConfig,
            config: (
                Qwen3_5TextConfig
                | Qwen3_5MoeTextConfig
                | Qwen3_5MoePredictorTextConfig
            ),
            quant_config: QuantizationConfig | None,
            prefix: str,
    ) -> nn.Module:
        if config.model_type == "qwen3_5_moe_predictor_text":
            return Qwen3_5PredictorSparseMoeBlock(
                cfie_config=cfie_config,
                prefix=prefix,
            )
        return super().build_mlp(
            cfie_config=cfie_config,
            config=config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def bind_active_layer_plan(
            self,
            layer_plan: PredictorLayerRoutingState | None,
    ) -> None:
        if isinstance(self.mlp, Qwen3_5PredictorSparseMoeBlock):
            self.mlp.bind_active_layer_plan(layer_plan)

    def bind_active_online_expert_state(
            self,
            online_expert_state: PredictorOnlineExpertState | None,
    ) -> None:
        if isinstance(self.mlp, Qwen3_5PredictorSparseMoeBlock):
            self.mlp.bind_active_online_expert_state(online_expert_state)


class Qwen3_5PredictorModel(Qwen3_5Model):
    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        super().__init__(cfie_config=cfie_config, prefix=prefix)
        self.predictor_runtime: PredictorRuntimeState | None = None
        self.predictor_forward_state = PredictorForwardState()

    def build_decoder_layer(
            self,
            *,
            cfie_config: CfieConfig,
            layer_type: str,
            prefix: str,
    ) -> Qwen3_5PredictorDecoderLayer:
        return Qwen3_5PredictorDecoderLayer(
            cfie_config,
            layer_type=layer_type,
            prefix=prefix,
        )

    def bind_predictor_runtime(
            self,
            runtime_state: PredictorRuntimeState | None,
    ) -> None:
        self.predictor_runtime = runtime_state
        for layer in self.layers:
            if isinstance(layer, Qwen3_5PredictorDecoderLayer) and isinstance(
                    layer.mlp, Qwen3_5PredictorSparseMoeBlock
            ):
                layer.mlp.bind_predictor_runtime(runtime_state)
                layer.mlp.bind_active_layer_plan(None)
                layer.mlp.bind_active_online_expert_state(None)

    def _restore_predictor_pp_state(
            self,
            intermediate_tensors: IntermediateTensors | None,
    ) -> None:
        if intermediate_tensors is None:
            return
        pending_layer_plans = intermediate_tensors.get(
            PREDICTOR_PENDING_LAYER_PLANS_KEY
        )
        if pending_layer_plans is None:
            return
        if not isinstance(pending_layer_plans, (tuple, list)):
            raise TypeError(
                "predictor PP state must be a tuple/list of "
                "PredictorLayerRoutingState objects"
            )
        restored_plans = tuple(
            routing_state
            for routing_state in pending_layer_plans
            if isinstance(routing_state, PredictorLayerRoutingState)
            and self.start_layer <= routing_state.layer_plan.future_layer_index
            < self.config.num_hidden_layers
        )
        self.predictor_forward_state.restore_pending_layer_plans(restored_plans)

    def _predictor_decoder_layer(
            self,
            *,
            layer_index: int,
    ) -> Qwen3_5PredictorDecoderLayer | None:
        if not (self.start_layer <= layer_index < self.end_layer):
            return None
        layer = self.layers[layer_index - self.start_layer]
        if not (
                isinstance(layer, Qwen3_5PredictorDecoderLayer)
                and isinstance(layer.mlp, Qwen3_5PredictorSparseMoeBlock)
        ):
            return None
        return layer

    def _resolve_online_expert_state(
            self,
            *,
            insertion_layer_index: int,
    ) -> tuple[PredictorOnlineExpertState | None, str | None]:
        runtime_state = self.predictor_runtime
        if runtime_state is not None:
            bound_online_state = runtime_state.get_explicit_online_expert_state(
                insertion_layer_index
            )
            if bound_online_state is not None:
                return bound_online_state, "explicit_runtime_state"
            observed_online_state = runtime_state.get_observed_online_expert_state(
                insertion_layer_index
            )
            if observed_online_state is not None:
                return observed_online_state, "observed_runtime_state"
        layer = self._predictor_decoder_layer(layer_index=insertion_layer_index)
        if layer is None:
            return None, None
        logical_replica_count = layer.mlp.experts.eplb_state.logical_replica_count
        if logical_replica_count is None:
            return None, None
        return (
            PredictorOnlineExpertState.from_logical_replica_count(
                logical_replica_count,
            ),
            "eplb_logical_replica_count",
        )

    def reset_predictor_forward_state(self) -> None:
        self.predictor_forward_state.reset()
        if self.predictor_runtime is not None:
            self.predictor_runtime.reset_observed_online_expert_step()
        for layer in self.layers:
            if isinstance(layer, Qwen3_5PredictorDecoderLayer):
                layer.bind_active_layer_plan(None)
                layer.bind_active_online_expert_state(None)

    def _maybe_emit_predictor_window_plan(
            self,
            *,
            layer_idx: int,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None,
            positions: torch.Tensor,
    ) -> None:
        del positions
        runtime_state = self.predictor_runtime
        if runtime_state is None:
            return
        stride_layers = max(int(runtime_state.schema.stride_layers), 1)
        if layer_idx % stride_layers != 0:
            return

        predictor_hidden_state = (
            hidden_states + residual if residual is not None else hidden_states
        )
        window_plan = runtime_state.planner.plan_window(
            predictor_hidden_state,
            insertion_layer_index=layer_idx,
            total_layers=self.config.num_hidden_layers,
        )
        filtered_layer_plans = tuple(
            layer_plan
            for layer_plan in window_plan.layer_plans
            if layer_idx < layer_plan.future_layer_index < self.config.num_hidden_layers
        )
        if not filtered_layer_plans:
            return
        window_plan = PredictorCandidatePlan(
            profile_name=window_plan.profile_name,
            selection_mode=window_plan.selection_mode,
            allow_candidate_mismatch=window_plan.allow_candidate_mismatch,
            insertion_layer_index=window_plan.insertion_layer_index,
            layer_plans=filtered_layer_plans,
        )
        self.predictor_forward_state.emitted_window_plans.append(window_plan)
        for layer_plan in window_plan.layer_plans:
            self.predictor_forward_state.upsert_pending_layer_plan(
                PredictorLayerRoutingState(
                    insertion_layer_index=layer_idx,
                    layer_plan=layer_plan,
                )
            )

    def forward(
            self,
            input_ids: torch.Tensor | None,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        self.reset_predictor_forward_state()

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        self._restore_predictor_pp_state(intermediate_tensors)

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
                islice(self.layers, self.start_layer, self.end_layer),
                start=self.start_layer,
        ):
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )

            if isinstance(layer, Qwen3_5PredictorDecoderLayer):
                layer.bind_active_layer_plan(
                    self.predictor_forward_state.pending_layer_plans.pop(
                        layer_idx,
                        None,
                    )
                )
                active_online_expert_state, _ = self._resolve_online_expert_state(
                    insertion_layer_index=layer_idx
                )
                layer.bind_active_online_expert_state(active_online_expert_state)

            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

            if isinstance(layer, Qwen3_5PredictorDecoderLayer):
                self._maybe_emit_predictor_window_plan(
                    layer_idx=layer_idx,
                    hidden_states=hidden_states,
                    residual=residual,
                    positions=positions,
                )
                layer.bind_active_layer_plan(None)
                layer.bind_active_online_expert_state(None)

        if not get_pp_group().is_last_rank:
            if self.predictor_runtime is not None:
                self.predictor_runtime.commit_observed_online_expert_states()
            output_tensors: dict[str, torch.Tensor | object] = {
                "hidden_states": hidden_states,
                "residual": residual,
            }
            pending_layer_plans = self.predictor_forward_state.export_pending_layer_plans(
                min_future_layer_index=self.end_layer,
            )
            if pending_layer_plans:
                output_tensors[PREDICTOR_PENDING_LAYER_PLANS_KEY] = (
                    pending_layer_plans
                )
            intermediate_tensors = IntermediateTensors(output_tensors)
            # PP 非最后 stage 也可能负责一部分需要采集的 hidden-state 层；
            # 返回 tuple 让 model runner 在本地落盘这些层，再继续按原 PP 字典传递主干状态。
            if aux_hidden_states:
                return intermediate_tensors, aux_hidden_states
            return intermediate_tensors
        if self.predictor_runtime is not None:
            self.predictor_runtime.commit_observed_online_expert_states()
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Qwen3_5MoePredictorForCausalLM(Qwen3_5MoeForCausalLM):
    # 显式挂出 hybrid 标记，避免 registry 基于本模块单独缓存 modelinfo 时
    # 丢失 attention + linear_attention 混合模型的能力声明。
    is_hybrid = True

    hf_to_cfie_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": None,
            "model.language_model.": "model.",
        }
    )

    def __init__(self, *, cfie_config: CfieConfig, prefix: str = ""):
        super().__init__(cfie_config=cfie_config, prefix=prefix)
        self.predictor_runtime: PredictorRuntimeState | None = None

        predictor_config = cfie_config.model_config.hf_config
        bundle_path = getattr(predictor_config, "predictor_bundle_path", None)
        if bundle_path:
            self.load_predictor_runtime(
                bundle_path=bundle_path,
                map_location=getattr(
                    predictor_config,
                    "predictor_map_location",
                    "cpu",
                ),
                device=getattr(
                    predictor_config,
                    "predictor_device",
                    "cpu",
                ),
            )

    def build_backbone_model(
            self,
            *,
            cfie_config: CfieConfig,
            prefix: str,
    ) -> Qwen3_5PredictorModel:
        return Qwen3_5PredictorModel(
            cfie_config=cfie_config,
            prefix=prefix,
        )

    def _require_predictor_runtime(self) -> PredictorRuntimeState:
        if self.predictor_runtime is None:
            raise RuntimeError("predictor runtime is not loaded")
        return self.predictor_runtime

    def bind_predictor_online_expert_state(
            self,
            *,
            layer_index: int,
            online_expert_state: PredictorOnlineExpertState | None,
    ) -> None:
        self._require_predictor_runtime().bind_online_expert_state(
            layer_index=layer_index,
            online_expert_state=online_expert_state,
        )

    def bind_predictor_online_expert_states(
            self,
            online_expert_states: dict[int, PredictorOnlineExpertState | None],
    ) -> None:
        self._require_predictor_runtime().bind_online_expert_states(
            online_expert_states
        )

    def clear_predictor_online_expert_states(self) -> None:
        if self.predictor_runtime is not None:
            self.predictor_runtime.clear_online_expert_states()

    def clear_observed_predictor_online_expert_states(self) -> None:
        if self.predictor_runtime is not None:
            self.predictor_runtime.clear_observed_online_expert_states()

    def load_predictor_runtime(
            self,
            *,
            bundle_path: str | Path,
            map_location: str = "cpu",
            device: str = "cpu",
    ) -> PredictorRuntimeState:
        predictor_model, bundle = load_predictor_model(
            bundle_path,
            map_location=map_location,
            device=device,
        )
        runtime_state = PredictorRuntimeState(
            bundle_path=Path(bundle_path).resolve(),
            bundle=bundle,
            planner=PredictorCandidatePlanner(
                schema=bundle.schema,
                model=predictor_model,
            ),
            parameter_bucket=bucketize_module_parameters(predictor_model),
        ).validate_runtime_support()
        self.predictor_runtime = runtime_state
        if isinstance(self.model, Qwen3_5PredictorModel):
            self.model.bind_predictor_runtime(runtime_state)
        return runtime_state

    def clear_predictor_runtime(self) -> None:
        self.predictor_runtime = None
        if isinstance(self.model, Qwen3_5PredictorModel):
            self.model.bind_predictor_runtime(None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_cfie_mapper)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5MoePredictorProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_5MoePredictorForConditionalGeneration(
    Qwen3_5MoeForConditionalGeneration
):
    def build_language_model(
            self,
            *,
            cfie_config: CfieConfig,
            prefix: str,
    ) -> Qwen3_5MoePredictorForCausalLM:
        return Qwen3_5MoePredictorForCausalLM(
            cfie_config=cfie_config,
            prefix=prefix,
        )

    def bind_predictor_online_expert_state(
            self,
            *,
            layer_index: int,
            online_expert_state: PredictorOnlineExpertState | None,
    ) -> None:
        self.language_model.bind_predictor_online_expert_state(
            layer_index=layer_index,
            online_expert_state=online_expert_state,
        )

    def bind_predictor_online_expert_states(
            self,
            online_expert_states: dict[int, PredictorOnlineExpertState | None],
    ) -> None:
        self.language_model.bind_predictor_online_expert_states(
            online_expert_states
        )

    def clear_predictor_online_expert_states(self) -> None:
        self.language_model.clear_predictor_online_expert_states()

    def clear_observed_predictor_online_expert_states(self) -> None:
        self.language_model.clear_observed_predictor_online_expert_states()
