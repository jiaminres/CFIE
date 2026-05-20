# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Set as AbstractSet
from dataclasses import replace
from itertools import product

from cfie.config import CUDAGraphMode, CfieConfig
from cfie.forward_context import BatchDescriptor
from cfie.logger import init_logger
from cfie.lora.utils import get_captured_lora_counts

logger = init_logger(__name__)


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to dispatch keys for multiple set of
    cudagraphs.

    The dispatcher stores two sets of dispatch keys, one for PIECEWISE and one
    for FULL cudagraph runtime mode. The keys are initialized depending on
    attention support and what cudagraph mode is set in CompilationConfig. The
    keys stored in dispatcher are the only source of truth for valid
    cudagraphs that can be dispatched at runtime.

    At runtime, the dispatch method generates the runtime cudagraph mode (FULL,
    PIECEWISE, or NONE for no cudagraph) and the valid key (batch descriptor)
    based on the input key. After dispatching (communicated via forward
    context), the cudagraph wrappers will trust the dispatch key to either
    capture or replay (if the mode matches), or pass through to the underlying
    runnable without cudagraph (if the mode does not match or mode is NONE).
    """

    def __init__(self, cfie_config: CfieConfig):
        self.cfie_config = cfie_config
        self.compilation_config = cfie_config.compilation_config
        self.uniform_decode_query_len = (
            1
            if not self.cfie_config.speculative_config
            else 1 + self.cfie_config.speculative_config.num_speculative_tokens
        )

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        assert (
            not self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            or self.compilation_config.is_attention_compiled_piecewise()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.compilation_config.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        self.keys_initialized = False
        self.specialize_lora_count = (
            self.cfie_config.lora_config.specialize_active_lora
            if self.cfie_config.lora_config is not None
            else False
        )
        # Default cudagraph_mode to NONE until initialize_cudagraph_keys is called
        self.cudagraph_mode = CUDAGraphMode.NONE

    def _decode_capture_sizes(self) -> list[int]:
        sizes = self.compilation_config.cudagraph_decode_capture_sizes
        if sizes is None:
            sizes = self.compilation_config.cudagraph_capture_sizes
        return list(sizes or [])

    def _mixed_capture_sizes(self) -> list[int]:
        sizes = self.compilation_config.cudagraph_prefill_capture_sizes
        if sizes is None:
            sizes = self.compilation_config.cudagraph_capture_sizes
        return list(sizes or [])

    def _has_separate_decode_capture_sizes(self) -> bool:
        return bool(self.compilation_config.cudagraph_decode_capture_sizes)

    def _supports_uniform_decode_key(self) -> bool:
        return self.cudagraph_mode.has_mode(
            CUDAGraphMode.FULL
        ) or (
            self.cudagraph_mode.has_mode(CUDAGraphMode.PIECEWISE)
            and self._has_separate_decode_capture_sizes()
        )

    @staticmethod
    def _build_bs_to_padded_graph_size(capture_sizes: list[int]) -> list[int]:
        if not capture_sizes:
            return [0]

        max_size = capture_sizes[-1]
        bs_to_padded_graph_size: list[int] = [0] * (max_size + 1)
        for end, start in zip(capture_sizes + [max_size + 1], [0] + capture_sizes):
            for bs in range(start, end):
                if bs == start:
                    bs_to_padded_graph_size[bs] = start
                else:
                    bs_to_padded_graph_size[bs] = end
        return bs_to_padded_graph_size

    def _compute_bs_to_padded_graph_size(self) -> None:
        """Pre-compute the mapping from batch size to padded graph size."""
        capture_sizes = self.compilation_config.cudagraph_capture_sizes
        assert capture_sizes is not None, (
            "Cudagraph capture sizes must be set when cudagraphs are enabled."
        )
        self._bs_to_padded_graph_size_by_uniform: dict[bool, list[int]] = {
            False: self._build_bs_to_padded_graph_size(self._mixed_capture_sizes()),
            True: self._build_bs_to_padded_graph_size(self._decode_capture_sizes()),
        }
        # Keep the legacy attribute for code/tests that only know about the
        # unified table.
        self._bs_to_padded_graph_size = self._bs_to_padded_graph_size_by_uniform[
            False
        ]

        # Validate that compile_sizes won't be changed by padding.
        # Only validate when cudagraphs are actually being used.
        if (
            self.compilation_config.compile_sizes
            and self.cudagraph_mode != CUDAGraphMode.NONE
        ):
            for size in self.compilation_config.compile_sizes:
                size = int(size)
                if size <= self.compilation_config.max_cudagraph_capture_size:
                    mapping = self._bs_to_padded_graph_size_by_uniform[False]
                    padded = mapping[size] if size < len(mapping) else 0
                    if padded != size:
                        raise ValueError(
                            f"compile_sizes contains {size} which would be "
                            f"padded to {padded}. All compile_sizes must be "
                            "values that won't be changed by cudagraph padding. "
                            "Use values from cudagraph_capture_sizes."
                        )

    def _get_lora_cases(self) -> list[int]:
        """
        Returns list of has_lora values for CUDA graph capture.
        This is the single source of truth for LoRA capture cases.
        """
        lora_config = self.cfie_config.lora_config
        if lora_config is None:
            # No LoRA configured - single case with no LoRA
            return [0]

        # LoRA is enabled - capture graphs based on cudagraph_specialize_lora
        if self.compilation_config.cudagraph_specialize_lora:
            captured_counts = get_captured_lora_counts(
                lora_config.max_loras, self.specialize_lora_count
            )
            # Specialize: capture separate graphs for with and without LoRA
            return [0] + captured_counts
        else:
            # No specialization: only capture graphs with LoRA active
            return [lora_config.max_loras + 1]

    def _create_padded_batch_descriptor(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        num_active_loras: int = 0,
    ) -> BatchDescriptor:
        max_num_seqs = self.cfie_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        use_uniform_decode_key = (
            uniform_decode and self._supports_uniform_decode_key()
        )
        mapping = self._bs_to_padded_graph_size_by_uniform[use_uniform_decode_key]
        if num_tokens >= len(mapping) or mapping[num_tokens] == 0:
            raise ValueError(
                f"No cudagraph capture size can cover num_tokens={num_tokens} "
                f"for {'decode' if use_uniform_decode_key else 'mixed'} mode."
            )
        num_tokens_padded = mapping[num_tokens]

        if use_uniform_decode_key:
            num_reqs = min(num_tokens_padded // uniform_decode_query_len, max_num_seqs)
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)

        return BatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
            num_active_loras=num_active_loras,
        )

    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    def initialize_cudagraph_keys(
        self,
        cudagraph_mode: CUDAGraphMode,
        uniform_decode_query_len: int | None = None,
    ):
        # This should be called only after attention backend is initialized. So we can
        # get the correct cudagraph mode after backend support is resolved.
        self.cudagraph_mode = cudagraph_mode
        if uniform_decode_query_len is not None:
            self.uniform_decode_query_len = uniform_decode_query_len
        uniform_decode_query_len = self.uniform_decode_query_len

        # Early exit if cudagraphs are disabled
        if cudagraph_mode == CUDAGraphMode.NONE:
            self.keys_initialized = True
            return

        self._compute_bs_to_padded_graph_size()

        # Get LoRA cases to capture
        lora_cases = self._get_lora_cases()
        self.captured_lora_counts = [
            lora_count for lora_count in lora_cases if lora_count
        ]

        # Note: we create all valid keys for cudagraph here but do not
        # guarantee all keys would be used. For example, if we allow lazy
        # capturing in future PR, some keys may never be triggered.
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            cudagraph_capture_sizes_for_mixed = self._mixed_capture_sizes()
            for bs, num_active_loras in product(
                cudagraph_capture_sizes_for_mixed, lora_cases
            ):
                batch_desc = self._create_padded_batch_descriptor(
                    bs, False, num_active_loras > 0, num_active_loras
                )
                # Only relax for PIECEWISE mode. FULL mode needs exact num_reqs
                # because FA3's scheduler_metadata computation depends on it.
                if cudagraph_mode.mixed_mode() == CUDAGraphMode.PIECEWISE:
                    batch_desc = replace(batch_desc, num_reqs=None, uniform=False)
                self.add_cudagraph_key(cudagraph_mode.mixed_mode(), batch_desc)

        # If decode uses a separate routine, add dedicated uniform decode keys.
        # PIECEWISE normally shares mixed keys, but explicit decode capture
        # sizes opt into a separate uniform decode key space so decode=1/2 is
        # not padded to large prefill graph shapes.
        if (
            (
                cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and cudagraph_mode.separate_routine()
            )
            or (
                cudagraph_mode.decode_mode() == CUDAGraphMode.PIECEWISE
                and self._has_separate_decode_capture_sizes()
            )
        ):
            max_num_tokens = (
                uniform_decode_query_len
                * self.cfie_config.scheduler_config.max_num_seqs
            )
            cudagraph_capture_sizes_for_decode = self._decode_capture_sizes()
            assert cudagraph_capture_sizes_for_decode, (
                "Cudagraph decode capture sizes must be set when decode "
                "graphs are enabled."
            )
            cudagraph_capture_sizes_for_decode = [
                x
                for x in cudagraph_capture_sizes_for_decode
                if (
                    x <= max_num_tokens
                    and x >= uniform_decode_query_len
                    and x % uniform_decode_query_len == 0
                )
            ]
            for bs, num_active_loras in product(
                cudagraph_capture_sizes_for_decode, lora_cases
            ):
                runtime_mode = cudagraph_mode.decode_mode()
                batch_desc = self._create_padded_batch_descriptor(
                    bs, True, num_active_loras > 0, num_active_loras
                )
                if runtime_mode == CUDAGraphMode.PIECEWISE:
                    batch_desc = replace(batch_desc, num_reqs=None, uniform=True)
                self.add_cudagraph_key(
                    runtime_mode,
                    batch_desc,
                )

        self.keys_initialized = True

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool = False,
        has_lora: bool = False,
        num_active_loras: int = 0,
        valid_modes: AbstractSet[CUDAGraphMode] | None = None,
        invalid_modes: AbstractSet[CUDAGraphMode] | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """
        Given conditions(e.g.,batch descriptor and if using piecewise only),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).

        Args:
            num_tokens: Number of tokens in the batch.
            uniform_decode: Whether the batch is uniform decode (i.e. uniform and query
                length is uniform_decode_query_len).
            has_lora: Whether LoRA is active.
            num_active_loras: Number of distinct active LoRA adapters.
            valid_modes: Set of cudagraph modes that are allowed. None means
                all modes are allowed.
            invalid_modes: Set of cudagraph modes to exclude. Subtracted from
                valid_modes to compute allowed modes. (e.g., {FULL} for
                features like cascade attention not supported by full
                cudagraphs). None means no modes are excluded.
        """
        allowed_modes = valid_modes or CUDAGraphMode.valid_runtime_modes()

        if invalid_modes:
            allowed_modes -= invalid_modes

        assert len(allowed_modes) >= 1, (
            f"No allowed cudagraph modes: valid_modes={valid_modes}, "
            f"invalid_modes={invalid_modes}"
        )

        if (
            not self.keys_initialized
            or self.cudagraph_mode == CUDAGraphMode.NONE
            or num_tokens > self.compilation_config.max_cudagraph_capture_size
            or allowed_modes <= {CUDAGraphMode.NONE}
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        effective_num_active_loras = num_active_loras
        if has_lora and num_active_loras > 0:
            if self.specialize_lora_count:
                # Find the smallest captured `num_active_loras` that is >= the current
                # `num_active_loras`. This is because we only capture graphs for
                # a subset of possible `num_active_loras` values (powers of 2).
                import bisect

                idx = bisect.bisect_left(self.captured_lora_counts, num_active_loras)
                if idx < len(self.captured_lora_counts):
                    effective_num_active_loras = self.captured_lora_counts[idx]
            else:
                # When not specializing, graphs are captured only with max_loras + 1,
                # so we must use max_loras + 1 for dispatch to find a matching graph.
                assert self.cfie_config.lora_config is not None, (
                    "LoRA config must be set when has_lora is True."
                )
                effective_num_active_loras = self.cfie_config.lora_config.max_loras + 1

        normalized_uniform = uniform_decode and (
            self.cudagraph_mode.separate_routine()
            or self._has_separate_decode_capture_sizes()
        )
        try:
            batch_desc = self._create_padded_batch_descriptor(
                num_tokens, normalized_uniform, has_lora, effective_num_active_loras
            )
        except ValueError:
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        if CUDAGraphMode.FULL in allowed_modes:
            # check if key exists for full cudagraph
            batch_desc_to_check = batch_desc
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc_to_check

        if CUDAGraphMode.PIECEWISE in allowed_modes:
            if batch_desc.uniform:
                batch_desc_to_check = replace(batch_desc, num_reqs=None, uniform=True)
                if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
                    return CUDAGraphMode.PIECEWISE, batch_desc_to_check

            # also check if the relaxed key exists for more "general"
            # piecewise cudagraph
            batch_desc_to_check = replace(batch_desc, num_reqs=None, uniform=False)
            if batch_desc_to_check in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
                return CUDAGraphMode.PIECEWISE, batch_desc_to_check

        assert CUDAGraphMode.NONE in allowed_modes, (
            f"No matching cudagraph found and NONE is not in "
            f"allowed_modes={allowed_modes}"
        )
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

    def get_capture_descs(self) -> list[tuple[CUDAGraphMode, list[BatchDescriptor]]]:
        """
        Returns capture descriptors for cudagraph capturing.

        Returns:
            List of (runtime_mode, batch_descriptors) tuples, ordered PIECEWISE
            first then FULL. Batch descriptors are sorted largest-first for
            memory efficiency.
        """
        if not self.keys_initialized or self.cudagraph_mode == CUDAGraphMode.NONE:
            return []

        result = []
        # Return in order: PIECEWISE first, then FULL
        for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
            for uniform in [False, True]:
                descs = [
                    desc
                    for desc in self.cudagraph_keys[mode]
                    if desc.uniform == uniform
                ]
                if not descs:
                    continue
                # Sort by (num_tokens, num_active_loras) descending
                descs.sort(
                    key=lambda d: (d.num_tokens, d.num_active_loras),
                    reverse=True,
                )
                result.append((mode, descs))

        return result
