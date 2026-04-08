"""Contiguous parameter bucket helpers for predictor runtime models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class PredictorParameterViewSpec:
    name: str
    shape: tuple[int, ...]
    start_offset: int
    end_offset: int
    dtype: torch.dtype
    device: torch.device

    @property
    def numel(self) -> int:
        return self.end_offset - self.start_offset


@dataclass(slots=True)
class PredictorParameterBucket:
    storage: torch.Tensor
    view_specs: dict[str, PredictorParameterViewSpec]

    @property
    def dtype(self) -> torch.dtype:
        return self.storage.dtype

    @property
    def device(self) -> torch.device:
        return self.storage.device

    @property
    def total_numel(self) -> int:
        return int(self.storage.numel())

    def view(self, name: str) -> torch.Tensor:
        spec = self.view_specs[name]
        return self.storage[spec.start_offset: spec.end_offset].view(spec.shape)


def bucketize_module_parameters(module: nn.Module) -> PredictorParameterBucket:
    named_parameters = list(module.named_parameters())
    if not named_parameters:
        raise ValueError("predictor module has no parameters to bucketize")

    first_param = named_parameters[0][1]
    bucket_dtype = first_param.dtype
    bucket_device = first_param.device
    for name, param in named_parameters[1:]:
        if param.dtype != bucket_dtype:
            raise ValueError(
                "predictor parameter bucket requires uniform dtype, but "
                f"{name} uses {param.dtype} instead of {bucket_dtype}"
            )
        if param.device != bucket_device:
            raise ValueError(
                "predictor parameter bucket requires a single device, but "
                f"{name} is on {param.device} instead of {bucket_device}"
            )

    total_numel = sum(int(param.numel()) for _, param in named_parameters)
    storage = torch.empty(
        total_numel,
        dtype=bucket_dtype,
        device=bucket_device,
    )
    view_specs: dict[str, PredictorParameterViewSpec] = {}

    offset = 0
    with torch.no_grad():
        for name, param in named_parameters:
            numel = int(param.numel())
            end_offset = offset + numel
            storage[offset:end_offset].copy_(param.detach().reshape(-1))
            param.data = storage[offset:end_offset].view_as(param)
            view_specs[name] = PredictorParameterViewSpec(
                name=name,
                shape=tuple(param.shape),
                start_offset=offset,
                end_offset=end_offset,
                dtype=param.dtype,
                device=param.device,
            )
            offset = end_offset

    return PredictorParameterBucket(
        storage=storage,
        view_specs=view_specs,
    )
