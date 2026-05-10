"""GPU 端 GPTQ 量化器——支持 per-group absmax 和激活感知 GPTQ。"""

from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

INT4_MIN, INT4_MAX = -8, 7


@dataclass(slots=True)
class GpuGptqConfig:
	bits: int = 4
	group_size: int = 128
	damp_percent: float = 0.01
	def __post_init__(self) -> None:
		if self.bits != 4: raise ValueError("only 4-bit supported")


@dataclass(slots=True)
class GpuGptqQuantizer:
	config: GpuGptqConfig = field(default_factory=GpuGptqConfig)
	_activation_buffer: list[torch.Tensor] = field(default_factory=list)
	_max_samples: int = 256

	def reset(self) -> None:
		self._activation_buffer.clear()

	def collect_activations(self, hidden_states: torch.Tensor) -> None:
		if len(self._activation_buffer) >= self._max_samples:
			return
		flat = hidden_states.detach().reshape(-1, hidden_states.shape[-1])
		n = min(flat.shape[0], self._max_samples - len(self._activation_buffer))
		self._activation_buffer.append(flat[:n].clone())

	def build_hessian(self) -> torch.Tensor:
		if not self._activation_buffer:
			raise RuntimeError("no activations collected")
		X = torch.cat(self._activation_buffer, dim=0).float()
		H = 2 * (X.T @ X) / X.shape[0]
		damp = self.config.damp_percent * torch.mean(torch.diag(H))
		H.diagonal().add_(damp)
		return H

	def quantize(self, weight: torch.Tensor, *, return_dict: bool = False) -> tuple:
		weight = weight.float()
		out_f, in_f = weight.shape
		gs = self.config.group_size
		if not self._activation_buffer:
			return self._quantize_fast(weight)
		H = self.build_hessian().to(weight.device)
		try:
			L = torch.linalg.cholesky(H)
		except torch.linalg.LinAlgError:
			H.diagonal().add_(0.01)
			L = torch.linalg.cholesky(H)
		Linv = torch.linalg.inv(L)
		Hinv = Linv.T @ Linv
		W = weight.clone()
		scales_list, qweight_list = [], []
		num_groups = (in_f + gs - 1) // gs
		for g in range(num_groups):
			gs_ = g * gs; ge = min(gs_ + gs, in_f); gz = ge - gs_
			mx = W[:, gs_:ge].abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
			scale = (mx / INT4_MAX).to(torch.float16)
			scales_list.append(scale)
			qw_g = []
			for j in range(gz):
				ci = gs_ + j
				wc = W[:, ci].clone()
				q = (wc / scale[:, 0]).round().clamp(INT4_MIN, INT4_MAX)
				qc = q * scale[:, 0]
				err = wc - qc
				qw_g.append(q.to(torch.int8))
				if j < gz - 1 and ci + 1 < in_f:
					rem = in_f - (ci + 1)
					hs = Hinv[ci, ci+1:ci+1+rem]
					hd = Hinv[ci, ci]
					if hd > 1e-10:
						W[:, ci+1:] -= err.unsqueeze(1) * hs.unsqueeze(0) / hd
			qw_g_t = torch.stack(qw_g, dim=1)
			qweight_list.append(_pack_int4_along_dim1(qw_g_t))
		qweight = torch.cat(qweight_list, dim=1)
		scales_t = torch.cat(scales_list, dim=1)
		if return_dict:
			return {"qweight": qweight, "scales": scales_t, "qzeros": torch.empty(0)}
		return qweight, scales_t, torch.empty(0)

	def _quantize_fast(self, weight: torch.Tensor) -> tuple:
		out_f, in_f = weight.shape
		gs = self.config.group_size
		num_groups = (in_f + gs - 1) // gs
		scales = torch.empty(out_f, num_groups, dtype=torch.float16, device=weight.device)
		qweight = torch.empty(out_f, (in_f + 1) // 2, dtype=torch.uint8, device=weight.device)
		for g in range(num_groups):
			gs_ = g * gs; ge = min(gs_ + gs, in_f)
			gw = weight[:, gs_:ge]
			mx = gw.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
			scales[:, g:g+1] = (mx / INT4_MAX).to(torch.float16)
			q = (gw / mx * INT4_MAX).round().clamp(INT4_MIN, INT4_MAX).to(torch.int8)
			packed = _pack_int4_along_dim1(q)
			qweight[:, gs_ // 2:gs_ // 2 + packed.shape[1]] = packed
		return qweight, scales, torch.zeros(out_f, (num_groups+1)//2, dtype=torch.uint8, device=weight.device)

	def quantize_logical(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		"""GPU 量化 → 逻辑 int4 [out_f,in_f] int32 + scales [out_f,num_groups] fp16。"""
		out_f, in_f = weight.shape
		gs = self.config.group_size
		num_groups = (in_f + gs - 1) // gs
		w = weight.float()
		scales = torch.empty(out_f, num_groups, dtype=torch.float16, device=weight.device)
		logical = torch.empty(out_f, in_f, dtype=torch.int32, device=weight.device)
		for g in range(num_groups):
			gs_ = g * gs; ge = min(gs_ + gs, in_f)
			gw = w[:, gs_:ge]
			mx = gw.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
			scales[:, g:g+1] = (mx / INT4_MAX).to(torch.float16)
			logical[:, gs_:ge] = (gw / mx * INT4_MAX).round().clamp(INT4_MIN, INT4_MAX).to(torch.int32)
		return logical, scales

	@staticmethod
	def decode(qweight, scales, qzeros, *, out_features, in_features, group_size=128):
		device = qweight.device
		weight = torch.empty(out_features, in_features, dtype=torch.float16, device=device)
		for g in range((in_features + group_size - 1) // group_size):
			gs_, ge = g * group_size, min(g * group_size + group_size, in_features)
			gz = ge - gs_
			qw_g = _unpack_int4_along_dim1(qweight[:, gs_//2:gs_//2 + (gz+1)//2], gz)
			sc_g = scales[:, g:g+1].expand(-1, gz)
			weight[:, gs_:ge] = qw_g.to(torch.float16) * sc_g
		return weight

	def quantize_and_store(self, weight: torch.Tensor) -> bytes:
		qweight, scales, qzeros = self.quantize(weight)
		qw = qweight.contiguous().cpu()
		sc = scales.contiguous().cpu()
		qz = qzeros.contiguous().cpu()
		hdr = torch.tensor([qw.numel(), sc.numel(), qz.numel()], dtype=torch.int32)
		return hdr.numpy().tobytes() + qw.numpy().tobytes() + sc.numpy().tobytes() + qz.numpy().tobytes()


def _pack_int4_along_dim1(x: torch.Tensor) -> torch.Tensor:
	n = x.shape[1]
	if n % 2: x = F.pad(x, (0, 1), value=0)
	return (x[:, 0::2].to(torch.uint8) & 0x0F) | ((x[:, 1::2].to(torch.uint8) & 0x0F) << 4)


def _unpack_int4_along_dim1(x: torch.Tensor, n: int) -> torch.Tensor:
	lo = x.to(torch.int16) & 0x0F
	hi = (x.to(torch.int16) >> 4) & 0x0F
	lo = torch.where(lo > 7, lo - 16, lo)
	hi = torch.where(hi > 7, hi - 16, hi)
	return torch.stack([lo, hi], dim=2).reshape(x.shape[0], -1)[:, :n].to(torch.int8)
