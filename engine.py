from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch
from torchvision.models.optical_flow import (
    Raft_Large_Weights,
    Raft_Small_Weights,
    raft_large,
    raft_small,
)

from concrete_cache import AdaptiveFlowPolicy, CacheConfig, frame_pair_sketch


@dataclass
class FlowResult:
    flow_bgr: np.ndarray
    overlay_bgr: np.ndarray
    action: str
    gpu_ms: float
    wall_ms: float
    input_drift: float
    predicted_error_px: float
    observed_error_px: float
    stats: dict


def _round8(v: int) -> int:
    return max(128, (int(v) // 8) * 8)


def resize_letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    width, height = _round8(width), _round8(height)
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(8, int(round(w * scale))), max(8, int(round(h * scale)))
    nw, nh = _round8(nw), _round8(nh)
    nw, nh = min(width, nw), min(height, nh)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def flow_to_bgr(flow: torch.Tensor) -> np.ndarray:
    f = flow[0].detach().float().cpu().numpy()
    fx, fy = f[0], f[1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    scale = float(np.percentile(mag, 95.0)) if mag.size else 0.0
    scale = max(scale, 1.0)
    hsv = np.zeros((f.shape[1], f.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(ang / 2.0, 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * (255.0 / scale), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flow_rms_error_px(current: torch.Tensor, reference: torch.Tensor | None) -> float:
    if reference is None or current.shape != reference.shape:
        return math.inf
    diff = current.float() - reference.float()
    err2 = torch.sum(diff * diff, dim=1)
    return float(torch.sqrt(torch.mean(err2)).item())


class RaftConcreteEngine:
    def __init__(
        self,
        model_size: str = "large",
        device: str = "cuda",
        width: int = 768,
        height: int = 432,
        updates: int = 20,
        cache_config: CacheConfig | None = None,
        status_cb: Callable[[str], None] | None = None,
    ):
        self.model_size = model_size
        self.device = torch.device(device)
        self.width = _round8(width)
        self.height = _round8(height)
        self.updates = int(updates)
        self.status_cb = status_cb or (lambda _: None)
        self.policy = AdaptiveFlowPolicy(cache_config or CacheConfig())
        self.model = None
        self.cached_flow: torch.Tensor | None = None
        self.cached_flow_bgr: np.ndarray | None = None
        self._load_model()

    def _status(self, text: str) -> None:
        self.status_cb(text)

    def _load_model(self) -> None:
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build.")
        self._status(f"Loading RAFT-{self.model_size} weights...")
        if self.model_size == "small":
            self.model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=True)
        else:
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True)
        self.model = self.model.eval().to(self.device)
        self._status(f"RAFT-{self.model_size} ready on {self.device}")

    def reset(self, keep_learning: bool = False) -> None:
        self.policy.reset(keep_learning=keep_learning)
        self.cached_flow = None
        self.cached_flow_bgr = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def configure(self, *, width: int | None = None, height: int | None = None, updates: int | None = None) -> None:
        changed_shape = False
        if width is not None and _round8(width) != self.width:
            self.width = _round8(width)
            changed_shape = True
        if height is not None and _round8(height) != self.height:
            self.height = _round8(height)
            changed_shape = True
        if updates is not None:
            self.updates = int(updates)
        if changed_shape:
            self.reset(keep_learning=False)

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device, non_blocking=True)
        return t.float().div_(127.5).sub_(1.0)

    @torch.inference_mode()
    def _run_raft(self, prev: np.ndarray, curr: np.ndarray) -> tuple[torch.Tensor, float, float]:
        a = self._to_tensor(prev)
        b = self._to_tensor(curr)
        wall0 = time.perf_counter()
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            flows = self.model(a, b, num_flow_updates=self.updates)
            end.record()
            end.synchronize()
            gpu_ms = float(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            flows = self.model(a, b, num_flow_updates=self.updates)
            gpu_ms = (time.perf_counter() - t0) * 1000.0
        wall_ms = (time.perf_counter() - wall0) * 1000.0
        return flows[-1].detach(), gpu_ms, wall_ms

    def process(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray, concrete_enabled: bool = True) -> FlowResult:
        prev = resize_letterbox(prev_bgr, self.width, self.height)
        curr = resize_letterbox(curr_bgr, self.width, self.height)
        sketch = frame_pair_sketch(prev, curr, self.policy.config)
        action, input_drift, predicted = self.policy.decide(sketch, enabled=concrete_enabled)

        if action == "REUSE" and self.cached_flow is not None and self.cached_flow_bgr is not None:
            self.policy.note_reuse()
            overlay = cv2.addWeighted(curr, 0.62, self.cached_flow_bgr, 0.38, 0.0)
            return FlowResult(
                self.cached_flow_bgr.copy(), overlay, action, 0.0, 0.0,
                input_drift, predicted, math.nan, self.policy.summary()
            )

        old_flow = self.cached_flow
        flow, gpu_ms, wall_ms = self._run_raft(prev, curr)
        observed = flow_rms_error_px(flow, old_flow)
        self.policy.observe_execution(
            sketch,
            input_drift=input_drift,
            output_error_px=observed,
            gpu_ms=gpu_ms,
            was_audit=(action == "AUDIT"),
        )
        self.cached_flow = flow
        self.cached_flow_bgr = flow_to_bgr(flow)
        overlay = cv2.addWeighted(curr, 0.62, self.cached_flow_bgr, 0.38, 0.0)
        return FlowResult(
            self.cached_flow_bgr.copy(), overlay, action, gpu_ms, wall_ms,
            input_drift, predicted, observed, self.policy.summary()
        )

    def gpu_info(self) -> dict:
        if self.device.type != "cuda":
            return {"device": str(self.device), "allocated_mb": 0.0, "reserved_mb": 0.0}
        idx = self.device.index or 0
        return {
            "device": torch.cuda.get_device_name(idx),
            "allocated_mb": torch.cuda.memory_allocated(idx) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(idx) / 1024**2,
        }
