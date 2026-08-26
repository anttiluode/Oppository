from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque

import cv2
import numpy as np


@dataclass
class CacheConfig:
    output_tolerance_px: float = 1.5
    min_observations: int = 8
    history: int = 96
    quantile: float = 0.90
    safety_margin: float = 1.6
    audit_rate: float = 0.10
    min_audit_rate: float = 0.02
    failure_boost: float = 0.35
    sketch_width: int = 48
    sketch_height: int = 27
    seed: int = 0
    eps: float = 1e-6


@dataclass
class CacheStats:
    calls: int = 0
    executions: int = 0
    reuses: int = 0
    audits: int = 0
    unsafe_audits: int = 0
    gpu_ms: float = 0.0
    estimated_gpu_ms_saved: float = 0.0
    last_input_drift: float = math.inf
    last_predicted_error_px: float = math.inf
    last_observed_error_px: float = math.inf
    learned_gain: float = math.inf
    last_action: str = "COLD"

    @property
    def reuse_rate(self) -> float:
        return self.reuses / self.calls if self.calls else 0.0

    @property
    def unsafe_rate(self) -> float:
        return self.unsafe_audits / self.audits if self.audits else 0.0

    @property
    def mean_gpu_ms(self) -> float:
        return self.gpu_ms / self.executions if self.executions else 0.0

    @property
    def compute_reduction(self) -> float:
        return self.reuses / self.calls if self.calls else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            reuse_rate=self.reuse_rate,
            unsafe_rate=self.unsafe_rate,
            mean_gpu_ms=self.mean_gpu_ms,
            compute_reduction=self.compute_reduction,
        )
        return d


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return math.inf
    xs = sorted(values)
    q = min(1.0, max(0.0, q))
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    f = pos - lo
    return xs[lo] * (1.0 - f) + xs[hi] * f


def frame_pair_sketch(prev_bgr: np.ndarray, curr_bgr: np.ndarray, cfg: CacheConfig) -> np.ndarray:
    """Tiny CPU certificate input: two low-resolution grayscale frames."""
    def one(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tiny = cv2.resize(gray, (cfg.sketch_width, cfg.sketch_height), interpolation=cv2.INTER_AREA)
        return tiny.astype(np.float32) / 255.0

    return np.stack((one(prev_bgr), one(curr_bgr)), axis=0)


def sketch_drift(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return math.inf
    diff = a.astype(np.float32, copy=False) - b.astype(np.float32, copy=False)
    return float(np.sqrt(np.mean(diff * diff)))


class AdaptiveFlowPolicy:
    """Learn when a previous optical-flow result is still worth reusing.

    The policy only decides; it knows nothing about PyTorch. Executed RAFT calls
    report the actual flow change through ``observe_execution``. Cache hits are
    periodically audited so the policy can discover false confidence.
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        self._rng = random.Random(self.config.seed)
        self._gains: Deque[float] = deque(maxlen=self.config.history)
        self.reference_sketch: np.ndarray | None = None
        self.has_output = False

    def reset(self, keep_learning: bool = False) -> None:
        self.reference_sketch = None
        self.has_output = False
        self.stats = CacheStats()
        if not keep_learning:
            self._gains.clear()

    def learned_gain(self) -> float:
        if len(self._gains) < self.config.min_observations:
            return math.inf
        base = _quantile(list(self._gains), self.config.quantile)
        return max(0.0, base * self.config.safety_margin)

    def audit_probability(self) -> float:
        base = max(self.config.min_audit_rate, self.config.audit_rate)
        return min(1.0, base + self.config.failure_boost * self.stats.unsafe_rate)

    def decide(self, sketch: np.ndarray, enabled: bool) -> tuple[str, float, float]:
        """Return (EXECUTE|AUDIT|REUSE, input_drift, predicted_flow_error_px)."""
        self.stats.calls += 1
        if not self.has_output or self.reference_sketch is None:
            self.stats.last_action = "EXECUTE"
            return "EXECUTE", math.inf, math.inf

        d = sketch_drift(sketch, self.reference_sketch)
        gain = self.learned_gain()
        predicted = d * gain if math.isfinite(gain) else math.inf
        self.stats.last_input_drift = d
        self.stats.last_predicted_error_px = predicted
        self.stats.learned_gain = gain

        # OFF still creates useful training data: execute everything and learn.
        if not enabled or predicted > self.config.output_tolerance_px or not math.isfinite(predicted):
            self.stats.last_action = "EXECUTE"
            return "EXECUTE", d, predicted

        if self._rng.random() < self.audit_probability():
            self.stats.audits += 1
            self.stats.last_action = "AUDIT"
            return "AUDIT", d, predicted

        self.stats.reuses += 1
        self.stats.last_action = "REUSE"
        return "REUSE", d, predicted

    def observe_execution(
        self,
        sketch: np.ndarray,
        input_drift: float,
        output_error_px: float,
        gpu_ms: float,
        was_audit: bool = False,
    ) -> None:
        self.stats.executions += 1
        self.stats.gpu_ms += max(0.0, float(gpu_ms))
        self.stats.last_observed_error_px = float(output_error_px)

        if math.isfinite(input_drift) and math.isfinite(output_error_px):
            if input_drift <= self.config.eps:
                gain = 0.0 if output_error_px <= self.config.output_tolerance_px else output_error_px / self.config.eps
            else:
                gain = output_error_px / input_drift
            if math.isfinite(gain):
                self._gains.append(float(gain))

        if was_audit and output_error_px > self.config.output_tolerance_px:
            self.stats.unsafe_audits += 1

        self.reference_sketch = sketch.copy()
        self.has_output = True
        self.stats.learned_gain = self.learned_gain()

    def note_reuse(self) -> None:
        self.stats.estimated_gpu_ms_saved += self.stats.mean_gpu_ms

    def summary(self) -> dict:
        d = self.stats.to_dict()
        d["observations"] = len(self._gains)
        d["audit_probability"] = self.audit_probability()
        return d
