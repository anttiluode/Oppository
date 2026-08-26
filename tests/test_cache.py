import math
import numpy as np

from concrete_cache import AdaptiveFlowPolicy, CacheConfig, sketch_drift


def make(v):
    return np.full((2, 8, 8), v, dtype=np.float32)


def test_sketch_drift_is_rms():
    assert abs(sketch_drift(make(0.0), make(0.25)) - 0.25) < 1e-6


def test_policy_learns_safe_reuse():
    cfg = CacheConfig(min_observations=3, audit_rate=0.0, min_audit_rate=0.0, output_tolerance_px=1.0, safety_margin=1.0)
    p = AdaptiveFlowPolicy(cfg)
    ref = make(0.0)
    action, d, pred = p.decide(ref, enabled=False)
    assert action == "EXECUTE"
    p.observe_execution(ref, d, math.inf, 10.0)
    for x in [0.01, 0.02, 0.03]:
        s = make(x)
        action, d, pred = p.decide(s, enabled=False)
        p.observe_execution(s, d, 0.2, 10.0)
    s = make(0.031)
    action, d, pred = p.decide(s, enabled=True)
    assert action == "REUSE"
    p.note_reuse()
    assert p.stats.reuses == 1
    assert p.stats.estimated_gpu_ms_saved > 0


def test_audit_failure_increases_distrust():
    cfg = CacheConfig(min_observations=1, audit_rate=1.0, output_tolerance_px=1.0, safety_margin=1.0)
    p = AdaptiveFlowPolicy(cfg)
    p.observe_execution(make(0), math.inf, math.inf, 5.0)
    p._gains.append(1.0)
    action, d, pred = p.decide(make(0.01), enabled=True)
    assert action == "AUDIT"
    p.observe_execution(make(0.01), d, 5.0, 5.0, was_audit=True)
    assert p.stats.unsafe_audits == 1
    assert p.audit_probability() >= cfg.audit_rate
