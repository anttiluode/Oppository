from __future__ import annotations

import argparse
import math
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from concrete_cache import CacheConfig
from engine import RaftConcreteEngine


def read_frames(path: str, limit: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < 2:
        raise SystemExit("Need at least two frames")
    return frames


def main() -> int:
    p = argparse.ArgumentParser(description="Replay one video through full RAFT and Concrete RAFT.")
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=80)
    p.add_argument("--size", default="768x432")
    p.add_argument("--updates", type=int, default=20)
    p.add_argument("--model", choices=["large", "small"], default="large")
    p.add_argument("--tolerance", type=float, default=1.5, help="audited flow RMS tolerance in pixels")
    p.add_argument("--audit", type=float, default=0.10)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark is intended for CUDA. torch.cuda.is_available() is False.")
    w, h = (int(x) for x in args.size.lower().split("x", 1))
    frames = read_frames(args.video, args.frames)
    cfg = CacheConfig(output_tolerance_px=args.tolerance, audit_rate=args.audit)
    engine = RaftConcreteEngine(args.model, "cuda", w, h, args.updates, cfg, print)

    print("\nPASS 1/2: Concrete OFF — full RAFT on every frame pair")
    engine.reset(keep_learning=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    off_gpu_ms = 0.0
    baseline_probes = []
    for i in range(1, len(frames)):
        r = engine.process(frames[i - 1], frames[i], concrete_enabled=False)
        off_gpu_ms += r.gpu_ms
        probe = F.adaptive_avg_pool2d(engine.cached_flow.float(), (27, 48))[0].cpu().numpy().astype(np.float16)
        baseline_probes.append(probe)
        print(f"\r  {i:4d}/{len(frames)-1}  GPU {r.gpu_ms:7.1f} ms", end="")
    torch.cuda.synchronize()
    off_wall = time.perf_counter() - t0
    print(f"\n  full GPU time: {off_gpu_ms/1000:.2f}s, wall: {off_wall:.2f}s, learned observations: {len(engine.policy._gains)}")

    engine.reset(keep_learning=True)
    print("\nPASS 2/2: Concrete ON — same video, same model")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    on_gpu_ms = 0.0
    concrete_probes = []
    for i in range(1, len(frames)):
        r = engine.process(frames[i - 1], frames[i], concrete_enabled=True)
        on_gpu_ms += r.gpu_ms
        probe = F.adaptive_avg_pool2d(engine.cached_flow.float(), (27, 48))[0].cpu().numpy().astype(np.float16)
        concrete_probes.append(probe)
        print(f"\r  {i:4d}/{len(frames)-1}  {r.action:7s} GPU {r.gpu_ms:7.1f} ms", end="")
    torch.cuda.synchronize()
    on_wall = time.perf_counter() - t0
    s = engine.policy.summary()

    probe_errors = []
    for base, got in zip(baseline_probes, concrete_probes):
        d = got.astype(np.float32) - base.astype(np.float32)
        probe_errors.append(float(np.sqrt(np.mean(np.sum(d * d, axis=0)))))
    mean_probe_epe = float(np.mean(probe_errors)) if probe_errors else math.nan
    p95_probe_epe = float(np.percentile(probe_errors, 95)) if probe_errors else math.nan

    print("\n")
    print(f"full GPU seconds       {off_gpu_ms/1000:.3f}")
    print(f"Concrete GPU seconds   {on_gpu_ms/1000:.3f}")
    print(f"GPU compute speedup    {off_gpu_ms/max(on_gpu_ms, 1e-9):.2f}x")
    print(f"full wall seconds      {off_wall:.3f}")
    print(f"Concrete wall seconds  {on_wall:.3f}")
    print(f"wall speedup           {off_wall/max(on_wall, 1e-9):.2f}x")
    print(f"reused                 {s['reuses']} / {s['calls']} ({s['reuse_rate']:.1%})")
    print(f"audits / unsafe        {s['audits']} / {s['unsafe_audits']}")
    print(f"estimated GPU saved    {s['estimated_gpu_ms_saved']/1000:.3f}s")
    print(f"mean probe flow error  {mean_probe_epe:.3f}px")
    print(f"p95 probe flow error   {p95_probe_epe:.3f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
