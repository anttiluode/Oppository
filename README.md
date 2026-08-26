# Oppository

## Concrete CUDA Video

**Make an expensive video AI earn every forward pass.**

Oppository is a live CUDA experiment/product prototype built around a real pretrained video model: Torchvision **RAFT-Large optical flow**. RAFT-Large is heavy enough to make the question nontrivial (Torchvision currently lists ~211 GFLOPs for the model). It takes two video frames and predicts a 2-D motion vector for every pixel.

The GUI runs a webcam or video through RAFT and lets you flip **CONCRETE ON/OFF** while watching the motion field and the GPU bill.

```text
                   SAME VIDEO STREAM
                         |
              cheap 48 x 27 certificate
                         |
              +----------+----------+
              |                     |
        CONCRETE OFF           CONCRETE ON
              |                     |
       run RAFT always       predicted flow change
              |                     |
         expensive               /     \
                              big       tiny
                               |          |
                           run RAFT     REUSE
                                          |
                                    sometimes audit
                                          |
                                    RAFT was wrong?
                                          |
                                      learn / trust less
```

This is not frame interpolation and not a fake workload. A `REUSE` means the RAFT forward call did not happen.

### What you see

The left pane is the input video. The right pane overlays RAFT's optical-flow field as color. The bottom bar shows:

- **EXECUTE** — full RAFT ran;
- **REUSE** — the previous flow result was reused and the GPU model call was skipped;
- **AUDIT** — Concrete predicted reuse was safe but deliberately ran RAFT anyway to measure the error;
- RAFT GPU milliseconds;
- live processed FPS;
- calls / executions / reuses;
- fraction of model calls actually removed;
- predicted and measured flow error in pixels;
- audit count and unsafe-audit count;
- CUDA memory.

Sit still and Concrete should progressively learn that many frame pairs do not justify another RAFT call. Move your hands or swing the camera and the input certificate should cross the learned validity boundary, waking RAFT again.

## Install on Windows / NVIDIA

Use a CUDA-enabled PyTorch build. PyTorch's current Windows install selector is the source of truth because the supported CUDA wheel changes over time:

https://pytorch.org/get-started/locally/

Then:

```bat
python3.13 -m pip install -r requirements.txt
```

Verify CUDA before doing anything else:

```bat
python3.13 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

You want `True` and your NVIDIA GPU name.

The first run downloads Torchvision's pretrained RAFT weights (~20 MB for RAFT-Large).

## Run the GUI

```bat
python3.13 app.py
```

Default is deliberately heavy:

```text
model       RAFT-Large
canvas      768 x 432
updates     20 recurrent flow updates
CUDA        on
Concrete    on
```

Higher load:

```bat
python3.13 app.py --preset "Burn 960x544" --updates 32
```

If that is too brutal:

```bat
python3.13 app.py --model small --preset "Medium 640x360" --updates 12
```

A CPU mode exists only as a sanity fallback:

```bat
python3.13 app.py --device cpu --model small
```

### Controls

**CONCRETE ON/OFF** is the important one. OFF still learns from every expensive execution, so spend 10-20 frames with it OFF if you want to seed the sensitivity model, then turn it ON and watch how many forwards disappear.

**Flow error tolerance** is the quality/speed knob. It is an audited RMS endpoint error in flow pixels. Smaller = cautious. Larger = aggressive.

**Audit %** is the exploration budget. A predicted-safe reuse is occasionally executed anyway. If those audits reveal > tolerance error, `unsafe` rises and the policy becomes more skeptical.

Changing resolution resets the learned validity map because the computation changed.

## Rigorous same-video benchmark

Live webcam toggling is fun but not a fair speed measurement because the scene is different each time. For a real contest, use one video twice:

```bat
python3.13 benchmark.py myvideo.mp4 --frames 100 --size 768x432 --updates 20
```

Pass 1 runs full RAFT on every pair and learns sensitivity. Pass 2 replays the exact same frames with Concrete enabled. It reports:

```text
full GPU seconds
Concrete GPU seconds
GPU compute speedup
full / Concrete wall seconds
wall speedup
reuse fraction
audits / unsafe audits
mean and p95 probe flow error
```

This is the number that decides whether the idea survives.

## What is learned

For every executed transition Concrete sees two quantities:

```text
cheap input drift d
    RMS change of two 48x27 grayscale frame sketches

actual output drift e
    RMS vector difference between new RAFT flow and cached RAFT flow, in pixels
```

It stores a bounded history of empirical gains `e / d`, uses a high quantile rather than the mean, multiplies that by a safety margin, and predicts:

```text
predicted flow error = current input drift * conservative learned gain
```

If predicted error is below the selected pixel tolerance, reuse becomes eligible. A fraction of eligible hits are audited. An unsafe audit increases future audit pressure.

The certificate stays on the CPU. That is intentional. If checking whether RAFT is necessary requires uploading full frames and running another GPU network, the certificate has already lost.

## Why RAFT

This is a clean first real target because:

1. it is genuinely video-native: two frames in, dense motion out;
2. the output is visible rather than hidden in a benchmark scalar;
3. the model is expensive enough for skipping a forward to matter;
4. ordinary video contains enormous temporal redundancy;
5. pretrained RAFT is in Torchvision, so there is no custom checkpoint zoo or model-service dependency.

The point is **not** that optical flow should be cached forever. The point is that a receiver should only wake when the world has changed enough to invalidate what it already knows.

## Failure conditions

Oppository loses if any of these happen on real CUDA video:

- the cheap certificate + GUI overhead destroys the saved time;
- useful scenes force RAFT to execute nearly every frame;
- aggressive reuse creates visible stale motion before audits catch it;
- safe settings give only ~1.05x wall speedup;
- a simpler fixed frame-difference threshold matches the adaptive policy.

That last attacker matters. The adaptive learner only earns its complexity if its learned validity boundary beats a boring hand-set threshold across different videos.

## First scoreboard

Run the same video at a load your GPU actually feels and write down:

```text
Concrete OFF GPU seconds     ______
Concrete ON GPU seconds      ______
GPU speedup                  ______ x
wall speedup                 ______ x
reuse                        ______ %
unsafe audits                ______ / ______
visible failure?             yes / no
```

Interpretation:

```text
~1.05x      crumble
~1.5x       useful
2x+         oh.
```

Let's find out.
