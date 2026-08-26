from __future__ import annotations

import argparse
import math
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
from PIL import Image, ImageTk

from concrete_cache import CacheConfig
from engine import RaftConcreteEngine


PRESETS = {
    "Medium 640x360": (640, 360),
    "Heavy 768x432": (768, 432),
    "Burn 960x544": (960, 544),
    "Very Burn 1280x720": (1280, 720),
}


class OppositoryApp:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.root.title("Oppository — Concrete CUDA Video")
        self.root.geometry("1450x900")
        self.root.minsize(1100, 700)
        self.stop_event = threading.Event()
        self.frame_q: queue.Queue = queue.Queue(maxsize=1)
        self.command_q: queue.Queue = queue.Queue()
        self.last_packet = None
        self.photo_left = None
        self.photo_right = None
        self.concrete_enabled = True
        self._build_ui()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        self.root.after(30, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        self.concrete = tk.BooleanVar(value=True)
        self.concrete_btn = ttk.Checkbutton(top, text="CONCRETE ON", variable=self.concrete, command=self._toggle_label)
        self.concrete_btn.grid(row=0, column=0, padx=4)

        ttk.Button(top, text="Webcam", command=lambda: self.command_q.put(("source", 0))).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Open video…", command=self._open_video).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Reset learning", command=lambda: self.command_q.put(("reset", False))).grid(row=0, column=3, padx=4)

        ttk.Label(top, text="Load:").grid(row=0, column=4, padx=(18, 2))
        self.preset = tk.StringVar(value=self.args.preset)
        cb = ttk.Combobox(top, textvariable=self.preset, values=list(PRESETS), width=19, state="readonly")
        cb.grid(row=0, column=5, padx=4)
        cb.bind("<<ComboboxSelected>>", lambda e: self.command_q.put(("preset", self.preset.get())))

        ttk.Label(top, text="RAFT updates:").grid(row=0, column=6, padx=(18, 2))
        self.updates = tk.IntVar(value=self.args.updates)
        cb2 = ttk.Combobox(top, textvariable=self.updates, values=[12, 20, 32], width=5, state="readonly")
        cb2.grid(row=0, column=7)
        cb2.bind("<<ComboboxSelected>>", lambda e: self.command_q.put(("updates", int(self.updates.get()))))

        controls = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        controls.pack(fill="x")
        ttk.Label(controls, text="Flow error tolerance (px)").grid(row=0, column=0, sticky="w")
        self.tolerance = tk.DoubleVar(value=self.args.tolerance)
        scale = ttk.Scale(controls, from_=0.25, to=6.0, variable=self.tolerance, command=self._policy_changed)
        scale.grid(row=0, column=1, sticky="ew", padx=6)
        self.tol_label = ttk.Label(controls, text=f"{self.tolerance.get():.2f}")
        self.tol_label.grid(row=0, column=2, padx=(0, 14))
        ttk.Label(controls, text="Audit %").grid(row=0, column=3)
        self.audit = tk.DoubleVar(value=self.args.audit * 100)
        scale2 = ttk.Scale(controls, from_=0, to=50, variable=self.audit, command=self._policy_changed)
        scale2.grid(row=0, column=4, sticky="ew", padx=6)
        self.audit_label = ttk.Label(controls, text=f"{self.audit.get():.0f}%")
        self.audit_label.grid(row=0, column=5)
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(4, weight=1)

        panes = ttk.Frame(self.root, padding=8)
        panes.pack(fill="both", expand=True)
        ttk.Label(panes, text="INPUT", anchor="center", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="ew")
        ttk.Label(panes, text="RAFT OPTICAL FLOW", anchor="center", font=("Segoe UI", 13, "bold")).grid(row=0, column=1, sticky="ew")
        self.left = ttk.Label(panes, anchor="center")
        self.right = ttk.Label(panes, anchor="center")
        self.left.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.right.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        panes.rowconfigure(1, weight=1)
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)

        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(fill="x")
        self.action = ttk.Label(bottom, text="LOADING", font=("Consolas", 20, "bold"), width=10)
        self.action.grid(row=0, column=0, rowspan=2, padx=(0, 16))
        self.stats1 = ttk.Label(bottom, text="", font=("Consolas", 10))
        self.stats1.grid(row=0, column=1, sticky="w")
        self.stats2 = ttk.Label(bottom, text="", font=("Consolas", 10))
        self.stats2.grid(row=1, column=1, sticky="w")
        self.status = ttk.Label(bottom, text="Starting…", anchor="e")
        self.status.grid(row=0, column=2, rowspan=2, sticky="e")
        bottom.columnconfigure(1, weight=1)

    def _toggle_label(self):
        self.concrete_enabled = bool(self.concrete.get())
        self.concrete_btn.configure(text="CONCRETE ON" if self.concrete_enabled else "CONCRETE OFF — FULL RAFT")

    def _policy_changed(self, _=None):
        self.tol_label.configure(text=f"{self.tolerance.get():.2f}")
        self.audit_label.configure(text=f"{self.audit.get():.0f}%")
        self.command_q.put(("policy", (float(self.tolerance.get()), float(self.audit.get()) / 100.0)))

    def _open_video(self):
        path = filedialog.askopenfilename(
            title="Open video",
            filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv *.webm"), ("All files", "*.*")],
        )
        if path:
            self.command_q.put(("source", path))

    def _open_capture(self, source):
        cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")
        return cap

    def _worker_loop(self):
        try:
            w, h = PRESETS[self.args.preset]
            cfg = CacheConfig(output_tolerance_px=self.args.tolerance, audit_rate=self.args.audit)
            engine = RaftConcreteEngine(
                model_size=self.args.model,
                device=self.args.device,
                width=w,
                height=h,
                updates=self.args.updates,
                cache_config=cfg,
                status_cb=lambda s: self._emit({"status": s}),
            )
            source = 0 if self.args.source == "webcam" else self.args.source
            cap = self._open_capture(source)
            source_is_camera = isinstance(source, int)
            prev = None
            last_t = time.perf_counter()
            ema_fps = 0.0
            while not self.stop_event.is_set():
                while True:
                    try:
                        cmd, value = self.command_q.get_nowait()
                    except queue.Empty:
                        break
                    if cmd == "source":
                        cap.release()
                        cap = self._open_capture(value)
                        source_is_camera = isinstance(value, int)
                        prev = None
                        engine.reset(keep_learning=False)
                    elif cmd == "reset":
                        engine.reset(keep_learning=bool(value))
                        prev = None
                    elif cmd == "preset":
                        w, h = PRESETS[value]
                        engine.configure(width=w, height=h)
                        prev = None
                    elif cmd == "updates":
                        engine.configure(updates=int(value))
                    elif cmd == "policy":
                        tol, audit = value
                        engine.policy.config.output_tolerance_px = tol
                        engine.policy.config.audit_rate = audit

                ok, frame = cap.read()
                if not ok:
                    if not source_is_camera:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.02)
                    prev = None
                    continue
                if prev is None:
                    prev = frame
                    continue

                result = engine.process(prev, frame, concrete_enabled=self.concrete_enabled)
                prev = frame
                now = time.perf_counter()
                dt = now - last_t
                last_t = now
                inst = 1.0 / dt if dt > 0 else 0.0
                ema_fps = inst if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst
                display_input = cv2.resize(frame, (result.flow_bgr.shape[1], result.flow_bgr.shape[0]))
                self._emit({
                    "input": display_input,
                    "flow": result.overlay_bgr,
                    "action": result.action,
                    "gpu_ms": result.gpu_ms,
                    "fps": ema_fps,
                    "stats": result.stats,
                    "gpu": engine.gpu_info(),
                    "pred": result.predicted_error_px,
                    "obs": result.observed_error_px,
                    "drift": result.input_drift,
                })
            cap.release()
        except Exception as e:
            self._emit({"status": f"ERROR: {type(e).__name__}: {e}", "error": True})

    def _emit(self, packet):
        try:
            if self.frame_q.full():
                self.frame_q.get_nowait()
            self.frame_q.put_nowait(packet)
        except (queue.Empty, queue.Full):
            pass

    @staticmethod
    def _photo(bgr, max_w: int, max_h: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale != 1.0:
            rgb = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def _poll(self):
        try:
            while True:
                self.last_packet = self.frame_q.get_nowait()
        except queue.Empty:
            pass
        p = self.last_packet
        if p:
            if "input" in p:
                max_w = max(320, self.root.winfo_width() // 2 - 40)
                max_h = max(260, self.root.winfo_height() - 300)
                self.photo_left = self._photo(p["input"], max_w, max_h)
                self.photo_right = self._photo(p["flow"], max_w, max_h)
                self.left.configure(image=self.photo_left)
                self.right.configure(image=self.photo_right)
                self.action.configure(text=p["action"])
                s = p["stats"]
                g = p["gpu"]
                pred = p["pred"]
                obs = p["obs"]
                self.stats1.configure(text=(
                    f"FPS {p['fps']:6.2f} | last RAFT GPU {p['gpu_ms']:7.1f} ms | "
                    f"calls {s['calls']}  exec {s['executions']}  reuse {s['reuses']} ({s['reuse_rate']:.1%}) | "
                    f"compute cut {s['compute_reduction']:.1%}"
                ))
                self.stats2.configure(text=(
                    f"certificate drift {p['drift']:.5f} | predicted flow error "
                    f"{pred if math.isfinite(pred) else float('nan'):.2f}px | observed "
                    f"{obs if math.isfinite(obs) else float('nan'):.2f}px | audits {s['audits']} unsafe {s['unsafe_audits']} | "
                    f"GPU mem {g['allocated_mb']:.0f}/{g['reserved_mb']:.0f} MB"
                ))
                self.status.configure(text=g["device"])
            elif "status" in p:
                self.status.configure(text=p["status"])
        self.root.after(30, self._poll)

    def _close(self):
        self.stop_event.set()
        self.root.destroy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="webcam", help="webcam or video path")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--model", default="large", choices=["large", "small"])
    p.add_argument("--preset", default="Heavy 768x432", choices=list(PRESETS))
    p.add_argument("--updates", type=int, default=20, choices=[12, 20, 32])
    p.add_argument("--tolerance", type=float, default=1.5)
    p.add_argument("--audit", type=float, default=0.10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = tk.Tk()
    app = OppositoryApp(root, args)
    root.mainloop()
