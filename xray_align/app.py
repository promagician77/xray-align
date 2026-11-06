"""Live alignment app.

    python -m xray_align.app --camera 0 --calib camera.json        # USB camera
    python -m xray_align.app --sim                                 # synthetic camera, keys move the tube
    python -m xray_align.app --sim --demo --record demo.mp4 --headless   # scripted demo video

Keys: X mark exposure · A acceptable · U unacceptable · Z zero (tube aligned) · E export CSV · Q quit
Sim keys: J / L turn the tube, I / K tilt it, F / H / T / G move it, R reset (no clash with X A U Z E Q).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from .detect import Smoother, detect, make_detector, ring_pose
from .geometry import RingLayout, alignment_from_pose, rot
from .overlay import W, H, compose, draw_camera
from .render import Camera, render_ring
from .risk import RiskModel, fit, recommendation
from .store import Store


class SimSource:
    """Synthetic camera. Pose is set by keys, or by the demo script."""

    def __init__(self, demo: bool = False, seed: int = 7):
        self.cam, self.layout = Camera(), RingLayout()
        self.rng = np.random.default_rng(seed)
        self.h, self.v, self.dx, self.dy, self.d, self.roll = -7.0, 3.0, 6.0, -3.0, 160.0, 12.0
        self.demo, self.t = demo, 0.0
        self.label = "SIMULATED CAMERA"

    def script(self, t):
        """Demo: start 7 deg off, correct to aligned, exposure (acceptable); drift, exposure (unacceptable)."""
        def ease(a, b, t0, t1):
            u = min(1, max(0, (t - t0) / (t1 - t0))); u = u * u * (3 - 2 * u)
            return a + (b - a) * u
        wob = 0.25 * math.sin(2.1 * t) + 0.15 * math.sin(5.3 * t)
        if t < 10:
            h = ease(-7.0, 0.6, 3.5, 8.5); v = ease(3.0, -0.4, 4.0, 8.0); dx = ease(6.0, 1.0, 5.0, 8.5); dy = ease(-3.0, 0.5, 5.0, 8.5)
        else:
            h = ease(0.6, 6.5, 12.0, 15.0); v = ease(-0.4, -2.0, 12.0, 15.0); dx = ease(1.0, -2.0, 12.0, 15.0); dy = 0.5
        events = {10.0: "x", 10.8: "a", 16.6: "x", 17.4: "u"}
        key = next((k for tt, k in events.items() if abs(t - tt) < 1 / 60), None)
        return h + wob * 0.3, v + wob * 0.2, dx, dy, key

    def read(self, key=None, dt=1 / 30):
        ev = None
        if self.demo:
            self.h, self.v, self.dx, self.dy, ev = self.script(self.t)
        elif key is not None:
            step = {ord("j"): ("h", -0.5), ord("l"): ("h", 0.5), ord("i"): ("v", 0.5), ord("k"): ("v", -0.5),
                    ord("f"): ("dx", -1), ord("h"): ("dx", 1), ord("t"): ("dy", -1), ord("g"): ("dy", 1)}.get(key)
            if step:
                setattr(self, step[0], getattr(self, step[0]) + step[1])
            if key == ord("r"):
                self.h = self.v = self.dx = self.dy = 0.0
        self.t += dt
        R = rot(self.h, self.v, self.roll)          # ring orientation seen from the tube
        img = render_ring(self.cam, self.layout, R, np.array([self.dx, self.dy, self.d]), self.rng, noise=3.0, blur=0.8)
        return img, ev


class CamSource:
    def __init__(self, index: int, width: int, height: int, focus: float | None):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)            # autofocus changes the focal length: keep it fixed
        if focus is not None:
            self.cap.set(cv2.CAP_PROP_FOCUS, focus)
        self.label = f"USB camera {index}"
        if not self.cap.isOpened():
            raise SystemExit(f"camera {index} could not be opened")

    def read(self, key=None, dt=None):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None


class VideoSource(CamSource):
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path)); self.label = Path(path).name


def load_calib(path, fallback: Camera | None):
    if path:
        c = json.loads(Path(path).read_text())
        return np.array(c["K"], float), np.array(c["dist"], float)
    if fallback is not None:
        return fallback.K, fallback.dist
    raise SystemExit("--calib camera.json is required for a real camera (python -m xray_align.calibrate)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int); src.add_argument("--video"); src.add_argument("--sim", action="store_true")
    ap.add_argument("--demo", action="store_true", help="with --sim: scripted 20 s demonstration")
    ap.add_argument("--calib"); ap.add_argument("--layout", help="ring layout JSON (default: 4 x 12 mm markers, 26 mm radius)")
    ap.add_argument("--db", default="data/exposures.sqlite"); ap.add_argument("--operator", default="")
    ap.add_argument("--tol-deg", type=float, default=2.0); ap.add_argument("--tol-mm", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=800); ap.add_argument("--focus", type=float)
    ap.add_argument("--record"); ap.add_argument("--headless", action="store_true"); ap.add_argument("--seconds", type=float, default=19.0)
    a = ap.parse_args(argv)

    layout = RingLayout.load(a.layout) if a.layout else RingLayout()
    if a.sim:
        source = SimSource(demo=a.demo); K, dist = load_calib(a.calib, source.cam)
    elif a.video:
        source = VideoSource(a.video); K, dist = load_calib(a.calib, None)
    else:
        source = CamSource(a.camera, a.width, a.height, a.focus); K, dist = load_calib(a.calib, None)
    det, smooth, store = make_detector(layout), Smoother(0.35), Store(a.db)
    xs, ys = store.labelled()
    model = fit(np.array(xs), np.array(ys)) or RiskModel()
    zero, marked, message = None, None, "Ready."
    writer = None
    if a.record:
        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
        writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.record], stdin=subprocess.PIPE)
    key, frames, t_start, fps, last = None, 0, time.perf_counter(), 0.0, time.perf_counter()
    while True:
        frame, ev = source.read(key)
        if frame is None:
            break
        key = ev and ord(ev) or key
        t_det = time.perf_counter()
        corners, ids = detect(det, frame)
        pose = ring_pose(corners, ids, layout, K, dist)
        proc_ms = 1000 * (time.perf_counter() - t_det)
        m = None
        if pose is not None:
            al = alignment_from_pose(pose.R, pose.t, zero).as_dict()
            disp = smooth({k: al[k] for k in ("horizontal_deg", "vertical_deg", "total_deg", "dx_mm", "dy_mm", "distance_mm")})
            m = {**disp, "markers_seen": len(pose.ids), "reproj_px": round(pose.reproj_px, 3), "raw": al}
        else:
            smooth.reset()
        aligned = m is not None and abs(m["horizontal_deg"]) <= a.tol_deg and abs(m["vertical_deg"]) <= a.tol_deg and abs(m["dx_mm"]) <= a.tol_mm and abs(m["dy_mm"]) <= a.tol_mm
        risk = model(m["total_deg"]) if m else 0.0
        # keys
        k = chr(key).lower() if key is not None and 0 < key < 256 else ""
        if k == "q":
            break
        if k == "z" and pose is not None:
            zero = pose.R.copy()                     # raw pose: alignment is measured relative to this
            message = "Zeroed: the current tube position is the reference (camera mounting offset removed)."
        if k == "x" and m is not None:
            marked = {"m": dict(m["raw"], markers_seen=m["markers_seen"], reproj_px=m["reproj_px"]), "risk": risk}
            message = f"Exposure marked at {m['total_deg']:.1f}°. Press A or U after viewing the image."
        if k in ("a", "u") and (marked or m):
            rec = marked or {"m": dict(m["raw"], markers_seen=m["markers_seen"], reproj_px=m["reproj_px"]), "risk": risk}
            fb = "acceptable" if k == "a" else "unacceptable"
            rid = store.add(rec["m"], fb, 100 * rec["risk"], model.label(), a.operator)
            xs, ys = store.labelled(); model = fit(np.array(xs), np.array(ys)) or model
            message = f"Saved #{rid}: image {fb} at {rec['m']['total_deg']:.1f}° (risk shown {100 * rec['risk']:.0f}%)."
            marked = None
        if k == "e":
            n = store.export_csv(Path(a.db).with_suffix(".csv")); message = f"Exported {n} rows to {Path(a.db).with_suffix('.csv')}"
        cam = draw_camera(frame, corners, ids, pose, K, dist, layout, aligned)
        state = {"m": m, "aligned": aligned, "tol_deg": a.tol_deg, "tol_mm": a.tol_mm, "risk": risk, "risk_label": model.label(),
                 "recs": recommendation(m["horizontal_deg"], m["vertical_deg"], m["dx_mm"], m["dy_mm"], a.tol_deg, a.tol_mm) if m else [],
                 "source": source.label, "fps": fps, "proc_ms": proc_ms, "message": message, "zero_label": "zeroed" if zero is not None else "not zeroed (press Z with the tube aligned)"}
        screen = compose(cam, state)
        if writer:
            writer.stdin.write(screen.tobytes())
        key = None
        if not a.headless:
            cv2.imshow("X-ray tube alignment", screen)
            kk = cv2.waitKey(1)
            key = kk if kk >= 0 else None
        frames += 1
        now = time.perf_counter(); fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-6)); last = now
        if a.sim and a.demo and source.t >= a.seconds:
            break
    if writer:
        writer.stdin.close(); writer.wait()
    store.close()
    if not a.headless:
        cv2.destroyAllWindows()
    print(f"{frames} frames, {frames / (time.perf_counter() - t_start):.1f} fps processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
