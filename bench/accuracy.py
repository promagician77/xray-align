"""Measured accuracy against ground truth (synthetic camera with lens distortion, blur, noise).

    python bench/accuracy.py            -> results/accuracy.json

1. Calibrate the camera from 25 rendered ChArUco views (as you would with a printed board).
2. With the *calibrated* intrinsics, estimate the ring pose at 144 known poses
   (4 distances x 12 orientations x 3 noise draws, random centring offset and ring roll),
   by the whole ring (all markers at once) and by a single marker (tutorial approach).
3. Jitter: 60 frames of one still pose.
4. Camera mounting offset with and without zeroing.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xray_align.geometry import RingLayout, rot, alignment_from_pose            # noqa: E402
from xray_align.render import Camera, render_ring, charuco_board, render_board  # noqa: E402
from xray_align.detect import make_detector, detect, ring_pose, single_marker_pose, Smoother  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def normal(R):
    return np.asarray(R) @ np.array([0.0, 0.0, 1.0])


def angle_between(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))))))


def calibrate(cam: Camera, rng) -> dict:
    board = charuco_board()
    det = cv2.aruco.CharucoDetector(board)
    sx, sy = board.getChessboardSize(); sq = board.getSquareLength()
    objs, imgs, used = [], [], 0
    for k in range(25):
        R = rot(rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-25, 25))
        d = rng.uniform(170, 300)
        # put the board centre at a random spot in view
        c = np.array([rng.uniform(-0.25, 0.25) * d, rng.uniform(-0.2, 0.2) * d, d])
        t = c - R @ np.array([sx * sq / 2, sy * sq / 2, 0])
        img = render_board(cam, board, R, t, rng)
        cc, ci, _, _ = det.detectBoard(img)
        if ci is None or len(ci) < 8:
            continue
        o, i = board.matchImagePoints(cc, ci)
        objs.append(o); imgs.append(i); used += 1
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, (cam.w, cam.h), None, None)
    return {"views": used, "rms_px": float(rms), "K": K, "dist": dist.ravel(),
            "fx_err_pct": float(100 * (K[0, 0] - cam.K[0, 0]) / cam.K[0, 0]), "fy_err_pct": float(100 * (K[1, 1] - cam.K[1, 1]) / cam.K[1, 1]),
            "cx_err_px": float(K[0, 2] - cam.K[0, 2]), "cy_err_px": float(K[1, 2] - cam.K[1, 2]),
            "k1_true": float(cam.dist[0]), "k1_est": float(dist.ravel()[0])}


def main() -> int:
    rng = np.random.default_rng(2026)
    cam, L = Camera(), RingLayout()
    det = make_detector(L)
    t0 = time.perf_counter()
    cal = calibrate(cam, rng)
    K, dist = cal["K"], cal["dist"]
    print(f"calibration: {cal['views']} views, RMS {cal['rms_px']:.3f} px, fx err {cal['fx_err_pct']:+.3f}%, k1 {cal['k1_est']:+.4f} (true {cal['k1_true']:+.4f})")

    orients = [(0, 0), (1, 0), (-2, 0), (3, 1), (-5, 2), (7, 0), (-7, -3), (10, -5), (-12, 6), (15, 0), (0, -8), (4, -10)]
    rows = []
    timings = []
    for d in (100, 150, 200, 250):
        for yaw, pitch in orients:
            for rep in range(3):
                R = rot(yaw, pitch, rng.uniform(-20, 20))
                t = np.array([rng.uniform(-8, 8), rng.uniform(-8, 8), d])
                img = render_ring(cam, L, R, t, rng, noise=3.0, blur=rng.uniform(0.5, 1.2))
                s0 = time.perf_counter()
                c, ids = detect(det, img)
                pr = ring_pose(c, ids, L, K, dist)
                timings.append(1000 * (time.perf_counter() - s0))
                ps = single_marker_pose(c, ids, L, K, dist)
                gt = alignment_from_pose(R, t)
                row = {"d": d, "yaw": yaw, "pitch": pitch, "markers": len(ids), "true_h": gt.horizontal_deg, "true_v": gt.vertical_deg, "true_total": gt.total_deg}
                for name, p in (("ring", pr), ("single", ps)):
                    if p is None:
                        row[name] = None; continue
                    a = alignment_from_pose(p.R, p.t)
                    row[name] = {"err_deg": angle_between(normal(p.R), normal(R)), "h": a.horizontal_deg, "v": a.vertical_deg,
                                 "pos_err_mm": float(np.linalg.norm(p.t[:2] - t[:2])), "dist_err_mm": float(p.t[2] - t[2]), "reproj_px": p.reproj_px}
                rows.append(row)

    def summary(name):
        ok = [r for r in rows if r[name] is not None]
        e = np.array([r[name]["err_deg"] for r in ok])
        flips, comps = 0, 0
        for r in ok:
            for tv, mv in ((r["true_h"], r[name]["h"]), (r["true_v"], r[name]["v"])):
                if abs(tv) >= 2.0:
                    comps += 1; flips += int(np.sign(tv) != np.sign(mv))
        by_d = {d: float(np.sqrt(np.mean([r[name]["err_deg"] ** 2 for r in ok if r["d"] == d]))) for d in (100, 150, 200, 250)}
        near = [r[name]["err_deg"] for r in ok if r["true_total"] < 3.0]
        return {"frames": len(rows), "solved": len(ok), "angle_err_rms_deg": float(np.sqrt(np.mean(e ** 2))), "angle_err_p95_deg": float(np.percentile(e, 95)),
                "angle_err_max_deg": float(e.max()), "near_aligned_err_rms_deg": float(np.sqrt(np.mean(np.square(near)))) if near else None,
                "sign_flips": flips, "sign_checks": comps, "sign_flip_pct": 100.0 * flips / max(comps, 1),
                "pos_err_rms_mm": float(np.sqrt(np.mean([r[name]["pos_err_mm"] ** 2 for r in ok]))),
                "dist_err_rms_mm": float(np.sqrt(np.mean([r[name]["dist_err_mm"] ** 2 for r in ok]))), "rms_by_distance_deg": by_d}

    res = {"camera": {"w": cam.w, "h": cam.h, "hfov_deg": 70.0, "K_true": cam.K.tolist(), "dist_true": cam.dist.tolist()},
           "layout": {"markers": len(L.marker_ids), "marker_mm": L.marker_mm, "radius_mm": L.radius_mm, "dictionary": L.dictionary},
           "calibration": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cal.items()},
           "ring": summary("ring"), "single": summary("single"),
           "detect_pose_ms": {"mean": float(np.mean(timings)), "p95": float(np.percentile(timings, 95))},
           "rows": rows}

    # jitter: one still pose (-7 deg horizontal, 150 mm), 60 frames
    R, t = rot(-7, 0, 10), np.array([3.0, -2.0, 150.0])
    raw = {"ring": [], "single": []}; sm = Smoother(0.35); smoothed = []
    for k in range(60):
        img = render_ring(cam, L, R, t, rng, noise=3.0, blur=0.8)
        c, ids = detect(det, img)
        for name, fn in (("ring", ring_pose), ("single", single_marker_pose)):
            p = fn(c, ids, L, K, dist)
            if p is not None:
                raw[name].append(alignment_from_pose(p.R, p.t).horizontal_deg)
        if raw["ring"]:
            smoothed.append(sm({"h": raw["ring"][-1]})["h"])
    res["jitter"] = {"pose": "horizontal -7.0 deg, 150 mm", "ring_std_deg": float(np.std(raw["ring"])), "single_std_deg": float(np.std(raw["single"])),
                     "ring_smoothed_std_deg": float(np.std(smoothed[10:])), "ring_mean_deg": float(np.mean(raw["ring"])), "single_mean_deg": float(np.mean(raw["single"])),
                     "series_ring": [round(x, 3) for x in raw["ring"]], "series_single": [round(x, 3) for x in raw["single"]], "series_smoothed": [round(x, 3) for x in smoothed]}

    # camera mounting offset: camera clamped 2 deg yaw + 2 deg pitch off the tube axis
    M = rot(2.0, 2.0)
    def view(R_tube, t_tube):
        return M.T @ R_tube, M.T @ t_tube
    Rz, tz = view(np.eye(3), np.array([0, 0, 150.0]))
    img = render_ring(cam, L, Rz, tz, rng); c, ids = detect(det, img); p0 = ring_pose(c, ids, L, K, dist)
    zero = p0.R.copy()
    out = []
    for yaw in (0.0, -7.0, 5.0):
        Rc, tc = view(rot(yaw, 0), np.array([0, 0, 150.0]))
        img = render_ring(cam, L, Rc, tc, rng); c, ids = detect(det, img); p = ring_pose(c, ids, L, K, dist)
        out.append({"true_h": yaw, "no_zero_h": alignment_from_pose(p.R, p.t).horizontal_deg, "no_zero_v": alignment_from_pose(p.R, p.t).vertical_deg,
                    "zeroed_h": alignment_from_pose(p.R, p.t, zero).horizontal_deg, "zeroed_v": alignment_from_pose(p.R, p.t, zero).vertical_deg})
    res["zeroing"] = {"mount_offset": "2 deg yaw + 2 deg pitch", "cases": out}
    res["elapsed_s"] = round(time.perf_counter() - t0, 1)

    OUT.mkdir(exist_ok=True)
    (OUT / "accuracy.json").write_text(json.dumps(res, indent=1))
    for n in ("ring", "single"):
        s = res[n]
        print(f"{n:6}: solved {s['solved']}/{s['frames']}  angle RMS {s['angle_err_rms_deg']:.3f}  p95 {s['angle_err_p95_deg']:.3f}  max {s['angle_err_max_deg']:.2f}  near-aligned RMS {s['near_aligned_err_rms_deg']:.3f}  sign flips {s['sign_flips']}/{s['sign_checks']} ({s['sign_flip_pct']:.1f}%)  pos {s['pos_err_rms_mm']:.2f} mm  dist {s['dist_err_rms_mm']:.2f} mm  by d {s['rms_by_distance_deg']}")
    j = res["jitter"]; print(f"jitter std: ring {j['ring_std_deg']:.3f}, single {j['single_std_deg']:.3f}, ring smoothed {j['ring_smoothed_std_deg']:.3f}; means {j['ring_mean_deg']:.2f} / {j['single_mean_deg']:.2f}")
    print("zeroing:", [{k: round(v, 2) for k, v in c.items()} for c in out])
    print("detect+pose ms:", res["detect_pose_ms"], "elapsed", res["elapsed_s"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
