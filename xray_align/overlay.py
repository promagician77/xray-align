"""Operator display: camera view with markers and axes, plus a side panel with the readings."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

A = Path(__file__).parent / "assets"
GREEN, RED, GREY, INK, MUTED, PANEL, LINE = (43, 182, 115), (229, 72, 77), (120, 128, 138), (236, 240, 244), (150, 160, 172), (22, 27, 34), (48, 56, 66)
W, H, CAM_W, CAM_H, TOP = 1600, 900, 1152, 720, 64


def _font(size, bold=False, mono=False):
    name = "DejaVuSansMono.ttf" if mono else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    try:
        return ImageFont.truetype(str(A / name), size)
    except OSError:
        return ImageFont.load_default()


F = {k: _font(*v) for k, v in {"h1": (34, True), "h2": (20, True), "b": (17, False), "s": (14, False), "v": (30, True), "mono": (15, False, True), "risk": (64, True)}.items()}


def draw_camera(frame_gray, corners, ids, pose, K, dist, layout, tol_ok: bool):
    """Camera image (BGR) with detected markers, ring outline/axes and the tube-axis crosshair."""
    img = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR) if frame_gray.ndim == 2 else frame_gray.copy()
    col = (115, 182, 43) if tol_ok else (77, 72, 229)                       # BGR
    for c in corners:
        cv2.polylines(img, [c.astype(np.int32).reshape(-1, 1, 2)], True, (115, 182, 43), 2, cv2.LINE_AA)
    cx, cy = int(K[0, 2]), int(K[1, 2])
    cv2.line(img, (cx - 40, cy), (cx + 40, cy), (240, 240, 240), 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 40), (cx, cy + 40), (240, 240, 240), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 6, (240, 240, 240), 1, cv2.LINE_AA)
    if pose is not None:
        rvec, _ = cv2.Rodrigues(pose.R)
        a = np.linspace(0, 2 * np.pi, 90)
        ring = np.stack([40 * np.cos(a), 40 * np.sin(a), np.zeros_like(a)], 1)
        p, _ = cv2.projectPoints(ring, rvec, pose.t, K, dist)
        cv2.polylines(img, [p.astype(np.int32).reshape(-1, 1, 2)], True, col, 2, cv2.LINE_AA)
        c0, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, pose.t, K, dist)
        c0 = tuple(int(v) for v in c0.ravel())
        cv2.arrowedLine(img, (cx, cy), c0, col, 2, cv2.LINE_AA, tipLength=0.15)
        cv2.drawFrameAxes(img, K, dist, rvec, pose.t, 25, 2)
    return img


def _bar(d, x, y, w, value, rng, tol, label, unit, fmt="{:+.1f}"):
    ok = abs(value) <= tol
    d.text((x, y), label, font=F["s"], fill=MUTED)
    d.text((x + w, y - 6), fmt.format(value) + unit, font=F["v"], fill=GREEN if ok else RED, anchor="ra")
    by = y + 34
    d.rounded_rectangle((x, by, x + w, by + 12), 6, fill=LINE)
    sx = lambda v: x + (max(-rng, min(rng, v)) + rng) / (2 * rng) * w
    d.rectangle((sx(-tol), by, sx(tol), by + 12), fill=(30, 96, 66))
    d.line((sx(0), by - 4, sx(0), by + 16), fill=MUTED, width=1)
    nx = sx(value)
    d.ellipse((nx - 9, by - 3, nx + 9, by + 15), fill=GREEN if ok else RED, outline=PANEL, width=2)


def compose(cam_bgr, state: dict) -> np.ndarray:
    """Full operator screen (W x H, BGR)."""
    canvas = Image.new("RGB", (W, H), PANEL)
    cam = cv2.resize(cam_bgr, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    canvas.paste(Image.fromarray(cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)), (0, TOP))
    d = ImageDraw.Draw(canvas)
    # top bar
    d.rectangle((0, 0, W, TOP), fill=(14, 18, 23))
    d.text((24, 18), "X-ray tube alignment", font=F["h2"], fill=INK)
    d.text((268, 22), "prototype for phantom testing · not for clinical decisions", font=F["s"], fill=MUTED)
    d.text((W - 24, 22), f"{state.get('source', '')}   ·   detect + pose {state.get('proc_ms', 0):.0f} ms", font=F["s"], fill=MUTED, anchor="ra")
    x, y, w = CAM_W + 28, TOP + 22, W - CAM_W - 56
    m = state.get("m")
    if m is None:
        d.text((x, y), "MARKERS NOT FOUND", font=F["h1"], fill=GREY)
        d.text((x, y + 48), "Point the camera at the ring.", font=F["b"], fill=MUTED)
    else:
        ok = state["aligned"]
        d.rounded_rectangle((x, y, x + w, y + 58), 10, fill=GREEN if ok else RED)
        d.text((x + w / 2, y + 29), "ALIGNED" if ok else "ADJUST TUBE", font=F["h1"], fill=(255, 255, 255), anchor="mm")
        y += 82
        tol, tmm = state["tol_deg"], state["tol_mm"]
        _bar(d, x, y, w, m["horizontal_deg"], 15, tol, "Horizontal angle", "°"); y += 72
        _bar(d, x, y, w, m["vertical_deg"], 15, tol, "Vertical angle", "°"); y += 72
        _bar(d, x, y, w, m["dx_mm"], 20, tmm, "Centring, left / right", " mm", "{:+.0f}"); y += 72
        _bar(d, x, y, w, m["dy_mm"], 20, tmm, "Centring, up / down", " mm", "{:+.0f}"); y += 70
        d.text((x, y), f"Total deviation {m['total_deg']:.1f}°   ·   distance {m['distance_mm']:.0f} mm   ·   {m['markers_seen']} markers", font=F["s"], fill=MUTED); y += 34
        # risk
        r = state["risk"]
        d.line((x, y, x + w, y), fill=LINE, width=1); y += 14
        d.text((x, y), "Estimated risk of a poor image", font=F["s"], fill=MUTED)
        rc = RED if r >= 0.5 else (GREEN if r < 0.2 else (240, 180, 60))
        d.text((x, y + 18), f"{100 * r:.0f}%", font=F["risk"], fill=rc)
        d.text((x + 170, y + 30), state["risk_label"], font=F["s"], fill=(240, 180, 60) if state["risk_label"] == "SIMULATED" else INK)
        d.text((x + 170, y + 52), "illustrative curve, not clinical" if state["risk_label"] == "SIMULATED" else "from operator feedback", font=F["s"], fill=MUTED)
        y += 98
        recs = state["recs"]
        d.text((x, y), "Recommendation", font=F["s"], fill=MUTED); y += 22
        for line in (recs or ["Hold position: within tolerance"])[:4]:
            d.text((x, y), ("→ " if recs else "✓ ") + line, font=F["b"], fill=INK if recs else GREEN); y += 24
    # bottom bar
    by = TOP + CAM_H
    d.rectangle((0, by, CAM_W, H), fill=(14, 18, 23))
    d.text((24, by + 16), "X  mark exposure    A  image acceptable    U  image unacceptable    Z  zero (tube aligned)    E  export CSV    Q  quit", font=F["mono"], fill=MUTED)
    msg = state.get("message")
    if msg:
        d.text((24, by + 50), msg, font=F["b"], fill=GREEN if "acceptable" in msg and "unacceptable" not in msg else (RED if "unacceptable" in msg else INK))
    d.text((CAM_W - 24, by + 50), state.get("zero_label", ""), font=F["s"], fill=MUTED, anchor="ra")
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
