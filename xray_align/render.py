"""Synthetic camera: renders the marker ring (or a ChArUco board) at an exactly known pose,
through a distorting lens, with blur and sensor noise. Used to measure accuracy against
ground truth and to run the app without hardware (--sim).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .geometry import RingLayout


class Camera:
    def __init__(self, w=1280, h=800, hfov_deg=70.0, k1=-0.12, k2=0.03, p1=0.0005, p2=-0.0003):
        f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
        self.w, self.h = w, h
        self.K = np.array([[f, 0, w / 2 + 3.5], [0, f, h / 2 - 2.0], [0, 0, 1]], np.float64)
        self.dist = np.array([k1, k2, p1, p2, 0.0], np.float64)
        # map: for each distorted output pixel, where to sample the ideal (pinhole) image
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(h, w, 2)
        self.mapx, self.mapy = und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)


def _marker_texture(dictionary, mid: int, px: int = 240, quiet: float = 0.25) -> tuple[np.ndarray, float]:
    """Marker bitmap with a white quiet zone. Returns image and quiet-zone fraction per side."""
    m = cv2.aruco.generateImageMarker(dictionary, mid, px)
    q = int(round(px * quiet))
    return cv2.copyMakeBorder(m, q, q, q, q, cv2.BORDER_CONSTANT, value=255), q / px


def _background(cam: Camera, rng) -> np.ndarray:
    y, x = np.mgrid[0:cam.h, 0:cam.w]
    bg = 95 + 35 * np.sin(x / 210.0 + rng.uniform(0, 6)) * np.cos(y / 170.0) + 0.02 * x
    return np.clip(bg, 0, 255).astype(np.float32)


def _warp(dst, src, src_quad, dst_quad, mask_val=None):
    H = cv2.getPerspectiveTransform(src_quad.astype(np.float32), dst_quad.astype(np.float32))
    warped = cv2.warpPerspective(src.astype(np.float32), H, (dst.shape[1], dst.shape[0]), flags=cv2.INTER_AREA if False else cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones_like(src, np.float32), H, (dst.shape[1], dst.shape[0]), flags=cv2.INTER_LINEAR)
    return dst * (1 - mask) + warped * mask


def render_ring(cam: Camera, layout: RingLayout, R: np.ndarray, t_mm: np.ndarray, rng=None,
                noise: float = 3.0, blur: float = 0.8, gain: float = 1.0) -> np.ndarray:
    """Grey image of the ring with its markers, camera at origin. R, t: ring -> camera."""
    rng = rng or np.random.default_rng()
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, layout.dictionary))
    img = _background(cam, rng)
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.asarray(t_mm, float).reshape(3, 1)
    zero = np.zeros(5)
    # positioning ring: a light plastic annulus (drawn as a polygon of the outer/inner circles)
    for radius, val in ((40.0, 205.0), (13.0, None)):
        a = np.linspace(0, 2 * math.pi, 120, endpoint=False)
        circle = np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros_like(a)], 1)
        p, _ = cv2.projectPoints(circle, rvec, tvec, cam.K, zero)
        poly = p.reshape(-1, 2).astype(np.int32)
        if val is None:
            cv2.fillPoly(img, [poly], 70.0)
        else:
            cv2.fillPoly(img, [poly], val)
    for k, mid in enumerate(layout.marker_ids):
        tex, q = _marker_texture(d, mid)
        corners = layout.marker_corners(k)
        c = corners.mean(axis=0)
        outer = c + (corners - c) * (1 + 2 * q)                  # include the white quiet zone
        p, _ = cv2.projectPoints(outer, rvec, tvec, cam.K, zero)
        s = tex.shape[0] - 1
        img = _warp(img, tex, np.array([[0, 0], [s, 0], [s, s], [0, s]]), p.reshape(4, 2))
    img = cv2.remap(img, cam.mapx, cam.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img = img * gain + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def charuco_board():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    return cv2.aruco.CharucoBoard((7, 5), 20.0, 15.0, d)   # 7 x 5 squares of 20 mm, markers 15 mm


def render_board(cam: Camera, board, R, t_mm, rng=None, noise=3.0, blur=0.8) -> np.ndarray:
    rng = rng or np.random.default_rng()
    sx, sy = board.getChessboardSize(); sq = board.getSquareLength()
    W, H = sx * sq, sy * sq
    px_per_mm = 8
    tex = board.generateImage((int(W * px_per_mm), int(H * px_per_mm)), marginSize=0)
    img = _background(cam, rng)
    rvec, _ = cv2.Rodrigues(R)
    quad = np.array([[0, 0, 0], [W, 0, 0], [W, H, 0], [0, H, 0]], np.float64)
    p, _ = cv2.projectPoints(quad, rvec, np.asarray(t_mm, float).reshape(3, 1), cam.K, np.zeros(5))
    s = np.array([[0, 0], [tex.shape[1] - 1, 0], [tex.shape[1] - 1, tex.shape[0] - 1], [0, tex.shape[0] - 1]])
    img = _warp(img, tex, s, p.reshape(4, 2))
    img = cv2.remap(img, cam.mapx, cam.mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img = img + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
