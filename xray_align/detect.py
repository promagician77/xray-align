"""Marker detection and ring pose."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import RingLayout


def make_detector(layout: RingLayout) -> cv2.aruco.ArucoDetector:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, layout.dictionary))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 4
    return cv2.aruco.ArucoDetector(d, p)


@dataclass
class Pose:
    R: np.ndarray
    t: np.ndarray
    ids: list
    reproj_px: float
    method: str


def detect(detector, gray):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return [], []
    return [c.reshape(4, 2) for c in corners], [int(i) for i in ids.ravel()]


def _reproj(obj, img, rvec, tvec, K, dist) -> float:
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))


def ring_pose(corners, ids, layout: RingLayout, K, dist, min_markers: int = 2) -> Pose | None:
    """Pose of the whole ring from every visible marker at once (16 corners when all 4 are seen).

    Corners spread across the ring make the tilt well conditioned; a single small square seen
    face-on is ambiguous (two mirror-image tilts fit almost equally well).
    """
    keep = [(c, i) for c, i in zip(corners, ids) if i in layout.marker_ids]
    if len(keep) < min_markers:
        return None
    obj = layout.object_points([i for _, i in keep])
    img = np.concatenate([c for c, _ in keep]).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    return Pose(R, tvec.reshape(3), [i for _, i in keep], _reproj(obj, img, rvec, tvec, K, dist), "ring")


def single_marker_pose(corners, ids, layout: RingLayout, K, dist, marker_id: int | None = None) -> Pose | None:
    """Baseline: ring pose from one marker (IPPE square), as many marker tutorials do."""
    for c, i in zip(corners, ids):
        if i in layout.marker_ids and (marker_id is None or i == marker_id):
            h = layout.marker_mm / 2
            # OpenCV's IPPE_SQUARE layout: marker frame x right, y up, z towards the camera
            obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, c.astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                return None
            Rm, _ = cv2.Rodrigues(rvec)
            R = Rm @ np.diag([1.0, -1.0, -1.0])        # marker frame -> ring frame (y down, z away)
            k = layout.marker_ids.index(i)
            centre = layout.marker_corners(k).mean(axis=0)
            t_ring = tvec.reshape(3) - R @ centre
            return Pose(R, t_ring, [i], _reproj(obj, c, rvec, tvec, K, dist), "single")
    return None


class Smoother:
    """Exponential smoothing of the displayed angles (raw values are still logged)."""

    def __init__(self, alpha: float = 0.35):
        self.alpha, self.state = alpha, None

    def __call__(self, values: dict) -> dict:
        if self.state is None:
            self.state = dict(values)
        else:
            self.state = {k: self.alpha * v + (1 - self.alpha) * self.state.get(k, v) for k, v in values.items()}
        return dict(self.state)

    def reset(self):
        self.state = None
