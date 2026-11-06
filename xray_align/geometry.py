"""Marker ring layout and alignment angles.

Ring frame: origin at the centre of the positioning ring, x right, y down (as seen from the
tube), z pointing away from the tube (into the patient). Markers lie flat in the ring plane.

Camera frame (OpenCV): x right, y down, z along the optical axis. The camera is mounted on
the tube, so after zeroing its axis stands in for the tube's central ray.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np


@dataclass
class RingLayout:
    dictionary: str = "DICT_4X4_50"
    marker_ids: list = field(default_factory=lambda: [0, 1, 2, 3])
    marker_mm: float = 12.0          # black square side, printed size
    radius_mm: float = 26.0          # ring centre -> marker centre
    angles_deg: list = field(default_factory=lambda: [45.0, 135.0, 225.0, 315.0])

    def marker_corners(self, i: int) -> np.ndarray:
        """3D corners (TL, TR, BR, BL as OpenCV reports them) of marker i in the ring frame, mm."""
        a = math.radians(self.angles_deg[i])
        cx, cy = self.radius_mm * math.cos(a), self.radius_mm * math.sin(a)
        h = self.marker_mm / 2
        return np.array([[cx - h, cy - h, 0], [cx + h, cy - h, 0], [cx + h, cy + h, 0], [cx - h, cy + h, 0]], np.float64)

    def object_points(self, ids) -> np.ndarray:
        idx = {m: k for k, m in enumerate(self.marker_ids)}
        return np.concatenate([self.marker_corners(idx[i]) for i in ids if i in idx])

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def load(path) -> "RingLayout":
        return RingLayout(**json.loads(Path(path).read_text()))


def rot(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """Ring orientation in the camera frame: yaw about y, pitch about x, roll about z."""
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


@dataclass
class Alignment:
    horizontal_deg: float      # + : ring turned to the camera's right (tube must swing right)
    vertical_deg: float        # + : ring tilted so its top leans away
    total_deg: float           # angle between tube axis and ring normal
    dx_mm: float               # ring centre right of the tube axis
    dy_mm: float               # ring centre below the tube axis
    distance_mm: float

    def as_dict(self) -> dict:
        return {k: round(v, 3) for k, v in asdict(self).items()}


def alignment_from_pose(R: np.ndarray, t: np.ndarray, zero: np.ndarray | None = None) -> Alignment:
    """Angles between the tube (camera) axis and the ring normal, plus centring offset.

    zero: 3x3 rotation captured with the tube correctly aligned (removes the camera mounting
    offset). Measurements are expressed relative to that reference.
    """
    R = np.asarray(R, float)
    t = np.asarray(t, float).reshape(3)
    if zero is not None:
        R = zero.T @ R
        t = zero.T @ t
    n = R @ np.array([0.0, 0.0, 1.0])          # ring normal in camera frame; aligned -> (0,0,1)
    horiz = math.degrees(math.atan2(n[0], n[2]))
    vert = math.degrees(math.atan2(-n[1], n[2]))
    total = math.degrees(math.acos(max(-1.0, min(1.0, n[2]))))
    return Alignment(horiz, vert, total, float(t[0]), float(t[1]), float(t[2]))


def zero_from_pose(R: np.ndarray) -> np.ndarray:
    """Reference rotation for zeroing: the measured ring orientation when the tube is aligned."""
    return np.asarray(R, float).copy()
