"""Estimated risk of a poor image from the angular deviation.

SIMULATED by default: an illustrative logistic curve, NOT clinically derived.
    risk(theta) = 1 / (1 + exp(-(|theta| - theta50) / spread))
With theta50 = 5.3 deg and spread = 1.0 deg, a 7 deg deviation gives 85% (the brief's example).

FITTED once enough operator feedback exists: the same curve, fitted by logistic regression
(iteratively reweighted least squares, numpy only) to the logged exposures, where
"unacceptable" = 1 and "acceptable" = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RiskModel:
    theta50: float = 5.3
    spread: float = 1.0
    source: str = "SIMULATED"     # or "FITTED (n=...)"
    n: int = 0

    def __call__(self, total_deg: float) -> float:
        z = (abs(total_deg) - self.theta50) / self.spread
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def label(self) -> str:
        return self.source if self.source == "SIMULATED" else f"FITTED from {self.n} exposures"


def fit(deg: np.ndarray, poor: np.ndarray, min_n: int = 20, iters: int = 50) -> RiskModel | None:
    """Fit theta50 and spread from (|deviation|, poor 0/1). None if too few or one-sided labels."""
    x = np.abs(np.asarray(deg, float)); y = np.asarray(poor, float)
    if len(x) < min_n or y.min() == y.max():
        return None
    X = np.stack([np.ones_like(x), x], 1)
    b = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -60, 60)))
        W = np.clip(p * (1 - p), 1e-6, None)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    if b[1] <= 0 or 1 / b[1] < 0.1:   # must rise with deviation; a near-step curve means the data separate perfectly (too few)
        return None
    return RiskModel(theta50=float(-b[0] / b[1]), spread=float(1 / b[1]), source=f"FITTED (n={len(x)})", n=len(x))


def recommendation(h: float, v: float, dx: float, dy: float, tol_deg: float, tol_mm: float) -> list[str]:
    """Plain corrections, in the camera image's directions (mapped to the room during setup)."""
    out = []
    if abs(h) > tol_deg:
        out.append(f"Turn tube {abs(h):.1f}° {'right' if h > 0 else 'left'}")
    if abs(v) > tol_deg:
        out.append(f"Tilt tube {abs(v):.1f}° {'up' if v > 0 else 'down'}")
    if abs(dx) > tol_mm:
        out.append(f"Move tube {abs(dx):.0f} mm {'right' if dx > 0 else 'left'}")
    if abs(dy) > tol_mm:
        out.append(f"Move tube {abs(dy):.0f} mm {'down' if dy > 0 else 'up'}")
    return out
