import math

import numpy as np
import pytest

from xray_align.geometry import RingLayout, rot, alignment_from_pose
from xray_align.risk import RiskModel, fit, recommendation
from xray_align.store import Store
from xray_align.render import Camera, render_ring
from xray_align.detect import make_detector, detect, ring_pose


@pytest.mark.parametrize("h,v", [(0, 0), (-7, 0), (5, -3), (12, 8), (-15, 10)])
def test_angles_from_known_pose(h, v):
    a = alignment_from_pose(rot(h, v, 17), np.array([3.0, -2.0, 150.0]))
    assert a.horizontal_deg == pytest.approx(h, abs=1e-9)
    # vertical is measured in the tube's vertical plane: atan2(sin v, cos v * cos h)
    assert a.vertical_deg == pytest.approx(math.degrees(math.atan2(math.sin(math.radians(v)), math.cos(math.radians(v)) * math.cos(math.radians(h)))), abs=1e-9)
    assert a.distance_mm == pytest.approx(150.0)


def test_roll_does_not_change_alignment():
    a = alignment_from_pose(rot(4, -2, 0), np.zeros(3) + [0, 0, 150])
    b = alignment_from_pose(rot(4, -2, 33), np.zeros(3) + [0, 0, 150])
    assert a.total_deg == pytest.approx(b.total_deg)


def test_zeroing_removes_mount_offset():
    M = rot(2, 2)
    zero = M.T @ np.eye(3)
    for h in (0.0, -7.0, 5.0):
        a = alignment_from_pose(M.T @ rot(h, 0), M.T @ np.array([0, 0, 150.0]), zero)
        assert a.horizontal_deg == pytest.approx(h, abs=1e-9)
        assert a.vertical_deg == pytest.approx(0, abs=1e-9)


def test_simulated_curve_matches_brief_example():
    assert RiskModel()(-7.0) == pytest.approx(0.85, abs=0.005)
    assert RiskModel()(0) < 0.01


def test_fit_recovers_curve():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 12, 400)
    y = (rng.uniform(0, 1, 400) < 1 / (1 + np.exp(-(x - 4.0) / 1.5))).astype(int)
    m = fit(x, y)
    assert m.theta50 == pytest.approx(4.0, abs=0.4) and m.spread == pytest.approx(1.5, abs=0.4)


def test_fit_refuses_one_sided_data():
    assert fit(np.ones(30), np.zeros(30)) is None


def test_recommendation_text():
    r = recommendation(-7.2, 2.7, 6, -1, 2.0, 5.0)
    assert r == ["Turn tube 7.2° left", "Tilt tube 2.7° up", "Move tube 6 mm right"]


def test_store_roundtrip(tmp_path):
    s = Store(tmp_path / "x.sqlite")
    s.add({"horizontal_deg": -7, "vertical_deg": 0, "total_deg": 7, "dx_mm": 1, "dy_mm": 0, "distance_mm": 150, "markers_seen": 4, "reproj_px": 0.3}, "unacceptable", 85, "SIMULATED")
    assert s.labelled() == ([7.0], [1])
    assert s.export_csv(tmp_path / "x.csv") == 1


def test_detect_and_pose_on_rendered_frame():
    cam, L = Camera(), RingLayout()
    img = render_ring(cam, L, rot(-7, 2, 10), np.array([2.0, 1.0, 150.0]), np.random.default_rng(3))
    c, ids = detect(make_detector(L), img)
    assert sorted(ids) == [0, 1, 2, 3]
    p = ring_pose(c, ids, L, cam.K, cam.dist)
    a = alignment_from_pose(p.R, p.t)
    assert a.horizontal_deg == pytest.approx(-7, abs=0.5) and a.distance_mm == pytest.approx(150, abs=2)
