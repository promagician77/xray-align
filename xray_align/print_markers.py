"""Printable marker sheet for the positioning ring, at true scale.

    python -m xray_align.print_markers --out ring_markers.pdf
    python -m xray_align.print_markers --write-layout ring.json     # then edit and pass --layout ring.json

Prints the four markers with their white margins, IDs and a 50 mm scale bar to check the
printer did not rescale. Stick them on the ring at the angles in the layout (45/135/225/315
degrees, 26 mm from the ring centre by default), top edge towards the top of the ring.
"""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import RingLayout

DPI = 600


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="ring_markers.pdf"); ap.add_argument("--layout")
    ap.add_argument("--write-layout", metavar="JSON", help="write the default ring layout to edit (marker size, radius, angles) and exit")
    a = ap.parse_args(argv)
    if a.write_layout:
        RingLayout().save(a.write_layout); print(f"wrote {a.write_layout}: edit it, then pass --layout {a.write_layout}"); return 0
    L = RingLayout.load(a.layout) if a.layout else RingLayout()
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, L.dictionary))
    mm = lambda v: int(round(v / 25.4 * DPI))
    page = Image.new("L", (mm(210), mm(297)), 255)                     # A4
    dr = ImageDraw.Draw(page)
    try:
        f = ImageFont.truetype(str(__import__("pathlib").Path(__file__).parent / "assets" / "DejaVuSans.ttf"), mm(3.2))
    except OSError:
        f = ImageFont.load_default()
    dr.text((mm(15), mm(12)), f"Ring markers: {L.dictionary}, {L.marker_mm:.0f} mm, IDs {L.marker_ids}. Print at 100% (no 'fit to page').", font=f, fill=0)
    dr.line((mm(15), mm(22), mm(65), mm(22)), fill=0, width=mm(0.4)); dr.text((mm(68), mm(19.5)), "50 mm check", font=f, fill=0)
    for k, mid in enumerate(L.marker_ids):
        side = mm(L.marker_mm); quiet = mm(L.marker_mm * 0.25)
        m = cv2.aruco.generateImageMarker(d, mid, side)
        tile = Image.fromarray(cv2.copyMakeBorder(m, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255))
        x, y = mm(20 + 45 * k), mm(35)
        page.paste(tile, (x, y))
        dr.rectangle((x, y, x + tile.width, y + tile.height), outline=180, width=2)
        dr.text((x, y + tile.height + mm(2)), f"ID {mid} · {L.angles_deg[k]:.0f}°", font=f, fill=0)
    # placement diagram
    cx, cy, r = mm(105), mm(140), mm(L.radius_mm * 1.6)
    dr.ellipse((cx - mm(64), cy - mm(64), cx + mm(64), cy + mm(64)), outline=0, width=mm(0.3))
    for k, mid in enumerate(L.marker_ids):
        a_ = math.radians(L.angles_deg[k]); px, py = cx + r * math.cos(a_), cy + r * math.sin(a_)
        h = mm(L.marker_mm * 0.8)
        dr.rectangle((px - h, py - h, px + h, py + h), outline=0, width=mm(0.3)); dr.text((px - h / 2, py - mm(1.5)), f"ID {mid}", font=f, fill=0)
    dr.text((mm(15), mm(210)), f"Placement (drawn 1.6x): marker centres {L.radius_mm:.0f} mm from the ring centre at {', '.join(f'{v:.0f}' for v in L.angles_deg)} degrees", font=f, fill=0)
    dr.text((mm(15), mm(216)), "(0 degrees = right, 90 = down, as seen from the tube). Measure your ring and edit the layout JSON if it differs.", font=f, fill=0)
    page.save(a.out, resolution=DPI)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
