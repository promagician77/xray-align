"""Camera calibration with a printed ChArUco board (7 x 5 squares of 20 mm, 15 mm markers).

    python -m xray_align.calibrate --board-png charuco.png          # print at 100% scale
    python -m xray_align.calibrate --camera 0 --out camera.json     # SPACE capture, C calibrate, Q quit
    python -m xray_align.calibrate --images calib/ --out camera.json

Capture 15-25 views: tilt the board up to about 30 degrees, move it across the whole image,
and work at the distance the tube will be from the ring. Fix the focus first; refocusing
changes the focal length and invalidates the calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .render import charuco_board


def collect(board, det, gray):
    cc, ci, _, _ = det.detectBoard(gray)
    if ci is None or len(ci) < 8:
        return None
    return board.matchImagePoints(cc, ci)


def solve(views, size):
    objs = [o for o, _ in views]; imgs = [i for _, i in views]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objs, imgs, size, None, None)
    per_view = []
    for o, i, r, t in zip(objs, imgs, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, 1)))))
    return {"K": K.tolist(), "dist": dist.ravel().tolist(), "rms_px": float(rms), "views": len(views),
            "image_size": list(size), "per_view_rms_px": per_view}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int); ap.add_argument("--images"); ap.add_argument("--out", default="camera.json")
    ap.add_argument("--board-png", help="write the board image (300 dpi) and exit")
    ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=800)
    a = ap.parse_args(argv)
    board = charuco_board()
    if a.board_png:
        sx, sy = board.getChessboardSize(); mm = board.getSquareLength()
        px = int(round(300 / 25.4 * mm))
        img = board.generateImage((sx * px, sy * px), marginSize=0)
        img = cv2.copyMakeBorder(img, px // 2, px // 2, px // 2, px // 2, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(a.board_png, img)
        print(f"wrote {a.board_png}: print at 300 dpi, 100% scale; squares must measure {mm:.0f} mm")
        return 0
    det = cv2.aruco.CharucoDetector(board)
    views, size = [], None
    if a.images:
        for p in sorted(Path(a.images).glob("*")):
            g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            size = g.shape[::-1]; v = collect(board, det, g)
            if v: views.append(v)
    else:
        cap = cv2.VideoCapture(a.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height); cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); size = g.shape[::-1]
            cc, ci, _, _ = det.detectBoard(g)
            if ci is not None:
                cv2.aruco.drawDetectedCornersCharuco(f, cc, ci)
            cv2.putText(f, f"views: {len(views)}  SPACE capture  C calibrate  Q quit", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("calibration", f)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(" "):
                v = collect(board, det, g)
                if v: views.append(v)
            elif k in (ord("c"), ord("q")):
                break
        cap.release(); cv2.destroyAllWindows()
    if len(views) < 8:
        raise SystemExit(f"only {len(views)} usable views; capture at least 8 (15-25 recommended)")
    res = solve(views, size)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"calibrated from {res['views']} views: RMS {res['rms_px']:.3f} px -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
