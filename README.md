# X-ray tube alignment: proof of concept

Live alignment between a dental X-ray tube and the sensor holder's positioning ring. It uses a USB camera on the tube and ArUco markers on the ring.

**Phantom testing only. Not for clinical decisions.** The risk percentage is an illustrative SIMULATED curve until it is fitted to your own phantom feedback.

## What it does

1. **Detects** the four ring markers (OpenCV ArUco, sub-pixel corners) in each camera frame.
2. **Estimates the 3D pose of the whole ring** from all detected marker corners at once (SQPnP, then Levenberg–Marquardt refinement), using a calibrated camera. It works with 2 or more markers visible.
3. **Measures alignment:**
   - horizontal and vertical angle between the tube axis and the ring normal;
   - total deviation;
   - centring offset (mm);
   - distance (mm).
4. **Guides the operator:**
   - ALIGNED (green) or ADJUST TUBE (red);
   - each value shown on a bar with its tolerance band;
   - plain corrections ("Turn tube 7.2° left").
5. **Shows an estimated risk** of a poor image from the total deviation. The default curve gives 7° → 85%, as in the brief, and is labelled SIMULATED on screen.
6. **Logs operator feedback.** X marks an exposure, then A (acceptable) or U (unacceptable) stores the verdict with the angles, centring, distance, tracking quality and the risk shown. Data goes to SQLite; E exports CSV.
7. **Learns from that feedback.** After 20 or more verdicts with both outcomes, the risk curve is refitted (logistic regression) and labelled FITTED (n = …).

## Install

Python 3.10 or newer on Windows, macOS or Linux.

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q             # 13 tests
```

## Try it without hardware

```bash
python -m xray_align.app --sim
```

A synthetic camera renders the ring through a distorting lens.

- J / L turn the tube, I / K tilt it, F / H / T / G move it, R resets.
- The live keys (below) work as normal.

To reproduce the demo video (needs ffmpeg on PATH):

```bash
python -m xray_align.app --sim --demo --record media/demo.mp4 --headless --db data/demo.sqlite
```

## Real camera setup

1. **Print the markers.**
   ```bash
   python -m xray_align.print_markers --out ring_markers.pdf
   ```
   Print at 100% and check the 50 mm bar with a ruler. Use matte labels, because glare hides corners.
2. **Stick them on the ring:** 4 × 12 mm markers, centres 26 mm from the ring centre, at 45°, 135°, 225° and 315°. If your ring differs, run `python -m xray_align.print_markers --write-layout ring.json`, edit it, and pass `--layout ring.json` to both the print and the app. The marker size must be the printed black-square size.
3. **Mount the camera** on the tube, as close to the beam axis as practical. Fix the focus at the tube-to-ring working distance; turn autofocus off, because refocusing changes the focal length.
4. **Calibrate once per camera and lens setting.**
   ```bash
   python -m xray_align.calibrate --board-png charuco.png     # print it; measure a square (should be 20 mm)
   python -m xray_align.calibrate --camera 0 --out camera.json
   ```
   SPACE captures a view (aim for 15–25, varied angles and distances), C solves, and Q quits. Expect an RMS under about 0.5 px. Rerun if you change the focus or zoom.
5. **Run.**
   ```bash
   python -m xray_align.app --camera 0 --calib camera.json --operator "initials"
   ```
6. **Zero once per mounting.** Align the tube with your current reference method (for example a jig), then press **Z**. This removes the camera's mounting offset from the tube axis. In the measurements, a 2° + 2° offset read −8.97° for a true −7° before zeroing and −6.95° after.

Options: `--tol-deg 2`, `--tol-mm 5` (green thresholds), `--width/--height` (default 1280×800), `--focus` (driver focus value), `--video file.mp4` (replay recorded footage).

## Keys

| Key | Action |
|---|---|
| X | Mark exposure (stores the alignment at that moment) |
| A / U | Image acceptable / unacceptable (saved with the marked alignment) |
| Z | Zero: the current pose becomes "aligned" |
| E | Export CSV next to the database |
| Q | Quit |

## Data

`data/exposures.sqlite`, table `exposures`:

| Column | Meaning |
|---|---|
| `ts` | Time of the verdict |
| `horizontal_deg`, `vertical_deg`, `total_deg` | Raw (unsmoothed) angles |
| `dx_mm`, `dy_mm`, `distance_mm` | Centring and distance |
| `markers_seen`, `reproj_px` | Tracking quality |
| `risk_pct`, `risk_model` | What the screen showed |
| `feedback`, `operator`, `note` | The verdict and who gave it |

## Measured (synthetic ground truth)

`bench/accuracy.py` calibrates a synthetic 1280×800, 70° camera from 25 rendered ChArUco views; the pipeline solves it, it is not given. It then measures 144 known poses: 100–250 mm, ±15°, with noise and blur.

| | Whole ring (this app) | Single marker |
|---|---|---|
| RMS angle error | 0.42° | 5.83° |
| 95th percentile | 0.96° | 12.04° |
| Tilt direction flipped (\|angle\| ≥ 2°) | 0 of 180 | 16 of 180 |
| Distance error RMS | 0.24 mm | 1.96 mm |

- Detection plus pose takes about 17 ms per frame (2 CPU cores).
- `bench/risk_fit.py` shows how many verdicts a fitted curve needs:
  - with 20, the fitted 50%-risk angle varies between 3.0° and 6.0° (90% of runs);
  - with 200, between 4.0° and 4.9°.
- Results are in `results/` and are also shown in `site/`.

**Not measured:**
- a physical camera, real prints and lighting;
- a moving tube;
- the phantom;
- Windows or macOS (the live window was run on Linux under a virtual display).

The first phantom session should check angle readings against a protractor or jig at several known angles.

## Layout

```
xray_align/  detect.py (markers, ring pose, single-marker baseline, smoothing) · geometry.py (layout, angles, zeroing)
             app.py (live app) · overlay.py (display) · risk.py (simulated/fitted risk, corrections)
             store.py (SQLite/CSV) · calibrate.py (ChArUco) · print_markers.py · render.py (synthetic camera)
bench/       accuracy.py, risk_fit.py  →  results/*.json
tests/       test_core.py
tools/       export_site.py (results → site/data, media)
site/        static review page (Vercel: outputDirectory "site", noindex)
```

## Recommended camera

**Arducam 1MP OV9281 USB board:**
- global shutter, mono, 1280×800 at up to 120 fps;
- M12 lens, 70° horizontal field of view;
- UVC, so no drivers are needed;
- 38 × 38 mm board.

**Lower-cost alternative: ELP USBFHD01M (1080p, 2.1 mm lens).**
- It has a rolling shutter and a wider lens with more distortion.
- Hold the tube still while reading, and calibrate carefully.

For either camera, use a fixed focus at the working distance.
