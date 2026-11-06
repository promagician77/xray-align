"""results/ + media/ -> site/data and site/media for the review site."""
import json, shutil, sqlite3, subprocess
from pathlib import Path
R = Path(__file__).resolve().parents[1]
S = R / "site"
acc = json.loads((R / "results/accuracy.json").read_text())
keep = {k: acc[k] for k in ("camera", "layout", "calibration", "ring", "single", "detect_pose_ms", "jitter", "zeroing")}
keep["calibration"] = {k: v for k, v in acc["calibration"].items() if k not in ("K",)}
keep["points"] = [{"d": r["d"], "true_h": round(r["true_h"], 3), "true_v": round(r["true_v"], 3), "true_total": round(r["true_total"], 3),
                   "ring": r["ring"] and {k: round(r["ring"][k], 3) for k in ("h", "v", "err_deg")},
                   "single": r["single"] and {k: round(r["single"][k], 3) for k in ("h", "v", "err_deg")}} for r in acc["rows"]]
(S / "data/accuracy.json").write_text(json.dumps(keep))
shutil.copy(R / "results/risk_fit.json", S / "data/risk_fit.json")
db = sqlite3.connect(R / "data/demo.sqlite"); db.row_factory = sqlite3.Row
(S / "data/demo_log.json").write_text(json.dumps([dict(r) for r in db.execute("select * from exposures order by id")]))
shutil.copy(R / "media/demo.mp4", S / "media/demo.mp4")
for t, name in ((2.0, "poster.jpg"), (11.2, "aligned.jpg"), (17.8, "drift.jpg")):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(R / "media/demo.mp4"), "-frames:v", "1", "-q:v", "3", str(S / "media" / name)], check=True)
for f in ("ring_markers.pdf", "charuco_board.png"):
    shutil.copy(R / "media" / f, S / "media" / f)
print("exported:", sorted(p.name for p in (S / "data").iterdir()), sorted(p.name for p in (S / "media").iterdir()))
