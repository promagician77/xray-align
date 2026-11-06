"""Exposure log: SQLite, with CSV export."""
from __future__ import annotations

import csv
import sqlite3
import time
from pathlib import Path

COLUMNS = ["id", "ts", "horizontal_deg", "vertical_deg", "total_deg", "dx_mm", "dy_mm", "distance_mm",
           "markers_seen", "reproj_px", "risk_pct", "risk_model", "feedback", "operator", "note"]


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS exposures (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
            horizontal_deg REAL, vertical_deg REAL, total_deg REAL, dx_mm REAL, dy_mm REAL, distance_mm REAL,
            markers_seen INTEGER, reproj_px REAL, risk_pct REAL, risk_model TEXT,
            feedback TEXT CHECK (feedback IN ('acceptable', 'unacceptable')), operator TEXT, note TEXT)""")
        self.db.commit()

    def add(self, m: dict, feedback: str, risk_pct: float, risk_model: str, operator: str = "", note: str = "") -> int:
        cur = self.db.execute(
            "INSERT INTO exposures (ts, horizontal_deg, vertical_deg, total_deg, dx_mm, dy_mm, distance_mm, markers_seen, reproj_px, risk_pct, risk_model, feedback, operator, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), m.get("horizontal_deg"), m.get("vertical_deg"), m.get("total_deg"), m.get("dx_mm"), m.get("dy_mm"),
             m.get("distance_mm"), m.get("markers_seen"), m.get("reproj_px"), round(risk_pct, 1), risk_model, feedback, operator, note))
        self.db.commit()
        return int(cur.lastrowid)

    def rows(self) -> list[dict]:
        cur = self.db.execute(f"SELECT {', '.join(COLUMNS)} FROM exposures ORDER BY id")
        return [dict(zip(COLUMNS, r)) for r in cur.fetchall()]

    def labelled(self):
        r = self.db.execute("SELECT total_deg, feedback FROM exposures WHERE feedback IS NOT NULL AND total_deg IS NOT NULL").fetchall()
        return [x[0] for x in r], [1 if x[1] == "unacceptable" else 0 for x in r]

    def export_csv(self, path: str | Path) -> int:
        rows = self.rows()
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS); w.writeheader(); w.writerows(rows)
        return len(rows)

    def close(self):
        self.db.close()
