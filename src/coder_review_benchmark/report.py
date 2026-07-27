from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_report(run_dir: Path) -> Path:
    rows = []
    for path in sorted(run_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    metrics = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = {"run_dir": str(run_dir), "total": len(rows), "completed": sum(r.get("status") == "completed" for r in rows), "malformed_calls": sum(r.get("malformed_calls", 0) for r in rows), "metrics": metrics, "rows": rows}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    md = ["# Benchmark summary", "", f"- Total: {summary['total']}", f"- Completed: {summary['completed']}", f"- Malformed tool calls: {summary['malformed_calls']}", ""]
    for key, value in metrics.items():
        md.append(f"- {key}: {value}")
    (run_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    return run_dir / "summary.json"
