from pathlib import Path
import json

class MetricsLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.case_name = log_dir.name
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict):
        with open(self.log_dir / f"{self.case_name}_metric.json", "a") as f:
            json.dump(metrics, f)
            f.write("\n")