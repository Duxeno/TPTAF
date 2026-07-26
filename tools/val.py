#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tptaf.config import load_config
from tptaf.metrics import METRIC_REGISTRY
from tptaf.validation import ValidationOptions, validate

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate pre-generated TPTAF fused images.")
    p.add_argument("--config", default=str(ROOT / "configs" / "tptaf_template.yaml"))
    p.add_argument("--fused-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--output-dir", default="outputs/tptaf_validation")
    p.add_argument("--metrics", nargs="+", default=list(METRIC_REGISTRY), choices=sorted(METRIC_REGISTRY))
    p.add_argument("--save-color", action="store_true")
    p.add_argument("--no-save-gray", action="store_true")
    p.add_argument("--max-samples", type=int)
    a = p.parse_args()
    result = validate(load_config(a.config), ValidationOptions(data_root=Path(a.data_root), split=a.split, output_dir=Path(a.output_dir), fused_dir=Path(a.fused_dir), metrics=tuple(a.metrics), save_grayscale=not a.no_save_gray, save_color=a.save_color, max_samples=a.max_samples))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
