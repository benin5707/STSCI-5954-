"""Fit the fixed six-group K-means model, evaluate, and save delivery outputs."""
import argparse
from pathlib import Path
import sys
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.clustering import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT/"data"/"raw"/"tracks_clean.csv")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        main(args.csv, args.project_root)


