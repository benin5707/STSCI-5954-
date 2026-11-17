"""Assign new songs to groups using the saved delivery model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.prediction import predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT/"data"/"examples"/"sample_new_songs.csv")
    parser.add_argument("--model", type=Path, default=ROOT/"output"/"models"/"playlist_model.joblib")
    parser.add_argument("--output", type=Path, default=ROOT/"output"/"examples"/"sample_predictions.csv")
    args = parser.parse_args()
    result = predict(args.csv, args.model, args.output)
    print(f"Assigned {len(result)} songs. Predictions saved to {args.output}")

