"""Run the analysis, figures, and English report in their required order."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT/"data"/"raw"/"tracks_clean.csv")
    parser.add_argument("--project-root", type=Path, default=ROOT,
                        help="Destination project root; defaults to this repository.")
    parser.add_argument("--skip-fit", action="store_true",
                        help="Rebuild figures and report from existing results.")
    parser.add_argument("--include-demo", action="store_true",
                        help="Also run prediction on the supplied demonstration input.")
    args = parser.parse_args()
    steps = []
    if not args.skip_fit:
        steps.append(("01_fit_clusters.py", ["--csv", str(args.csv)]))
    steps += [("02_make_figures.py", []), ("03_generate_report.py", [])]
    for script, extra in steps:
        command = [sys.executable, str(ROOT/"scripts"/script),
                   "--project-root", str(args.project_root), *extra]
        print(f"Running {script}", flush=True)
        subprocess.run(command, check=True)
    if args.include_demo:
        subprocess.run([sys.executable, str(ROOT/"scripts"/"04_predict_new_songs.py"),
                        "--csv", str(ROOT/"data"/"examples"/"sample_new_songs.csv"),
                        "--model", str(args.project_root/"output"/"models"/"playlist_model.joblib"),
                        "--output", str(args.project_root/"output"/"examples"/"sample_predictions.csv")],
                       check=True)

if __name__ == "__main__":
    main()

