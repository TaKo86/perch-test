#!/usr/bin/env python3
"""
Analyse bird sounds with Google Perch 2.0.

Pipeline:
  audio file(s) -> resample to 32 kHz mono -> 5 s windows -> Perch 2.0
  -> per-window class logits (~14.8k species/sound classes) + 1536-d embedding
  -> detections CSV, species summary, optional embeddings (.npy)

Install:
  pip install --upgrade bioacoustics-model-zoo opensoundscape tensorflow kagglehub pandas numpy
  (Perch 2 needs tensorflow >= 2.20. Lighter alternatives: bmz.Perch2ONNX / bmz.Perch2LiteRT.)

Usage:
  python perch_analyse.py recording.wav
  python perch_analyse.py ./recordings --threshold 0.3 --top-k 3 --embeddings
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def find_audio(path: Path) -> list[str]:
    if path.is_file():
        return [str(path)]
    return sorted(str(p) for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_model(backend: str):
    import bioacoustics_model_zoo as bmz

    # Same interface, different runtimes. Use onnx/litert if you want to avoid TensorFlow.
    cls = {"tf": "Perch2", "onnx": "Perch2ONNX", "litert": "Perch2LiteRT"}[backend]
    return getattr(bmz, cls)()


def detections(scores: pd.DataFrame, threshold: float, top_k: int) -> pd.DataFrame:
    """Turn a (windows x classes) score table into a long detections table."""
    rows = []
    for idx, row in scores.iterrows():
        file, start, end = idx  # OpenSoundscape's (file, start_time, end_time) index
        for species, p in row.nlargest(top_k).items():
            if p >= threshold:
                rows.append((file, start, end, species, float(p)))
    return pd.DataFrame(rows, columns=["file", "start_s", "end_s", "species", "confidence"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="audio file or folder")
    ap.add_argument("--out", type=Path, default=Path("perch_out"))
    ap.add_argument("--threshold", type=float, default=0.25, help="min confidence (0-1)")
    ap.add_argument("--top-k", type=int, default=3, help="max species per 5 s window")
    ap.add_argument("--backend", choices=["tf", "onnx", "litert"], default="tf")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--embeddings", action="store_true", help="also save 1536-d embeddings")
    args = ap.parse_args()

    files = find_audio(args.input)
    if not files:
        raise SystemExit(f"No audio files found in {args.input}")
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading Perch 2.0 ({args.backend})...")
    model = load_model(args.backend)

    print(f"Analysing {len(files)} file(s)...")
    logits = model.predict(files, batch_size=args.batch_size)  # raw logits per 5 s window
    scores = logits.apply(sigmoid)  # independent per-class confidence, since several birds can overlap

    dets = detections(scores, args.threshold, args.top_k)
    dets.to_csv(args.out / "detections.csv", index=False)

    if dets.empty:
        print("No detections above threshold. Try lowering --threshold.")
    else:
        summary = (
            dets.groupby("species")
            .agg(windows=("confidence", "size"), max_conf=("confidence", "max"), mean_conf=("confidence", "mean"))
            .sort_values("windows", ascending=False)
        )
        summary.to_csv(args.out / "species_summary.csv")
        print("\nTop species:")
        print(summary.head(15).round(3).to_string())

    if args.embeddings:
        emb = model.embed(files, batch_size=args.batch_size)
        np.save(args.out / "embeddings.npy", emb.to_numpy())
        emb.index.to_frame(index=False).to_csv(args.out / "embeddings_index.csv", index=False)
        print(f"\nSaved embeddings {emb.shape} (for clustering, or training your own classifier on top)")

    print(f"\nResults in {args.out}/")


if __name__ == "__main__":
    main()