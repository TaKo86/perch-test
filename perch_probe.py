#!/usr/bin/env python3
"""
Train a small classifier ("linear probe") on frozen Perch 2.0 embeddings for NZ species.

Expected training data layout (folder name = label):
  data/
    tui/            *.wav
    bellbird/       *.wav
    kereru/         *.wav
    background/     *.wav     <- strongly recommended: wind, rain, silence, non-target noise

Train:    python perch_probe.py train data/ --model nz_probe.joblib
Predict:  python perch_probe.py predict recordings/ --model nz_probe.joblib --threshold 0.6

Install: pip install --upgrade bioacoustics-model-zoo opensoundscape tensorflow kagglehub scikit-learn joblib pandas numpy
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def find_audio(path: Path) -> list[str]:
    if path.is_file():
        return [str(path)]
    return sorted(str(p) for p in path.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def embed(files: list[str], backend: str, batch_size: int) -> pd.DataFrame:
    import bioacoustics_model_zoo as bmz

    cls = {"tf": "Perch2", "onnx": "Perch2ONNX", "litert": "Perch2LiteRT"}[backend]
    model = getattr(bmz, cls)()
    return model.embed(files, batch_size=batch_size)  # index: (file, start_time, end_time), 1536 cols


def cmd_train(args):
    files = find_audio(args.data)
    if not files:
        raise SystemExit(f"No audio found in {args.data}")
    print(f"Embedding {len(files)} files with Perch 2.0...")
    emb = embed(files, args.backend, args.batch_size)

    files_idx = emb.index.get_level_values(0)
    labels = np.array([Path(f).parent.name for f in files_idx])  # label = parent folder name
    X = emb.to_numpy()
    print(f"{len(X)} windows, classes: {dict(pd.Series(labels).value_counts())}")

    # Split by FILE, not by window. Windows from one recording are near-duplicates, so a
    # window-level split leaks and gives an optimistic score.
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=0)
    train_idx, test_idx = next(splitter.split(X, labels, groups=files_idx))
    missing = set(labels) - set(labels[test_idx])
    if missing:
        print(f"Warning: no test files for {sorted(missing)}; add more recordings per class.")

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced"),
    )
    clf.fit(X[train_idx], labels[train_idx])

    print("\nHeld-out performance (split by recording):")
    print(classification_report(labels[test_idx], clf.predict(X[test_idx]), zero_division=0))

    # Refit on everything for the saved model
    clf.fit(X, labels)
    joblib.dump(clf, args.model)
    print(f"Saved {args.model}")


def cmd_predict(args):
    clf = joblib.load(args.model)
    files = find_audio(args.input)
    if not files:
        raise SystemExit(f"No audio found in {args.input}")
    emb = embed(files, args.backend, args.batch_size)

    proba = pd.DataFrame(clf.predict_proba(emb.to_numpy()), index=emb.index, columns=clf.classes_)
    rows = []
    for (file, start, end), row in proba.iterrows():
        for species, p in row.nlargest(args.top_k).items():
            if p >= args.threshold and species not in args.ignore:
                rows.append((file, start, end, species, float(p)))
    out = pd.DataFrame(rows, columns=["file", "start_s", "end_s", "species", "confidence"])
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "probe_detections.csv", index=False)
    print(out.groupby("species").size().sort_values(ascending=False).to_string() if len(out) else "No detections")
    print(f"Saved {args.out / 'probe_detections.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "predict"):
        p = sub.add_parser(name)
        p.add_argument("data" if name == "train" else "input", type=Path)
        p.add_argument("--model", type=Path, default=Path("nz_probe.joblib"))
        p.add_argument("--backend", choices=["tf", "onnx", "litert"], default="tf")
        p.add_argument("--batch-size", type=int, default=16)
        if name == "train":
            p.add_argument("--test-size", type=float, default=0.25)
            p.add_argument("--C", type=float, default=1.0, help="inverse regularisation; lower = stronger")
            p.set_defaults(func=cmd_train)
        else:
            p.add_argument("--out", type=Path, default=Path("perch_out"))
            p.add_argument("--threshold", type=float, default=0.6)
            p.add_argument("--top-k", type=int, default=2)
            p.add_argument("--ignore", nargs="*", default=["background"], help="labels to hide from output")
            p.set_defaults(func=cmd_predict)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
