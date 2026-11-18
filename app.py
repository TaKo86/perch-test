#!/usr/bin/env python3
"""Upload bird audio and classify it with the saved NZ Perch probe."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import joblib
from flask import Flask, abort, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import bioacoustics_model_zoo as bmz
from bird_facts import BIRD_FACTS, get_bird_facts, slugify_bird_label

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "nz_birds_probe.joblib"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
TOP_K = 3

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

classifier = joblib.load(MODEL_PATH)
perch_model = None
inference_lock = Lock()


def get_perch_model():
    global perch_model
    if perch_model is None:
        perch_model = bmz.Perch2()
    return perch_model


def classify_file(audio_path: Path) -> list[dict]:
    with inference_lock:
        embeddings = get_perch_model().embed([str(audio_path)], batch_size=16)

    probabilities = classifier.predict_proba(embeddings.to_numpy())
    results = []
    for (_audio_file, start, end), row in zip(
        embeddings.index, probabilities, strict=True
    ):
        scores = [
            (label, float(score))
            for label, score in zip(classifier.classes_, row, strict=True)
            if label.lower() != "background"
        ]
        top = sorted(scores, key=lambda item: item[1], reverse=True)[:TOP_K]
        results.append(
            {
                "start": float(start),
                "end": float(end),
                "predictions": [
                    {"label": label, "score": score} for label, score in top
                ],
            }
        )
    return results


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return (
        render_template(
            "index.html",
            error="That file is over the 100 MB upload limit.",
            max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
        ),
        413,
    )


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/birds/<slug>")
def bird_detail(slug):
    for label, bird in BIRD_FACTS.items():
        if slugify_bird_label(label) == slug:
            return render_template("bird.html", bird=bird)
    abort(404)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template(
            "index.html",
            max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
            get_bird_facts=get_bird_facts,
            slugify_bird_label=slugify_bird_label,
        )

    upload = request.files.get("audio")
    if upload is None or not upload.filename:
        return render_template(
            "index.html",
            error="Choose an audio file to analyze.",
            max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
            get_bird_facts=get_bird_facts,
            slugify_bird_label=slugify_bird_label,
        ), 400

    filename = secure_filename(upload.filename)
    extension = Path(filename).suffix.lower()
    if extension not in AUDIO_EXTS:
        return render_template(
            "index.html",
            error="Use a WAV, FLAC, MP3, OGG, or M4A audio file.",
            max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
            get_bird_facts=get_bird_facts,
            slugify_bird_label=slugify_bird_label,
        ), 400

    with TemporaryDirectory(prefix="perch-upload-") as temp_dir:
        audio_path = Path(temp_dir) / f"audio{extension}"
        upload.save(audio_path)
        results = classify_file(audio_path)

    return render_template(
        "index.html",
        filename=filename,
        results=results,
        max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
        get_bird_facts=get_bird_facts,
        slugify_bird_label=slugify_bird_label,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
