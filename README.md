# NZ Bird Sound Classifier

A responsive Flask web app for uploading or recording a bird sound sample and
classifying it with the saved NZ probe (`nz_birds_probe.joblib`). Perch 2.0
creates embeddings for 5-second audio segments; the probe returns the top three
non-background labels for each segment.

Each recognized label links to a bird facts page with its scientific name,
an overview, and a photo loaded from Wikipedia when available. Photos are
fetched by the viewer's browser; the facts page remains usable if the image
service is unavailable.

## Run locally

Use Python in an environment supported by TensorFlow and Perch 2.0:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. Choose a WAV, FLAC, MP3, OGG, or M4A file (up to
100 MB), or allow microphone access and record up to two minutes. Microphone
recording requires HTTPS in deployment (localhost is also considered secure by
browsers). The browser converts a recording to WAV before upload. Uploaded
files are processed from a temporary directory and removed after analysis.

The first analysis loads the Perch model and can take longer than later
analyses. The `/healthz` endpoint is a lightweight liveness check; it does not
load or verify the inference model.

## Deploy

The repository includes a `wsgi.py` entry point and a `Procfile` for Linux
platforms that support Procfiles. Install `requirements.txt`, make sure
`nz_birds_probe.joblib` is included in the deployed application, and use a
Python/runtime combination supported by the installed Perch and TensorFlow
packages. The start command binds to the platform's `PORT` and runs one Gunicorn
worker with a longer request timeout for model inference.

This is a deployment starting point, not a fully hardened public service.
Terminate HTTPS at the hosting platform so microphone access works. Before
opening the app to the public, add appropriate abuse protection (such as
authentication or rate limits), monitoring, and resource controls. Model
inference is CPU/memory intensive, and each instance should be sized and
load-tested accordingly. Audio is temporarily stored on the server for
inference; document that handling in the deployed service's privacy notice.

The displayed scores are model estimates, not calibrated probabilities or a
guarantee of species identification.
