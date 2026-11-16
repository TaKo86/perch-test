# NZ Bird Sound Classifier

A small Flask web app for uploading a bird recording and classifying it with the
saved NZ probe (`nz_birds_probe.joblib`). Perch 2.0 creates an embedding for each
5-second audio segment; the probe returns the top three non-background labels for
each segment.

## Run locally

From the project directory, install the packages in `requirements.txt` in an
environment that supports TensorFlow and Perch 2.0, then start the app:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000> and upload a WAV, FLAC, MP3, OGG, or M4A file
(maximum 100 MB). The first analysis loads the Perch model and can take longer
than later analyses. Uploaded files are processed from a temporary directory
and removed after analysis.

The displayed scores are model estimates, not calibrated probabilities. This
app uses Flask's development server and binds to the local machine; it is not
configured for public deployment. Before exposing it to other people, use a
production WSGI server and HTTPS, and consider authentication, request limits,
and how uploaded audio and server logs are handled.
