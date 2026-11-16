from io import BytesIO

import app as app_module


def test_health_endpoint():
    response = app_module.app.test_client().get("/healthz")

    assert response.status_code == 200
    assert response.json == {"status": "ok"}


def test_upload_renders_analysis_result(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "classify_file",
        lambda _path: [
            {
                "start": 0.0,
                "end": 5.0,
                "predictions": [{"label": "tui", "score": 0.9}],
            }
        ],
    )

    response = app_module.app.test_client().post(
        "/",
        data={"audio": (BytesIO(b"audio data"), "recording.wav")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"Analysis results" in response.data
    assert b"recording.wav" in response.data
    assert b"tui" in response.data


def test_upload_rejects_unsupported_file_type():
    response = app_module.app.test_client().post(
        "/",
        data={"audio": (BytesIO(b"not audio"), "recording.txt")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert b"Use a WAV, FLAC, MP3, OGG, or M4A" in response.data
