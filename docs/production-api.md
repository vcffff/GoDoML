# ForteBank Datathon production API

Contract source: the complete `Кейс_ForteBank.docx`, including section 10 and the FAQ.
SHA-256: `F73A951C87ACA9D540076AACC5B4FD8A688810A4EC827B1A382512C770D278A0`.

## Start

From the project root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Mode local -BindAddress 127.0.0.1 -Port 8001
```

The launcher uses `ML_API_KEY`, or reads the existing private key from
`tmp/ml-api-key.txt`. The key file is excluded from Git. Supply its value in the
`X-API-Key` header. No key is returned by the API or written to evaluation logs.

Weights are loaded from `ML_MODEL_DIR`; the default is the existing local
`Qwen3-VL-2B-Instruct` installation. Model loading and inference use offline mode.
No video, frame or description is sent to a cloud AI service by this pipeline.

## Routes

| Request | Response |
|---|---|
| `GET /health`, no API key | HTTP 200 when the model is ready; HTTP 503 otherwise |
| `POST /api/v1/analyze`, multipart `file`, filename ending in `.mov` | HTTP 202, `{"job_id":"..."}` |
| `GET /api/v1/jobs/{job_id}` | HTTP 200, job status `processing`, `done` or `failed` |
| Missing/wrong `X-API-Key` | HTTP 401 before reading the upload |
| Unknown job with a valid key | HTTP 404 |
| Model already reserved by another request | HTTP 503 |

A completed job has exactly `job_id`, `status`, `verdict`, and `violations`.
Its verdict is `violation` or `no_violation`. Each violation contains `type`,
`timecode_sec`, `description`, and 1–3 JPEG base64 `evidence_frames`. The original
rule ID and regulatory source are included in its description.

A failed job has `job_id`, `status: "failed"`, and an `error` with a machine-readable
`code` and a safe `message`. It has **no binary verdict**. Codes distinguish
`analysis_inconclusive`, `analysis_timeout`, `invalid_video`, `analysis_error`,
and `inference_error`. Failed jobs must never be displayed as “no violation”.

`POST /analyze` remains available with its existing synchronous result shape and
regulatory `rule` object. It now requires the same API key in local mode.
`GET /health` remains public and exposes readiness, mode and busy state, not secrets.

## Backend health check

Call `GET /health` on the ML service base URL, without a request body or
`X-API-Key`. The check reads model state and does not run inference.

```sh
curl --max-time 5 -i "http://<ml-host>:8001/health"
```

Example response from an idle local service (HTTP 200):

```json
{
  "mode": "local",
  "ready": true,
  "busy": false,
  "mock": false,
  "classification_validated": false,
  "inference_only": true,
  "weights_updated": false,
  "assessment_schema_version": 2,
  "verdict_semantics": "candidate_for_human_review"
}
```

- HTTP 200 with `ready: true`: the model is loaded. A running analysis keeps
  HTTP 200 and sets `busy: true`; wait before submitting another video.
- HTTP 503 with `ready: false`: the model is not ready.
- A connection error or client timeout means the service is unavailable to the
  backend. Initial model loading happens before the server starts serving requests,
  so startup does not necessarily produce an HTTP 503 response.

Before sending real videos, check `mode == "local"`, `mock == false`,
`ready == true`, and `busy == false`. This is a snapshot, not a reservation:
another request can take the model, so submission must still handle HTTP 503.
Health does not certify recognition quality or successful analysis; inspect each
job's `status` and `error` separately. The same contract is described in `/docs`.

## Canonical execution

Both upload routes call the same `execute_model` helper and the same loaded
`Analyzer.analyze_video` instance. The existing `Video.sample` extraction feeds
Qwen screening, followed by individual relevant-rule assessments and deterministic
`AnalysisResult` validation. A shared lock prevents overlapping model calls.
No evaluation script or experimental worktree is part of this execution path.

Each rule assessment uses the installed Transformers `prefix_allowed_tokens_fn`
capability to constrain JSON syntax and the Y/N/U status alphabet. Each source
condition and exception has its own ordered status; evidence uses actual sampled
frames. These constraints do not force positive or negative judgments. Missing
assessments remain unknown. Unsupported model output fails the job explicitly.

## Practical limits

The processing deadline is 285 seconds from the beginning of upload. Polling
reports a failed timeout when that deadline passes; an outstanding native GPU or
decoder call retains its lock until it exits. This is not a guarantee against a
driver hang. Measure the complete upload/poll round trip against the official
five-minute limit.

Job state is in memory and is lost on restart. Keep the single-worker service
running during judging. This local launch does not configure a public jury URL,
TLS, firewall or routing. Recognition quality is separate from API/JSON validity;
failed or inconclusive jobs are excluded from conditional binary metrics and
reported separately with decision coverage.
