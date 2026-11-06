# What the frontend expects from the API

**For Person C.** The workplan lists the endpoints but not their exact formats, so the frontend was built against the formats below. These are proposals, not decisions: if yours differ, tell Person D or change `src/api/types.ts` and `src/api/client.ts`, which are the only files that know the formats.

## Rules for every endpoint

- Same origin as the frontend, under `/api`.
- Probabilities and rates are fractions from 0 to 1, not percentages.
- Errors use FastAPI's format, `{"detail": "A sentence the user can read"}`. The app shows `detail` as written.
- Images in a response are anything usable as an `<img src>`: a data URL, an `/api/...` path, or a full URL.

## Serving the frontend

`npm run build` writes `frontend/dist/`. FastAPI must serve those files **and return `index.html` for any other non-`/api` path**, so that `/studio/<job id>` still works when reloaded or opened from a link.

## Recognition

### `POST /api/recognize`

Multipart, field `image`: a JPEG the browser has already rotated upright and scaled to 2048 px or less on the long side. Box coordinates must refer to the image as uploaded.

```json
{
  "image_size": { "width": 1600, "height": 1000 },
  "detection": { "box": [120, 280, 880, 760], "label": "car", "score": 0.98 },
  "top5": [{ "class_id": 11, "label": "Audi TT Hatchback 2011", "probability": 0.71 }],
  "confidence": { "state": "confident", "top1": 0.71, "margin": 0.57, "reason": null },
  "timings_ms": { "detection": 412, "classification": 96, "total": 540 }
}
```

- `box` is `[x1, y1, x2, y2]` in pixels.
- `detection` is `null` when no vehicle is found. Still classify the full frame and return `200`, as the proposal's §11 says. The app warns the user.
- `confidence.state` is `confident`, `ambiguous` or `rejected`. `reason` is optional text shown to the user.
- `timings_ms` can hold any keys. A trailing `_ms` is removed for display.

### `POST /api/classical`

Multipart, field `image`.

```json
{
  "steps": [{ "key": "clahe", "title": "CLAHE", "image": "data:image/jpeg;base64,...", "caption": "Optional" }],
  "features": { "aspect_ratio": 1.58, "hsv_histogram": [0.1, 0.4] },
  "timings_ms": { "total": 180 }
}
```

Feature values are a number, a string or an array of numbers. Arrays of up to 64 values are drawn as a small histogram.

## References

### `GET /api/references?class_id=11`

```json
{
  "query": "Audi TT 2011",
  "candidates": [{
    "id": "abc",
    "image_url": "/api/...",
    "source": "Wikimedia Commons",
    "source_url": "https://commons.wikimedia.org/...",
    "author": "Name",
    "licence": "CC BY-SA 4.0",
    "licence_url": "https://creativecommons.org/...",
    "viewpoint": "side",
    "match": "confirmed"
  }]
}
```

- `viewpoint` is `front`, `front_three_quarter`, `side`, `rear_three_quarter`, `rear` or `null`. `null` puts the image under Unsorted, and users can always regroup by hand, so the cut-order fallback needs no frontend change.
- `match` is `confirmed` (the class is in the classifier's Top-k), `approximate` or `user`.

### `POST /api/references/upload`

Multipart, repeated field `images`. Returns the same shape, with `match: "user"`.

## 3D

### `POST /api/reconstruct`

Multipart with **either** an `image` file **or** a `reference_id` from the two endpoints above, plus an optional `class_id`. Returns `{"job_id": "..."}`.

### `GET /api/jobs/{id}`

Polled every 1.5 s until the status is final.

```json
{
  "id": "...",
  "status": "succeeded",
  "stage": "Reconstructing mesh",
  "queue_position": null,
  "logs": ["Job accepted", "Removing background"],
  "error": null,
  "result": {
    "asset_id": "...",
    "generation_seconds": 3.1,
    "files": [
      { "name": "model.glb", "format": "glb", "size_bytes": 123456 },
      { "name": "model.obj", "format": "obj", "size_bytes": 123456 },
      { "name": "model.stl", "format": "stl", "size_bytes": 123456 },
      { "name": "validation.json", "format": "json", "size_bytes": 200 }
    ]
  }
}
```

- `status` is `queued`, `running`, `succeeded` or `failed`.
- `queue_position` is shown while queued, where 1 means next.
- Return `404` for an unknown or cleaned-up job. The app stops polling on any 4xx and keeps retrying on network errors and 5xx.
- The viewer loads the file whose `format` is `glb`. Every listed file gets a download button, so FBX appears by itself if it is ever added.

### `GET /api/assets/{asset_id}/{file}`

Serves the listed files. The file with `format: "json"` is read as the validation report. Any field can be missing, and the app then shows "Not reported":

```json
{
  "watertight": false,
  "non_manifold_edges": 3,
  "degenerate_faces": 0,
  "connected_components": 4,
  "vertices": 32000,
  "faces": 64000,
  "has_texture": false,
  "has_vertex_colors": true
}
```

## Metrics

### `GET /api/metrics`

Each key is either `"not available"` or an object that names the saved file it came from:

```json
{
  "classifier": { "source": "path/to/file.json", "generated_at": "2026-11-06T12:00:00Z", "data": {} },
  "ablation": "not available"
}
```

| Key | `data` | Owner |
|---|---|---|
| `classifier` | `{model, top1, top5, top1_detector_crop, top5_detector_crop, test_images}` | A |
| `detection` | `{rows: [{model, mean_iou, map, cpu_ms}]}` | B |
| `optimisation` | `{rows: [{variant, size_mb, cpu_ms, gpu_ms, top1}]}` | A |
| `robustness` | `{rows: [{perturbation, severity, top1}]}` | B |
| `confidence` | `{chosen_threshold, rows: [{threshold, coverage, accuracy}]}` | B |
| `comparison` | `{rows: [{approach, top1, top5, inference_ms, size_mb}]}` | B |
| `ablation` | `{baseline_top1, rows: [{removed, top1, note}]}` | B |
| `confused_pairs` | `{rows: [{a, b, count}]}` | A or B |
| `failure_cases` | `{cases: [{title, image, predicted, actual, explanation}]}` | B |

A section in the wrong format shows a "couldn't be displayed" notice without affecting the rest of the page.

## Open question

**Correcting the identity is limited to the Top-5.** The contract has no endpoint listing all 196 classes, so if the true class isn't in the Top-5, users can only choose "None of these". The proposal only says "user correction", so this may be enough. A `GET /api/classes` returning `[{class_id, label}]` would allow a full search.
