# How the frontend uses the backend API

The formats are Person C's, from [docs/API_CONTRACT.md](../docs/API_CONTRACT.md) on the `backend-c` branch. `src/api/adapter.ts` is the only file that translates them into what the pages use, so a format change touches that file and `adapter.test.ts`.

Tested end to end against the real backend on 2026-11-06: recognition, classical CV, reference search, reference upload path, 3D job, validation, downloads and metrics all worked.

## Differences the frontend works around

| Contract | What the frontend does |
|---|---|
| Upload field is `file` | Sends `file` |
| `/api/recognize` returns no image size | Uses the size of the image it sent, which it resized itself |
| Job status `completed` | Shown as succeeded |
| `queue_position: 0` while running | Treats 0 as "not waiting" |
| Validation arrives inside the job, not only as a file | Reads it from the job. The file is still offered as a download |
| `/api/references/upload` takes one file | Sends one request per file |
| `/api/reconstruct` can fetch an `image_url` itself | The frontend never uses it. It downloads a reference in the browser and uploads the image as a file, so the server is not asked to fetch web addresses (Person A's integration note 8) |
| Uploaded references have a relative `image_url` that the server can't fetch | Downloads the image in the browser and sends it as a file |
| `/api/metrics` returns flat numbers | Shows each as a section. Missing or non-numeric values show "Not available" |

## Requests for Person C

1. **Say when a result is not real.** The classifier fallback (`feature_heuristic`) picks classes from a hash of the image, and the reconstruction fallback returns a fixed box-shaped car. The app can't tell these from real output, so a judge would see "Confident: Dodge Durango" for any photo. Suggested fields: `model_type` on `/api/recognize` and `is_placeholder` on a finished job. The frontend would then show a clear "placeholder, not a real prediction" label. The workplan's honesty rule asks for exactly this.
2. **Add `GET /api/classes`** returning `[{class_id, class_name}]`, so a user can correct the identity beyond the Top-5.
3. **Return `image_size`** from `/api/recognize`, so the box doesn't depend on the frontend's resize.
4. **Serve `.glb` without `Content-Disposition: attachment`** if possible. It works today, but inline is the safer default for a viewer.
5. **Metrics shape.** The frontend can draw robustness curves, a confidence curve, the ablation table and failure cases if `metrics` also includes `{rows: [...]}` sections. The shapes are in `src/api/types.ts`. Today's flat numbers give single values only.
6. **Start-up time.** The first recognition took about 12 s on CPU, mostly loading the detector. A warm-up call at start would help the 10 s target.

## Saved result files as a fallback

`getMetrics` asks `/api/metrics` first. For any section the backend has no data for, it uses Person A's saved files, which the build copies to `dist/results/` (`vite.config.ts`, `RESULT_FILES`). Only `classifier_metrics.json` and `optimization_metrics.json` are exposed. The backend's own numbers always win when both exist, and a section with no file behind it stays "Not available". `src/api/client.test.ts` covers these cases.

## Feeding `/api/metrics` with Person A's results

The endpoint returns whichever single file the backend finds first (`notebooks/metrics.json`, then `eval_P5_metrics.json`, then `docs/metrics.json`). Person A saves two files under `results/`, which the backend does not read, so the Research Lab shows "Not available" until one is put in a path it reads.

The Research Lab understands three shapes, all from saved files, and shows only what a file contains:

| Shape | Sections it fills |
|---|---|
| Flat keys such as `test_top1_acc` (Person C's contract) | Classifier accuracy, Size and speed (two rows), and the Person B sections |
| Person A's `classifier_metrics.json` as it is | Classifier accuracy, Training run, Confidence states, Calibration |
| Person A's `optimization_metrics.json` as it is | Size and speed, with every variant and the deployed one marked |

One file can hold several of these at once. Use these names for the parts: `classifier_metrics`, `optimization_metrics`, plus the usual `confused_pairs`, `robustness`, `comparison` and `ablation` at the top level.

To combine Person A's two files and their confused-pairs table into the file the backend reads, save this as `combine_metrics.py` and run it from the repo root with `python combine_metrics.py`:

```python
import csv, json

a = json.load(open("results/classifier_metrics.json"))
b = json.load(open("results/optimization_metrics.json"))
pairs = [
    {"a": r["class_a"], "b": r["class_b"], "count": int(r["total"])}
    for r in csv.DictReader(open("results/classifier_confused_pairs.csv"))
]
combined = {"classifier_metrics": a, "optimization_metrics": b, "confused_pairs": {"rows": pairs}}
json.dump(combined, open("notebooks/metrics.json", "w"), indent=1)
print("wrote notebooks/metrics.json with", len(pairs), "confused pairs")
```

Person B's results (`robustness`, `comparison`, `ablation`) can be added to the same file as top-level keys. The "Errors" column for confused pairs is `total`, which is both directions added together. Person A's `results/classifier_per_class.csv` and plots are not shown on the page.

Person A's real files are kept as test fixtures in `src/api/fixtures/`, and `src/api/adapter.test.ts` checks that their numbers come out right.
