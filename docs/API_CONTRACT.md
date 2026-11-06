# Apex3D — API Contract & Endpoint Specification

**Status:** APPROVED (Hour 0–1 Checkpoint)  
**Base URL:** `http://localhost:8000` (Local) or Tunnel URL (Live judging)  
**API Prefix:** `/api`  
**OpenAPI / Swagger Documentation:** `http://localhost:8000/docs`

This document defines the formal request and response schemas agreed upon between Backend (Person C), Frontend (Person D), ML/GPU (Person A), and Classical CV (Person B).

---

## Quick Reference Table

| Method | Endpoint | Description | Input | Output Summary |
|---|---|---|---|---|
| `POST` | `/api/recognize` | Vehicle detection & fine-grained classification | Image file (`multipart/form-data`) | Bounding box, Top-5, confidence state, latency |
| `POST` | `/api/classical` | Classical CV visualisations & descriptors | Image file (`multipart/form-data`) | Base64 stages (CLAHE, Canny, HOG, etc.) & numerical features |
| `GET` | `/api/references` | Search & filter Wikimedia reference candidates | Query params: `class_id` or `query` | Deduplicated, verified candidates grouped by viewpoint |
| `POST` | `/api/references/upload` | Upload user reference photo | Image file (`multipart/form-data`) | Indexed reference candidate |
| `POST` | `/api/reconstruct` | Enqueue background 3D mesh generation | Uploaded image or reference URL | `job_id`, queue status |
| `GET` | `/api/jobs/{id}` | Poll real job progress, status and logs | Path param: `id` | Status, stage, progress 0-1, logs, asset URLs |
| `GET` | `/api/assets/{id}/{file}` | Download generated 3D files | Path param: `id`, `file` (glb/obj/stl) | Binary mesh stream or validation JSON |
| `GET` | `/api/metrics` | Retrieve offline notebook evaluation results | None | Real metrics JSON or "not available" state |
| `GET` | `/api/health` | Service health and GPU readiness | None | GPU status, queue size, uptime |

---

## 1. Recognition Endpoint: `POST /api/recognize`

Performs vehicle detection using R-CNN, crops the primary vehicle, feeds it to the fine-grained classifier, and determines the confidence state.

### Request
- **Content-Type:** `multipart/form-data`
- **Form Fields:**
  - `file`: `UploadFile` (Required) — Supported: `image/jpeg`, `image/png`, `image/webp`. Max size: 20 MB.
  - `force_full_frame`: `boolean` (Optional, default `false`) — Force classification on the full frame even if a detector box is present.

### Response `200 OK`
```json
{
  "success": true,
  "car_detected": true,
  "crop_used": "detected_bbox",
  "bounding_box": {
    "x1": 42.0,
    "y1": 115.0,
    "x2": 612.0,
    "y2": 450.0,
    "confidence": 0.94,
    "label": "car",
    "normalized": [0.24, 0.06, 0.94, 0.96]
  },
  "all_boxes": [
    {
      "x1": 42.0,
      "y1": 115.0,
      "x2": 612.0,
      "y2": 450.0,
      "confidence": 0.94,
      "label": "car",
      "normalized": [0.24, 0.06, 0.94, 0.96]
    }
  ],
  "predictions": [
    {
      "class_id": 7,
      "class_name": "Aston Martin V8 Vantage Convertible 2012",
      "make": "Aston Martin",
      "model": "V8 Vantage",
      "year": 2012,
      "body_style": "Convertible",
      "probability": 0.8842
    },
    {
      "class_id": 8,
      "class_name": "Aston Martin V8 Vantage Coupe 2012",
      "make": "Aston Martin",
      "model": "V8 Vantage",
      "year": 2012,
      "body_style": "Coupe",
      "probability": 0.0611
    },
    {
      "class_id": 9,
      "class_name": "Aston Martin Virage Convertible 2012",
      "make": "Aston Martin",
      "model": "Virage",
      "year": 2012,
      "body_style": "Convertible",
      "probability": 0.0245
    },
    {
      "class_id": 10,
      "class_name": "Aston Martin Virage Coupe 2012",
      "make": "Aston Martin",
      "model": "Virage",
      "year": 2012,
      "body_style": "Coupe",
      "probability": 0.0150
    },
    {
      "class_id": 98,
      "class_name": "Ferrari California Convertible 2012",
      "make": "Ferrari",
      "model": "California",
      "year": 2012,
      "body_style": "Convertible",
      "probability": 0.0042
    }
  ],
  "confidence_state": "confident",
  "confidence_score": 0.8842,
  "margin": 0.8231,
  "message": "High-confidence vehicle identification.",
  "timings_ms": {
    "detection": 45.2,
    "classification": 32.1,
    "total": 81.3
  }
}
```

### Confidence States
- `"confident"`: Top-1 probability $\ge 0.50$ AND Top-1 - Top-2 margin $\ge 0.15$.
- `"ambiguous"`: Top-1 probability $\ge 0.50$ BUT margin $< 0.15$ (near-identical model years or trim levels; UI highlights top candidates).
- `"rejected"`: Top-1 probability $< 0.20$ (out-of-taxonomy vehicle or non-car image; UI flags as unrecognised and allows manual search).

### Error Responses
- `400 Bad Request`: Unsupported file type (e.g. PDF or text).
- `413 Payload Too Large`: File exceeds 20 MB.
- `422 Unprocessable Entity`: Corrupt image that cannot be decoded.

---

## 2. Classical CV Endpoint: `POST /api/classical`

Computes traditional computer vision pipeline transforms and feature descriptors required by P5.

### Request
- **Content-Type:** `multipart/form-data`
- **Form Fields:**
  - `file`: `UploadFile` (Required) — Target image.
  - `bbox_x1`: `float` (Optional) — Custom bounding box.
  - `bbox_y1`: `float` (Optional)
  - `bbox_x2`: `float` (Optional)
  - `bbox_y2`: `float` (Optional)

### Response `200 OK`
```json
{
  "success": true,
  "intermediate_images": {
    "original": "data:image/png;base64,iVBORw0KGgo...",
    "clahe": "data:image/png;base64,iVBORw0KGgo...",
    "edges": "data:image/png;base64,iVBORw0KGgo...",
    "contours": "data:image/png;base64,iVBORw0KGgo...",
    "largest_contour_roi": "data:image/png;base64,iVBORw0KGgo...",
    "hog_visualization": "data:image/png;base64,iVBORw0KGgo..."
  },
  "features": {
    "geometric": {
      "aspect_ratio": 2.14,
      "extent": 0.72,
      "solidity": 0.88,
      "contour_area": 142500.0,
      "perimeter": 1940.5,
      "hu_moments": [0.0012, 0.00004, 0.000001, ...]
    },
    "hog": {
      "vector_length": 3780,
      "mean": 0.045,
      "std": 0.082,
      "feature_sample": [0.12, 0.04, 0.00, 0.21, 0.15]
    },
    "lbp": {
      "num_points": 24,
      "radius": 3,
      "histogram": [0.03, 0.05, 0.12, ...]
    },
    "color_hsv": {
      "mean_hue": 24.5,
      "mean_saturation": 142.1,
      "mean_value": 185.0,
      "hue_histogram": [0.15, 0.22, ...]
    }
  },
  "timings_ms": {
    "clahe": 8.1,
    "canny_contours": 12.4,
    "hog": 24.5,
    "lbp": 16.2,
    "geometric": 3.1,
    "total": 66.8
  }
}
```

---

## 3. Reference Retrieval: `GET /api/references`

Queries Wikimedia Commons for openly licensed images matching the vehicle class, removes duplicates via perceptual hashing, filters through the classifier, and groups into viewpoints.

### Request
- **Parameters (Query string):**
  - `class_id`: `int` (Optional) — Stanford Cars class index (0–195).
  - `query`: `string` (Optional) — Custom vehicle search string (e.g. `"Aston Martin V8 Vantage"`).
  - `limit`: `int` (Optional, default `12`, max `30`).

### Response `200 OK`
```json
{
  "success": true,
  "query": "Aston Martin V8 Vantage 2012",
  "class_id": 7,
  "class_name": "Aston Martin V8 Vantage Convertible 2012",
  "total_candidates": 6,
  "candidates": [
    {
      "id": "wm_9482104",
      "title": "File:Aston Martin V8 Vantage front.jpg",
      "image_url": "https://upload.wikimedia.org/wikipedia/commons/...",
      "thumb_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/...",
      "source": "wikimedia_commons",
      "license": "CC BY-SA 4.0",
      "license_url": "https://creativecommons.org/licenses/by-sa/4.0",
      "author": "Vauxford",
      "viewpoint": "front_three_quarter",
      "viewpoint_confidence": 0.89,
      "match_score": 0.94,
      "is_approximate": false,
      "width": 1920,
      "height": 1080
    }
  ],
  "viewpoint_groups": {
    "front": ["wm_12345"],
    "front_three_quarter": ["wm_9482104"],
    "side": ["wm_67890"],
    "rear": [],
    "rear_three_quarter": ["wm_44556"]
  },
  "honesty_notice": "References are images of the same vehicle model, not the identical physical car. Trim lines, wheel options, and condition may differ."
}
```

---

## 4. Manual Reference Upload: `POST /api/references/upload`

Enables users to contribute their own high-res reference photos.

### Request
- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `file`: `UploadFile` (Required)
  - `viewpoint`: `string` (Optional, default `"unknown"`, values: `front`, `front_three_quarter`, `side`, `rear`, `rear_three_quarter`)
  - `class_id`: `int` (Optional)
  - `label`: `string` (Optional)

---

## 5. 3D Reconstruction Job: `POST /api/reconstruct`

Submits an image to the background 3D reconstruction queue. Single-view inference via TripoSR.

### Request (Option A: Direct Multipart Upload)
- **Content-Type:** `multipart/form-data`
- **Fields:**
  - `file`: `UploadFile` (Required if `image_url` is omitted)
  - `remove_background`: `boolean` (Default `true`)
  - `mc_resolution`: `int` (Default `256`)

### Request (Option B: JSON with Reference URL)
- **Content-Type:** `application/json`
```json
{
  "image_url": "https://upload.wikimedia.org/...",
  "reference_id": "wm_9482104",
  "options": {
    "model": "triposr",
    "remove_background": true,
    "mc_resolution": 256
  }
}
```

### Response `202 Accepted`
```json
{
  "job_id": "8fa213e4-c119-4cb5-8d54-159846b0de91",
  "status": "queued",
  "queue_position": 1,
  "created_at": "2026-11-06T12:00:00Z",
  "message": "Reconstruction job enqueued successfully."
}
```

---

## 6. Job Status & Logs: `GET /api/jobs/{id}`

Returns the real-time state of the 3D generation worker.

### Response `200 OK` (While Running)
```json
{
  "job_id": "8fa213e4-c119-4cb5-8d54-159846b0de91",
  "status": "running",
  "stage": "triposr_inference",
  "progress": 0.65,
  "queue_position": 0,
  "created_at": "2026-11-06T12:00:00Z",
  "started_at": "2026-11-06T12:00:02Z",
  "completed_at": null,
  "logs": [
    "[12:00:00] Job created and enqueued",
    "[12:00:02] Worker picked up job. Segmenting vehicle foreground...",
    "[12:00:05] Running TripoSR image-to-3D representation...",
    "[12:00:18] Extracting isosurface via Marching Cubes (res=256)..."
  ],
  "error": null,
  "assets": null,
  "validation": null
}
```

### Response `200 OK` (When Completed)
```json
{
  "job_id": "8fa213e4-c119-4cb5-8d54-159846b0de91",
  "status": "completed",
  "stage": "done",
  "progress": 1.0,
  "queue_position": 0,
  "created_at": "2026-11-06T12:00:00Z",
  "started_at": "2026-11-06T12:00:02Z",
  "completed_at": "2026-11-06T12:00:26Z",
  "logs": [
    "[12:00:00] Job created and enqueued",
    "[12:00:02] Background removed successfully",
    "[12:00:18] Mesh extracted (52,140 vertices, 104,276 faces)",
    "[12:00:22] Trimesh validation completed: watertight 2-manifold",
    "[12:00:26] Formats exported: GLB, OBJ, STL"
  ],
  "error": null,
  "assets": {
    "glb_url": "/api/assets/8fa213e4-c119-4cb5-8d54-159846b0de91/mesh.glb",
    "obj_url": "/api/assets/8fa213e4-c119-4cb5-8d54-159846b0de91/mesh.obj",
    "stl_url": "/api/assets/8fa213e4-c119-4cb5-8d54-159846b0de91/mesh.stl",
    "validation_report_url": "/api/assets/8fa213e4-c119-4cb5-8d54-159846b0de91/validation.json"
  },
  "validation": {
    "is_watertight": true,
    "is_manifold": true,
    "vertex_count": 52140,
    "face_count": 104276,
    "degenerate_faces": 0,
    "connected_components": 1,
    "has_vertex_colors": true,
    "has_uv_texture": false,
    "bounding_box_extents": [1.92, 1.34, 4.45],
    "asset_readiness": "draft",
    "scale_status": "unknown",
    "notes": [
      "Mesh is topologically valid and watertight.",
      "Unseen sides were inferred from single view.",
      "Real-world scale is uncalibrated. Do not use for engineering without scaling."
    ]
  }
}
```

---

## 7. Asset Download: `GET /api/assets/{id}/{file}`

Streams the generated 3D files:
- `mesh.glb` (`model/gltf-binary`) — Used directly by React Three Fiber viewer.
- `mesh.obj` (`text/plain`) — For 3D modeling packages (Blender, Maya).
- `mesh.stl` (`model/stl`) — For CAD and 3D slicing.
- `validation.json` (`application/json`) — Full geometric validation telemetry.

---

## 8. Offline Metrics Endpoint: `GET /api/metrics`

Reads genuine offline training and evaluation results produced by Jupyter notebooks.

### Response `200 OK` (When Notebooks Have Run)
```json
{
  "status": "available",
  "source_file": "notebooks/metrics.json",
  "metrics": {
    "test_top1_acc": 0.884,
    "test_top5_acc": 0.962,
    "detector_mAP50": 0.912,
    "detector_mean_iou": 0.845,
    "model_size_mb": 112.4,
    "quantized_size_mb": 29.1,
    "inference_time_gpu_ms": 14.2,
    "inference_time_cpu_ms": 48.6,
    "inference_time_quantized_cpu_ms": 18.5,
    "robustness": {
      "noise_gaussian_top1": 0.812,
      "blur_motion_top1": 0.795,
      "brightness_low_top1": 0.834,
      "occlusion_synthetic_top1": 0.742
    },
    "comparison": {
      "classical_svm_top1": 0.312,
      "deep_convnext_top1": 0.884,
      "hybrid_fusion_top1": 0.891
    }
  }
}
```

### Response `200 OK` (If Notebook Has Not Finished)
```json
{
  "status": "not available",
  "message": "Classifier training metrics have not been generated yet. Notebook 01 has not finished training.",
  "metrics": null
}
```

---

## 9. Health & System Status: `GET /api/health`

### Response `200 OK`
```json
{
  "status": "ok",
  "version": "1.0.0",
  "service": "Apex3D Backend",
  "gpu_available": false,
  "device": "cpu",
  "queue_size": 0
}
```

