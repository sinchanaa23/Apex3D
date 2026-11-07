"""
Pydantic data models and schemas defining the Apex3D API contract.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


# -------------------------------------------------------------------------
# Detection & Recognition Models
# -------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x1: float = Field(..., description="Top-left X coordinate (pixels)")
    y1: float = Field(..., description="Top-left Y coordinate (pixels)")
    x2: float = Field(..., description="Bottom-right X coordinate (pixels)")
    y2: float = Field(..., description="Bottom-right Y coordinate (pixels)")
    confidence: float = Field(..., description="Detector confidence score (0.0 to 1.0)")
    label: str = Field("car", description="Detected class label (e.g. car, truck, bus)")
    normalized: Optional[List[float]] = Field(None, description="[ymin, xmin, ymax, xmax] normalized (0.0 to 1.0)")


class VehiclePrediction(BaseModel):
    class_id: int = Field(..., description="Stanford Cars class index (0-195)")
    class_name: str = Field(..., description="Full class label (Make Model BodyStyle Year)")
    make: str = Field(..., description="Vehicle manufacturer (e.g. Audi, BMW)")
    model: str = Field(..., description="Vehicle model name (e.g. A4, M3)")
    year: int = Field(..., description="Model year")
    body_style: str = Field(..., description="Body style (e.g. Sedan, Coupe, SUV)")
    probability: float = Field(..., description="Softmax predicted probability (0.0 to 1.0)")


class RecognitionResponse(BaseModel):
    success: bool = True
    car_detected: bool = Field(..., description="Whether a vehicle was detected by the detector")
    crop_used: str = Field("detected_bbox", description="'detected_bbox' or 'full_frame'")
    bounding_box: Optional[BoundingBox] = Field(None, description="Primary vehicle bounding box")
    all_boxes: Optional[List[BoundingBox]] = Field(default=[], description="All detected vehicle candidate boxes")
    predictions: List[VehiclePrediction] = Field(..., description="Top-5 predicted fine-grained classes")
    confidence_state: str = Field(..., description="One of 'confident', 'ambiguous', 'rejected'")
    confidence_score: float = Field(..., description="Top-1 probability score")
    margin: float = Field(..., description="Difference between Top-1 and Top-2 probabilities")
    message: Optional[str] = Field(None, description="Human-readable status or guidance message")
    timings_ms: Dict[str, float] = Field(..., description="Breakdown of inference latency in milliseconds")


# -------------------------------------------------------------------------
# Classical CV Models
# -------------------------------------------------------------------------

class GeometricFeatures(BaseModel):
    aspect_ratio: float
    extent: float
    solidity: float
    contour_area: float
    perimeter: float
    hu_moments: List[float]


class HOGSummary(BaseModel):
    vector_length: int
    mean: float
    std: float
    feature_sample: List[float]


class LBPSummary(BaseModel):
    num_points: int
    radius: int
    histogram: List[float]


class HSVSummary(BaseModel):
    mean_hue: float
    mean_saturation: float
    mean_value: float
    hue_histogram: List[float]


class ClassicalFeatures(BaseModel):
    geometric: GeometricFeatures
    hog: HOGSummary
    lbp: LBPSummary
    color_hsv: HSVSummary


class ClassicalCVResponse(BaseModel):
    success: bool = True
    intermediate_images: Dict[str, str] = Field(
        ...,
        description="Base64 data URIs: original, clahe, edges, contours, largest_contour_roi, hog_visualization"
    )
    features: ClassicalFeatures
    timings_ms: Dict[str, float]


# -------------------------------------------------------------------------
# Reference Retrieval Models
# -------------------------------------------------------------------------

class ReferenceCandidate(BaseModel):
    id: str = Field(..., description="Unique identifier for the reference image")
    title: str = Field(..., description="Image title / file name")
    image_url: str = Field(..., description="Full image URL")
    thumb_url: str = Field(..., description="Thumbnail image URL")
    source: str = Field("wikimedia_commons", description="Source: 'wikimedia_commons', 'local_stanford', or 'manual_upload'")
    license: str = Field(..., description="Licence type (e.g. CC BY-SA 4.0, Public Domain)")
    license_url: Optional[str] = Field(None, description="Direct URL to licence terms")
    author: Optional[str] = Field(None, description="Author or contributor name")
    viewpoint: str = Field("unknown", description="'front', 'front_three_quarter', 'side', 'rear', 'rear_three_quarter', or 'unknown'")
    viewpoint_confidence: float = Field(0.0, description="Confidence in viewpoint classification (0.0 to 1.0)")
    match_score: float = Field(0.0, description="Classifier ranking or match score (0.0 to 1.0)")
    is_approximate: bool = Field(False, description="True if reference is marked as approximate match")
    width: Optional[int] = None
    height: Optional[int] = None


class ReferencesResponse(BaseModel):
    success: bool = True
    query: str
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    total_candidates: int
    candidates: List[ReferenceCandidate]
    viewpoint_groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Dictionary mapping viewpoint names to candidate IDs"
    )
    honesty_notice: str = (
        "References are images of the same vehicle model, not the identical physical car. "
        "Trim lines, wheel options, and condition may differ."
    )


class ReferenceUploadResponse(BaseModel):
    success: bool = True
    candidate: ReferenceCandidate
    message: str = "Reference image uploaded and indexed successfully"


# -------------------------------------------------------------------------
# 3D Reconstruction & Validation Models
# -------------------------------------------------------------------------

class ReconstructionOptions(BaseModel):
    model: str = Field("triposr", description="Reconstruction backend model")
    remove_background: bool = Field(True, description="Whether to apply foreground segmentation before mesh generation")
    mc_resolution: int = Field(256, description="Marching cubes resolution")
    bake_texture: bool = Field(True, description="Whether to bake vertex color or texture")


class ReconstructionJobCreate(BaseModel):
    image_url: Optional[str] = None
    reference_id: Optional[str] = None
    options: Optional[ReconstructionOptions] = Field(default_factory=ReconstructionOptions)


class MeshValidationReport(BaseModel):
    is_watertight: bool = Field(..., description="Whether mesh is a closed watertight 2-manifold")
    is_manifold: bool = Field(..., description="Whether mesh topology is 2-manifold")
    vertex_count: int
    face_count: int
    degenerate_faces: int = Field(..., description="Faces with zero or near-zero area")
    connected_components: int = Field(..., description="Number of disconnected mesh components")
    has_vertex_colors: bool
    has_uv_texture: bool
    bounding_box_extents: List[float] = Field(..., description="[width, height, length] in arbitrary model units")
    asset_readiness: str = Field("draft", description="'draft', 'needs_repair', or 'unvalidated'")
    scale_status: str = Field("unknown", description="Scale is uncalibrated single-view estimate; not real-world dimensions")
    notes: List[str] = Field(default_factory=list)


class ReconstructionAssets(BaseModel):
    glb_url: str = Field(..., description="URL to download or stream the GLB mesh")
    obj_url: str = Field(..., description="URL to download OBJ mesh")
    stl_url: str = Field(..., description="URL to download STL mesh")
    validation_report_url: str = Field(..., description="URL to full validation JSON")


class ReconstructionJobResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="'queued', 'running', 'completed', 'failed'")
    stage: str = Field("waiting", description="Current execution stage")
    progress: float = Field(0.0, description="0.0 to 1.0 completion ratio")
    queue_position: int = Field(0, description="Position in GPU job queue")
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    assets: Optional[ReconstructionAssets] = None
    validation: Optional[MeshValidationReport] = None
    mesh_source: Optional[str] = Field(None, description="'catalog', 'reference' or 'photo': what the mesh was made from")
    class_id: Optional[int] = None
    class_name: Optional[str] = None


# -------------------------------------------------------------------------
# Metrics & System Health Models
# -------------------------------------------------------------------------

class MetricsResponse(BaseModel):
    status: str = Field(..., description="'available' or 'not available'")
    message: Optional[str] = None
    source_file: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    service: str = "Apex3D Backend"
    gpu_available: bool = False
    device: str = "cpu"
    queue_size: int = 0
    mesh_builder: bool = True  # False when TripoSR isn't installed on this server
    detector: str = "unknown"  # "onnx", "torch" or "none" (the weak OpenCV fallback: something is wrong)

