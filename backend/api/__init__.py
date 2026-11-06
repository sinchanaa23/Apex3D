"""
Apex3D API Routers.
"""
from backend.api.routes_recognition import router as recognition_router
from backend.api.routes_classical import router as classical_router
from backend.api.routes_references import router as references_router
from backend.api.routes_reconstruct import router as reconstruct_router
from backend.api.routes_metrics import router as metrics_router

__all__ = [
    "recognition_router",
    "classical_router",
    "references_router",
    "reconstruct_router",
    "metrics_router",
]

