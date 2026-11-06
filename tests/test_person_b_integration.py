"""Checks that Person B's work still runs, and that the app uses the evaluated rules."""

import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["prepare_dataset", "evaluate_detector", "evaluate_with_classifier", "evaluate_classical_roi", "build_failure_cases", "build_eval_report", "build_app_metrics"]


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_starts_outside_pytest(script):
    """Pytest puts the repo root on the path, which hides import errors teammates would hit."""
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{script}.py"), "--help"], capture_output=True, text=True, env=environment, cwd=ROOT)
    assert result.returncode == 0, result.stderr


def test_dataset_loader_imports_alongside_the_app_class_list():
    code = "import sys; sys.path.insert(0, 'backend'); from data.stanford_cars import StanfordCars; from data import STANFORD_CARS_CLASSES; print(len(STANFORD_CARS_CLASSES))"
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=environment, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "196"


def test_app_picks_the_main_car_not_a_tiny_high_scoring_one():
    detector_service = pytest.importorskip("backend.services.detector_service")
    # The boxes the detector really returns for test image 00010 (Chevrolet HHR SS).
    tiny = SimpleNamespace(x1=16, y1=59, x2=33, y2=71, confidence=0.951)
    main = SimpleNamespace(x1=32, y1=59, x2=239, y2=170, confidence=0.874)
    assert detector_service.choose_primary([tiny, main]) is main


def test_app_crop_margin_matches_training():
    detector_service = pytest.importorskip("backend.services.detector_service")
    assert inspect.signature(detector_service.get_crop).parameters["padding_pct"].default == 0.08
