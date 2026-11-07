"""The ONNX adapter used by the evaluation scripts. Skipped when the model file isn't in checkpoints/."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

MODEL = REPO / "checkpoints" / "model_quantized.onnx"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="needs checkpoints/model_quantized.onnx (Person A's handover zip)")


@pytest.fixture(scope="module")
def classifier():
    from recognition.onnx_classifier import OnnxClassifier

    return OnnxClassifier(MODEL, threads=2)


def photos(count=3):
    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, (120 + 10 * i, 180, 3), dtype=np.uint8)) for i in range(count)]


def test_probabilities_are_a_distribution_over_196_classes(classifier):
    probabilities = classifier.predict_proba(photos())
    assert probabilities.shape == (3, 196)
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)
    assert (probabilities >= 0).all()


def test_embeddings_are_the_768_values_before_the_classification_layer(classifier):
    assert classifier.embed(photos()).shape == (3, 768)


def test_one_pass_gives_the_same_answers_as_two(classifier):
    images = photos()
    probabilities, embeddings = classifier.predict_and_embed(images)
    assert np.allclose(probabilities, classifier.predict_proba(images), atol=1e-5)
    assert np.allclose(embeddings, classifier.embed(images), atol=1e-5)


def test_reading_the_embedding_does_not_change_the_prediction(classifier):
    import onnxruntime as ort

    images = photos()
    plain = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"]).run(None, {"input": classifier._batch(images)})[0]
    assert (plain.argmax(axis=1) == classifier.predict_proba(images).argmax(axis=1)).all()


def test_calibration_can_be_applied_on_top_without_changing_the_class(classifier):
    from evaluation import confidence as conf

    probabilities = classifier.predict_proba(photos())
    calibrated = conf.apply_temperature(probabilities, classifier.temperature)
    assert (calibrated.argmax(axis=1) == probabilities.argmax(axis=1)).all()
    assert (calibrated.max(axis=1) >= probabilities.max(axis=1) - 1e-6).all()
