"""How does the app treat photos that are NOT one of its 196 cars?

    python scripts/evaluate_out_of_taxonomy.py

The classifier always names one of 196 classes, so the only protection is the confidence rule (rejected below
0.52, ambiguous when the lead over the runner-up is under 0.34). This measures how often that rule fires on
things the model has never been taught, and how often it still gives a confident answer.

Outside photos: the Caltech-101 test split (Hugging Face, Multimodal-Fatima/Caltech101_not_background_test, 78 MB),
in two groups. Other vehicles: airplanes, motorbikes, helicopters, ferries, ketches and schooners. Other things:
animals, objects and scenes. Caltech's own car category, its background class and its face classes are left out.

Limits, repeated in the result file: these are whole photos with no detector crop, Caltech-101 images are older and
smaller than Stanford Cars, and it does not contain newer car models, which is the likeliest real case. For a fair
comparison the in-class images are also scored uncropped.
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from data.stanford_cars import default_root, load_manifest  # noqa: E402
from evaluation import classifier_experiments as ex  # noqa: E402
from evaluation import confidence as conf  # noqa: E402
from recognition.onnx_classifier import OnnxClassifier  # noqa: E402

URL = "https://huggingface.co/datasets/Multimodal-Fatima/Caltech101_not_background_test/resolve/main/data/test-00000-of-00001-e2f326374823b432.parquet"
SIZE = 78_304_363
OTHER_VEHICLES = ("airplanes", "motorbikes", "helicopter", "ferry", "ketch", "schooner")
LEFT_OUT = ("car_side", "background_google", "faces", "faces_easy")
PER_VEHICLE_CLASS, OTHER_THINGS, IN_CLASS = 100, 600, 1200
SEED = 42

# The parquet carries no label names, so these are Caltech-101's standard alphabetical categories (index = label id).
# main() checks that the six vehicle names land on images whose captions describe them, so a wrong order stops the run.
CALTECH_NAMES = (
    "accordion airplanes anchor ant background_google barrel bass beaver binocular bonsai brain brontosaurus buddha butterfly "
    "camera cannon car_side ceiling_fan cellphone chair chandelier cougar_body cougar_face crab crayfish crocodile crocodile_head "
    "cup dalmatian dollar_bill dolphin dragonfly electric_guitar elephant emu euphonium ewer faces faces_easy ferry flamingo "
    "flamingo_head garfield gerenuk gramophone grand_piano hawksbill headphone hedgehog helicopter ibis inline_skate joshua_tree "
    "kangaroo ketch lamp laptop leopards llama lobster lotus mandolin mayfly menorah metronome minaret motorbikes nautilus octopus "
    "okapi pagoda panda pigeon pizza platypus pyramid revolver rhino rooster saxophone schooner scissors scorpion sea_horse snoopy "
    "soccer_ball stapler starfish stegosaurus stop_sign strawberry sunflower tick trilobite umbrella watch water_lilly wheelchair "
    "wild_cat windsor_chair wrench yin_yang"
).split()
CAPTION_CHECK = {"airplanes": "airplane", "motorbikes": "motorcycle", "helicopter": "helicopter", "stapler": "stapleer"}


def download(path: Path) -> None:
    if path.exists() and path.stat().st_size == SIZE:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Caltech-101 test split (78 MB)...", flush=True)
    urllib.request.urlretrieve(URL, path)
    if path.stat().st_size != SIZE:
        raise SystemExit(f"Expected {SIZE} bytes, got {path.stat().st_size}.")


def states_of(probabilities: np.ndarray, temperature: float, thresholds: dict):
    calibrated = conf.apply_temperature(probabilities, temperature)
    top, margin = conf.top_two(calibrated)
    return top, conf.decide(top, margin, thresholds["confident_p1"], thresholds["confident_margin"], thresholds["reject_below_p1"])


def shares(states: np.ndarray) -> dict:
    return {state: float((states == state).mean()) for state in ("rejected", "ambiguous", "confident")}


def main() -> None:
    from sklearn.metrics import roc_auc_score

    metrics = json.loads((ROOT / "results" / "classifier_metrics.json").read_text(encoding="utf-8"))
    temperature, thresholds = metrics["calibration"]["temperature"], metrics["thresholds"]
    classifier = OnnxClassifier()

    path = default_root().parent / "caltech101" / "test.parquet"
    download(path)
    table = pq.read_table(path)
    names = CALTECH_NAMES
    captions = table.select(["label", "blip_caption"]).to_pylist()
    for name, word in CAPTION_CHECK.items():
        mine = [c["blip_caption"] for c in captions if c["label"] == names.index(name)]
        if not mine or sum(word in c for c in mine) < 0.5 * len(mine):
            raise SystemExit(f"Label order check failed for {name}: the captions do not say '{word}'.")
    rows = table.select(["image", "label"]).to_pylist()
    rng = np.random.default_rng(SEED)

    def pick(label_names: tuple[str, ...] | None, count: int) -> list[dict]:
        pool = [r for r in rows if (names[r["label"]] in label_names if label_names else names[r["label"]] not in OTHER_VEHICLES + LEFT_OUT)]
        return [pool[i] for i in rng.permutation(len(pool))[:count]]

    vehicles = [r for name in OTHER_VEHICLES for r in pick((name,), PER_VEHICLE_CLASS)]
    others = pick(None, OTHER_THINGS)

    def classify(samples: list[dict]) -> np.ndarray:
        outputs, batch = [], []
        for sample in samples:
            batch.append(Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB"))
            if len(batch) == ex.BATCH_SIZE:
                outputs.append(classifier.predict_proba(batch))
                batch = []
        if batch:
            outputs.append(classifier.predict_proba(batch))
        return np.vstack(outputs)

    print(f"Scoring {len(vehicles)} other vehicles and {len(others)} other things...", flush=True)
    vehicle_probabilities, other_probabilities = classify(vehicles), classify(others)

    manifest = load_manifest(default_root())
    in_class = manifest[manifest["split"] == "test"].sample(IN_CLASS, random_state=SEED)
    print(f"Scoring {len(in_class)} in-class test photos, uncropped, for a fair comparison...", flush=True)
    in_probabilities = ex.run_batches(classifier.predict_proba, in_class, default_root(), "none")

    groups = {"other_vehicles": vehicle_probabilities, "other_things": other_probabilities, "in_class_uncropped": in_probabilities}
    result = {"experiment": "out_of_taxonomy", "model": "checkpoints/model_quantized.onnx (deployed)", "thresholds": thresholds, "temperature": temperature,
              "source": "Caltech-101 test split, Hugging Face Multimodal-Fatima/Caltech101_not_background_test",
              "other_vehicle_classes": list(OTHER_VEHICLES), "groups": {}}
    tops = {}
    for name, probabilities in groups.items():
        top, states = states_of(probabilities, temperature, thresholds)
        tops[name] = top
        result["groups"][name] = {"images": len(probabilities), **shares(states), "mean_top_probability": float(top.mean())}

    outside = np.concatenate([tops["other_vehicles"], tops["other_things"]])
    scores = np.concatenate([tops["in_class_uncropped"], outside])
    truth = np.concatenate([np.ones(len(tops["in_class_uncropped"])), np.zeros(len(outside))])
    result["separation_auroc"] = float(roc_auc_score(truth, scores))
    result["sweep"] = [
        {"threshold": round(float(t), 2), "in_class_accepted": float((tops["in_class_uncropped"] >= t).mean()),
         "other_vehicles_accepted": float((tops["other_vehicles"] >= t).mean()), "other_things_accepted": float((tops["other_things"] >= t).mean())}
        for t in np.linspace(0.1, 0.95, 18)
    ]
    result["limits"] = [
        "Whole photos with no detector crop. The app would crop to a detected car first.",
        "Caltech-101 photos are older and smaller than Stanford Cars.",
        "Caltech-101 has no newer car models, which is the likeliest real case, so this measures non-car inputs only.",
        "In-class images are scored uncropped here so both sides are treated alike.",
    ]
    ex.save_json(result, ROOT / "results" / "out_of_taxonomy.json")
    for name, values in result["groups"].items():
        print(f"{name:20s} n={values['images']:5d}  rejected {values['rejected']:.1%}  ambiguous {values['ambiguous']:.1%}  confident {values['confident']:.1%}")
    print(f"AUROC (in-class vs outside, by top probability): {result['separation_auroc']:.3f}")


if __name__ == "__main__":
    main()
