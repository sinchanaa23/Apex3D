import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from data.stanford_cars import StanfordCars, default_root, load_class_names, load_manifest, parse_class_name


@pytest.mark.parametrize(
    "name, make, model, body, year",
    [
        ("AM General Hummer SUV 2000", "AM General", "Hummer", "SUV", 2000),
        ("Aston Martin V8 Vantage Coupe 2012", "Aston Martin", "V8 Vantage", "Coupe", 2012),
        ("Ford F-150 Regular Cab 2007", "Ford", "F-150", "Regular Cab", 2007),
        ("smart fortwo Convertible 2012", "smart", "fortwo", "Convertible", 2012),
        ("Acura Integra Type R 2001", "Acura", "Integra Type R", "", 2001),
    ],
)
def test_parse_class_name(name, make, model, body, year):
    parsed = parse_class_name(name)
    assert (parsed["make"], parsed["model"], parsed["body_type"], parsed["year"]) == (make, model, body, year)


needs_data = pytest.mark.skipif(not (default_root() / "manifest.csv").exists(), reason="dataset not prepared")


@needs_data
def test_split_sizes_and_classes():
    manifest = load_manifest()
    counts = manifest["split"].value_counts()
    assert counts["test"] == 8041
    assert counts["train"] + counts["val"] == 8144
    assert counts["val"] == pytest.approx(8144 * 0.15, abs=2)
    assert manifest.groupby("split")["label"].nunique().eq(196).all()
    assert len(load_class_names()) == 196


@needs_data
def test_splits_do_not_overlap():
    manifest = load_manifest()
    assert manifest["path"].is_unique


@needs_data
def test_every_image_file_exists():
    manifest = load_manifest()
    assert all((default_root() / path).exists() for path in manifest["path"])


@needs_data
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_loader_returns_rgb_images_and_crops(split):
    full, label = StanfordCars(split)[0]
    cropped, same_label = StanfordCars(split, crop="box")[0]
    assert full.mode == "RGB" and 0 <= label < 196 and label == same_label
    assert cropped.size[0] <= full.size[0] and cropped.size[1] <= full.size[1]
