# Colab training notebook (backup)

> **Status:** Person A trained the classifier on their own GPU laptop (92.2% test Top-1, see `results/classifier_metrics.json`). This Colab notebook is kept as an independent backup and cross-check. It is not needed to run the app.

## 01_classifier_training_colab.ipynb (backup)

Trains and evaluates the car classifier in **Google Colab** (free T4 GPU is enough). It replaces the plan to train on Person A's laptop.

### Run it

1. Open Colab, then `File > Upload notebook`, and choose `01_classifier_training_colab.ipynb`. (Once this branch is on GitHub you can also use `File > Open notebook > GitHub`.)
2. `Runtime > Change runtime type > T4 GPU`.
3. Add a **Kaggle API key** so the notebook can download the dataset. In Kaggle, go to Settings, then API, then Create New Token. In Colab, open the key icon on the left and add two secrets, `KAGGLE_USERNAME` and `KAGGLE_KEY`, with notebook access turned on. If you skip this, the notebook asks you to upload `kaggle.json`.
4. `Runtime > Run all`, and allow Google Drive access when asked. Results are saved to `MyDrive/apex3d/`.

**About the dataset:** the Kaggle mirror named in the project proposal (`jessicali9530/stanford-cars-dataset`) has been removed. The notebook tries the copies in its `SLUGS` list in order, prints what it finds in each, and uses the first it can read. If none work, download Stanford Cars yourself, put the folder in Google Drive, and set `MANUAL_DATA_DIR` in the download cell. A 403 error from Kaggle for a dataset usually means the dataset is gone, not that your key is wrong.

Expect roughly 30 to 60 minutes on a T4 for training, plus about 15 minutes for evaluation, the detector-crop test and optimisation. Colab can disconnect: run all again and training resumes from the last saved epoch.

### What it does and does not do

Does: dataset checks, a stratified 85/15 train/validation split, staged fine-tuning of ConvNeXt-Tiny, Top-1 and Top-5 on the unseen test set (ground-truth and detector crops), per-class precision, recall and F1, the confusion matrix and most-confused pairs, temperature scaling, threshold analysis, and an ONNX plus INT8 export with size, speed and accuracy before and after.

Does not: the classical baseline, hybrid fusion, ablation study, robustness tests, and detector IoU and mAP. Those belong to Person B's notebooks (`02` and `03`).

### Things it checks and will stop for

- Image counts, class counts and decoding, and that each test label agrees with the folder its image sits in. The proposal flags the test labels as unverified.
- That the class list has the dataset's 196 names in the backend's order. **At the time of writing the backend's `STANFORD_CARS_CLASSES` has 197 entries**, so the notebook will report which names differ. Person C needs to fix that list before the trained model is connected, or every class after the extra entry gets the wrong name in the app.

### Files it produces (in `MyDrive/apex3d/artifacts/`)

| File | Goes to |
|---|---|
| `model.onnx`, `model_quantized.onnx` | the repo's `checkpoints/`. The backend loads them. The temperature is built in, so its softmax gives calibrated probabilities |
| `metrics.json` | `notebooks/metrics.json`. The Research Lab reads it. It merges with any existing file, so Person B's results are kept |
| `classes.json` | reference, to confirm the class order |
| `per_class_metrics.csv`, `confusion_matrix.png`, `confidence_curve.png`, `training_curves.png`, `failure_candidates.png` | the evaluation report |
| `convnext_stanford_cars_state.pth` | weights and class names, for anyone retraining or inspecting |

The backend needs `onnxruntime` added to `requirements.txt` to load the ONNX files.

### Honest limits

- Timings are measured on the Colab machine, not on the final host. Re-measure there for the report.
- INT8 is dynamic quantization. It shrinks the model and speeds up CPU inference, and the notebook measures its accuracy cost on a shared test subset.
- The detector-crop test uses torchvision's Faster R-CNN, as the backend does, but it is not Person B's detector evaluation.
- Failure cases: the notebook only suggests candidates. A person has to write the explanations.
