# Bundles Person A's large files (kept out of Git) into one zip for the team drive.
#   powershell -ExecutionPolicy Bypass -File scripts\package_handover.ps1
# Output: outputs\apex3d_person_a_handover.zip. Unzip it at the repo root: paths are preserved.
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Items = @(
    "checkpoints\model_quantized.onnx",                 # Person C: what the app loads (calibrated INT8)
    "checkpoints\apex3d_convnext_tiny.pt",              # Person B/C: full PyTorch bundle (cls_lib.load_classifier)
    "artifacts\embeddings_train.npz", "artifacts\embeddings_val.npz", "artifacts\embeddings_test.npz",   # Person B: hybrid
    "artifacts\logits_train.npz", "artifacts\logits_val.npz", "artifacts\logits_test.npz",               # Person B: thresholds/ablation
    "outputs\triposr_test"                              # Person D: real sample meshes for the viewer
)
foreach ($i in $Items) { if (-not (Test-Path $i)) { throw "Missing $i. Run the notebooks first." } }

$Stage = Join-Path $Repo "outputs\handover_stage"
if (Test-Path $Stage) { Remove-Item -Recurse -Force $Stage }
foreach ($i in $Items) {
    $dest = Join-Path $Stage $i
    New-Item -ItemType Directory -Force (Split-Path $dest -Parent) | Out-Null
    Copy-Item -Recurse $i $dest
}
@"
Apex3D Person A handover ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))
Unzip at the repo root.
checkpoints/model_quantized.onnx   -> the classifier the backend auto-loads (temperature built in)
checkpoints/apex3d_convnext_tiny.pt -> PyTorch bundle; load with notebooks/cls_lib.load_classifier
artifacts/*.npz                     -> 768-d embeddings + logits for train/val/test (split = results/split.json)
outputs/triposr_test/*/mesh.glb     -> real TripoSR meshes for the 3D viewer
Details: docs/PERSON_A_ML_COMPUTE.md, docs/INTEGRATION_NOTES_A_TO_C.md
"@ | Set-Content -Encoding utf8 (Join-Path $Stage "README_HANDOVER.txt")

$Zip = Join-Path $Repo "outputs\apex3d_person_a_handover.zip"
if (Test-Path $Zip) { Remove-Item $Zip }
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $Zip
Remove-Item -Recurse -Force $Stage
"{0} ({1:N0} MB)" -f $Zip, ((Get-Item $Zip).Length / 1MB)
