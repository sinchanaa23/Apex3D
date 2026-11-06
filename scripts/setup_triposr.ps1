# Clones TripoSR (MIT) at the commit Apex3D was tested with into third_party/TripoSR.
# Weights (stabilityai/TripoSR, ~1.7 GB) download from Hugging Face on first use.
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Repo "third_party\TripoSR"
$Commit = "107cefdc244c39106fa830359024f6a2f1c78871"

function Invoke-Checked { param([scriptblock]$Cmd, [string]$What)
    & $Cmd
    if ($LASTEXITCODE -ne 0) { throw "$What failed (exit code $LASTEXITCODE)" }
}

if (-not (Test-Path (Join-Path $Dest "tsr\system.py"))) {
    New-Item -ItemType Directory -Force (Join-Path $Repo "third_party") | Out-Null
    Invoke-Checked { git clone https://github.com/VAST-AI-Research/TripoSR.git $Dest } "git clone"
}
Invoke-Checked { git -C $Dest fetch --depth 1 origin $Commit } "git fetch"
Invoke-Checked { git -C $Dest checkout --quiet $Commit } "git checkout"
Write-Host "TripoSR ready at $Dest ($Commit)"
