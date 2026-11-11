# Starts Apex3D on this computer and opens it in the browser at http://localhost:8000
#
#   powershell -ExecutionPolicy Bypass -File scripts\Start-Apex3D.ps1 [-NoBrowser] [-Port 8000]
#
# Keep the window open while you use the app; closing it (or pressing Ctrl+C) stops the server.
# If Apex3D is already running, it only opens the browser. Needs the app's setup done once (.venv and frontend\dist); see README.
param([switch]$NoBrowser, [int]$Port = 8000)

$ErrorActionPreference = "Stop"
$App = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $App
$Url = "http://localhost:$Port"
$Host.UI.RawUI.WindowTitle = "Apex3D server - close this window to stop it"

function Test-Apex3D {
    try { return ((Invoke-WebRequest "$Url/api/health" -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200) } catch { return $false }
}

function Stop-WithMessage([string]$Message) {
    Write-Host ""
    Write-Host $Message -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to close"
    exit 1
}

if (Test-Apex3D) {
    Write-Host "Apex3D is already running at $Url."
    if (-not $NoBrowser) { Start-Process $Url }
    exit 0
}

$Python = Join-Path $App ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { Stop-WithMessage "Apex3D isn't set up in $App yet: the .venv folder is missing. See the README's quickstart." }
if (-not (Test-Path (Join-Path $App "frontend\dist\index.html"))) { Stop-WithMessage "The web app isn't built yet. Run: cd frontend ; npm ci ; npm run build" }
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { Stop-WithMessage "Something else is already using port $Port, and it isn't Apex3D. Close it, or run this script with -Port 8001." }

Write-Host "Starting Apex3D at $Url ... the first start takes about half a minute while the models load."
$Server = Start-Process -FilePath $Python -ArgumentList "start_server.py", "--no-reload", "--host", "127.0.0.1", "--port", $Port -NoNewWindow -PassThru
try {
    $ready = $false
    for ($i = 0; $i -lt 180; $i++) {
        if ($Server.HasExited) { break }
        if (Test-Apex3D) { $ready = $true; break }
        Start-Sleep -Seconds 1
    }
    if (-not $ready) { Stop-WithMessage "Apex3D didn't start. The messages above say why." }
    Write-Host ""
    Write-Host "Apex3D is ready at $Url. Sign up on first use. Close this window to stop it." -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process $Url }
    $Server.WaitForExit()
}
finally {
    if ($Server -and -not $Server.HasExited) { Stop-Process -Id $Server.Id -Force -ErrorAction SilentlyContinue }
}
