# Exposes the local Apex3D server through a Cloudflare quick tunnel (no account needed).
#   powershell -ExecutionPolicy Bypass -File scripts\run_tunnel.ps1 [-Port 8000]
# Start the FastAPI server first (Person C's one-command startup), then run this.
# The public https://<random>.trycloudflare.com URL is printed by cloudflared below.
# It changes every time the tunnel restarts: update the stable link page when it does.
param([int]$Port = 8000)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Exe = Join-Path $Repo "tools\cloudflared.exe"

if (-not (Test-Path $Exe)) {
    $Found = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($Found) { $Exe = $Found.Source }
    else {
        Write-Host "cloudflared not found. Install it once with:"
        Write-Host "  winget install --id Cloudflare.cloudflared"
        Write-Host "or download cloudflared-windows-amd64.exe from"
        Write-Host "  https://github.com/cloudflare/cloudflared/releases/latest"
        Write-Host "and save it as $Exe"
        exit 1
    }
}

try { Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/" -TimeoutSec 5 | Out-Null }
catch {
    if (-not $_.Exception.Response) { Write-Host "Nothing is answering on port $Port. Start the server first."; exit 1 }
}

Write-Host "Tunnelling http://127.0.0.1:$Port. Keep this window open; Ctrl+C stops the public URL."
& $Exe tunnel --no-autoupdate --url "http://127.0.0.1:$Port"
