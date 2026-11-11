# Puts an "Apex3D" shortcut on your desktop that starts the app (scripts\Start-Apex3D.ps1) and opens http://localhost:8000
#
#   powershell -ExecutionPolicy Bypass -File scripts\make_desktop_shortcut.ps1
#
# The shortcut points at the folder this script is in, so run it from the copy of Apex3D you want to keep using.
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Desktop = [Environment]::GetFolderPath("Desktop")
$Link = Join-Path $Desktop "Apex3D.lnk"
$Icon = Join-Path $App "scripts\apex3d.ico"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($Link)
$Shortcut.TargetPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$(Join-Path $App 'scripts\Start-Apex3D.ps1')`""
$Shortcut.WorkingDirectory = $App
$Shortcut.Description = "Start Apex3D and open it at http://localhost:8000"
if (Test-Path $Icon) { $Shortcut.IconLocation = $Icon }
$Shortcut.Save()
Write-Host "Created $Link"
