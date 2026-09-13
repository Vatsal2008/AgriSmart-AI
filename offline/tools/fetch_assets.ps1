# Downloads the offline app's models and voice clips (too big for git) from the GitHub release, once.
# Called by run.bat when web\models\india_v1.onnx is missing. Needs internet only this one time.
param(
    [string]$Release = "offline-app-v1",
    [string]$Repo = "Vatsal2008/AgriSmart-AI",
    [string]$Asset = "AgriSmart-offline-windows.zip"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # the progress bar makes Invoke-WebRequest many times slower
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot          # ...\offline
$web = Join-Path $root "web"
$tmp = Join-Path $env:TEMP ("agrismart-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force $tmp | Out-Null
try {
    $url = "https://github.com/$Repo/releases/download/$Release/$Asset"
    Write-Host "Downloading $url"
    $zip = Join-Path $tmp $Asset
    Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
    Write-Host ("Downloaded {0:N0} MB, unpacking..." -f ((Get-Item $zip).Length / 1MB))
    # .NET's unzip: Expand-Archive in Windows PowerShell 5.1 takes many minutes on the ~4,400 files
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($zip, $tmp)
    $src = Get-ChildItem $tmp -Recurse -Directory -Filter web | Where-Object { Test-Path (Join-Path $_.FullName "models\india_v1.onnx") } | Select-Object -First 1
    if (-not $src) { throw "The download does not contain web\models\india_v1.onnx" }
    New-Item -ItemType Directory -Force (Join-Path $web "models") | Out-Null
    Copy-Item (Join-Path $src.FullName "models\*") (Join-Path $web "models") -Force
    if (Test-Path (Join-Path $src.FullName "audio")) {
        Copy-Item (Join-Path $src.FullName "audio") $web -Recurse -Force
    }
    Write-Host "Models and voice clips are ready."
} finally {
    Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
