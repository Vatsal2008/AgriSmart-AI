# Packs the two downloads for the GitHub release into offline\build\release:
#   AgriSmart-offline-windows.zip  run.bat + serve.ps1 + the web app with models and voice clips
#   AgriSmart-offline.apk          the Android app (same web app inside)
# Run after export_onnx.py, build_offline_data.py, build_audio.py and "gradlew assembleRelease".
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot           # ...\offline
$out = Join-Path $root "build\release"
$stage = Join-Path $root "build\stage\AgriSmart-offline"
if (Test-Path (Join-Path $root "build\stage")) { Remove-Item (Join-Path $root "build\stage") -Recurse -Force }
New-Item -ItemType Directory -Force $out, $stage | Out-Null

foreach ($need in "web\models\india_v1.onnx", "web\models\india_v2.onnx", "web\models\core.onnx", "web\data\app.json", "web\audio\en\index.json") {
    if (-not (Test-Path (Join-Path $root $need))) { throw "Missing $need - build it first" }
}
Copy-Item (Join-Path $root "run.bat"), (Join-Path $root "serve.ps1") $stage
New-Item -ItemType Directory -Force (Join-Path $stage "tools") | Out-Null
Copy-Item (Join-Path $root "tools\fetch_assets.ps1") (Join-Path $stage "tools")
robocopy (Join-Path $root "web") (Join-Path $stage "web") /E /XD _test /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed ($LASTEXITCODE)" }
Copy-Item (Join-Path $root "README.md") (Join-Path $stage "README.md") -ErrorAction SilentlyContinue

$zip = Join-Path $out "AgriSmart-offline-windows.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $stage -DestinationPath $zip -CompressionLevel Optimal
$apk = Join-Path $root "android\app\build\outputs\apk\release\app-release.apk"
if (Test-Path $apk) { Copy-Item $apk (Join-Path $out "AgriSmart-offline.apk") -Force } else { Write-Warning "No APK built yet" }

Get-ChildItem $out | ForEach-Object {
    "{0,-34} {1,8:N1} MB  sha256 {2}" -f $_.Name, ($_.Length / 1MB), (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
}
