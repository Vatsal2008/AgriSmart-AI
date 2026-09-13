@echo off
rem AgriSmart offline: double-click to open the crop-disease app in your browser. Works without internet.
cd /d "%~dp0"
if not exist "web\models\india_v1.onnx" (
    echo The model and voice files are not here yet. Downloading them once from GitHub ^(a few hundred MB^)...
    powershell -NoProfile -ExecutionPolicy Bypass -File "tools\fetch_assets.ps1"
    if errorlevel 1 (
        echo Download failed. Check the internet connection and run this again.
        pause
        exit /b 1
    )
)
title AgriSmart offline
powershell -NoProfile -ExecutionPolicy Bypass -File "serve.ps1"
