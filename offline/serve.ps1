# AgriSmart offline: a tiny local web server for the offline app, using only what ships with Windows.
# Started by run.bat. Serves offline\web at http://localhost:8777/ and opens the browser.
# No internet and no Python needed; everything (models, voices, data) is inside offline\web.
param([int]$Port = 8777, [switch]$NoBrowser)

$ErrorActionPreference = "Stop"
$root = Join-Path $PSScriptRoot "web"
if (-not (Test-Path (Join-Path $root "index.html"))) {
    Write-Host "offline\web\index.html is missing. Run offline\tools\fetch_assets.ps1 once, or re-download the app." -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
}

$mime = @{
    ".html" = "text/html; charset=utf-8"; ".js" = "text/javascript; charset=utf-8"; ".mjs" = "text/javascript; charset=utf-8"
    ".css" = "text/css; charset=utf-8"; ".json" = "application/json; charset=utf-8"; ".wasm" = "application/wasm"
    ".onnx" = "application/octet-stream"; ".ogg" = "audio/ogg"; ".mp3" = "audio/mpeg"; ".wav" = "audio/wav"
    ".png" = "image/png"; ".jpg" = "image/jpeg"; ".svg" = "image/svg+xml"; ".ico" = "image/x-icon"; ".webmanifest" = "application/manifest+json"
}

$listener = New-Object System.Net.HttpListener
$prefix = "http://localhost:$Port/"
$listener.Prefixes.Add($prefix)
try { $listener.Start() } catch {
    Write-Host "Port $Port is busy. Close the other AgriSmart window, or run: serve.ps1 -Port $($Port + 1)" -ForegroundColor Red
    Read-Host "Press Enter to close"; exit 1
}
Write-Host ""
Write-Host "  AgriSmart offline is running at $prefix" -ForegroundColor Green
Write-Host "  Keep this window open while you use the app. Close it to stop."
Write-Host ""
if (-not $NoBrowser) { Start-Process $prefix }

$rootFull = [IO.Path]::GetFullPath($root)
while ($listener.IsListening) {
    $ctx = $listener.GetContext()
    $res = $ctx.Response
    try {
        $rel = [Uri]::UnescapeDataString($ctx.Request.Url.AbsolutePath.TrimStart("/"))
        if ($rel -eq "") { $rel = "index.html" }
        $path = [IO.Path]::GetFullPath((Join-Path $rootFull $rel))
        if (-not $path.StartsWith($rootFull) -or -not (Test-Path $path -PathType Leaf)) {
            $res.StatusCode = 404
            $bytes = [Text.Encoding]::UTF8.GetBytes("Not found")
            $res.OutputStream.Write($bytes, 0, $bytes.Length)
        } else {
            $ext = [IO.Path]::GetExtension($path).ToLower()
            $res.ContentType = if ($mime.ContainsKey($ext)) { $mime[$ext] } else { "application/octet-stream" }
            # cross-origin isolation lets the model use every CPU core (WebAssembly threads); "credentialless"
            # keeps that while still showing the online check's photos from Pl@ntNet
            $res.AddHeader("Cross-Origin-Opener-Policy", "same-origin")
            $res.AddHeader("Cross-Origin-Embedder-Policy", "credentialless")
            $res.AddHeader("Cache-Control", "no-cache")
            $fs = [IO.File]::OpenRead($path)
            try { $res.ContentLength64 = $fs.Length; $fs.CopyTo($res.OutputStream) } finally { $fs.Dispose() }
        }
    } catch {
        # the browser closed the connection early (e.g. it stopped an audio clip) -- nothing to do
    } finally {
        try { $res.OutputStream.Close() } catch {}
    }
}
