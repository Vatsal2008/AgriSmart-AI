package com.agrismart.offline;

import android.Manifest;
import android.app.Activity;
import android.content.ClipData;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Base64;
import android.util.Log;
import android.webkit.ConsoleMessage;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.ValueCallback;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

import org.json.JSONObject;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/**
 * AgriSmart offline: a full-screen WebView showing the offline web app from the APK's assets
 * (assets/web, copied from offline/web at build time). The page is served from a private https origin with
 * cross-origin isolation headers so the model can use several CPU threads. Everything works without a network;
 * the optional online check (turned on by the user) talks to Pl@ntNet and an online AI directly from the page.
 */
public class MainActivity extends Activity {
    private static final String HOST = "appassets.androidplatform.net";
    private static final String TAG = "AgriSmartWeb";
    private static final int REQ_CAMERA = 1;
    private static final int REQ_FILE = 2;

    private WebView web;
    private PermissionRequest pendingCamera;
    private ValueCallback<Uri[]> fileCallback;
    private TextToSpeech tts;
    private volatile boolean ttsReady;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        web = new WebView(this);
        web.setBackgroundColor(Color.parseColor("#f3f5ef"));
        setContentView(web);

        WebSettings s = web.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);                 // history and settings (localStorage)
        s.setMediaPlaybackRequiresUserGesture(false); // the answer is spoken as soon as it is ready
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(true);
        web.addJavascriptInterface(new Bridge(), "AgriSmartAndroid");

        // the phone's own voice, for online answers (the offline answers use recorded clips)
        tts = new TextToSpeech(this, status -> ttsReady = status == TextToSpeech.SUCCESS);
        tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override public void onStart(String id) { }
            @Override public void onDone(String id) { spoken(id); }
            @Override public void onError(String id) { spoken(id); }
            @Override public void onStop(String id, boolean interrupted) { spoken(id); }
        });

        web.setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                Uri url = request.getUrl();
                if (!HOST.equals(url.getHost())) {
                    return null;                      // online check: let the WebView fetch it normally
                }
                return serveAsset(url.getPath());
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                Uri url = request.getUrl();
                if (HOST.equals(url.getHost())) return false;
                try {
                    startActivity(new Intent(Intent.ACTION_VIEW, url));   // links (e.g. key sign-up) open in the browser
                } catch (Exception ignored) {
                    // no browser installed
                }
                return true;
            }
        });

        web.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onConsoleMessage(ConsoleMessage m) {
                // the page's messages in logcat (tag AgriSmartWeb), for field debugging with "adb logcat -s AgriSmartWeb"
                Log.i(TAG, m.messageLevel() + " " + m.message() + " (" + m.sourceId() + ":" + m.lineNumber() + ")");
                return true;
            }

            @Override
            public void onPermissionRequest(PermissionRequest request) {
                runOnUiThread(() -> {
                    boolean wantsCamera = false;
                    for (String r : request.getResources()) {
                        if (PermissionRequest.RESOURCE_VIDEO_CAPTURE.equals(r)) wantsCamera = true;
                    }
                    if (!wantsCamera) {
                        request.deny();
                    } else if (hasCamera()) {
                        request.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
                    } else {
                        pendingCamera = request;
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
                    }
                });
            }

            @Override
            public boolean onShowFileChooser(WebView view, ValueCallback<Uri[]> callback, FileChooserParams params) {
                if (fileCallback != null) fileCallback.onReceiveValue(null);
                fileCallback = callback;
                Intent pick = new Intent(Intent.ACTION_GET_CONTENT);
                pick.addCategory(Intent.CATEGORY_OPENABLE);
                pick.setType("image/*");
                try {
                    startActivityForResult(Intent.createChooser(pick, null), REQ_FILE);
                } catch (Exception e) {
                    fileCallback = null;
                    return false;
                }
                return true;
            }
        });

        if (savedInstanceState != null) {
            web.restoreState(savedInstanceState);
        } else {
            web.loadUrl("https://" + HOST + "/index.html");
        }
    }

    private boolean hasCamera() {
        return Build.VERSION.SDK_INT < 23 || checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private WebResourceResponse serveAsset(String path) {
        if (path == null || path.equals("/") || path.isEmpty()) path = "/index.html";
        if (path.contains("..")) return notFound();
        String asset = "web" + path;
        Map<String, String> headers = baseHeaders();
        try {
            InputStream in = getAssets().open(asset);
            try (AssetFileDescriptor fd = getAssets().openFd(asset)) {   // only works for uncompressed assets
                headers.put("Content-Length", String.valueOf(fd.getLength()));
            } catch (IOException compressed) {
                // size unknown for compressed assets; the page copes without it
            }
            String mime = mimeType(asset);
            String charset = mime.startsWith("text/") || mime.endsWith("javascript") || mime.endsWith("json") ? "utf-8" : null;
            return new WebResourceResponse(mime, charset, 200, "OK", headers, in);
        } catch (IOException e) {
            return notFound();
        }
    }

    private static Map<String, String> baseHeaders() {
        Map<String, String> h = new HashMap<>();
        h.put("Cross-Origin-Opener-Policy", "same-origin");
        // "credentialless" keeps the page cross-origin isolated (threads) and still shows Pl@ntNet's photos
        h.put("Cross-Origin-Embedder-Policy", "credentialless");
        h.put("Cross-Origin-Resource-Policy", "same-origin");
        h.put("Cache-Control", "no-cache");
        return h;
    }

    private static WebResourceResponse notFound() {
        return new WebResourceResponse("text/plain", "utf-8", 404, "Not Found", baseHeaders(),
                new ByteArrayInputStream(new byte[0]));
    }

    private static String mimeType(String name) {
        String n = name.toLowerCase(Locale.ROOT);
        if (n.endsWith(".html")) return "text/html";
        if (n.endsWith(".js") || n.endsWith(".mjs")) return "text/javascript";
        if (n.endsWith(".css")) return "text/css";
        if (n.endsWith(".json")) return "application/json";
        if (n.endsWith(".wasm")) return "application/wasm";
        if (n.endsWith(".ogg")) return "audio/ogg";
        if (n.endsWith(".svg")) return "image/svg+xml";
        if (n.endsWith(".png")) return "image/png";
        if (n.endsWith(".jpg") || n.endsWith(".jpeg")) return "image/jpeg";
        return "application/octet-stream";
    }

    /** A voice for this language on the phone (hi-IN, ta-IN...), or null if the phone has none. */
    private Locale voiceFor(String lang) {
        if (!ttsReady || lang == null) return null;
        for (Locale l : new Locale[]{new Locale(lang, "IN"), new Locale(lang)}) {
            if (tts.isLanguageAvailable(l) >= TextToSpeech.LANG_AVAILABLE) return l;
        }
        return null;
    }

    private void spoken(String id) {
        runOnUiThread(() -> web.evaluateJavascript(
                "window.__agriTtsDone && window.__agriTtsDone(" + JSONObject.quote(id) + ")", null));
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        super.onRequestPermissionsResult(requestCode, permissions, results);
        if (requestCode == REQ_CAMERA && pendingCamera != null) {
            if (results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
                pendingCamera.grant(new String[]{PermissionRequest.RESOURCE_VIDEO_CAPTURE});
            } else {
                pendingCamera.deny();
            }
            pendingCamera = null;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_FILE && fileCallback != null) {
            Uri[] uris = WebChromeClient.FileChooserParams.parseResult(resultCode, data);
            if (uris == null && resultCode == RESULT_OK && data != null) {
                // some pickers (e.g. the Android photo picker behind a chooser) return the photo only in ClipData
                if (data.getData() != null) {
                    uris = new Uri[]{data.getData()};
                } else if (data.getClipData() != null && data.getClipData().getItemCount() > 0) {
                    uris = new Uri[]{data.getClipData().getItemAt(0).getUri()};
                }
            }
            Log.i(TAG, "photo picker result=" + resultCode + " photo=" + (uris == null ? "none" : uris[0]));
            fileCallback.onReceiveValue(uris);
            fileCallback = null;
        }
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        web.saveState(outState);
    }

    @Override
    public void onBackPressed() {
        if (web.canGoBack()) web.goBack(); else super.onBackPressed();
    }

    @Override
    protected void onDestroy() {
        if (tts != null) tts.shutdown();
        web.destroy();
        super.onDestroy();
    }

    /** Called from the page as window.AgriSmartAndroid.*. */
    private class Bridge {
        /** Opens the phone's share sheet (WhatsApp, SMS...). */
        @JavascriptInterface
        public void share(String text) {
            runOnUiThread(() -> {
                Intent send = new Intent(Intent.ACTION_SEND);
                send.setType("text/plain");
                send.putExtra(Intent.EXTRA_TEXT, text);
                startActivity(Intent.createChooser(send, null));
            });
        }

        /** Speaks text with the phone's voice; false if there is no voice for the language. Calls __agriTtsDone(id) when done. */
        @JavascriptInterface
        public boolean speak(String text, String lang, String id) {
            Locale voice = voiceFor(lang);
            if (voice == null || text == null || text.isEmpty()) return false;
            tts.setLanguage(voice);
            tts.setSpeechRate(0.95f);
            return tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, id) == TextToSpeech.SUCCESS;
        }

        /** Hands the leaf photo (base64 JPEG) to Google Lens for a picture search; the share menu if Lens is missing. */
        @JavascriptInterface
        public void openLens(String base64Jpeg) {
            try {
                File dir = PhotoProvider.shareDir(MainActivity.this);
                if (!dir.isDirectory() && !dir.mkdirs()) throw new IOException("no cache folder");
                File f = new File(dir, "leaf.jpg");
                try (FileOutputStream out = new FileOutputStream(f)) {
                    out.write(Base64.decode(base64Jpeg, Base64.DEFAULT));
                }
                Uri uri = Uri.parse("content://" + PhotoProvider.AUTHORITY + "/leaf.jpg");
                Intent send = new Intent(Intent.ACTION_SEND);
                send.setType("image/jpeg");
                send.putExtra(Intent.EXTRA_STREAM, uri);
                send.setClipData(ClipData.newRawUri("leaf", uri));
                send.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
                runOnUiThread(() -> {
                    for (String pkg : new String[]{"com.google.ar.lens", "com.google.android.googlequicksearchbox"}) {
                        Intent lens = new Intent(send).setPackage(pkg);
                        if (lens.resolveActivity(getPackageManager()) != null) {
                            try {
                                startActivity(lens);
                                return;
                            } catch (Exception ignored) {
                                // try the next one
                            }
                        }
                    }
                    startActivity(Intent.createChooser(send, null));
                });
            } catch (Exception e) {
                Log.w(TAG, "Google Lens: " + e);
            }
        }

        @JavascriptInterface
        public boolean hasVoice(String lang) {
            return voiceFor(lang) != null;
        }

        @JavascriptInterface
        public void stopSpeaking() {
            if (tts != null) tts.stop();
        }
    }
}
