plugins {
    id("com.android.application")
}

// The APK is a thin shell around the same offline web app that run.bat serves: offline/web is copied into
// the APK's assets on every build (models, voice clips, onnxruntime-web and the page itself).
val webDir = rootProject.file("../web")
val syncWeb by tasks.registering(Sync::class) {
    from(webDir)
    into(layout.projectDirectory.dir("src/main/assets/web"))
    doFirst {
        require(File(webDir, "models/india_v1.onnx").exists()) {
            "offline/web/models has no models yet: run offline/tools/export_onnx.py (or tools/fetch_assets.ps1) first"
        }
    }
}
tasks.named("preBuild") { dependsOn(syncWeb) }

android {
    namespace = "com.agrismart.offline"
    compileSdk = 37

    defaultConfig {
        applicationId = "com.agrismart.offline"
        minSdk = 24
        // 34 keeps the classic window (no forced edge-to-edge drawing under the status bar on Android 15+)
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    androidResources {
        // stored as-is so the WebView can stream them straight from the APK
        noCompress += listOf("onnx", "ogg", "wasm")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // signed with the local debug key so the APK can be installed directly (not for Play upload)
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
