# AgriSmart Architecture & Design Decisions

This document records key architectural and design choices made during the development of the AgriSmart Android app, following section 1 and section 15 of `docs/android_app_prompt.md`.

---

## 1. Project Layout & Directory Structure
- **Decision:** Keep `app/` as the standard primary Android application module (`com.agrismart.app`).
- **Rationale:** standard Android project convention. Existing standalone Python camera scripts (`live_camera.py`, `live_camera.html`) are relocated to `server/live_camera/`.
- **Root Files:** `.env` and `.env.example` placed at repository root, read directly by `app/build.gradle.kts` into `BuildConfig.API_BASE_URL`.

## 2. Min SDK Level & Locale Support
- **Decision:** `minSdk = 24`, `targetSdk = 35`, `compileSdk = 35`.
- **Rationale:** `minSdk 24` (Android 7.0) is required for `b+` BCP-47 resource qualifiers used by languages such as Maithili (`values-b+mai`), Santali (`values-b+sat`), and Kashmiri (`values-b+ks`).

## 3. Dependency Injection & UI Framework
- **Decision:** Hilt (`2.51.1`+) for DI, Jetpack Compose with BOM for UI, Material 3 with disabled dynamic color (to guarantee consistent high-contrast accessibility branding).
- **Rationale:** Adheres strictly to Section 4 tech stack recommendations.

## 4. Fake API Fallback in Debug
- **Decision:** When `API_BASE_URL` is empty, `FakeAgriSmartApi` serves mock fixtures located in `app/src/main/assets/mock/` after a 1–2s artificial delay.
- **Rationale:** Ensures every screen, state, and flow can be executed end-to-end without an active server connection during development and review.

## 5. Local On-Device Model Download & Offline Inference
- **Decision:** On first launch after language selection, the app shows a dedicated `ModelDownloadScreen` that downloads the 86.6 MB plant disease classification model (`model.pt`) directly from `https://github.com/wpzvqrs8/SIH_2026/releases/download/model-v2/model.pt` to the device local private storage (`context.filesDir/model/model.pt`).
- **Verification & Status:** The download displays a real-time progress bar (0–100%), bytes/MB downloaded, and verifies SHA-256 checksum `361daa8f299733046ec8c241107cfa3e9433737dc3abf7ed353c7cb97abf35cc`.
- **Inference Mode:** The app uses local model inference on device using the downloaded model file, providing full offline capability without requiring a remote backend server.

