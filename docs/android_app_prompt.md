# AgriSmart Android App: Build Prompt

> Give this whole document to your AI coding assistant, or to an Android developer. It describes a complete,
> production-quality Android app. Build it in the phases in section 15 and stop for review after each phase.
> When something here is ambiguous, pick the simplest option that satisfies section 2, and record the choice in
> `DECISIONS.md` at the project root.

---

## 1. Your role and the goal

You are a senior Android engineer and product designer. Build **AgriSmart**, a Kotlin + Jetpack Compose Android app
that lets any farmer check a plant leaf for disease with a phone photo, in their own language, and keeps a history of
past checks on the phone.

The app is a thin, careful client. The photo goes to the AgriSmart backend API, which runs the disease model and
returns the result and advice. **The API address is not known yet.** It comes from an `.env` file (section 5). Until
it is set, the app must still run end to end on built-in sample data.

Background: this is the mobile front end of *AgriSmart AI* (SIH 2026 hackathon). The model behind the API classifies
leaf photos of 13 crops into 28 classes (appendix A). That list will change, so the app loads it from the API and keeps
a bundled fallback.

---

## 2. Who uses it, and what that demands

**Users:** small and marginal farmers across India. Many are elderly, some read little or not at all, most use
low-cost phones (2–3 GB RAM, Android 8 or newer), with patchy mobile data, in bright outdoor light. Young users and
agriculture officers use it too, so it must look modern and precise, not childish.

**Design principles.** Every screen must pass all of these:

1. **One clear action per screen.** One big primary button. Secondary actions are visibly smaller.
2. **Icons always come with words.** No icon-only buttons except the top-bar menu and back arrow, and those have
   spoken labels. Never use colour as the only signal.
3. **Big and readable.** Body text 18 sp, touch targets at least 56 dp (primary buttons 64 dp). Everything works at
   200% system font size.
4. **The user's language everywhere,** including spoken results and error messages.
5. **Forgiving.** Back always works. Nothing is deleted without confirmation. Every error says what to do next.
6. **Light on the phone.** Small download, low memory, little storage. If there is no internet, photos are saved and
   sent later.
7. **Honest.** Say "not sure" rather than guess. Always make clear that results are a screening aid.

**Hard rules:**

- **No emoji anywhere**: not in UI text, code, notifications or store listing. Use icons (section 7.4).
- **Few colours** (section 7.1). Minimal, flat, sharp.
- **Navigation through a hamburger menu** (navigation drawer). Home also shows the two main actions as big buttons,
  because elderly users often miss menus.
- **No sign-up and no login.**

---

## 3. Scope

**In this build:**

- First launch: language choice, then a guided tour on the real screens with a practice check.
- Check a plant: choose the plant from the supported list or "Others", then take a photo or pick one from the phone.
  Quality check, send to the API, show the result, save it.
- Recent tests: a local history with thumbnails, detail view and delete.
- 16 languages (section 8), voice read-out (section 9), offline queue (section 11.5).
- Settings, help, about.
- Mock mode when no API address is set.

**Out of scope for now** (leave clean seams, not code): weather advice, sustainability score, chat assistant,
on-device model, accounts, cloud sync, push notifications.

---

## 4. Tech stack and project setup

| Area | Choice |
|---|---|
| Language / build | Kotlin (latest stable 2.x), Android Gradle Plugin (latest stable), Gradle version catalog `gradle/libs.versions.toml` |
| SDK levels | `minSdk 24` (Android 7.0: needed for the BCP-47 `b+` resource folders used by Maithili, Santali and Kashmiri). `compileSdk` / `targetSdk` = latest stable level that Google Play requires |
| UI | Jetpack Compose (BOM), Material 3, Navigation Compose, `material3-window-size-class` |
| Architecture | Single activity (`AppCompatActivity`, for per-app languages), MVVM with unidirectional data flow, coroutines + Flow |
| Dependency injection | Hilt |
| Network | Retrofit + OkHttp + kotlinx.serialization |
| Local data | Room (test history), DataStore Preferences (settings), app-private files (images) |
| Background work | WorkManager (send pending tests when online) |
| Camera | CameraX: preview, image capture, image analysis |
| Pick from phone | Photo Picker (`ActivityResultContracts.PickVisualMedia`). No storage permission |
| Images | Coil 3 for Compose, with bounded caches |
| Speech | Android `TextToSpeech` |
| Tests | JUnit, Turbine, MockWebServer, Room testing, Compose UI tests; screenshot tests with Roborazzi or Paparazzi |

**Do not add:** Firebase or any analytics SDK, Lottie, `material-icons-extended` (several MB; import only the icons
you use), bundled custom fonts (section 8.3), or any dependency over 1 MB without a written reason in `DECISIONS.md`.

- **Package:** `com.agrismart.app` (placeholder).
- **App name:** "AgriSmart" in every language (it's a brand), with a translated one-line subtitle
  ("Check your plant's health").

---

## 5. Configuration: backend address from `.env`

Two files at the repository root:

`.env`, which git ignores:

```
# AgriSmart backend. Leave empty until the server is ready.
API_BASE_URL=
```

`.env.example`, committed, with the same content.

In `app/build.gradle.kts`:

```kotlin
import java.util.Properties

val env = Properties().apply {
    rootProject.file(".env").takeIf { it.exists() }?.inputStream()?.use { load(it) }
}

android {
    buildFeatures { buildConfig = true }
    defaultConfig {
        buildConfigField("String", "API_BASE_URL", "\"${env.getProperty("API_BASE_URL", "").trim()}\"")
    }
}
```

**Rules:**

- `ApiConfig` adds a trailing `/` if missing, and exposes `isConfigured = API_BASE_URL.isNotBlank()`.
- **Empty address, debug build:** use `FakeAgriSmartApi`, which serves the fixtures in `assets/mock/` (appendix C)
  after a 1–2 s delay, so every screen and result state can be seen and tested. Show a thin banner:
  "Sample results: no server set".
- **Empty address, release build:** the check flow shows "This service is not set up yet". The practice check and
  Recent tests still work.
- **HTTPS only in release** (network security config with cleartext disabled). Debug builds may allow cleartext to
  `10.0.2.2` and `localhost`, for a backend running on a developer's laptop.
- The address ships inside the APK, so it is **not a secret**. Never put private keys in `.env` or BuildConfig.
  Protect the API on the server side (rate limits).

---

## 6. User flows and screens

### 6.1 First launch

1. **Splash** (Android SplashScreen API): app icon on the background colour. No artificial delay.
2. **Choose your language** (section 8.1). Tapping a tile switches the whole UI to that language at once, so the user
   sees it working. A big **Continue** button moves on. Nothing else on this screen.
3. **Guided tour ("How AgriSmart works").** It runs on the **real Home screen**, not on slides. Pop-ups (coach marks)
   dim the screen, cut a spotlight around the real control, and explain it in one or two short sentences:

   | Step | Spotlight on | Text (English source) |
   |---|---|---|
   | 1 | Big **Check a plant** button | "Tap here to check a plant." |
   | 2 | Plant grid (the tour opens *Choose your plant*) | "Pick your plant. Not in the list? Choose Others." |
   | 3 | **Take photo** / **Choose from phone** | "Take a photo of one leaf, or choose a photo from your phone." |
   | 4 | Result card (shown with the practice result) | "Here is the result. Tap Listen to hear it." |
   | 5 | Recent tests on Home | "Your past checks are saved here." |
   | 6 | Menu button | "Open the menu for language, help and settings." |

   - **Each pop-up has:** an icon; a step counter ("2 of 6"); **Back**, **Next** (primary) and **Skip**.
   - **It reacts in real time:** tapping the highlighted control does the real action and moves the tour on.
   - **Voice:** it reads each step aloud when a voice is available for the chosen language. There's a speaker button
     to repeat.
   - **Accessibility:** TalkBack focuses the pop-up and announces what the spotlight is on.
   - **The last step offers two buttons:** **Practice now** runs the whole check with a bundled sample leaf photo and a
     bundled result, needing no network, and every screen is labelled "Practice". **Start using AgriSmart** goes Home.
   - **Replay and versioning:** the tour can be replayed from the menu under **How to use**. Store `tour_version_seen`,
     so a later change to the tour shows again once.
4. **Camera permission** is requested only the first time the user taps **Take photo**, after a one-screen explanation
   (section 13).

### 6.2 Check a plant (the core flow)

`Home → Choose your plant → Add a photo → (Camera | Photo Picker) → Photo preview → Checking → Result`. The result is
saved to Recent tests automatically.

1. **Choose your plant**
   - A grid of crop tiles, each with a crop icon and the translated name, from `GET /v1/crops` (cached) or the bundled
     fallback.
   - The last used crop comes first, then alphabetical order in the current language. **Others** ("Not in the list")
     is always last.
   - **Limited crops get a note on the tile** (appendix A):
     - healthy only: "Can only confirm a healthy leaf";
     - diseases only: "Can't confirm a healthy leaf".
     Selecting one of these shows a one-line explanation before the photo step.
   - A search field appears only when there are more than 16 crops.
2. **Add a photo:** two big buttons: **Take photo** (camera icon) and **Choose from phone** (photo library icon), plus
   a small tip line: "One leaf, in daylight, filling the frame".
3. **Camera**
   - A full-width preview with a square guide frame.
   - **Live hints** come from on-device analysis of small preview frames 2–3 times a second, using only the luminance
     plane so no bitmaps are made:
     - mean brightness too low: "Too dark: move into daylight";
     - too high with low contrast: "Too much glare";
     - low sharpness (variance of the Laplacian): "Hold still".
     Tune the thresholds on real phones.
   - A 72 dp shutter button labelled **Take photo**, and a flash toggle. The camera-switch button appears only if the
     phone has two cameras.
   - After capture: a still preview with **Use this photo** (primary) and **Retake**.
4. **Choose from phone:** the Photo Picker (images only, one photo), then the same preview screen.
5. **Photo check** (instant, on device): if the photo is too dark or blurry, show a friendly warning with
   **Take another** (primary) and **Use anyway**.
6. **Checking:** a progress indicator, "Checking your plant…", and a Cancel button. Time out after 30 s and show the
   error state.
7. **No internet at send time:** save the test as *Pending*. Say "No internet. Your photo is saved. We will check it
   when you are online." Go to Recent tests, where the item shows a *Waiting for internet* status. WorkManager sends it
   later (section 11.5).

### 6.3 Result screen

**Always shown:**
- the photo;
- the plant's name;
- a large title;
- a status chip (icon + word, never colour alone);
- "How sure", in plain words;
- a **Listen** button;
- a short "What to do" list;
- a **Check another plant** button;
- a footer: "This is a screening aid, not a diagnosis. Confirm with your local agriculture officer or Krishi Vigyan
  Kendra." with a **Call Kisan Call Centre** button, which dials 1800-180-1551 through the dial screen, so no call
  permission is needed.

"How sure" comes from the API's `confidence_level`:

| Level | Words |
|---|---|
| `high` | "Very likely" |
| `medium` | "Likely" |
| `low` | Never shown as a result: becomes the *Not sure* state |

The percentage appears only in small text on a "Details" line.

| State | Icon | Colour role | Title (English source) | Main actions |
|---|---|---|---|---|
| Disease | `warning` | attention (amber) | Disease name, e.g. "Late blight" | Listen · What to do · Check another plant |
| Healthy | `check_circle` | primary (green) | "Your tomato leaf looks healthy" | Listen · Tips to keep it healthy · Check another plant |
| Not sure | `help` | neutral | "We are not sure" | **Try another photo** (primary), with photo tips |
| Plant not supported | `block` | neutral | "We can't check this plant yet" | Show supported plants · Check another plant |
| Plant looks different | `swap_horiz` | attention | "This looks like potato, not tomato" | **Yes, it's potato** (re-check as potato) · **No, keep tomato** |
| Waiting for internet | `cloud_off` | neutral | "Saved. We will check it when you are online" | Go to Recent tests |
| Error | `error` | error (red) | Plain cause, e.g. "The server is busy" | **Try again** · Back |

- **Crop-coverage caveats.** When the chosen crop has limited coverage (appendix A), add the API's `coverage_note`
  under the title. For example: "For corn, AgriSmart can name 3 diseases but cannot yet confirm a healthy leaf."
- **Others.** A result for **Others** says "Best match among supported plants". If confidence is low, it shows
  *Plant not supported*.

### 6.4 Recent tests

- **Home** shows the latest 3 tests as cards, with **See all**.
- **The Recent tests screen:**
  - a paged list, newest first;
  - each row: a 64 dp thumbnail, the plant, the result word with its status icon, and a relative time ("Today, 10:30");
  - a status icon for *Waiting for internet*, *Sending* and *Failed*;
  - **Failed** rows show **Try again**.
- **Tapping a row** opens the same Result screen, read-only, with **Listen**, **Delete** (with confirmation) and
  **Share** (the system share sheet: text plus image).
- **Names follow the current language.** History stores ids and renders names in the current language, falling back
  to the saved API text.
- **An empty state:** icon + "No tests yet" + **Check a plant**.
- **No swipe-only actions.** Every action also has a visible button.

### 6.5 Menu (navigation drawer)

The top app bar has the hamburger (`menu`) on the start side. The drawer has an icon and a label for each item, and
highlights the current one:

1. Check a plant (`eco`)
2. Recent tests (`history`)
3. Language (`translate`), which shows the current language in its own script
4. How to use (`help`): replays the tour
5. Settings (`settings`)
6. Help and about (`info`)

At the bottom: the app version, and the model version reported by the API.

- **On wide screens** (section 7.7) the drawer is permanent.

### 6.6 Settings

- **Language:** opens the same picker as first launch.
- **Read results aloud:** on/off (default on). **Voice speed:** normal or slow.
- **Text size inside the app:** Normal / Large / Extra large. This multiplies on top of the system font size.
- **Storage:** space used; **Free up space**, which removes large images older than 90 days but keeps results and
  thumbnails; **Delete all tests** (confirmation required).
- **Show the tour again.**

### 6.7 Help and about

- Short photo tips, with simple line illustrations.
- Supported plants and their coverage.
- The disclaimer.
- **Call Kisan Call Centre** (1800-180-1551, toll-free).
- Privacy summary (section 13).
- Licences: open-source libraries, and the credit for the practice photo (appendix C).
- The team.

---

## 7. Visual design system

**Minimal, flat and sharp:** generous white space, crisp 1 dp outlines instead of shadows, small corner radii, strong
type hierarchy. Turn off dynamic colour (Material You), so the brand colours and contrast are always the same.

### 7.1 Colour: only these

| Role | Light | Dark | Used for |
|---|---|---|---|
| Primary (leaf green) | `#1E6B3C` | `#7FD49A` | Primary buttons, healthy status, selection |
| On primary | `#FFFFFF` | `#00391C` | Text and icons on primary |
| Attention (amber) | `#9A4A08` | `#F2B26B` | Disease status, "plant looks different" |
| Error (red) | `#B3261E` | `#F2B8B5` | Errors only |
| Background | `#F7F8F3` | `#111411` | Screen background |
| Surface | `#FFFFFF` | `#1A1E1A` | Cards, sheets, the drawer |
| Text | `#1A1C19` | `#E2E3DD` | Main text |
| Muted text | `#555C52` | `#A9B0A5` | Secondary text |
| Outline | `#D5DAD0` | `#3A413A` | 1 dp borders and dividers |

- **Contrast:** main text at least 7:1 on its background, everything else at least 4.5:1. Verify both themes.
- **Tinted backgrounds:** status chips use a 12% tint of their colour behind the icon and label. No other tints.
- **Camera screen:** always dark (`#000000` preview background), with white controls.

### 7.2 Typography

Use the system font family. Android uses Noto for Indic scripts, and this keeps the app small.

| Style | Size / line height | Weight | Use |
|---|---|---|---|
| Display | 32 / 44 sp | Semibold | Result title |
| Headline | 26 / 36 sp | Semibold | Screen titles |
| Title | 22 / 30 sp | Medium | Card titles, crop names |
| Body | 18 / 28 sp | Regular | Default text |
| Label | 18 / 24 sp | Medium | Button text |
| Caption | 15 / 22 sp | Regular | Details line, timestamps (the smallest size allowed) |

- **Indic scripts need tall lines:** keep line height at least 1.4× the font size, and never put text in a fixed-height
  box. Check for clipped vowel signs in Hindi, Tamil, Malayalam and Odia.

### 7.3 Spacing, shape, elevation

- **Spacing:** an 8 dp grid. Screen padding is 20 dp on phones and 32 dp on tablets. Stack items 16 dp apart.
- **Corner radius:** 8 dp for buttons, cards and tiles; 12 dp for sheets and dialogs.
- **Elevation:** 0 everywhere, with 1 dp outlines instead. Only the camera shutter button and dialogs get a small
  shadow.

### 7.4 Icons

- **Style:** Material Symbols, *Rounded*, weight 400. Size 24 dp in lists, 28 dp in buttons, 48 dp for status.
- **Import each icon as a vector drawable** (Android Studio: Vector Asset). No icon font, and no icons-extended
  library.
- **Directional icons** (back, forward) set `autoMirrored` for right-to-left languages.

**Icons used:**
- `menu`, `arrow_back`, `close`
- `eco`, `history`, `translate`, `help`, `settings`, `info`
- `photo_camera`, `photo_library`, `flash_on`, `flash_off`, `cameraswitch`
- `check_circle`, `warning`, `error`, `block`, `swap_horiz`, `cloud_off`, `schedule`, `refresh`
- `volume_up`, `stop_circle`
- `delete`, `share`, `call`, `chevron_right`, `search`, `storage`, `text_fields`, `wb_sunny`, `back_hand`

**Crop icons:** 13 custom single-colour line illustrations on a 48 dp grid, with a 2 dp stroke in the Text colour,
matching Material Symbols' look. Each should be under 3 KB as a vector. **Others** uses `help`.

### 7.5 Components

| Component | Spec |
|---|---|
| Primary button | Filled primary, 64 dp tall, full width on phones, icon + label |
| Secondary button | 1 dp outline, 56 dp tall, icon + label |
| Text button | Only for tertiary actions such as Skip and Details |
| Crop tile | Min 120 dp square, 8 dp radius, 1 dp outline; 48 dp icon above the name; the selected tile has a 2 dp primary border and a check mark |
| Status chip | Icon + word on a 12% tint; never colour alone |
| Result card | Photo (4:3, fills the width), then chip, title, "How sure" line, then actions |
| Dialog | Title, one sentence, two buttons stacked vertically at full width; the destructive one in error colour |
| Coach mark | Scrim at 60% black; spotlight with an 8 dp radius and 8 dp padding; a pop-up card holding icon, text, counter and buttons |

### 7.6 Motion

- **Timing:** 150–250 ms fades and slides only.
- **Reduced motion:** when the system's animator duration scale is 0, remove all animation.
- **Haptics:** a light tap on capture and when a result appears, only if system haptics are on.

### 7.7 Responsive layout

Use window size classes:

| Width | Layout |
|---|---|
| Compact (< 600 dp): most phones | One column; modal drawer; crop grid of 2 columns, or 3 when the tiles still fit at 120 dp |
| Medium (600–840 dp): large phones in landscape, small tablets, foldables | Modal drawer; crop grid of 3–4 columns; result shows photo and text side by side |
| Expanded (> 840 dp): tablets | Permanent drawer; Recent tests as list plus detail in two panes; 4–6 column crop grid |

- **Every screen:** works in portrait and landscape, scrolls when content doesn't fit, and never cuts off essential
  text at 200% font size.

### 7.8 Accessibility checklist

- **Touch targets:** at least 56 dp. Primary actions 64 dp.
- **TalkBack:**
  - every control has a label in the current language, and icons are never announced as "button";
  - headings are marked as headings;
  - focus order matches the visual order;
  - result changes are announced.
- **Font scaling:** works with the system font size at 200% and display size at maximum.
- **Colour:** status is always icon + word; contrast as in section 7.1.
- **No time limits,** except the network timeout, which offers **Try again**.
- **Keyboard and switch access** can reach everything.
- **The camera screen** can be used without seeing the preview: the live hints are also announced, at most every
  3 seconds.

---

## 8. Languages

### 8.1 The 16 languages

The 15 most spoken Indian languages (first-language speakers, Census 2011), plus English.

**Language picker:**
- **Suggested first:** the phone's own language is pinned at the top when it's in the list.
- **Then this order:** Hindi, English, then the rest in the table's order.
- **Each tile** shows the language in its own script (large) and in English (small).
- **Each tile has a speaker button** that says the language's name. Use a bundled clip (Opus, about 8 KB each) if
  available, otherwise text-to-speech; hide the button if neither works.

| # | Language | Native name | Locale tag | Resource folder | Script | Direction | Speakers (approx.) |
|---|---|---|---|---|---|---|---|
| 1 | Hindi | हिन्दी | `hi` | `values-hi` | Devanagari | LTR | 528 M |
| 2 | English | English | `en` | `values` (default) | Latin | LTR | widely used |
| 3 | Bengali | বাংলা | `bn` | `values-bn` | Bengali | LTR | 97 M |
| 4 | Marathi | मराठी | `mr` | `values-mr` | Devanagari | LTR | 83 M |
| 5 | Telugu | తెలుగు | `te` | `values-te` | Telugu | LTR | 81 M |
| 6 | Tamil | தமிழ் | `ta` | `values-ta` | Tamil | LTR | 69 M |
| 7 | Gujarati | ગુજરાતી | `gu` | `values-gu` | Gujarati | LTR | 55 M |
| 8 | Urdu | اردو | `ur` | `values-ur` | Perso-Arabic | **RTL** | 51 M |
| 9 | Kannada | ಕನ್ನಡ | `kn` | `values-kn` | Kannada | LTR | 44 M |
| 10 | Odia | ଓଡ଼ିଆ | `or` | `values-or` | Odia | LTR | 38 M |
| 11 | Malayalam | മലയാളം | `ml` | `values-ml` | Malayalam | LTR | 35 M |
| 12 | Punjabi | ਪੰਜਾਬੀ | `pa` | `values-pa` | Gurmukhi | LTR | 33 M |
| 13 | Assamese | অসমীয়া | `as` | `values-as` | Bengali-Assamese | LTR | 15 M |
| 14 | Maithili | मैथिली | `mai` | `values-b+mai` | Devanagari | LTR | 14 M |
| 15 | Santali | ᱥᱟᱱᱛᱟᱲᱤ | `sat` | `values-b+sat` | Ol Chiki | LTR | 7 M |
| 16 | Kashmiri | کٲشُر | `ks` | `values-b+ks` | Perso-Arabic | **RTL** | 7 M |

### 8.2 Implementation

- **Switching language:** use per-app language preferences. Call
  `AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags(tag))`. The activity must extend
  `AppCompatActivity`.
- **Supporting older Android:**
  - on Android below 13, add the AppCompat `AppLocalesMetadataHolderService` with `autoStoreLocales=true` in the
    manifest;
  - on Android 13 and newer, declare `android:localeConfig="@xml/locales_config"`, listing all 16 tags, so the system
    language settings stay in sync.
- **Remembering the choice:** store the tag in DataStore too, so the choice survives reinstall-restore and the
  language screen knows what is selected.
- **Trimming library translations:** keep only these 16 locales from libraries (`androidResources.localeFilters`, or
  `resourceConfigurations` on older plugin versions). This drops unused translations.
- **App Bundle:** turn **language splits off** (`bundle { language { enableSplit = false } }`). Users pick languages
  other than the phone's, so every translation must be inside the app. Strings are small.

### 8.3 Scripts and fonts

- **Rely on system fonts.** On the oldest supported Android version, check that every script renders, especially
  Ol Chiki (Santali), Odia and the Perso-Arabic scripts.
- **If a script is missing on some phones,** use Downloadable Fonts (Google Fonts provider, Noto family) for that
  script only. Never bundle a full font set.

### 8.4 Right-to-left (Urdu, Kashmiri)

- **Manifest:** `android:supportsRtl="true"`.
- **Layout:** use start/end, never left/right. Directional icons are mirrored.
- **Test:** check every screen in Urdu.
- **Mixed text:** numbers and Latin words inside RTL text must read correctly, so wrap user or API text with
  `BidiFormatter` where they mix.

### 8.5 Writing and translation rules

- **Source copy** in `values/strings.xml` is plain English: short sentences (aim for 8 words or fewer), everyday
  words, and verbs on buttons ("Take photo", "Try again"). No jargon: say "How sure", not "confidence".
- **No string concatenation.** Use placeholders (`%1$s`) and `plurals`. Never build sentences in code.
- **Translation:** native speakers review every language. Machine translation is allowed only as a draft, marked for
  review, and tracked in `TRANSLATIONS.md` (language, translator, status).
- **Crop and disease names:** these are translated in the bundled strings (keys in appendix A), so history and offline
  screens work. The API also returns translated names and advice for the language in `Accept-Language`.
- **Fallback order:** API text in the chosen language, then bundled translation, then English.
- **Numbers and dates:** use locale-aware formatting. Relative dates ("Today", "Yesterday") in the chosen language.

---

## 9. Voice (text-to-speech)

- **Engine setup:** create one `TextToSpeech` for the whole app and set it to the chosen language.
  - Check `isLanguageAvailable` for each language.
  - **If the voice is missing:** offer once to install it (`TextToSpeech.Engine.ACTION_INSTALL_TTS_DATA`). Otherwise
    fall back to Hindi, then English, with a one-time notice ("Voice not available in Santali; reading in Hindi").
  - **If there is no engine at all,** hide the Listen buttons.
- **Where it speaks:**
  - the result, automatically, when "Read results aloud" is on;
  - the **Listen** button, which replays;
  - tour steps;
  - camera hints for TalkBack users.
- **How it speaks:** short sentences; speech rate 0.9 by default (0.75 for "slow"). Stop speaking when the user leaves
  the screen.
- **What it reads:** the status and title, then "How sure", then the first two advice steps. Never the percentage.

---

## 10. Backend API contract (v1)

- **Base address:** `API_BASE_URL`. HTTPS, JSON in UTF-8.
- **Every request sends these headers:**
  - `Accept-Language: <tag>`
  - `X-App-Version: <versionName>`
  - `X-Install-Id: <random UUID made on first launch>`, for rate limiting only; not personal data.
- **The backend team implements this contract.** The app's `FakeAgriSmartApi` follows it exactly.

### 10.1 `GET v1/health`

```json
{ "status": "ok", "model_version": "model-v2" }
```

### 10.2 `GET v1/crops?lang=gu`

```json
{
  "version": "2026-09-11",
  "crops": [
    {
      "id": "tomato",
      "name": "ટામેટાં",
      "coverage": "full",
      "coverage_note": null,
      "classes": ["tomato_bacterial_spot", "tomato_early_blight", "tomato_healthy"]
    },
    {
      "id": "corn",
      "name": "મકાઈ",
      "coverage": "diseases_only",
      "coverage_note": "Can name 3 corn diseases; cannot confirm a healthy leaf yet.",
      "classes": ["corn_gray_leaf_spot", "corn_common_rust", "corn_northern_leaf_blight"]
    }
  ]
}
```

- **`coverage` values:**
  - `full`: has diseases and a healthy class;
  - `healthy_only`: only a healthy class, so it can't name diseases;
  - `diseases_only`: no healthy class, so it can't confirm a healthy leaf.
- **Caching:** keep the list for 24 hours or until `version` changes. The bundled fallback is
  `assets/crops_fallback.json`, holding appendix A.

### 10.3 `POST v1/predict` (multipart/form-data)

| Part | Type | Notes |
|---|---|---|
| `image` | file, `image/jpeg` | Longest side at most 1024 px, quality 85, EXIF stripped (section 12) |
| `crop` | text | A crop id from `/v1/crops`, or `other` |
| `lang` | text | Locale tag, e.g. `ta` |
| `client_test_id` | text | UUID made on the device; the server treats repeats as the same request (idempotent retries) |

Response `200`:

```json
{
  "test_id": "srv_8f2c1e",
  "client_test_id": "3b0d1f7e-6a0c-4a4f-9d7e-0c1b2a3d4e5f",
  "model_version": "model-v2",
  "status": "disease",
  "crop": { "id": "tomato", "name": "Tomato" },
  "crop_check": { "matches_selection": true, "detected_crop": { "id": "tomato", "name": "Tomato" } },
  "disease": { "id": "tomato_late_blight", "name": "Late blight", "label": "Tomato___Late_blight" },
  "confidence": 0.82,
  "confidence_level": "high",
  "alternatives": [
    { "id": "tomato_early_blight", "name": "Early blight", "confidence": 0.07 }
  ],
  "coverage_note": null,
  "advice": {
    "summary": "A fungus-like disease that spreads fast in cool, wet weather.",
    "steps": [
      "Remove and destroy the affected leaves.",
      "Water at the base of the plant, not on the leaves.",
      "Ask your local agriculture officer which spray to use."
    ],
    "prevention": ["Leave space between plants so leaves dry quickly."],
    "source": "Reviewed extension guidance"
  },
  "quality": { "warnings": [] }
}
```

- **`status` values:**
  - `disease`, `healthy`, `uncertain`;
  - `unsupported`: for `other` when nothing matches well;
  - `crop_mismatch`: `crop_check.matches_selection` is false and the detected crop is confident.
- **`confidence_level`:**

  | Value | Confidence |
  |---|---|
  | `high` | ≥ 0.80 |
  | `medium` | 0.55–0.80 |
  | `low` | < 0.55 |

  The app treats `low` as `uncertain`.

**Guidance for the backend team** (so the app's result states make sense):
- use the crop hint to detect a mismatch;
- restrict predictions to the chosen crop's classes **only when that crop has `full` coverage**;
- for `healthy_only` and `diseases_only` crops, never return a confident answer the model cannot actually tell apart,
  and fill in `coverage_note`.

### 10.4 Errors

```json
{ "error": { "code": "IMAGE_UNREADABLE", "message": "We could not read this photo. Please take another." } }
```

| HTTP status | `code` | What the app shows |
|---|---|---|
| 400 | `IMAGE_UNREADABLE`, `UNSUPPORTED_MEDIA`, `IMAGE_TOO_SMALL` | Photo problem: **Take another** |
| 413 | `IMAGE_TOO_LARGE` | Shouldn't happen (the app compresses); recompress smaller and retry once |
| 429 | `RATE_LIMITED` | "Too many checks. Please wait a minute." |
| 503 | `SERVER_BUSY` | "The server is busy." **Try again**, and queue the test |
| 5xx | `INTERNAL` | "Something went wrong on our side." **Try again** |

The `message` comes already translated by the server. The app shows its own bundled text if `message` is missing.

### 10.5 Client behaviour

- **OkHttp timeouts:** connect 10 s, read 30 s, write 30 s.
- **Retries:** GET requests retry automatically up to 2 times with backoff. `POST predict` retries only through the
  offline queue, using the same `client_test_id`.
- **Compression and caching:** gzip responses; a 5 MB OkHttp cache, used for `/crops` only.
- **Logging:** logging interceptor in debug builds only, and never logs image bytes.

---

## 11. Data, storage and offline

### 11.1 Room

```kotlin
@Entity(tableName = "tests", indices = [Index("createdAt")])
data class TestEntity(
    @PrimaryKey val id: String,          // client_test_id (UUID)
    val createdAt: Long,
    val cropId: String,                  // "other" allowed
    val lang: String,                    // language at the time of the test
    val sendStatus: SendStatus,          // PENDING, SENDING, DONE, FAILED
    val resultStatus: ResultStatus?,     // DISEASE, HEALTHY, UNCERTAIN, UNSUPPORTED, CROP_MISMATCH
    val diseaseId: String?,
    val confidence: Float?,
    val confidenceLevel: String?,
    val resultJson: String?,             // full API response, so the result can be shown again offline
    val thumbPath: String,               // 256 px WebP
    val displayPath: String?,            // 720 px WebP; null after "Free up space"
    val uploadPath: String?,             // 1024 px JPEG; deleted once sent
    val serverTestId: String?,
    val modelVersion: String?,
    val errorCode: String?,
    val attempts: Int = 0,
    val isPractice: Boolean = false      // practice runs are never saved to history
)
```

- **DAO:** `pagingSource()` ordered by `createdAt DESC`; `latest(3)`; `pending()`; `byId(id)`; `delete(id)`;
  `deleteAll()`; `sizeStats()`.
- **Schema changes:** export the schema, and write real migrations. `fallbackToDestructiveMigration` is allowed only
  in debug builds.

### 11.2 DataStore keys

| Key | Holds |
|---|---|
| `language_tag` | The chosen language |
| `onboarding_done` | Whether first launch has finished |
| `tour_version_seen` | Which tour version the user has seen |
| `read_aloud` | Read results aloud on/off |
| `voice_speed` | Normal or slow |
| `app_text_scale` | In-app text size: 1.0, 1.15 or 1.3 |
| `last_crop_id` | The crop to show first |
| `install_id` | Random UUID for the `X-Install-Id` header |
| `crops_cache_json`, `crops_cache_version`, `crops_cache_time` | The cached crop list |
| `camera_primer_shown` | Whether the camera explanation has been shown |
| `privacy_notice_accepted` | Whether the privacy notice has been accepted |

### 11.3 Files

All in app-private storage. No storage permission; deleted when the app is uninstalled.

```
filesDir/tests/<id>/thumb.webp      256 px, quality 70     (~10 KB)
filesDir/tests/<id>/display.webp    720 px, quality 80     (~60–100 KB)
cacheDir/upload/<id>.jpg            1024 px, quality 85    (~150–250 KB, deleted once sent)
cacheDir/camera/                    raw captures, deleted as soon as they are processed
```

### 11.4 Retention

- **Keep the 100 most recent tests.** Older ones are deleted with their files.
- **After 90 days,** `display.webp` is removed; the thumbnail and result stay. The same happens early if app storage
  goes over 50 MB.
- **When the phone is low on storage** (`StorageManager` or a low-storage broadcast), prune first and tell the user in
  Settings.
- **At startup, in the background,** delete orphaned files that no row points to.

### 11.5 Offline queue

- **Sending:** `SendTestWorker(testId)` runs as unique work per test, needs a network connection, and uses
  exponential backoff starting at 30 s.
- **Status flow:** PENDING, then SENDING, then DONE or FAILED. After 5 attempts it becomes FAILED, which shows
  **Try again**.
- **When a result arrives in the background,** update the row. If the user is in the app, show an in-app banner:
  "Your tomato result is ready". No push notifications in this build.

---

## 12. Space and memory budgets

| Budget | Target |
|---|---|
| Download size (App Bundle, typical phone) | ≤ 8 MB |
| Universal APK | ≤ 15 MB |
| Cold start to the first usable screen | ≤ 1.5 s on a mid-range phone; ≤ 3 s on a 2 GB phone |
| Memory, typical use | ≤ 120 MB PSS |
| Memory, camera screen | ≤ 180 MB PSS |
| Image pipeline peak (extra) | ≤ 25 MB |
| Stored data | about 10–15 MB for 100 tests; hard cap 50 MB for display images |
| Data used per check | about 150–250 KB upload, < 5 KB response |

**How to meet them:**

- **Build:**
  - R8 in full mode with `isMinifyEnabled` and `isShrinkResources`;
  - vector drawables for all icons and crop art;
  - WebP for the few raster images;
  - locale filters (section 8.2);
  - no icons-extended, Lottie or Firebase;
  - a baseline profile for faster start-up.
- **Image pipeline** (on `Dispatchers.Default`, one bitmap alive at a time):
  1. Decode with a target size (`ImageDecoder` with `setTargetSize` on API 28+, otherwise `BitmapFactory` with
     `inSampleSize`), to at most 1600 px on the longest side. Apply the EXIF rotation.
  2. Run the quality checks on a 256 px copy: mean brightness and variance of the Laplacian.
  3. Write the upload JPEG at 1024 px, quality 85, with **EXIF removed** (no location leaks).
  4. Write the `display.webp` (720 px) and `thumb.webp` (256 px) files.
  5. Recycle the bitmaps. Never keep a full-resolution bitmap in memory or in UI state; pass file paths.
- **Camera:**
  - capture resolution at most 1920×1440;
  - image analysis at 320×240 using `STRATEGY_KEEP_ONLY_LATEST`, reading only the luminance plane, throttled to
    3 fps;
  - unbind the camera as soon as the screen leaves.
- **Coil:** memory cache at most 15% of app memory; disk cache 32 MB. Always request thumbnails at the size they are
  displayed.
- **Lists:** `LazyColumn` with stable keys and Paging 3 for history. No large objects in `rememberSaveable`.
- **Leaks:** add LeakCanary to debug builds only. Zero known leaks before release.
- **Network:** compressed uploads, gzip, and no polling. WorkManager runs only when there is queued work.

---

## 13. Privacy, permissions and safety

- **Permissions:**
  - `CAMERA`: asked when first needed, after an explanation screen ("AgriSmart needs the camera to photograph your
    plant.");
  - `INTERNET` and `ACCESS_NETWORK_STATE`.
  - **No** storage permission (the Photo Picker doesn't need one), **no** location, **no** contacts.
  - **If the camera is refused:** offer **Choose from phone**, and explain how to allow it in Settings.
- **Privacy notice** (once, before the first check, in the chosen language): "Your photo is sent to AgriSmart only to
  check your plant. We do not ask for your name or location." Link to the full privacy text in Help.
  - Keep data to the minimum, in line with India's Digital Personal Data Protection Act, 2023.
- **Photo data:** EXIF (GPS, device) is removed before upload. Photos stay in app-private storage.
- **Safety:**
  - every result carries the screening-aid disclaimer and the Kisan Call Centre button;
  - the app never shows pesticide names or doses unless they come from the API as reviewed advice.
- **Practice mode** is clearly labelled and never saved as a real test.

---

## 14. Architecture and code structure

```
com.agrismart.app
├─ AgriSmartApp.kt                 Hilt application, WorkManager config, TTS warm-up
├─ MainActivity.kt                 AppCompatActivity + Compose host, window size class
├─ core/
│  ├─ designsystem/                theme (colours, type, shapes), components (buttons, tiles, chips, coach mark)
│  ├─ i18n/                        Language list, LanguageManager (AppCompat locales + DataStore)
│  ├─ tts/                         SpeechManager
│  ├─ image/                       ImagePipeline (decode, quality check, compress, thumbnails)
│  └─ util/                        Result types, dispatchers, connectivity
├─ data/
│  ├─ api/                         AgriSmartApi (Retrofit), DTOs, FakeAgriSmartApi, ApiConfig
│  ├─ db/                          Room database, TestEntity, TestDao, converters
│  ├─ prefs/                       SettingsRepository (DataStore)
│  ├─ repo/                        CropRepository, TestRepository, PredictionRepository
│  └─ work/                        SendTestWorker
├─ domain/model/                   Crop, Coverage, TestRecord, Prediction, ResultStatus
└─ feature/
   ├─ onboarding/                  LanguageScreen, TourController + overlay, PracticeFlow
   ├─ home/                        HomeScreen
   ├─ scan/                        ChoosePlant, AddPhoto, Camera, PhotoPreview, Checking, Result
   ├─ history/                     RecentTests, TestDetail
   ├─ settings/                    Settings
   └─ help/                        HelpAbout
```

- **ViewModels** expose a single `StateFlow<UiState>` (immutable data classes) and take events as function calls. The
  UI never touches repositories directly.
- **The tour** (`TourController`): screens register targets with `Modifier.tourTarget(key)`, which records the target's
  bounds through `onGloballyPositioned`. The overlay draws the scrim with a spotlight cut-out and places the pop-up
  card near the target. Tour state lives in `TourController`, not in each screen.
- **Navigation routes:** `language`, `home`, `choose_plant`, `add_photo`, `camera`, `preview`, `checking`,
  `result/{testId}`, `history`, `settings`, `help`. They are type-safe routes, so no raw strings are passed around.
- **Errors:** one sealed `AppError` type (NoInternet, Timeout, Server(code), PhotoProblem(reason), NotConfigured,
  Unknown), mapped to translated messages in one place.
- **Code quality:**
  - no hard-coded user-visible text (lint rule `HardcodedText` as an error);
  - ktlint plus detekt;
  - each screen has Compose previews in English, Hindi and Urdu, light and dark, at font scale 1.0 and 2.0.

---

## 15. Build phases and "done when"

| Phase | Build | Done when |
|---|---|---|
| 1. Skeleton | Project, version catalog, Hilt, theme, typography, drawer navigation, empty screens, `.env` to BuildConfig, `FakeAgriSmartApi` | App runs; drawer works; the theme passes contrast checks; the address is read from `.env` |
| 2. Languages | All 16 locale folders (English final, others as marked drafts), language screen, per-app locale switching, RTL | Switching language updates every screen instantly and survives a restart; Urdu mirrors correctly; no hard-coded text |
| 3. Check a plant | Choose plant (API + fallback + coverage notes), Photo Picker, CameraX with live hints, image pipeline, predict call, every result state | With the mock API, all 7 result states can be reached; image pipeline memory peak ≤ 25 MB |
| 4. History and storage | Room, Recent tests, detail, delete, retention and pruning, Home recent cards | 100+ tests scroll smoothly; storage stays within budget; deleting removes the files |
| 5. Offline, voice, tour | WorkManager queue, TTS with fallback, guided tour with practice run, privacy notice, camera explanation | Airplane-mode test: a queued test is sent automatically after reconnecting; the tour reacts to real taps; TalkBack works through the whole flow |
| 6. Polish and QA | Tablet layouts, baseline profile, R8, size and memory measurement, screenshot tests, section 16 checklist | Every item in section 16 passes |

---

## 16. Final QA checklist

- [ ] First launch: language, tour, practice run, Home, all working without a server.
- [ ] Every screen checked in Hindi, Tamil, Urdu (RTL), Santali and English, in light and dark themes.
- [ ] Font scale 200% and display size at maximum: nothing essential cut off, and everything scrolls.
- [ ] TalkBack: the whole check flow can be completed by ear, in two languages.
- [ ] Phone with 2 GB RAM on Android 8: no crashes; the camera opens within 2 s; the budgets in section 12 are met
  (measured with Android Studio's memory profiler).
- [ ] Download size ≤ 8 MB (Play Console / bundletool); universal APK ≤ 15 MB.
- [ ] Offline: the test is saved and sent later. Server errors map to the right messages. A 30 s timeout shows
  **Try again**.
- [ ] Camera permission refused: **Choose from phone** still works.
- [ ] Uploaded JPEGs contain no EXIF location data.
- [ ] No emoji anywhere (search the code and resources for emoji code points).
- [ ] Every colour pair meets the contrast rules in both themes.
- [ ] Delete and Delete all remove the rows and their files; storage use goes down.
- [ ] With an empty `API_BASE_URL`: debug shows sample results; release shows "not set up yet".

---

## Appendix A: Crops and classes (draft list from model v2)

This mirrors the current model. The official hackathon list will replace it: update `assets/crops_fallback.json` and
the string keys, while the API stays the source of truth.

| Crop id | English name | Class ids (model label) | Coverage |
|---|---|---|---|
| `apple` | Apple | `apple_scab` (Apple___Apple_scab), `apple_cedar_rust` (Apple___Cedar_apple_rust), `apple_healthy` (Apple___healthy) | full |
| `bell_pepper` | Bell pepper | `pepper_bacterial_spot` (Pepper,_bell___Bacterial_spot), `pepper_healthy` (Pepper,_bell___healthy) | full |
| `blueberry` | Blueberry | `blueberry_healthy` (Blueberry___healthy) | healthy_only |
| `cherry` | Cherry | `cherry_healthy` (Cherry_(including_sour)___healthy) | healthy_only |
| `corn` | Corn (maize) | `corn_gray_leaf_spot` (Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot), `corn_common_rust` (Corn_(maize)___Common_rust_), `corn_northern_leaf_blight` (Corn_(maize)___Northern_Leaf_Blight) | diseases_only |
| `grape` | Grape | `grape_black_rot` (Grape___Black_rot), `grape_healthy` (Grape___healthy) | full |
| `peach` | Peach | `peach_healthy` (Peach___healthy) | healthy_only |
| `potato` | Potato | `potato_early_blight` (Potato___Early_blight), `potato_late_blight` (Potato___Late_blight) | diseases_only |
| `raspberry` | Raspberry | `raspberry_healthy` (Raspberry___healthy) | healthy_only |
| `soybean` | Soybean | `soybean_healthy` (Soybean___healthy) | healthy_only |
| `squash` | Squash | `squash_powdery_mildew` (Squash___Powdery_mildew) | diseases_only |
| `strawberry` | Strawberry | `strawberry_healthy` (Strawberry___healthy) | healthy_only |
| `tomato` | Tomato | `tomato_bacterial_spot`, `tomato_early_blight`, `tomato_late_blight`, `tomato_leaf_mold`, `tomato_septoria_leaf_spot`, `tomato_spider_mites`, `tomato_mosaic_virus`, `tomato_yellow_leaf_curl_virus`, `tomato_healthy` | full |
| `other` | Others (not in the list) | all classes; the result says "best match among supported plants" | n/a |

String keys: `crop_<id>` for crop names, and `disease_<class id>` for disease names, e.g. `crop_tomato` and
`disease_tomato_late_blight`. Healthy classes use the shared text "Healthy".

---

## Appendix B: Core English copy (source strings)

Keep this tone: short, calm, concrete. Translators work from these.

```xml
<string name="app_subtitle">Check your plant's health</string>
<string name="language_title">Choose your language</string>
<string name="language_suggested">Suggested</string>
<string name="action_continue">Continue</string>
<string name="tour_title">How AgriSmart works</string>
<string name="tour_step_counter">%1$d of %2$d</string>
<string name="tour_check">Tap here to check a plant.</string>
<string name="tour_plant">Pick your plant. Not in the list? Choose Others.</string>
<string name="tour_photo">Take a photo of one leaf, or choose a photo from your phone.</string>
<string name="tour_result">Here is the result. Tap Listen to hear it.</string>
<string name="tour_recent">Your past checks are saved here.</string>
<string name="tour_menu">Open the menu for language, help and settings.</string>
<string name="tour_practice">Practice now</string>
<string name="tour_start">Start using AgriSmart</string>
<string name="action_next">Next</string>
<string name="action_back">Back</string>
<string name="action_skip">Skip</string>
<string name="home_check">Check a plant</string>
<string name="home_recent">Recent tests</string>
<string name="see_all">See all</string>
<string name="choose_plant_title">Choose your plant</string>
<string name="crop_other">Others</string>
<string name="crop_other_hint">Not in the list</string>
<string name="coverage_healthy_only">Can only confirm a healthy leaf</string>
<string name="coverage_diseases_only">Can't confirm a healthy leaf</string>
<string name="add_photo_title">Add a photo</string>
<string name="take_photo">Take photo</string>
<string name="choose_from_phone">Choose from phone</string>
<string name="photo_tip">One leaf, in daylight, filling the frame</string>
<string name="hint_too_dark">Too dark: move into daylight</string>
<string name="hint_glare">Too much glare</string>
<string name="hint_hold_still">Hold still</string>
<string name="use_photo">Use this photo</string>
<string name="retake">Retake</string>
<string name="photo_problem_title">This photo may be hard to read</string>
<string name="take_another">Take another</string>
<string name="use_anyway">Use anyway</string>
<string name="checking">Checking your plant…</string>
<string name="cancel">Cancel</string>
<string name="status_disease">Disease found</string>
<string name="status_healthy">Healthy</string>
<string name="status_not_sure">Not sure</string>
<string name="result_healthy_title">Your %1$s leaf looks healthy</string>
<string name="result_not_sure_title">We are not sure</string>
<string name="result_not_sure_body">Take another photo: closer, in daylight, one leaf.</string>
<string name="result_unsupported_title">We can't check this plant yet</string>
<string name="result_mismatch_title">This looks like %1$s, not %2$s</string>
<string name="result_mismatch_yes">Yes, it's %1$s</string>
<string name="result_mismatch_no">No, keep %1$s</string>
<string name="how_sure_high">Very likely</string>
<string name="how_sure_medium">Likely</string>
<string name="listen">Listen</string>
<string name="what_to_do">What to do</string>
<string name="check_another">Check another plant</string>
<string name="best_match_note">Best match among supported plants</string>
<string name="disclaimer">This is a screening aid, not a diagnosis. Confirm with your local agriculture officer or Krishi Vigyan Kendra.</string>
<string name="call_kcc">Call Kisan Call Centre</string>
<string name="saved_offline">No internet. Your photo is saved. We will check it when you are online.</string>
<string name="status_waiting">Waiting for internet</string>
<string name="status_sending">Sending</string>
<string name="status_failed">Could not send</string>
<string name="try_again">Try again</string>
<string name="no_tests_yet">No tests yet</string>
<string name="delete_test_title">Delete this test?</string>
<string name="delete">Delete</string>
<string name="delete_all_title">Delete all tests?</string>
<string name="error_server_busy">The server is busy. Please try again.</string>
<string name="error_timeout">This is taking too long. Please try again.</string>
<string name="error_not_configured">This service is not set up yet.</string>
<string name="camera_primer">AgriSmart needs the camera to photograph your plant.</string>
<string name="allow_camera">Allow camera</string>
<string name="privacy_notice">Your photo is sent to AgriSmart only to check your plant. We do not ask for your name or location.</string>
<string name="practice_label">Practice</string>
<string name="mock_banner">Sample results: no server set</string>
```

---

## Appendix C: Bundled assets and mock fixtures

| Path | Purpose | Size budget |
|---|---|---|
| `assets/crops_fallback.json` | Appendix A in the `/v1/crops` format, English names (translations come from strings) | < 5 KB |
| `assets/mock/predict_disease.json` | Tomato late blight, `high` | < 3 KB |
| `assets/mock/predict_healthy.json` | Apple healthy | < 3 KB |
| `assets/mock/predict_uncertain.json` | `low` confidence | < 3 KB |
| `assets/mock/predict_unsupported.json` | For `other` | < 3 KB |
| `assets/mock/predict_mismatch.json` | Chose tomato, looks like potato | < 3 KB |
| `assets/mock/error_busy.json` | 503 `SERVER_BUSY` | < 1 KB |
| `assets/practice/leaf.webp` | Practice photo: a tomato late blight leaf from PlantDoc (CC BY 4.0; credit in Help and about) | < 60 KB |
| `assets/practice/result.json` | The practice result | < 3 KB |
| `res/raw/lang_<tag>.opus` | Optional spoken language names for the picker | < 10 KB each |

**`FakeAgriSmartApi` picks a fixture by crop:**
- `tomato`: disease;
- `apple`: healthy;
- `potato`: mismatch;
- `other`: unsupported;
- any other crop: uncertain.

A debug-only setting cycles through the error fixtures. That way every state in section 6.3 can be reached from the
UI.
