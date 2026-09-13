// AgriSmart offline: the crop-disease model runs inside this page (onnxruntime-web, WebAssembly), no server
// and no internet. Flow: point the camera, capture one still (by button or automatically when the picture is
// steady), then the model checks that still; the answer, its cause and what to do are shown and read aloud
// in the chosen language. The voice keeps going when the phone is moved, because the camera is no longer
// being checked once a still is captured.
// With internet (no keys, nothing to set up): a Wikipedia article about the answer in the chosen language,
// similar photos from Wikimedia Commons, and a "Search with Google Lens" button that hands the photo to Google
// Lens (the Lens app on Android; Lens in the browser on Windows) - which also names plants the offline model
// does not know, such as neem or tulsi.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const video = $("video"), frozen = $("frozen"), photo = $("photo"), viewport = $("viewport"), box = $("box");

  const BOX = 0.7;            // share of the shorter side inside the on-screen box (what gets checked)
  const STILL = 448;          // the captured square, in pixels (enough for the 336 px model)
  const UNSURE = 0.28;        // best disease below this -> "not sure"
  const CROP_SURE = 0.50, CROP_MARGIN = 0.35, MISMATCH = 0.15;  // same rules as the live-camera server
  const STEADY_FRAMES = 5;    // auto capture after this many calm checks in a row (every 200 ms)
  const HISTORY_MAX = 20;
  const WEB_TIMEOUT = 12000;

  const store = {
    get(key, fallback) { try { const v = localStorage.getItem("agri_" + key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
    set(key, value) { try { localStorage.setItem("agri_" + key, JSON.stringify(value)); } catch { /* storage blocked or full */ } },
  };
  const android = window.AgriSmartAndroid || null;   // the APK's bridge (share sheet, phone voice, Google Lens)

  let APP = null, MODELS = [];
  const AUDIO = {};           // language -> promise of {phrase key: clip url}
  let lang = store.get("lang", (navigator.language || "en").split("-")[0]);
  let modelId = store.get("model", "india_v1");
  let chosenCrop = store.get("crop", "");
  let voiceOn = store.get("voice", true), careful = store.get("careful", false), autoCap = store.get("autocap", true);
  let online = store.get("online", true);   // internet extras: only crop/disease names are sent (to Wikipedia)
  let trustedCrop = "";       // the farmer answered "yes, this is my crop" for the chosen crop
  let session = null, sessionId = null, loading = null;
  let stream = null, mode = "idle";   // idle | live | frozen | photo
  let last = null;                    // the result on screen: {spec, probs, res, still, web}
  let watchTimer = null, steady = 0, prevSig = null, liveSince = 0;

  // ---- text ----------------------------------------------------------------------------------------------
  const tr = (dict, key, fallback) => { const r = dict && dict[key]; return (r && (r[lang] || r.en)) || fallback || key; };
  const t = (key) => tr(APP.ui, key);
  const fill = (text, vars) => text.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? vars[k] : m));
  const cropName = (id) => tr(APP.crops, id, id.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase()));
  const termName = (label) => tr(APP.terms, label, label);
  const itemName = (item) => (item.healthy ? `${cropName(item.crop)} · ${termName("Healthy")}` : `${cropName(item.crop)} · ${termName(item.label)}`);
  const pct = (p) => Math.round(p * 100) + "%";

  function showStatus(text, bad) { const s = $("status"); s.textContent = text; s.className = "status" + (bad ? " bad" : ""); s.hidden = !text; }
  let toastTimer = null;
  function toast(text, ms) { const el = $("toast"); el.textContent = text; el.hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.hidden = true; }, ms || 2200); }
  function button(text, cls, onclick) { const b = document.createElement("button"); b.type = "button"; b.textContent = text; if (cls) b.className = cls; b.onclick = onclick; return b; }

  function applyText() {
    document.documentElement.lang = lang;
    document.documentElement.dir = APP.languages.find((l) => l.code === lang)?.rtl ? "rtl" : "ltr";
    $("brandTitle").textContent = t("title");
    $("langSelect").setAttribute("aria-label", t("choose_language"));
    $("historyBtn").textContent = t("history");
    $("modelLabel").textContent = t("model_label");
    $("cropLabel").textContent = t("choose_crop");
    $("startBtn").textContent = t("start_camera");
    $("uploadBtn").textContent = $("uploadBtn2").textContent = t("load_from_device");
    $("captureText").textContent = t("capture");
    $("againBtn").textContent = t("scan_again");
    $("lensBtn").textContent = t("lens_search");
    $("voiceBtn").textContent = voiceOn ? t("voice_on") : t("voice_off");
    $("voiceBtn").setAttribute("aria-pressed", String(voiceOn));
    $("autoCapText").textContent = t("auto_capture");
    $("carefulText").textContent = t("careful_mode");
    $("onlineText").textContent = t("online_mode");
    $("thinkingText").textContent = t("thinking");
    $("listenBtn").textContent = t("listen");
    $("stopBtn").textContent = t("stop_voice");
    $("shareBtn").textContent = t("share");
    $("causeTitle").textContent = t("cause_title");
    $("adviceTitle").textContent = t("advice_title");
    $("wikiTitle").textContent = t("learn_more");
    $("similarTitle").textContent = t("similar_photos");
    $("historyTitle").textContent = t("history");
    $("closeHistory").textContent = t("close");
    $("disclaimer").textContent = t("disclaimer");
    $("hint").textContent = mode === "live" ? t("tips") : "";
    buildCropSelect();
    if (last) { if (last.res && last.web && !last.web.loading) loadWebExtras(last); render(); } else renderIdle();
  }

  // ---- pickers -------------------------------------------------------------------------------------------
  function buildLangSelect() {
    const sel = $("langSelect"); sel.innerHTML = "";
    APP.languages.forEach((l) => sel.add(new Option(l.native, l.code, false, l.code === lang)));
    sel.onchange = () => { lang = sel.value; store.set("lang", lang); stopVoice(); applyText(); };
  }

  const spec = () => MODELS.find((m) => m.id === modelId) || MODELS[0];

  function buildModelSelect() {
    const sel = $("modelSelect"); sel.innerHTML = "";
    MODELS.forEach((m) => sel.add(new Option(m.name, m.id, false, m.id === modelId)));
    sel.onchange = () => {
      modelId = sel.value; store.set("model", modelId); showModelInfo(); buildCropSelect();
      ensureModel().then(() => { if (last && last.still) think(last.still, true); }).catch(modelError);
    };
    showModelInfo();
  }

  function showModelInfo() {
    const m = spec();
    const bits = [`${Object.keys(m.crop_classes).length} crops`, `${m.labels.length} classes`, `${m.size_mb} MB`];
    if (m.scores) bits.push(`field F1 ${m.scores.field_macro_f1.toFixed(2)}`);
    $("modelInfo").textContent = bits.join(" · ");
  }

  function buildCropSelect() {
    const sel = $("cropSelect"); sel.innerHTML = "";
    sel.add(new Option(t("let_model_guess"), ""));
    Object.keys(spec().crop_classes).map((id) => ({ id, name: cropName(id) }))
      .sort((a, b) => a.name.localeCompare(b.name, lang))
      .forEach(({ id, name }) => sel.add(new Option(name, id)));
    if (!(chosenCrop in spec().crop_classes)) chosenCrop = "";
    sel.value = chosenCrop;
    sel.onchange = () => { setCrop(sel.value); };
  }

  function setCrop(id) {
    chosenCrop = id; trustedCrop = ""; store.set("crop", id); $("cropSelect").value = id;
    if (last && last.probs && last.spec.id === modelId) {   // no need to run the model again
      last.res = analyse(last.spec, last.probs, chosenCrop);
      answerChanged();
    }
  }

  // the answer on screen changed (crop chosen, "yes" pressed): show it, say it, fetch its internet extras
  function answerChanged() {
    last.web = null;
    render(); speakResult();
    if (kindOf(last.res) === "sure") { saveHistory(last.res, last.still); loadWebExtras(last); }
  }

  // ---- model ---------------------------------------------------------------------------------------------
  ort.env.wasm.wasmPaths = new URL("ort/", location.href).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

  async function fetchBytes(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    const total = Number(res.headers.get("Content-Length")) || 0;
    if (!res.body) return new Uint8Array(await res.arrayBuffer());
    const reader = res.body.getReader(), chunks = [];
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length;
      if (total) onProgress(Math.min(1, got / total));
    }
    const out = new Uint8Array(got);
    let at = 0;
    for (const c of chunks) { out.set(c, at); at += c.length; }
    return out;
  }

  function modelError(err) { showStatus(`${spec().name}: ${err && err.message ? err.message : err}`, true); }

  async function ensureModel() {
    const m = spec();
    if (session && sessionId === m.id) return { spec: m, session };
    if (loading && loading.id === m.id) return loading.promise;
    const promise = (async () => {
      if (session) { const old = session; session = null; sessionId = null; try { await old.release(); } catch { /* already gone */ } }
      showStatus(`${t("loading_model")} 0%`);
      const bytes = await fetchBytes(m.file, (f) => showStatus(`${t("loading_model")} ${pct(f)}`));
      showStatus(t("loading_model"));
      const s = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
      if (modelId !== m.id) { try { await s.release(); } catch { /* ignore */ } return ensureModel(); }  // switched meanwhile
      session = s; sessionId = m.id; showStatus("");
      return { spec: m, session: s };
    })();
    loading = { id: m.id, promise };
    try { return await promise; } finally { if (loading && loading.promise === promise) loading = null; }
  }

  // same preprocessing the model was trained/checked with: shorter side -> size/0.9, centre crop, normalise
  function pixels(src, m, flip) {
    const S = m.img_size, sw = src.width, sh = src.height;
    const side = S / (Math.floor(S / m.crop_pct) / Math.min(sw, sh));
    const c = document.createElement("canvas"); c.width = c.height = S;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    if (flip) { ctx.translate(S, 0); ctx.scale(-1, 1); }
    ctx.drawImage(src, (sw - side) / 2, (sh - side) / 2, side, side, 0, 0, S, S);
    const d = ctx.getImageData(0, 0, S, S).data, n = S * S, out = new Float32Array(3 * n);
    const [m0, m1, m2] = m.mean, [s0, s1, s2] = m.std;
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      out[i] = (d[p] / 255 - m0) / s0;
      out[n + i] = (d[p + 1] / 255 - m1) / s1;
      out[2 * n + i] = (d[p + 2] / 255 - m2) / s2;
    }
    return out;
  }

  async function classify(still) {
    const { spec: m, session: s } = await ensureModel();
    const S = m.img_size, per = 3 * S * S;
    const views = careful ? [false, true] : [false];     // careful mode also checks the mirror image
    const data = new Float32Array(views.length * per);
    views.forEach((flip, i) => data.set(pixels(still, m, flip), i * per));
    const t0 = performance.now();
    const out = await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", data, [views.length, 3, S, S]) });
    const logits = out[s.outputNames[0]].data, C = m.labels.length, probs = new Float64Array(C);
    for (let v = 0; v < views.length; v++) {
      let mx = -Infinity, sum = 0;
      for (let k = 0; k < C; k++) mx = Math.max(mx, logits[v * C + k]);
      const e = new Float64Array(C);
      for (let k = 0; k < C; k++) { e[k] = Math.exp(logits[v * C + k] - mx); sum += e[k]; }
      for (let k = 0; k < C; k++) probs[k] += e[k] / sum / views.length;
    }
    return { spec: m, probs, ms: Math.round(performance.now() - t0) };
  }

  // plant first (each crop's classes added up), then the disease within that plant
  function analyse(m, probs, crop) {
    const share = {};
    m.labels.forEach((l, i) => { share[l.crop] = (share[l.crop] || 0) + probs[i]; });
    const ranked = Object.entries(share).sort((a, b) => b[1] - a[1]);
    const detected = ranked[0][0];
    let idx = m.labels.map((_, i) => i), ambiguous = false, matches = true;
    if (crop && m.crop_classes[crop]) {
      idx = m.crop_classes[crop];
      matches = detected === crop || (share[crop] || 0) >= MISMATCH || crop === trustedCrop;
    } else {
      crop = "";
      ambiguous = ranked.length > 1 && (ranked[0][1] < CROP_SURE || ranked[0][1] - ranked[1][1] < CROP_MARGIN);
      if (!ambiguous) idx = m.crop_classes[detected];
    }
    const mass = idx.reduce((s, i) => s + probs[i], 0) || 1;
    const top = idx.map((i) => ({ crop: m.labels[i].crop, label: m.labels[i].label, p: probs[i] / mass, healthy: m.labels[i].label === "Healthy" }))
      .sort((a, b) => b.p - a.p).slice(0, 3);
    return { top, best: top[0], detected, cropScores: ranked.slice(0, 4).map(([c, p]) => ({ crop: c, p })), ambiguous, matches, cropGiven: crop };
  }

  function kindOf(res) {
    if (res.dark) return "dark";
    if (res.ambiguous) return "ask";
    if (!res.matches) return "mismatch";
    if (!res.best || res.best.p < UNSURE) return "unsure";
    return "sure";
  }

  // ---- cause and advice ------------------------------------------------------------------------------------
  function groupOf(item) {
    const A = APP.advice;
    return A.crop_terms[`${item.crop}::${item.label}`] || A.terms[item.label] || (item.healthy ? "healthy" : null);
  }
  function steps(item) {
    const A = APP.advice, g = groupOf(item) || A.default;
    return (A.groups[g] || []).map(([key, params], i) => {
      const tpl = A.templates[key] || {};
      const text = (tpl[lang] || tpl.en || "").replace(/\{(\w+)\}/g, (m, k) => (params && k in params ? params[k] : m));
      return { key: `step:${g}:${i}`, text };
    });
  }
  function causeOf(item) {
    if (item.healthy) return null;
    const C = APP.causes, g = groupOf(item);
    const cat = C.terms[`${item.crop}::${item.label}`] || C.terms[item.label] || (g && C.groups[g]);
    return cat ? { key: `cause:${cat}`, text: tr(C.templates, cat) } : null;
  }

  // ---- internet extras (no keys): Wikipedia in the chosen language + similar photos from Wikimedia Commons ----
  async function fetchJson(url) {
    const ctl = new AbortController(), timer = setTimeout(() => ctl.abort(), WEB_TIMEOUT);
    try {
      const r = await fetch(url, { signal: ctl.signal });
      return r.ok ? await r.json() : null;
    } finally { clearTimeout(timer); }
  }
  const englishName = (text) => String(text || "").replace(/\s*\(.*?\)/g, "").trim();

  // the words that say which disease (or crop) a page is about; "leaf", "spot", "plant" are in every title
  const GENERIC = new Set(["leaf", "leaves", "plant", "plants", "disease", "diseases", "fruit", "spot", "spots", "pest", "and", "the", "healthy"]);
  const keyWords = (text) => englishName(text).toLowerCase().split(/[^a-z]+/).filter((w) => w.length >= 3 && !GENERIC.has(w));
  const namesAny = (title, words) => { const t = title.toLowerCase().split(/[^a-z]+/); return words.some((w) => t.includes(w)); };

  // an English article whose title names the disease (or the crop), then its version in the chosen language when one
  // exists; no article at all rather than a wrong one (a plain search for "Alternaria leaf blotch apple" ranks an
  // apple variety first)
  async function wikiArticle(query, mustName) {
    const words = keyWords(mustName);
    if (!words.length) return null;
    const q = new URLSearchParams({ action: "query", list: "search", srsearch: query, srlimit: "8", srprop: "", format: "json", origin: "*" });
    const s = await fetchJson(`https://en.wikipedia.org/w/api.php?${q}`);
    const hit = ((s && s.query && s.query.search) || []).find((h) => namesAny(h.title, words));
    if (!hit) return null;
    let code = "en", title = hit.title;
    if (lang !== "en") {
      const l = await fetchJson(`https://en.wikipedia.org/w/api.php?${new URLSearchParams({ action: "query", titles: hit.title, prop: "langlinks", lllang: lang, format: "json", origin: "*" })}`);
      const page = l && l.query && l.query.pages && Object.values(l.query.pages)[0];
      const local = page && page.langlinks && page.langlinks[0] && page.langlinks[0]["*"];
      if (local) { code = lang; title = local; }
    }
    const sum = await fetchJson(`https://${code}.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title.replace(/ /g, "_"))}`);
    if (!sum || !sum.extract) return null;
    return { title: sum.title, extract: sum.extract, lang: code,
             url: (sum.content_urls && sum.content_urls.mobile && sum.content_urls.mobile.page) || `https://${code}.wikipedia.org/wiki/${encodeURIComponent(title)}`,
             thumb: sum.thumbnail && sum.thumbnail.source };
  }

  // file names that show a sick plant: kept out of the photos for a healthy leaf ("Tomato septoria leaf spot" is not
  // what a healthy tomato looks like)
  const SICK = /spot|blight|rot\b|mildew|rust|virus|mosaic|wilt|scab|mou?ld|curl|canker|septoria|disease|infect|pest|lesion|necro|chloro|yellow|miner|deficien|damage|insect|larva|beetle|aphid|mite|borer|caterpill|scorch|burn|symptom|sick/i;

  // photos from Wikimedia Commons whose file names mention the disease (or crop); the first query that finds some wins
  async function commonsPhotos(queries, mustName, healthy) {
    const words = keyWords(mustName);
    for (const query of queries) {
      const q = new URLSearchParams({ action: "query", generator: "search", gsrnamespace: "6", gsrsearch: `${query} filetype:bitmap`,
                                      gsrlimit: "12", prop: "imageinfo", iiprop: "url", iiurlwidth: "240", format: "json", origin: "*" });
      const j = await fetchJson(`https://commons.wikimedia.org/w/api.php?${q}`);
      const pages = j && j.query && j.query.pages ? Object.values(j.query.pages).sort((a, b) => a.index - b.index) : [];
      const found = pages.filter((p) => p.imageinfo && p.imageinfo[0] && p.imageinfo[0].thumburl && words.some((w) => p.title.toLowerCase().includes(w))
                                      && !(healthy && SICK.test(p.title)))
        .map((p) => ({ src: p.imageinfo[0].thumburl, href: p.imageinfo[0].descriptionurl,
                       label: p.title.replace(/^File:/, "").replace(/\.\w+$/, "").slice(0, 60) }))
        .filter((p, i, all) => all.findIndex((o) => o.label === p.label) === i)   // the same photo uploaded twice
        .slice(0, 8);
      if (found.length) return found;
    }
    return [];
  }

  async function loadWebExtras(entry) {
    if (!online || !navigator.onLine || !entry.res || kindOf(entry.res) !== "sure") return;
    const b = entry.res.best, crop = englishName((APP.crops[b.crop] || {}).en) || b.crop.replace(/_/g, " ");
    const cropWords = `${crop} ${b.crop.replace(/_/g, " ")}`;   // "Corn (maize)" + "corn_maize": either name counts
    const disease = englishName(b.label);
    entry.web = { loading: true };
    if (last === entry) render();
    const [wiki, photos] = await Promise.all([
      (b.healthy ? wikiArticle(`${crop} plant`, cropWords) : wikiArticle(`${disease} ${crop}`, disease))
        .catch((e) => { console.warn("wikipedia:", e && e.message); return null; }),
      (b.healthy ? commonsPhotos([`${crop} leaf`, `${crop} plant`], cropWords, true) : commonsPhotos([`${disease} ${crop}`, disease], disease, false))
        .catch((e) => { console.warn("commons:", e && e.message); return []; }),
    ]);
    if (!entry.web || !entry.web.loading || last !== entry) return;   // the user moved on
    entry.web = { wiki, photos };
    render();
  }

  // Google Lens: the Lens app on Android; on Windows the photo is copied and Google Lens opens in the browser
  async function openLens() {
    const still = last && last.still;
    if (!still) return;
    if (android && android.openLens) { android.openLens(still.toDataURL("image/jpeg", 0.9).split(",")[1]); return; }
    try {
      const blob = await new Promise((r) => still.toBlob(r, "image/png"));
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
      toast(t("photo_copied"), 7000);
    } catch (err) { console.warn("could not copy the photo:", err && err.message); }
    window.open("https://lens.google.com/", "_blank", "noopener");
  }

  // ---- voice: recorded clips in each language (works offline), the device's own voice as a fallback --------
  const player = new Audio();
  let speakToken = 0, clipDone = null, ttsSeq = 0;
  const ttsWaiting = {};
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  window.__agriTtsDone = (id) => { const done = ttsWaiting[id]; delete ttsWaiting[id]; if (done) done(); };

  function playClip(url) {
    return new Promise((resolve) => {
      clipDone = resolve;
      player.onended = player.onerror = () => resolve();
      player.src = url;
      player.play().catch(() => resolve());
    });
  }
  function deviceVoice(text) {
    if (!text) return Promise.resolve();
    if (android && android.speak) {                    // the phone's text-to-speech (Android)
      return new Promise((resolve) => {
        const id = "u" + (++ttsSeq);
        ttsWaiting[id] = resolve; clipDone = resolve;
        if (!android.speak(text, lang, id)) { delete ttsWaiting[id]; resolve(); }
      });
    }
    const synth = window.speechSynthesis;
    const v = synth && synth.getVoices().find((x) => (x.lang || "").toLowerCase().startsWith(lang));
    if (!v) return Promise.resolve();                  // never read Hindi text with an English voice
    return new Promise((resolve) => {
      const u = new SpeechSynthesisUtterance(text);
      u.voice = v; u.lang = v.lang; u.rate = 0.95; u.onend = u.onerror = () => resolve();
      clipDone = resolve; synth.speak(u);
    });
  }
  // each language's clip list is loaded when that language is first needed
  function loadVoice(code) {
    if (!(code in AUDIO)) {
      AUDIO[code] = fetch(`audio/${code}/index.json`).then((r) => (r.ok ? r.json() : {})).then((j) => j.clips || {}).catch(() => ({}));
    }
    return AUDIO[code];
  }
  async function speak(parts) {
    const token = ++speakToken;
    halt();
    if (!voiceOn) return;
    const clips = await loadVoice(lang);
    if (token !== speakToken) return;
    for (const part of parts) {
      if (token !== speakToken) return;
      if (part.key && clips[part.key]) await playClip(clips[part.key]); else await deviceVoice(part.text);
      if (token !== speakToken) return;
      await sleep(220);
    }
  }
  function halt() {
    player.pause(); player.removeAttribute("src");
    if (window.speechSynthesis) speechSynthesis.cancel();
    if (android && android.stopSpeaking) android.stopSpeaking();
    if (clipDone) { const d = clipDone; clipDone = null; d(); }
  }
  function stopVoice() { speakToken++; halt(); }

  function speechParts(res) {
    const kind = kindOf(res), ui = (k) => ({ key: "ui:" + k, text: t(k) });
    if (kind === "dark") return [ui("too_dark")];
    if (kind === "ask") return [ui("which_crop"), ui("which_crop_hint")];
    if (kind === "mismatch") return [{ key: "crop:" + res.cropGiven, text: cropName(res.cropGiven) }, ui("confirm_crop")];
    if (kind === "unsure") return [ui("not_sure")];
    const b = res.best, parts = [{ key: "crop:" + b.crop, text: cropName(b.crop) }];
    if (b.healthy) parts.push(ui("healthy_result"));
    else {
      parts.push({ key: "term:" + b.label, text: termName(b.label) });
      const c = causeOf(b); if (c) parts.push(c);
    }
    const s = steps(b);
    if (s.length) parts.push(ui("advice_title"), ...s);
    return parts;
  }
  function speakResult() { if (last && last.res) speak(speechParts(last.res)); }

  // ---- camera ----------------------------------------------------------------------------------------------
  function cameraErrorText(err) {
    switch (err && err.name) {
      case "NotAllowedError": return "Camera access is blocked. Allow the camera for this app, then press the button again.";
      case "NotFoundError": return "No camera was found on this device.";
      case "NotReadableError": return "The camera is busy in another app. Close it, then press the button again.";
      default: return "The camera could not start: " + (err && err.message ? err.message : String(err));
    }
  }

  async function startCamera() {
    $("startMsg").textContent = "";
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      $("startMsg").textContent = "This browser cannot open the camera here. Use the photo button instead."; return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: false,
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 960 } } });
    } catch (err) { stream = null; $("startMsg").textContent = cameraErrorText(err); return; }
    video.srcObject = stream;
    video.play().catch(() => {});   // not awaited: some WebViews never settle this promise; capture waits for real frames
    const facing = stream.getVideoTracks()[0].getSettings().facingMode;
    const mirror = facing === "user" || (!facing && !/Android|iPhone|iPad/i.test(navigator.userAgent));
    video.classList.toggle("mirror", mirror); frozen.classList.toggle("mirror", mirror);
    goLive();
  }

  function fitViewport(w, h) {
    viewport.style.aspectRatio = `${w} / ${h}`;
    viewport.classList.toggle("portrait", h > w);
  }
  video.addEventListener("loadedmetadata", () => fitViewport(video.videoWidth, video.videoHeight));

  function goLive() {
    if (!stream) { $("start").hidden = false; return; }
    stream.getVideoTracks().forEach((tk) => { tk.enabled = true; });
    if (video.paused) video.play().catch(() => {});
    mode = "live"; liveSince = performance.now(); steady = 0; prevSig = null;
    video.hidden = false; frozen.hidden = true; photo.hidden = true; box.hidden = false; box.className = "box";
    $("start").hidden = true; $("captureBtn").hidden = false; $("againBtn").hidden = true; $("lensBtn").hidden = true;
    $("hint").textContent = t("tips");
    if (video.videoWidth) fitViewport(video.videoWidth, video.videoHeight);
    clearInterval(watchTimer); watchTimer = setInterval(watch, 200);
  }

  // auto capture: a small grey copy of the box, compared with the previous one; calm + bright enough -> capture
  const sigCanvas = document.createElement("canvas"); sigCanvas.width = sigCanvas.height = 24;
  const sigCtx = sigCanvas.getContext("2d", { willReadFrequently: true });
  function watch() {
    if (mode !== "live" || !video.videoWidth) return showSteady(0);
    const vw = video.videoWidth, vh = video.videoHeight, side = Math.min(vw, vh) * BOX;
    sigCtx.drawImage(video, (vw - side) / 2, (vh - side) / 2, side, side, 0, 0, 24, 24);
    const d = sigCtx.getImageData(0, 0, 24, 24).data, sig = new Float32Array(576);
    let mean = 0;
    for (let i = 0; i < 576; i++) { sig[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]; mean += sig[i]; }
    mean /= 576;
    let spread = 0; for (let i = 0; i < 576; i++) spread += (sig[i] - mean) ** 2; spread = Math.sqrt(spread / 576);
    let diff = 255;
    if (prevSig) { diff = 0; for (let i = 0; i < 576; i++) diff += Math.abs(sig[i] - prevSig[i]); diff /= 576; }
    prevSig = sig;
    const calm = diff < 5 && mean > 40 && spread > 10;
    steady = calm ? steady + 1 : 0;
    if (!autoCap) return showSteady(0);
    showSteady(steady / STEADY_FRAMES);
    if (steady >= STEADY_FRAMES && performance.now() - liveSince > 1500) capture();
  }
  function showSteady(f) {
    $("steady").hidden = !(autoCap && mode === "live");
    $("steadyFill").style.width = Math.min(100, f * 100) + "%";
    box.classList.toggle("ready", f >= 0.6 && mode === "live");
  }

  async function capture() {
    if (mode !== "live" || !video.videoWidth) return;
    clearInterval(watchTimer); showSteady(0);
    const vw = video.videoWidth, vh = video.videoHeight;
    frozen.width = vw; frozen.height = vh;
    frozen.getContext("2d").drawImage(video, 0, 0);
    frozen.hidden = false; video.hidden = true;
    stream.getVideoTracks().forEach((tk) => { tk.enabled = false; });   // the photo is taken; rest the camera
    mode = "frozen"; $("captureBtn").hidden = true; $("againBtn").hidden = false; $("hint").textContent = "";
    const side = Math.round(Math.min(vw, vh) * BOX);
    const still = document.createElement("canvas"); still.width = still.height = STILL;
    still.getContext("2d").drawImage(frozen, (vw - side) / 2, (vh - side) / 2, side, side, 0, 0, STILL, STILL);
    await think(still);
  }

  // a photo from the gallery or files: createImageBitmap first, an <img> as the fallback some WebViews need
  async function decodePhoto(file) {
    try { return await createImageBitmap(file); }
    catch (err) { console.warn("createImageBitmap failed:", err && err.message); }
    const url = URL.createObjectURL(file);
    try {
      const img = new Image(); img.src = url; await img.decode();
      const c = document.createElement("canvas"); c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext("2d").drawImage(img, 0, 0);
      return c;
    } finally { URL.revokeObjectURL(url); }
  }

  async function handleFile(file) {
    console.log(`photo chosen: ${file.type || "unknown type"}, ${file.size} bytes`);
    let bmp;
    try { bmp = await decodePhoto(file); }
    catch (err) { console.warn("photo could not be read:", err && err.message); toast("That file could not be read as a photo."); return; }
    clearInterval(watchTimer);
    if (stream) stream.getVideoTracks().forEach((tk) => { tk.enabled = false; });
    const scale = Math.min(1, 1280 / Math.max(bmp.width, bmp.height));
    const still = document.createElement("canvas");
    still.width = Math.round(bmp.width * scale); still.height = Math.round(bmp.height * scale);
    still.getContext("2d").drawImage(bmp, 0, 0, still.width, still.height);
    if (bmp.close) bmp.close();
    if (photo.src) URL.revokeObjectURL(photo.src);
    photo.src = URL.createObjectURL(file);
    fitViewport(still.width, still.height);
    mode = "photo"; photo.hidden = false; video.hidden = true; frozen.hidden = true; box.hidden = true; showSteady(0);
    $("start").hidden = true; $("captureBtn").hidden = true; $("againBtn").hidden = false; $("hint").textContent = "";
    await think(still);
  }

  function scanAgain() {
    if (stream) goLive(); else startCamera();
  }

  // ---- checking a still ------------------------------------------------------------------------------------
  function quality(canvas) {
    const n = 128, c = document.createElement("canvas"); c.width = c.height = n;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    const side = Math.min(canvas.width, canvas.height);
    ctx.drawImage(canvas, (canvas.width - side) / 2, (canvas.height - side) / 2, side, side, 0, 0, n, n);
    const d = ctx.getImageData(0, 0, n, n).data, g = new Float32Array(n * n);
    let mean = 0;
    for (let i = 0; i < n * n; i++) { g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]; mean += g[i]; }
    mean /= n * n;
    let s = 0, s2 = 0, cnt = 0;                         // variance of the Laplacian = sharpness
    for (let y = 1; y < n - 1; y++) for (let x = 1; x < n - 1; x++) {
      const i = y * n + x, lap = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - n] - g[i + n];
      s += lap; s2 += lap * lap; cnt++;
    }
    const sharp = s2 / cnt - (s / cnt) ** 2;
    return { dark: mean < 35, blurry: sharp < 30 };
  }

  let thinkingNow = 0;
  async function think(still, quiet) {
    const my = ++thinkingNow;
    const q = quality(still);
    if (q.dark) {
      last = { still, res: { dark: true } }; render();
      if (!quiet) speakResult();
      return;
    }
    $("thinking").hidden = false;
    let r;
    try { r = await classify(still); }
    catch (err) { $("thinking").hidden = true; modelError(err); return; }
    if (my !== thinkingNow) return;                      // a newer photo is being checked
    $("thinking").hidden = true;
    const res = analyse(r.spec, r.probs, chosenCrop);
    res.blurry = q.blurry; res.ms = r.ms;
    last = { spec: r.spec, probs: r.probs, res, still, web: null };
    render();
    speakResult();
    if (kindOf(res) === "sure") { saveHistory(res, still); loadWebExtras(last); }
  }

  // ---- rendering ---------------------------------------------------------------------------------------------
  function setChip(kind, text) { $("chip").className = "chip" + (kind ? " " + kind : ""); $("chip").textContent = text; }
  function clearCard() {
    ["askCrop", "voiceRow", "cause", "advice", "warnNote", "onlineNote", "wiki", "similar"].forEach((id) => { $(id).hidden = true; });
    $("fill").className = "fill"; $("fill").style.width = "0%"; $("conf").textContent = ""; $("crop").textContent = "";
    $("meter").hidden = false;
  }

  function renderIdle() {
    clearCard(); setChip("", ""); $("disease").textContent = t("subtitle"); $("meter").hidden = true;
  }

  function showList(items) {
    const list = $("adviceList"); list.innerHTML = "";
    items.forEach((text) => { const li = document.createElement("li"); li.textContent = text; list.appendChild(li); });
    $("advice").hidden = items.length === 0;
  }

  function render() {
    const res = last.res, kind = kindOf(res);
    clearCard();
    box.className = "box";
    // Google Lens can name any plant (also ones the offline model does not know): offered whenever there is a photo
    $("lensBtn").hidden = !(last.still && kind !== "dark" && navigator.onLine && mode !== "live");
    if (kind === "dark") {
      setChip("bad", t("not_sure")); $("disease").textContent = t("too_dark"); $("meter").hidden = true;
      return;
    }
    if (kind === "ask") return renderAskCrop(res);
    const b = res.best;
    if (kind === "mismatch") {
      // the photo does not look like the chosen crop: ask the farmer, who knows their own field
      setChip("bad", t("crop_mismatch"));
      $("disease").textContent = fill(t("is_this"), { crop: cropName(res.cropGiven) });
      $("meter").hidden = true;
      $("askCropHint").textContent = t("confirm_crop");
      const btns = $("cropBtns"); btns.innerHTML = "";
      btns.append(
        button(t("yes"), "primary", () => {
          trustedCrop = res.cropGiven;
          last.res = Object.assign({}, res, { matches: true });
          answerChanged();
        }),
        button(t("no"), "", () => setCrop("")),
      );
      $("askCrop").hidden = false;
      return;
    }
    if (kind === "unsure") {
      setChip("", t("not_sure")); $("disease").textContent = t("not_sure"); $("meter").hidden = true;
      $("warnNote").textContent = t("tips"); $("warnNote").hidden = false;
      return;
    }
    const k = b.healthy ? "healthy" : "disease";
    setChip(k, b.healthy ? termName("Healthy") : t("disease_found"));
    $("crop").textContent = cropName(b.crop);
    $("disease").textContent = b.healthy ? t("healthy_result") : termName(b.label);
    $("fill").className = "fill " + k; $("fill").style.width = pct(b.p);
    $("conf").textContent = `${t("how_sure")}: ${pct(b.p)}`;
    box.className = "box " + k;
    if (res.blurry) { $("warnNote").textContent = t("too_blurry"); $("warnNote").hidden = false; }
    const c = causeOf(b);
    if (c) { $("causeText").textContent = c.text; $("cause").hidden = false; }
    showList(steps(b).map((x) => x.text));
    $("adviceNote").textContent = b.healthy ? "" : t("advice_note");
    $("voiceRow").hidden = false;
    renderWeb(last);
  }

  // Wikipedia article and Commons photos, when they have arrived
  function renderWeb(entry) {
    const w = entry.web;
    if (!w) return;
    if (w.loading) {
      const note = $("onlineNote"); note.textContent = t("online_checking"); note.className = "onlineNote busy"; note.hidden = false;
      return;
    }
    if (w.wiki) {
      $("wikiName").textContent = w.wiki.title;
      $("wikiText").textContent = w.wiki.extract;
      $("wikiLink").href = w.wiki.url;
      $("wikiLink").textContent = `${w.wiki.lang}.wikipedia.org`;
      $("wikiThumb").hidden = !w.wiki.thumb;
      if (w.wiki.thumb) $("wikiThumb").src = w.wiki.thumb;
      $("wiki").hidden = false;
    }
    const strip = $("strip"); strip.innerHTML = "";
    (w.photos || []).forEach((p) => {
      const a = document.createElement("a"); a.href = p.href; a.target = "_blank"; a.rel = "noopener";
      const img = document.createElement("img"); img.src = p.src; img.alt = ""; img.loading = "lazy";
      const cap = document.createElement("span"); cap.textContent = p.label;
      a.append(img, cap); strip.appendChild(a);
    });
    $("similar").hidden = !(w.photos && w.photos.length);
  }

  function renderAskCrop(res) {
    setChip("", t("which_crop")); $("disease").textContent = t("which_crop"); $("meter").hidden = true;
    $("askCropHint").textContent = t("which_crop_hint");
    renderCropButtons(res.cropScores.filter((c, i) => i < 2 || (i < 3 && c.p >= 0.05)).map((c) => c.crop), res);
  }

  function renderCropButtons(crops, res) {
    const btns = $("cropBtns"); btns.innerHTML = "";
    crops.forEach((id) => {
      const b = button(cropName(id), "", () => setCrop(id));
      const score = res.cropScores.find((c) => c.crop === id);
      if (score) { const small = document.createElement("small"); small.textContent = pct(score.p); b.appendChild(small); }
      btns.appendChild(b);
    });
    $("askCrop").hidden = false;
  }

  // ---- history & share ---------------------------------------------------------------------------------------
  function saveHistory(res, still) {
    if (!still) return;
    const c = document.createElement("canvas"); c.width = c.height = 96;
    const side = Math.min(still.width, still.height);
    c.getContext("2d").drawImage(still, (still.width - side) / 2, (still.height - side) / 2, side, side, 0, 0, 96, 96);
    const items = store.get("history", []).filter((it) => it.best);
    items.unshift({ ts: Date.now(), model: modelId, thumb: c.toDataURL("image/jpeg", 0.7), best: res.best });
    store.set("history", items.slice(0, HISTORY_MAX));
  }

  function openHistory() {
    const list = $("historyList"); list.innerHTML = "";
    const items = store.get("history", []).filter((it) => it.best);
    if (!items.length) { const li = document.createElement("li"); li.className = "empty"; li.textContent = t("no_history"); list.appendChild(li); }
    items.forEach((it) => {
      const li = document.createElement("li"), b = document.createElement("button"); b.type = "button";
      const img = document.createElement("img"); img.src = it.thumb; img.alt = "";
      const txt = document.createElement("span");
      const h1 = document.createElement("span"); h1.className = "h1";
      h1.textContent = it.best.healthy ? `${cropName(it.best.crop)} · ${t("healthy_result")}` : itemName(it.best);
      const h2 = document.createElement("span"); h2.className = "h2";
      h2.textContent = `${new Date(it.ts).toLocaleString(lang)} · ${pct(it.best.p)}`;
      txt.append(h1, h2); b.append(img, txt);
      b.onclick = () => {
        $("historyDlg").close();
        last = { res: { best: it.best, top: [it.best], matches: true, ambiguous: false, cropScores: [] }, web: null };
        render(); speakResult(); loadWebExtras(last);
      };
      li.appendChild(b); list.appendChild(li);
    });
    $("historyDlg").showModal();
  }

  function shareText() {
    const res = last && last.res;
    if (!res || kindOf(res) !== "sure") return "";
    const b = res.best, lines = [`AgriSmart: ${cropName(b.crop)} - ${b.healthy ? t("healthy_result") : termName(b.label)} (${pct(b.p)})`];
    const c = causeOf(b); if (c) lines.push("", `${t("cause_title")}: ${c.text}`);
    const s = steps(b);
    if (s.length) { lines.push("", t("advice_title") + ":"); s.forEach((x, i) => lines.push(`${i + 1}. ${x.text}`)); }
    if (!b.healthy) lines.push("", t("advice_note"));
    if (last.web && last.web.wiki) lines.push("", `${t("learn_more")}: ${last.web.wiki.url}`);
    return lines.join("\n");
  }
  async function share() {
    const text = shareText(); if (!text) return;
    if (android && android.share) { android.share(text); return; }
    if (navigator.share) { try { await navigator.share({ title: "AgriSmart", text }); return; } catch (e) { if (e.name === "AbortError") return; } }
    try { await navigator.clipboard.writeText(text); toast("✓ " + t("share")); } catch { toast(text.slice(0, 80) + "…"); }
  }

  // ---- controls ------------------------------------------------------------------------------------------------
  $("startBtn").onclick = startCamera;
  $("captureBtn").onclick = capture;
  $("againBtn").onclick = scanAgain;      // does not stop the voice: it finishes what it was saying
  $("lensBtn").onclick = openLens;
  $("uploadBtn").onclick = $("uploadBtn2").onclick = () => $("fileInput").click();
  $("fileInput").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) handleFile(f); };
  $("voiceBtn").onclick = () => {
    voiceOn = !voiceOn; store.set("voice", voiceOn); if (!voiceOn) stopVoice();
    $("voiceBtn").textContent = voiceOn ? t("voice_on") : t("voice_off"); $("voiceBtn").setAttribute("aria-pressed", String(voiceOn));
  };
  $("listenBtn").onclick = () => { if (!voiceOn) $("voiceBtn").click(); speakResult(); };
  $("stopBtn").onclick = stopVoice;
  $("shareBtn").onclick = share;
  $("historyBtn").onclick = openHistory;
  $("closeHistory").onclick = () => $("historyDlg").close();
  $("autoCap").onchange = (e) => { autoCap = e.target.checked; store.set("autocap", autoCap); steady = 0; showSteady(0); };
  $("careful").onchange = (e) => { careful = e.target.checked; store.set("careful", careful); };
  $("onlineMode").onchange = (e) => {
    online = e.target.checked; store.set("online", online);
    if (last && last.res) { last.web = null; if (online) loadWebExtras(last); else render(); }
  };
  window.addEventListener("online", () => { if (last && last.res) { render(); if (!last.web) loadWebExtras(last); } });
  window.addEventListener("offline", () => { if (last && last.res) render(); });
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("button, select, input, dialog")) return;
    if (e.code === "Space") { e.preventDefault(); if (mode === "live") capture(); else if (mode !== "idle") scanAgain(); }
    else if (e.key === "v" || e.key === "V") $("voiceBtn").click();
  });

  // ---- start -----------------------------------------------------------------------------------------------------
  async function boot() {
    if (location.protocol === "file:") {
      showStatus("Open this app with run.bat (it starts a small local server). Opening index.html directly cannot load the model.", true);
    }
    try {
      const [app, models] = await Promise.all([
        fetch("data/app.json").then((r) => r.json()),
        fetch("models/models.json").then((r) => r.json()),
      ]);
      APP = app; MODELS = models.models;
    } catch (err) {
      showStatus("The app files could not be loaded: " + err.message, true); return;
    }
    if (!APP.languages.some((l) => l.code === lang)) lang = "en";
    if (!MODELS.some((m) => m.id === modelId)) modelId = MODELS[0].id;
    try { localStorage.removeItem("agri_keys"); } catch { /* the old online check's keys are no longer used */ }
    $("autoCap").checked = autoCap; $("careful").checked = careful; $("onlineMode").checked = online;
    buildLangSelect(); buildModelSelect(); applyText();
    ensureModel().catch(modelError);   // load while the user gets the camera ready
  }
  boot();
})();
