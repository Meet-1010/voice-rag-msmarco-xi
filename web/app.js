const $ = (id) => document.getElementById(id);
const CORE_BUDGET = 200;

const STAGE_ORDER = ["stt", "guard_input", "cache", "embed", "retrieve", "rerank",
                     "guard_relevance", "extract", "generate", "generate_general",
                     "guard_grounding"];
const STAGE_LABEL = {
  stt: "STT", guard_input: "input guard", cache: "cache", embed: "embed",
  retrieve: "retrieve", rerank: "rerank", guard_relevance: "relevance",
  extract: "extract", generate: "generate", generate_general: "LLM (general)",
  guard_grounding: "grounding",
};
// Stages outside the 200ms core budget, drawn differently so the distinction is
// visible rather than asserted in a footnote.
const EXTERNAL = new Set(["stt", "generate", "generate_general"]);
const SPEECH_LANG = { en: "en-IN", hi: "hi-IN", gu: "gu-IN" };

let recorder = null, chunks = [], busy = false;
let recTimer = null, recStart = 0, recog = null;
let liveText = "", finalText = "", speechEndedAt = 0;

/* ---------------------------------------------------------------------------
 * Speculative retrieval.
 *
 * The documented way to get a voice pipeline under 200ms is to stop waiting for
 * the user to finish. Interim speech results are good enough to retrieve on, so
 * every time the transcript settles for a moment we fire the retrieval pipeline
 * against what has been said so far and keep the promise. When the user actually
 * stops, the answer for that text is usually already in flight or done, and the
 * time between "stopped speaking" and "answer on screen" collapses to near zero.
 *
 * Speculative calls run retrieval-only (no LLM): they are cheap, they cannot
 * burn provider quota on half-sentences, and the extractive path is what serves
 * corpus questions anyway. If the final transcript needs an LLM we pay for it
 * once, at the end, on the real request.
 * ------------------------------------------------------------------------- */
const spec = new Map();
let specTimer = null;

function speculate(text) {
  const q = text.trim();
  if (q.length < 8 || spec.has(q)) return;
  if (spec.size > 12) spec.clear();
  spec.set(q, fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: q, lang: $("lang").value || null, use_cache: true,
      allow_generative: false, allow_general: false,
    }),
  }).then((r) => (r.ok ? r.json() : null)).catch(() => null));
}

function scheduleSpeculation(text) {
  clearTimeout(specTimer);
  specTimer = setTimeout(() => speculate(text), 200);
}

async function health() {
  const el = $("health");
  if (!el) return;   // status pills were removed from the header
  try {
    const h = await (await fetch("/health")).json();
    const n = (h.indexed_points || 0).toLocaleString();
    const langs = (h.manifest?.langs || []).join(" · ") || "–";
    el.innerHTML = `<span class="dot ok"></span><span>${n} chunks</span>`;
  } catch {
    el.innerHTML = `<span class="dot bad"></span><span>offline</span>`;
  }
}

function setBusy(on) {
  busy = on;
  $("send").disabled = on;
  if (on) {
    $("answer").innerHTML = `<p class="idle">Thinking…</p>`;
    $("pathBadge").hidden = $("reasonBadge").hidden = $("conf").hidden = true;
    $("budgetTag").hidden = true;
    $("bars").innerHTML = "";
  }
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));

function tick(el, value) {
  if (value == null) { el.textContent = "–"; return; }
  const fmt = (v) => (v < 10 ? v.toFixed(1) : v.toFixed(0));
  const dur = 420, t0 = performance.now();
  let settled = false;

  const step = (now) => {
    const p = Math.min(1, (now - t0) / dur);
    el.textContent = fmt(value * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(step); else settled = true;
  };
  requestAnimationFrame(step);

  // requestAnimationFrame is paused in a background tab, so without this the
  // number never appears at all - the count-up is decoration and must not be
  // what decides whether the measurement is shown.
  setTimeout(() => { if (!settled) el.textContent = fmt(value); }, dur + 120);
}

function renderTimings(t, voiceMs) {
  const stages = { ...(t.stages || {}) };
  if (t.stt_ms) stages.stt = t.stt_ms;

  tick($("coreMs"), t.core_ms ?? 0);
  tick($("totalMs"), t.wall_ms ?? t.total_ms ?? 0);
  // The number a person actually feels: silence to answer on screen.
  tick($("voiceMs"), voiceMs);

  const tag = $("budgetTag");
  const within = (t.core_ms ?? 0) <= CORE_BUDGET;
  tag.hidden = false;
  tag.className = "budget-tag " + (within ? "in" : "out");
  tag.textContent = within ? `core under ${CORE_BUDGET}ms` : `core over ${CORE_BUDGET}ms`;

  const entries = STAGE_ORDER.filter((s) => stages[s] !== undefined).map((s) => [s, stages[s]]);
  const max = Math.max(...entries.map(([, v]) => v), 1);
  $("bars").innerHTML = entries.map(([s, v]) => `
    <div class="bar ${EXTERNAL.has(s) ? "ext" : ""}">
      <span class="name">${STAGE_LABEL[s] || s}</span>
      <span class="track"><span class="fill" style="width:${Math.max(2, (v / max) * 100)}%"></span></span>
      <span class="ms">${v < 10 ? v.toFixed(1) : v.toFixed(0)}</span>
    </div>`).join("");
}

function renderCitations(cites) {
  $("citeCount").textContent = cites.length ? `${cites.length} passages` : "grounded sources";
  $("citesIdle").hidden = cites.length > 0;
  if (!cites.length) {
    $("cites").innerHTML = "";
    $("citesIdle").textContent = "No passages cited — this answer is not grounded in the corpus.";
    return;
  }
  $("cites").innerHTML = cites.map((c, i) => `
    <div class="cite" style="animation-delay:${(i * 45)}ms">
      <span class="wash"></span>
      <span class="cite-idx">${String(i + 1).padStart(2, "0")}</span>
      <div class="cite-inner">
        <div class="cite-head">
          <span class="mono">${esc(c.passage_id)} · ${esc(c.lang)}</span>
          <span class="score">${c.score.toFixed(3)}</span>
        </div>
        <div class="cite-body">${esc(c.text)}</div>
      </div>
    </div>`).join("");
}

function render(res, voiceMs = null) {
  const ans = $("answer");
  ans.className = "answer" + (res.refused ? " refused" : res.path === "general" ? " general" : "");
  ans.innerHTML = `<p>${esc(res.answer)}</p>`;

  const pb = $("pathBadge");
  pb.hidden = false;
  pb.dataset.p = res.path;
  pb.textContent = (res.path === "general" ? "general knowledge" : res.path)
    + (res.cache_similarity ? ` ${res.cache_similarity.toFixed(2)}` : "");

  const rb = $("reasonBadge");
  rb.hidden = !res.reason_code;
  if (res.reason_code) rb.textContent = res.reason_code;

  const tr = $("transcript");
  if (res.transcript) {
    tr.hidden = false;
    tr.innerHTML = `heard <b>${esc(res.transcript)}</b>${res.detected_lang ? ` · ${esc(res.detected_lang)}` : ""}`;
  } else tr.hidden = true;

  const cf = $("conf");
  cf.hidden = !!res.refused;
  if (!res.refused) {
    cf.innerHTML = `confidence
      <span class="track"><span class="fill" style="width:${(res.confidence * 100).toFixed(0)}%"></span></span>
      <span>${(res.confidence * 100).toFixed(0)}%</span>`;
  }

  renderTimings(res.timings || {}, voiceMs);
  renderCitations(res.citations || []);
}

function fail(msg) {
  $("answer").innerHTML = `<p class="err">${esc(msg)}</p>`;
}

async function askText(q, voiceMs = null) {
  if (!q.trim() || busy) return;
  setBusy(true);
  try {
    const r = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q, lang: $("lang").value || null,
        allow_general: !$("strict").checked,
      }),
    });
    if (!r.ok) throw new Error(`server returned ${r.status}`);
    render(await r.json(), voiceMs);
  } catch (e) {
    fail(e.message);
  } finally {
    setBusy(false);
  }
}

/* Resolve a spoken question, preferring work already done while you were talking. */
async function answerSpoken(text) {
  const q = text.trim();
  if (!q) return;
  $("q").value = q;

  const hit = spec.get(q);
  if (hit) {
    const res = await hit;
    // A speculative result only stands if it actually answered. A refusal there
    // just means retrieval alone was not enough, so fall through to the real
    // request which may use the LLM.
    if (res && !res.refused) {
      render(res, performance.now() - speechEndedAt);
      return;
    }
  }
  await askText(q, performance.now() - speechEndedAt);
}

function startLiveTranscript() {
  const Impl = window.SpeechRecognition || window.webkitSpeechRecognition;
  liveText = finalText = "";
  if (!Impl) return null;

  const r = new Impl();
  r.continuous = true;
  r.interimResults = true;
  r.lang = SPEECH_LANG[$("lang").value] || SPEECH_LANG.en;
  r.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const chunk = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText += chunk;
      else interim += chunk;
    }
    liveText = (finalText + interim).trim();
    $("q").value = liveText;
    $("q").classList.add("live");
    $("micHint").textContent = "listening — click the mic to send";
    scheduleSpeculation(liveText);
  };
  r.onend = () => { if (recorder) { try { r.start(); } catch { /* already restarting */ } } };
  r.onerror = (e) => { if (e.error === "not-allowed") $("micState").textContent = "Mic blocked"; };
  try { r.start(); } catch { return null; }
  return r;
}

function stopLiveTranscript() {
  if (!recog) return;
  recog.onend = null;
  try { recog.stop(); } catch { /* already stopped */ }
  recog = null;
}

async function startRec() {
  if (busy || recorder) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    const msg = e && e.name === "NotAllowedError"
      ? "Microphone permission denied. Click the padlock in the address bar, allow the mic, then reload."
      : e && e.name === "NotFoundError"
      ? "No microphone found. Check your input device."
      : "Microphone unavailable — needs HTTPS or localhost. The text box still works.";
    $("micState").textContent = "Mic blocked";
    fail(msg);
    return;
  }

  chunks = [];
  spec.clear();
  // Safari cannot emit webm and records silence if we insist on it, so take
  // whatever the browser will actually produce.
  const type = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"]
    .find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
  const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";

  recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    speechEndedAt = performance.now();
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(chunks, { type: type || "audio/webm" });
    const heard = liveText.trim();
    recorder = null;
    clearInterval(recTimer);
    stopLiveTranscript();
    $("mic").classList.remove("rec");
    window.setCursorLabel?.("");
    $("q").classList.remove("live");
    $("micState").textContent = "Click to speak";
    $("micHint").textContent = "click the mic, speak, click again";
    $("q").placeholder = "Ask anything…";

    if (heard) {
      // Answer from the local transcript immediately. Sarvam still runs, but as
      // the authoritative record rather than something the user waits behind.
      answerSpoken(heard);
      if (blob.size > 1200) verifyWithSarvam(blob, ext, heard);
      return;
    }
    if (blob.size > 1200) askVoice(blob, ext);
    else $("micState").textContent = "Didn't catch that — try again";
  };

  recorder.start();
  recog = startLiveTranscript();
  recStart = Date.now();
  $("mic").classList.add("rec");
  window.setCursorLabel?.("click to send");
  $("q").value = "";
  $("q").placeholder = recog ? "listening…" : "recording…";
  $("micHint").textContent = recog ? "speak now…" : "click again to stop";
  recTimer = setInterval(() => {
    $("micState").textContent = `Listening ${((Date.now() - recStart) / 1000).toFixed(1)}s`;
  }, 100);
}

/* Sarvam is the STT of record. It runs after the answer is already on screen and
 * only corrects the transcript line, so its ~400ms never sits in the user's path. */
async function verifyWithSarvam(blob, ext, heard) {
  try {
    const fd = new FormData();
    fd.append("file", blob, `audio.${ext}`);
    if ($("lang").value) fd.append("lang", $("lang").value);
    fd.append("transcribe_only", "true");
    const r = await fetch("/transcribe", { method: "POST", body: fd });
    if (!r.ok) return;
    const { transcript, lang, duration_ms } = await r.json();
    if (!transcript) return;
    const tr = $("transcript");
    tr.hidden = false;
    const same = transcript.trim().toLowerCase() === heard.trim().toLowerCase();
    tr.innerHTML = `heard <b>${esc(transcript)}</b>${lang ? ` · ${esc(lang)}` : ""}
      <span class="src">Sarvam ${duration_ms ? `· ${duration_ms.toFixed(0)}ms` : ""}${same ? " · matches live" : ""}</span>`;
  } catch { /* the answer is already shown; a failed verification changes nothing */ }
}

async function askVoice(blob, ext = "webm") {
  setBusy(true);
  $("micState").textContent = "Transcribing…";
  try {
    const fd = new FormData();
    fd.append("file", blob, `audio.${ext}`);
    if ($("lang").value) fd.append("lang", $("lang").value);
    fd.append("allow_general", String(!$("strict").checked));
    const r = await fetch("/ask-voice", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `server returned ${r.status}`);
    const res = await r.json();
    if (res.transcript) $("q").value = res.transcript;
    render(res);
  } catch (e) {
    fail(e.message);
  } finally {
    setBusy(false);
    $("micState").textContent = "Click to speak";
  }
}

function toggleRec() {
  if (recorder) recorder.stop();
  else startRec();
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); askText($("q").value); });
// Delegated rather than bound per element: the preset markup has been renamed
// twice (chip -> try-card -> pq) and each rename silently unbound every preset,
// because a querySelectorAll that matches nothing fails without an error.
// Delegation keys off the data attribute that actually matters instead.
document.addEventListener("click", (e) => {
  const el = e.target instanceof Element && e.target.closest("[data-q]");
  if (!el) return;
  // Presets carry the language their passage is indexed under; without it a
  // Gujarati question in an "Auto" box relies on script detection alone, which
  // is correct but slower to demo.
  if (el.dataset.lang !== undefined) $("lang").value = el.dataset.lang;
  $("q").value = el.dataset.q;
  askText(el.dataset.q);
});
$("mic").addEventListener("click", toggleRec);

health();


/* Custom cursor. The ring trails with easing so movement reads as weight, and a
   label appears while recording so the click target is never ambiguous. */
(() => {
  if (!window.matchMedia("(hover:hover) and (pointer:fine)").matches) return;
  const dot = $("cur"), ring = $("curRing"), label = $("curLabel");
  let x = -100, y = -100, rx = -100, ry = -100;

  addEventListener("mousemove", (e) => {
    x = e.clientX; y = e.clientY;
    dot.style.transform = `translate(${x}px,${y}px) translate(-50%,-50%)`;
    const hot = e.target instanceof Element &&
      e.target.closest("button,a,input,select,label,.cite");
    dot.style.width = dot.style.height = hot ? "6px" : "14px";
    ring.style.opacity = hot ? ".6" : ".3";
    ring.style.width = ring.style.height = hot ? "58px" : "40px";
  });

  (function loop() {
    rx += (x - rx) * 0.18; ry += (y - ry) * 0.18;
    ring.style.transform = `translate(${rx}px,${ry}px) translate(-50%,-50%)`;
    if (!label.hidden) label.style.transform = `translate(${rx + 18}px,${ry + 16}px)`;
    requestAnimationFrame(loop);
  })();

  window.setCursorLabel = (text) => {
    label.hidden = !text;
    if (text) label.textContent = text;
  };
})();
