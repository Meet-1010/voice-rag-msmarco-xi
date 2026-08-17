const $ = (id) => document.getElementById(id);
const CORE_BUDGET = 200;

// Order matters: the HUD reads as a pipeline, so keep it in execution order rather
// than sorting by duration.
const STAGE_ORDER = ["stt", "guard_input", "cache", "embed", "retrieve", "rerank",
                     "guard_relevance", "extract", "generate", "generate_general", "guard_grounding"];
const STAGE_LABEL = {
  stt: "STT", guard_input: "input guard", cache: "cache", embed: "embed",
  retrieve: "retrieve", rerank: "rerank", guard_relevance: "relevance",
  extract: "extract", generate: "generate", generate_general: "LLM (general)",
  guard_grounding: "grounding",
};
// Stages outside the 200ms core budget, drawn in a different colour so the
// distinction is visible rather than asserted in a footnote.
const EXTERNAL = new Set(["stt", "generate", "generate_general"]);

let recorder = null, chunks = [], busy = false;

async function health() {
  const el = $("health");
  try {
    const r = await fetch("/health");
    const h = await r.json();
    const n = (h.indexed_points || 0).toLocaleString();
    const langs = (h.manifest?.langs || []).join("/") || "–";
    const llm = Object.entries(h.providers || {}).filter(([, v]) => v.configured).map(([k]) => k);
    const mode = llm.length ? llm.join("+") : "extractive-only";
    el.innerHTML = `<span class="dot ok"></span><span>${n} chunks · ${langs} · ${mode}</span>`;
  } catch {
    el.innerHTML = `<span class="dot bad"></span><span>backend unreachable</span>`;
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

function renderTimings(t) {
  const stages = { ...(t.stages || {}) };
  if (t.stt_ms) stages.stt = t.stt_ms;

  $("coreMs").textContent = (t.core_ms ?? 0).toFixed(1);
  $("totalMs").textContent = (t.wall_ms ?? t.total_ms ?? 0).toFixed(1);
  $("sttMs").textContent = t.stt_ms ? t.stt_ms.toFixed(0) : "–";

  const tag = $("budgetTag");
  const within = (t.core_ms ?? 0) <= CORE_BUDGET;
  tag.hidden = false;
  tag.className = "budget-tag " + (within ? "in" : "out");
  tag.textContent = within ? `core under ${CORE_BUDGET}ms` : `core over ${CORE_BUDGET}ms`;

  const entries = STAGE_ORDER.filter((s) => stages[s] !== undefined)
    .map((s) => [s, stages[s]]);
  const max = Math.max(...entries.map(([, v]) => v), 1);
  $("bars").innerHTML = entries.map(([s, v]) => `
    <div class="bar ${EXTERNAL.has(s) ? "ext" : ""}">
      <span class="name">${STAGE_LABEL[s] || s}</span>
      <span class="track"><span class="fill" style="width:${Math.max(2, (v / max) * 100)}%"></span></span>
      <span class="ms">${v < 10 ? v.toFixed(1) : v.toFixed(0)}</span>
    </div>`).join("");
}

function renderCitations(cites) {
  $("citeCount").textContent = cites.length ? `${cites.length} passage${cites.length > 1 ? "s" : ""}` : "";
  if (!cites.length) {
    $("cites").innerHTML = `<p class="idle">No passages cited.</p>`;
    return;
  }
  $("cites").innerHTML = cites.map((c, i) => `
    <div class="cite">
      <div class="cite-head">
        <span>[${i + 1}] ${esc(c.passage_id)} · ${esc(c.lang)}</span>
        <span class="score">${c.score.toFixed(3)}</span>
      </div>
      <div class="cite-body">${esc(c.text)}</div>
    </div>`).join("");
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (m) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[m]));

function render(res) {
  const ans = $("answer");
  ans.className = "answer" + (res.refused ? " refused" : res.path === "general" ? " general" : "");
  ans.innerHTML = `<p>${esc(res.answer)}</p>`;

  const pb = $("pathBadge");
  pb.hidden = false;
  pb.dataset.p = res.path;
  pb.textContent = (res.path === "general" ? "general knowledge" : res.path)
    + (res.cache_similarity ? ` ${res.cache_similarity.toFixed(2)}` : "")
    + (res.provider && (res.path === "generative" || res.path === "general") ? ` · ${res.provider}` : "");

  const rb = $("reasonBadge");
  rb.hidden = !res.reason_code;
  if (res.reason_code) rb.textContent = res.reason_code;

  const tr = $("transcript");
  if (res.transcript) {
    tr.hidden = false;
    tr.innerHTML = `heard: <b>${esc(res.transcript)}</b>${res.detected_lang ? ` · ${esc(res.detected_lang)}` : ""}`;
  } else tr.hidden = true;

  const cf = $("conf");
  if (res.refused) cf.hidden = true;
  else {
    cf.hidden = false;
    cf.innerHTML = `confidence
      <span class="track"><span class="fill" style="width:${(res.confidence * 100).toFixed(0)}%"></span></span>
      <span>${(res.confidence * 100).toFixed(0)}%</span>`;
  }

  renderTimings(res.timings || {});
  renderCitations(res.citations || []);
}

function fail(msg) {
  $("answer").innerHTML = `<p class="err">${esc(msg)}</p>`;
}

async function askText(q) {
  if (!q.trim() || busy) return;
  setBusy(true);
  try {
    const r = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q,
        lang: $("lang").value || null,
        allow_general: !$("strict").checked,
      }),
    });
    if (!r.ok) throw new Error(`server returned ${r.status}`);
    render(await r.json());
  } catch (e) {
    fail(e.message);
  } finally {
    setBusy(false);
  }
}

async function askVoice(blob, ext = "webm", heard = "") {
  setBusy(true);
  $("micState").textContent = "Transcribing…";
  try {
    const fd = new FormData();
    fd.append("file", blob, `audio.${ext}`);
    if ($("lang").value) fd.append("lang", $("lang").value);
    fd.append("allow_general", String(!$("strict").checked));
    const r = await fetch("/ask-voice", { method: "POST", body: fd });
    if (!r.ok) {
      const body = await r.json().catch(() => ({}));
      throw new Error(body.detail || `server returned ${r.status}`);
    }
    const res = await r.json();
    if (!res.transcript && heard) {
      // Sarvam returned nothing usable but the browser did hear speech; asking
      // with what we heard beats telling the user their microphone is broken.
      render(await (await fetch("/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: heard, lang: $("lang").value || null,
                               allow_general: !$("strict").checked }),
      })).json());
      $("q").value = heard;
      return;
    }
    if (res.transcript) $("q").value = res.transcript;
    render(res);
  } catch (e) {
    if (heard) {
      // askText bails out while busy is set, so clear it before handing over.
      setBusy(false);
      $("q").value = heard;
      await askText(heard);
      return;
    }
    fail(e.message);
  } finally {
    setBusy(false);
    $("micState").textContent = "Click to speak";
    $("q").placeholder = "Ask anything…";
  }
}

let recTimer = null, recStart = 0, recog = null, liveText = "", finalText = "";

// Browser speech recognition runs locally and emits interim results while you
// are still talking, so words appear as they are spoken. Sarvam still produces
// the transcript we actually query with - this is the live feedback layer, and
// the fallback if Sarvam is unreachable.
const SPEECH_LANG = { en: "en-IN", hi: "hi-IN", gu: "gu-IN" };

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
    $("micHint").textContent = liveText ? "listening — click the mic to send" : "speak now…";
  };
  // Recognition dies on its own after a pause; restart while still recording.
  r.onend = () => { if (recorder) { try { r.start(); } catch { /* already starting */ } } };
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
    // Distinguish the three failure modes, because "mic not working" was
    // impossible to debug when they all produced the same message.
    const msg = e && e.name === "NotAllowedError"
      ? "Microphone permission denied. Click the padlock in the address bar, allow the mic, then reload."
      : e && e.name === "NotFoundError"
      ? "No microphone found. Check your input device."
      : "Microphone unavailable — this needs HTTPS or localhost. The text box still works.";
    $("micState").textContent = "Mic blocked";
    fail(msg);
    return;
  }

  chunks = [];
  // Let the browser pick a container it can actually produce. Safari does not
  // emit webm and records silence if we insist on it. Sarvam accepts all of
  // these, so the extension follows whatever we end up with.
  const type = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"]
    .find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
  const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";

  recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(chunks, { type: type || "audio/webm" });
    const held = Date.now() - recStart;
    const heard = liveText.trim();
    recorder = null;
    clearInterval(recTimer);
    stopLiveTranscript();
    $("mic").classList.remove("rec");
    $("q").classList.remove("live");
    $("micHint").textContent = "click the mic, speak, click again";

    if (held < 400 || blob.size < 1200) {
      // Only complain if we genuinely captured nothing. If live recognition
      // heard words, the recording was real and we can just use them.
      if (heard) { $("micState").textContent = "Click to speak"; askText(heard); return; }
      $("micState").textContent = "Too short — speak for a second";
      return;
    }
    askVoice(blob, ext, heard);
  };
  recorder.start();
  recog = startLiveTranscript();
  recStart = Date.now();
  $("mic").classList.add("rec");
  $("q").value = "";
  $("q").placeholder = recog ? "listening…" : "recording… (live transcript unavailable)";
  $("micHint").textContent = recog ? "speak now…" : "click again to stop";
  recTimer = setInterval(() => {
    $("micState").textContent = `Listening… ${((Date.now() - recStart) / 1000).toFixed(1)}s`;
  }, 100);
}

function stopRec() {
  if (!recorder) return;
  recorder.stop();
}

function toggleRec() {
  if (recorder) stopRec();
  else startRec();
}

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  askText($("q").value);
});
document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => {
    $("q").value = c.dataset.q;
    askText(c.dataset.q);
  }));

// Click to start, click to stop. The previous hold-to-talk binding stopped on
// mouseup and on mouseleave, so an ordinary click recorded ~50ms and a small
// drag off the button killed the recording mid-sentence - it read as "the mic
// does not work".
$("mic").addEventListener("click", toggleRec);

health();
