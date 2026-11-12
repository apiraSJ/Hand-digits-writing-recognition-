const API_BASE = location.protocol === "file:" ? "http://localhost:8000" : "";
const HISTORY_LIMIT = 30;

const BAR_COLORS = [
  "#6366f1", "#f97316", "#22c55e", "#ef4444", "#a855f7",
  "#eab308", "#ec4899", "#64748b", "#14b8a6", "#0ea5e9",
];

const $ = (id) => document.getElementById(id);

const els = {
  cameraStatus: $("cameraStatus"),
  apiStatus: $("apiStatus"),
  fpsChip: $("fpsChip"),
  detectionCount: $("detectionCount"),
  overlayState: $("overlayState"),
  stage: $("stage"),
  video: $("video"),
  overlay: $("overlay"),
  cameraEmpty: $("cameraEmpty"),
  noDetectionHint: $("noDetectionHint"),
  roiHint: $("roiHint"),
  startCamBtn: $("startCamBtn"),
  stopCamBtn: $("stopCamBtn"),
  roiBtn: $("roiBtn"),
  resetRoiBtn: $("resetRoiBtn"),
  overlayToggle: $("overlayToggle"),
  cameraAlert: $("cameraAlert"),
  binaryImageA: $("binaryImageA"),
  binaryImageB: $("binaryImageB"),
  binaryEmpty: $("binaryEmpty"),
  binaryBadge: $("binaryBadge"),
  thresholdValue: $("thresholdValue"),
  frameId: $("frameId"),
  predictionValue: $("predictionValue"),
  latencyValue: $("latencyValue"),
  processingValue: $("processingValue"),
  confidenceValue: $("confidenceValue"),
  minArea: $("minArea"),
  minAreaVal: $("minAreaVal"),
  maxArea: $("maxArea"),
  maxAreaVal: $("maxAreaVal"),
  mergeGap: $("mergeGap"),
  mergeGapVal: $("mergeGapVal"),
  padding: $("padding"),
  paddingVal: $("paddingVal"),
  maxDigits: $("maxDigits"),
  maxDigitsVal: $("maxDigitsVal"),
  minConfidence: $("minConfidence"),
  minConfidenceVal: $("minConfidenceVal"),
  minAspect: $("minAspect"),
  minAspectVal: $("minAspectVal"),
  maxAspect: $("maxAspect"),
  maxAspectVal: $("maxAspectVal"),
  inferenceFps: $("inferenceFps"),
  useTta: $("useTta"),
  settingsToggle: $("settingsToggle"),
  settingsBody: $("settingsBody"),
  digitSelector: $("digitSelector"),
  chartCaption: $("chartCaption"),
  chartNote: $("chartNote"),
  chart: $("chart"),
  chartEmpty: $("chartEmpty"),
  historyCount: $("historyCount"),
  historyList: $("historyList"),
  historySpark: $("historySpark"),
  uploadBadge: $("uploadBadge"),
  dropzone: $("dropzone"),
  fileInput: $("fileInput"),
  uploadPreview: $("uploadPreview"),
  analyzeUploadBtn: $("analyzeUploadBtn"),
  uploadAlert: $("uploadAlert"),
  uploadResult: $("uploadResult"),
  uploadResultNumber: $("uploadResultNumber"),
  annotatedImage: $("annotatedImage"),
  binaryImageStatic: $("binaryImageStatic"),
  digitsContainer: $("digitsContainer"),
  toast: $("toast"),
  cameraSelect: $("cameraSelect"),
};

const state = {
  stream: null,
  running: false,
  inFlight: false,
  inferenceTimer: null,
  abortController: null,
  frameId: 0,
  fpsTarget: 6,
  currentFps: 0,
  fpsSamples: [],
  selectedIndex: 0,
  liveResponse: null,
  liveMeta: null,
  chartData: null,
  chartMode: "live",
  history: [],
  consecutiveErrors: 0,
  overlayEnabled: true,
  roiMode: false,
  roi: null,
  dragStart: null,
  dragCurrent: null,
  uploadFile: null,
  uploadPreviewUrl: null,
  binaryFront: 0,
  chartBars: [],
  captureCanvas: document.createElement("canvas"),
  settingsVisible: true,
  toastTimer: null,
  availableCameras: [],
  currentDeviceId: null,
};

/* ── Small helpers ───────────────────────────────────────── */
function setChip(el, text, className) {
  el.textContent = text;
  el.className = className;
}

function setApiStatus(kind, text) {
  setChip(els.apiStatus, text, "status-chip " + kind);
}

function setCameraStatus(kind, text) {
  setChip(els.cameraStatus, text, "status-chip " + kind);
}

function showToast(message, isError = false) {
  els.toast.textContent = message;
  els.toast.className = "toast" + (isError ? " error" : "");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => els.toast.classList.add("hidden"), 4500);
}

function showCameraAlert(message) {
  els.cameraAlert.textContent = message;
  els.cameraAlert.className = "alert";
}

function hideCameraAlert() {
  els.cameraAlert.textContent = "";
  els.cameraAlert.className = "alert hidden";
}

function showUploadAlert(message, info = false) {
  els.uploadAlert.textContent = message;
  els.uploadAlert.className = "alert" + (info ? " info" : "");
}

function hideUploadAlert() {
  els.uploadAlert.textContent = "";
  els.uploadAlert.className = "alert hidden";
}

async function responseError(res) {
  const text = await res.text();
  try {
    const body = JSON.parse(text);
    return body.detail || text || res.statusText;
  } catch {
    return text || res.statusText;
  }
}

/* ── Health ──────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const res = await fetch(API_BASE + "/health", { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    if (data.model_loaded) {
      setApiStatus("ready", "API READY");
    } else {
      setApiStatus("error", "MODEL ERROR");
    }
  } catch {
    setApiStatus("error", "API OFFLINE");
  }
}

/* ── Settings controls ───────────────────────────────────── */
function bindRange(input, output, formatter = (v) => v) {
  const update = () => { output.textContent = formatter(input.value); };
  input.addEventListener("input", update);
  update();
}

bindRange(els.minArea, els.minAreaVal);
bindRange(els.maxArea, els.maxAreaVal);
bindRange(els.mergeGap, els.mergeGapVal);
bindRange(els.padding, els.paddingVal);
bindRange(els.maxDigits, els.maxDigitsVal);
bindRange(els.minConfidence, els.minConfidenceVal, (v) => v + "%");
bindRange(els.minAspect, els.minAspectVal, (v) => Number(v).toFixed(2));
bindRange(els.maxAspect, els.maxAspectVal, (v) => Number(v).toFixed(1));

els.inferenceFps.addEventListener("change", () => {
  state.fpsTarget = Number(els.inferenceFps.value) || 6;
});

els.settingsToggle.addEventListener("click", () => {
  state.settingsVisible = !state.settingsVisible;
  els.settingsBody.classList.toggle("hidden", !state.settingsVisible);
  els.settingsToggle.textContent = state.settingsVisible ? "Hide" : "Show";
});

els.overlayToggle.addEventListener("change", () => {
  state.overlayEnabled = els.overlayToggle.checked;
  els.overlayState.textContent = state.overlayEnabled ? "Overlay enabled" : "Overlay hidden";
  if (!state.overlayEnabled) clearOverlay();
});

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const panel = $("panel-" + btn.dataset.tab);
    if (panel) panel.classList.add("active");
  });
});

function getParams() {
  const minAspect = Number(els.minAspect.value);
  const maxAspect = Math.max(Number(els.maxAspect.value), minAspect + 0.05);
  return {
    min_area: Number(els.minArea.value),
    max_area: Number(els.maxArea.value),
    merge_gap: Number(els.mergeGap.value),
    padding: Number(els.padding.value),
    max_digits: Number(els.maxDigits.value),
    min_confidence: Number(els.minConfidence.value),
    use_tta: els.useTta.checked,
    min_aspect: minAspect,
    max_aspect: maxAspect,
  };
}

/* ── Chart ───────────────────────────────────────────────── */
function initChart() {
  els.chart.innerHTML = "";
  state.chartBars = [];
  for (let i = 0; i < 10; i += 1) {
    const bar = document.createElement("div");
    bar.className = "bar";

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    track.appendChild(fill);

    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = "0%";

    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = String(i);

    bar.append(track, value, label);
    els.chart.appendChild(bar);
    state.chartBars.push({ bar, fill, value });
  }
}

function resetChartBars() {
  state.chartBars.forEach(({ bar, fill, value }) => {
    bar.classList.remove("active");
    fill.style.height = "0%";
    value.textContent = "0%";
  });
}

function updateChart(data) {
  const detection = data && data.digits && data.digits[state.selectedIndex];
  if (!detection) {
    resetChartBars();
    els.chartEmpty.classList.remove("hidden");
    els.chartCaption.textContent = "Waiting for a detection...";
    return;
  }

  els.chartEmpty.classList.add("hidden");
  els.chartCaption.textContent =
    `Selected digit ${detection.digit} · ${(detection.confidence * 100).toFixed(1)}% confidence`;

  const probs = detection.probabilities || [];
  state.chartBars.forEach(({ bar, fill, value }, i) => {
    const p = Math.max(0, Math.min(1, Number(probs[i] || 0)));
    fill.style.height = `${(p * 100).toFixed(1)}%`;
    value.textContent = `${(p * 100).toFixed(0)}%`;
    bar.classList.toggle("active", i === detection.digit);
  });
}

function updateDigitSelector(data) {
  const digits = (data && data.digits) || [];
  if (!digits.length) {
    els.digitSelector.innerHTML = "";
    return;
  }

  if (els.digitSelector.children.length !== digits.length) {
    els.digitSelector.innerHTML = "";
    digits.forEach((detection, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "digit-chip";
      button.dataset.index = String(index);
      button.addEventListener("click", () => {
        state.selectedIndex = index;
        state.chartMode = data === state.liveResponse ? "live" : state.chartMode;
        updateDigitSelector(state.chartData);
        updateChart(state.chartData);
      });
      els.digitSelector.appendChild(button);
    });
  }

  Array.from(els.digitSelector.children).forEach((button, index) => {
    const detection = digits[index];
    button.classList.toggle("active", index === state.selectedIndex);
    button.textContent =
      `#${index + 1} · ${detection.digit} · ${(detection.confidence * 100).toFixed(0)}%`;
  });
}

/* ── History / sparkline ─────────────────────────────────── */
function addHistory(item) {
  state.history.push(item);
  if (state.history.length > HISTORY_LIMIT) state.history.shift();
  els.historyCount.textContent = `${state.history.length} / ${HISTORY_LIMIT}`;
  drawHistory();
}

function maybeRecordHistory(item) {
  if (!item.full_number) return;
  const last = state.history[state.history.length - 1];
  const now = Date.now();
  if (last && last.full_number === item.full_number && now - (last.ts || 0) < 5000) {
    return;
  }
  addHistory({ ...item, ts: now });
}

function drawHistory() {
  els.historyList.innerHTML = "";
  if (!state.history.length) {
    const empty = document.createElement("span");
    empty.className = "muted";
    empty.textContent = "No history";
    els.historyList.appendChild(empty);
    drawSparkline();
    return;
  }

  state.history.forEach((item) => {
    const chip = document.createElement("div");
    chip.className = "history-chip" + (item.source === "upload" ? " upload" : "");
    const value = document.createElement("b");
    value.textContent = item.full_number || "-";
    chip.appendChild(value);
    chip.appendChild(document.createTextNode(` ${Math.round((item.confidence || 0) * 100)}%`));
    chip.appendChild(document.createElement("br"));
    chip.appendChild(document.createTextNode(
      `${item.source === "upload" ? "UPLOAD" : "FRAME"} #${item.frame_id ?? "-"}`
    ));
    els.historyList.appendChild(chip);
  });

  els.historyList.scrollLeft = els.historyList.scrollWidth;
  drawSparkline();
}

function drawSparkline() {
  const canvas = els.historySpark;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;

  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(50 * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  const items = state.history.slice(-HISTORY_LIMIT);
  if (items.length < 2) return;

  const pad = 5 * dpr;
  const plotW = width - pad * 2;
  const plotH = height - pad * 2;
  const step = plotW / Math.max(1, items.length - 1);

  const points = items.map((item, index) => ({
    x: pad + index * step,
    y: pad + plotH * (1 - Math.max(0, Math.min(1, item.confidence || 0))),
  }));

  const gradient = ctx.createLinearGradient(0, pad, 0, height - pad);
  gradient.addColorStop(0, "rgba(34, 211, 238, .35)");
  gradient.addColorStop(1, "rgba(34, 211, 238, 0)");

  ctx.beginPath();
  ctx.moveTo(points[0].x, height - pad);
  points.forEach((point) => ctx.lineTo(point.x, point.y));
  ctx.lineTo(points[points.length - 1].x, height - pad);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) ctx.moveTo(point.x, point.y);
    else ctx.lineTo(point.x, point.y);
  });
  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = Math.max(1, 1.5 * dpr);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.stroke();

  const last = points[points.length - 1];
  ctx.beginPath();
  ctx.arc(last.x, last.y, Math.max(2, 2.2 * dpr), 0, Math.PI * 2);
  ctx.fillStyle = "#67e8f9";
  ctx.fill();
}

/* ── Binary preview ──────────────────────────────────────── */
function updateBinary(data) {
  if (!data || !data.binary_image) return;

  const front = state.binaryFront === 0 ? els.binaryImageA : els.binaryImageB;
  const incoming = state.binaryFront === 0 ? els.binaryImageB : els.binaryImageA;

  incoming.onload = () => {
    incoming.classList.remove("hidden");
    front.classList.add("hidden");
    state.binaryFront = 1 - state.binaryFront;
    incoming.onload = null;
  };
  incoming.src = data.binary_image;

  els.binaryEmpty.classList.add("hidden");
  els.binaryBadge.textContent = "LIVE";
  els.thresholdValue.textContent =
    data.threshold && data.threshold.value != null
      ? Math.round(data.threshold.value)
      : "-";
  els.frameId.textContent = `#${data.frame_id ?? "-"}`;
}

/* ── Overlay drawing ─────────────────────────────────────── */
function clearOverlay() {
  const canvas = els.overlay;
  if (!canvas.width || !canvas.height) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function pathRoundRect(ctx, x, y, w, h, r) {
  const radius = Math.min(r, w / 2, h / 2);
  if (ctx.roundRect) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, radius);
    return;
  }
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + w - radius, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
  ctx.lineTo(x + w, y + h - radius);
  ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
  ctx.lineTo(x + radius, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function confidenceColor(confidence) {
  if (confidence >= 0.9) return "#22d3ee";
  if (confidence >= 0.7) return "#f59e0b";
  return "#ef4444";
}

function mapBox(box, meta, data) {
  const sentW = (data && data.image_size && data.image_size.width) || meta.sentWidth;
  const sentH = (data && data.image_size && data.image_size.height) || meta.sentHeight;
  const scaleX = meta.roi.w / Math.max(1, sentW);
  const scaleY = meta.roi.h / Math.max(1, sentH);
  return {
    x: meta.roi.x + box.x * scaleX,
    y: meta.roi.y + box.y * scaleY,
    w: box.w * scaleX,
    h: box.h * scaleY,
  };
}

function drawDetection(ctx, box, digit, confidence, selected) {
  const canvas = ctx.canvas;
  const x = Math.max(0, box.x);
  const y = Math.max(0, box.y);
  const w = Math.min(canvas.width - x, box.w);
  const h = Math.min(canvas.height - y, box.h);
  if (w <= 2 || h <= 2) return;

  const color = confidenceColor(confidence);
  const line = Math.max(2, Math.round(canvas.width / 520));
  const corner = Math.max(8, Math.min(28, Math.min(w, h) * 0.18));

  ctx.save();
  ctx.lineWidth = line;
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.28;
  ctx.strokeRect(x, y, w, h);

  ctx.globalAlpha = 1;
  if (selected) {
    ctx.shadowColor = color;
    ctx.shadowBlur = line * 6;
  }
  ctx.beginPath();
  ctx.moveTo(x, y + corner);
  ctx.lineTo(x, y);
  ctx.lineTo(x + corner, y);
  ctx.moveTo(x + w - corner, y);
  ctx.lineTo(x + w, y);
  ctx.lineTo(x + w, y + corner);
  ctx.moveTo(x + w, y + h - corner);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x + w - corner, y + h);
  ctx.moveTo(x + corner, y + h);
  ctx.lineTo(x, y + h);
  ctx.lineTo(x, y + h - corner);
  ctx.stroke();
  ctx.shadowBlur = 0;

  const label = `${digit} · ${(confidence * 100).toFixed(0)}%`;
  const fontSize = Math.max(13, Math.round(canvas.width / 48));
  ctx.font = `700 ${fontSize}px "Cascadia Mono", Consolas, monospace`;
  const textW = ctx.measureText(label).width;
  const padX = fontSize * 0.48;
  const padY = fontSize * 0.28;
  const labelW = textW + padX * 2;
  const labelH = fontSize + padY * 2;
  const lx = Math.max(2, Math.min(x, canvas.width - labelW - 2));
  let ly = y - labelH - 4;
  if (ly < 2) ly = y + 4;

  ctx.fillStyle = color;
  pathRoundRect(ctx, lx, ly, labelW, labelH, 5);
  ctx.fill();
  ctx.fillStyle = "#04121a";
  ctx.textBaseline = "middle";
  ctx.fillText(label, lx + padX, ly + labelH / 2 + 1);
  ctx.restore();
}

function drawRoi(ctx, roi, preview = false, label = "ROI") {
  const canvas = ctx.canvas;
  const x = Math.max(0, Math.min(canvas.width, roi.x));
  const y = Math.max(0, Math.min(canvas.height, roi.y));
  const w = Math.max(0, Math.min(canvas.width - x, roi.w));
  const h = Math.max(0, Math.min(canvas.height - y, roi.h));
  if (w <= 1 || h <= 1) return;

  ctx.save();
  if (!preview) {
    ctx.fillStyle = "rgba(2, 6, 12, .48)";
    ctx.fillRect(0, 0, canvas.width, y);
    ctx.fillRect(0, y + h, canvas.width, canvas.height - y - h);
    ctx.fillRect(0, y, x, h);
    ctx.fillRect(x + w, y, canvas.width - x - w, h);
  }
  ctx.setLineDash([8, 6]);
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 500));
  ctx.strokeStyle = preview ? "#67e8f9" : "#22d3ee";
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);

  const fontSize = Math.max(12, Math.round(canvas.width / 55));
  ctx.font = `700 ${fontSize}px "Cascadia Mono", Consolas, monospace`;
  const labelW = ctx.measureText(label).width + 14;
  ctx.fillStyle = "#22d3ee";
  pathRoundRect(ctx, x, Math.max(2, y - fontSize - 8), labelW, fontSize + 8, 4);
  ctx.fill();
  ctx.fillStyle = "#04121a";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + 7, Math.max(2, y - fontSize - 8) + (fontSize + 8) / 2 + 1);
  ctx.restore();
}

function drawOverlayLoop() {
  requestAnimationFrame(drawOverlayLoop);

  const video = els.video;
  const canvas = els.overlay;
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  if (!state.running || !vw || !vh) {
    clearOverlay();
    return;
  }

  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width = vw;
    canvas.height = vh;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (state.roi) {
    drawRoi(ctx, state.roi, false, "CUSTOM ROI");
  } else {
    drawRoi(ctx, centeredRoi(vw, vh), false, "CENTER 40%");
  }
  if (state.dragStart && state.dragCurrent) {
    drawRoi(ctx, normalizeRect(state.dragStart, state.dragCurrent), true);
  }

  if (!state.overlayEnabled || !state.liveResponse || !state.liveMeta) return;

  state.liveResponse.digits.forEach((detection, index) => {
    const box = mapBox(detection.box, state.liveMeta, state.liveResponse);
    drawDetection(
      ctx,
      box,
      detection.digit,
      detection.confidence,
      index === state.selectedIndex && state.chartData === state.liveResponse
    );
  });
}

/* ── ROI / pointer interaction ───────────────────────────── */
function eventPoint(event) {
  const canvas = els.overlay;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !rect.height) return { x: 0, y: 0 };
  return {
    x: (event.clientX - rect.left) * (canvas.width / rect.width),
    y: (event.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function normalizeRect(a, b) {
  const canvas = els.overlay;
  const x1 = Math.max(0, Math.min(a.x, b.x));
  const y1 = Math.max(0, Math.min(a.y, b.y));
  const x2 = Math.min(canvas.width, Math.max(a.x, b.x));
  const y2 = Math.min(canvas.height, Math.max(a.y, b.y));
  return { x: x1, y: y1, w: Math.max(0, x2 - x1), h: Math.max(0, y2 - y1) };
}

function centeredRoi(width, height) {
  const w = width * 0.4;
  const h = height * 0.4;
  return { x: (width - w) / 2, y: (height - h) / 2, w, h };
}

function selectDetectionAt(point) {
  if (!state.liveResponse || !state.liveMeta) return;
  const digits = state.liveResponse.digits;
  for (let i = digits.length - 1; i >= 0; i -= 1) {
    const box = mapBox(digits[i].box, state.liveMeta, state.liveResponse);
    if (
      point.x >= box.x && point.x <= box.x + box.w &&
      point.y >= box.y && point.y <= box.y + box.h
    ) {
      state.selectedIndex = i;
      state.chartMode = "live";
      state.chartData = state.liveResponse;
      updateDigitSelector(state.chartData);
      updateChart(state.chartData);
      return;
    }
  }
}

els.overlay.addEventListener("pointerdown", (event) => {
  if (!state.running) return;
  if (state.roiMode) {
    state.dragStart = eventPoint(event);
    state.dragCurrent = state.dragStart;
    els.overlay.setPointerCapture(event.pointerId);
    event.preventDefault();
  } else {
    selectDetectionAt(eventPoint(event));
  }
});

els.overlay.addEventListener("pointermove", (event) => {
  if (!state.roiMode || !state.dragStart) return;
  state.dragCurrent = eventPoint(event);
});

els.overlay.addEventListener("pointerup", (event) => {
  if (!state.roiMode || !state.dragStart) return;
  const rect = normalizeRect(state.dragStart, eventPoint(event));
  state.dragStart = null;
  state.dragCurrent = null;
  if (rect.w >= 24 && rect.h >= 24) {
    state.roi = rect;
    state.liveResponse = null;
    state.liveMeta = null;
    state.roiMode = false;
    els.stage.classList.remove("roi-mode");
    els.resetRoiBtn.disabled = false;
    els.roiHint.classList.add("hidden");
    els.roiBtn.textContent = "ROI: CUSTOM";
  }
  event.preventDefault();
});

els.overlay.addEventListener("pointercancel", () => {
  state.dragStart = null;
  state.dragCurrent = null;
});

els.roiBtn.addEventListener("click", () => {
  state.roiMode = !state.roiMode;
  els.stage.classList.toggle("roi-mode", state.roiMode);
  els.roiBtn.textContent = state.roiMode
    ? "ROI: SELECT"
    : state.roi ? "ROI: CUSTOM" : "ROI: CENTER";
  els.roiHint.classList.toggle("hidden", !state.roiMode || Boolean(state.roi));
  if (!state.roiMode) {
    state.dragStart = null;
    state.dragCurrent = null;
  }
});

els.resetRoiBtn.addEventListener("click", () => {
  state.roi = null;
  state.roiMode = false;
  state.dragStart = null;
  state.dragCurrent = null;
  state.liveResponse = null;
  state.liveMeta = null;
  els.stage.classList.remove("roi-mode");
  els.roiBtn.textContent = "ROI: CENTER";
  els.roiHint.classList.add("hidden");
  els.resetRoiBtn.disabled = true;
  clearOverlay();
});

/* ── Live inference ──────────────────────────────────────── */
function captureFrame() {
  return new Promise((resolve) => {
    const video = els.video;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) {
      resolve(null);
      return;
    }

    const roi = state.roi || centeredRoi(vw, vh);
    const sx = Math.max(0, Math.min(vw - 1, roi.x));
    const sy = Math.max(0, Math.min(vh - 1, roi.y));
    const sw = Math.max(1, Math.min(vw - sx, roi.w));
    const sh = Math.max(1, Math.min(vh - sy, roi.h));

    const maxWidth = 640;
    const scale = Math.min(1, maxWidth / sw);
    const tw = Math.max(1, Math.round(sw * scale));
    const th = Math.max(1, Math.round(sh * scale));

    const canvas = state.captureCanvas;
    canvas.width = tw;
    canvas.height = th;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, tw, th);

    canvas.toBlob(
      (blob) => {
        if (!blob) {
          resolve(null);
          return;
        }
        resolve({
          blob,
          meta: {
            roi: { x: sx, y: sy, w: sw, h: sh },
            sentWidth: tw,
            sentHeight: th,
          },
        });
      },
      "image/jpeg",
      0.82
    );
  });
}

function scheduleNextInference(delay = 0) {
  clearTimeout(state.inferenceTimer);
  if (!state.running) return;
  state.inferenceTimer = setTimeout(runInference, Math.max(0, delay));
}

async function runInference() {
  if (!state.running || state.inFlight) return;
  if (document.hidden) {
    scheduleNextInference(300);
    return;
  }

  state.inFlight = true;
  const started = performance.now();
  const frame = await captureFrame();
  if (!frame) {
    state.inFlight = false;
    scheduleNextInference(250);
    return;
  }

  const controller = new AbortController();
  state.abortController = controller;
  const form = new FormData();
  form.append("file", frame.blob, "frame.jpg");
  form.append("frame_id", String(state.frameId));
  state.frameId += 1;

  const params = getParams();
  Object.entries(params).forEach(([key, value]) => form.append(key, String(value)));
  form.append("include_annotated", "false");
  form.append("include_canonical", "false");
  form.append("include_gray", "false");

  try {
    const res = await fetch(API_BASE + "/predict", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(await responseError(res));
    const data = await res.json();
    handleLiveResult(data, frame.meta);
    setApiStatus("ready", "API READY");
  } catch (error) {
    if (error.name !== "AbortError") {
      state.consecutiveErrors += 1;
      setApiStatus("error", "API ERROR");
      showCameraAlert("API error: " + error.message);
      if (state.consecutiveErrors >= 5) {
        stopCamera();
        showCameraAlert("Stopped after 5 consecutive API errors. Check the server console.");
      }
    }
  } finally {
    state.inFlight = false;
    state.abortController = null;
    if (state.running) {
      const elapsed = performance.now() - started;
      scheduleNextInference(Math.max(0, 1000 / state.fpsTarget - elapsed));
    }
  }
}

function handleLiveResult(data, meta) {
  state.liveResponse = data;
  state.liveMeta = meta;
  state.consecutiveErrors = 0;
  hideCameraAlert();

  if (!data.digits.length) {
    state.selectedIndex = 0;
  } else if (state.selectedIndex >= data.digits.length) {
    state.selectedIndex = 0;
  }

  recordFps();

  els.predictionValue.textContent = data.full_number || "-";
  els.latencyValue.textContent = `${Math.round(data.processing_time_ms || 0)} ms`;
  els.confidenceValue.textContent = data.digits.length
    ? `${(data.average_confidence * 100).toFixed(1)}%`
    : "-";
  els.detectionCount.textContent =
    `${data.digit_count} detection${data.digit_count === 1 ? "" : "s"}`;
  els.noDetectionHint.classList.toggle("hidden", data.digit_count > 0);

  updateBinary(data);

  if (state.chartMode === "live") {
    state.chartData = data;
    updateDigitSelector(data);
    updateChart(data);
  }

  maybeRecordHistory({
    frame_id: data.frame_id,
    full_number: data.full_number,
    confidence: data.average_confidence,
    latency: data.processing_time_ms,
    source: "camera",
  });
}

function recordFps() {
  const now = performance.now();
  state.fpsSamples.push(now);
  while (state.fpsSamples.length > 2 && now - state.fpsSamples[0] > 4000) {
    state.fpsSamples.shift();
  }
  const span = state.fpsSamples[state.fpsSamples.length - 1] - state.fpsSamples[0];
  state.currentFps = span > 0
    ? ((state.fpsSamples.length - 1) * 1000) / span
    : 0;
  const text = `${state.currentFps.toFixed(1)} FPS`;
  els.fpsChip.textContent = text;
  els.processingValue.textContent = text;
}

/* ── Camera selection ─────────────────────────────────────── */
async function populateCameraSelect() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter((d) => d.kind === "videoinput");
    state.availableCameras = videoDevices;

    const select = els.cameraSelect;
    const currentValue = select.value;
    select.innerHTML = '<option value="">-- Auto (Environment) --</option>';
    videoDevices.forEach((device) => {
      const opt = document.createElement("option");
      opt.value = device.deviceId;
      opt.text = device.label || `Camera ${select.length}`;
      select.appendChild(opt);
    });

    // Restore previous selection if still available
    if (currentValue && videoDevices.some((d) => d.deviceId === currentValue)) {
      select.value = currentValue;
    }
    select.disabled = false;
    select.title = "";
  } catch (err) {
    console.warn("Camera enumeration failed:", err);
  }
}

async function switchCamera(deviceId) {
  if (!state.running) return;
  if (state.currentDeviceId === deviceId) return;

  // Stop current inference
  clearTimeout(state.inferenceTimer);
  state.inferenceTimer = null;
  if (state.abortController) {
    state.abortController.abort();
    state.abortController = null;
  }
  state.inFlight = false;

  // Stop current stream tracks
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }

  // Start new stream
  els.video.srcObject = null;
  els.startCamBtn.disabled = true;
  els.startCamBtn.textContent = "Switching...";
  setCameraStatus("checking", "CAMERA SWITCHING");

  try {
    const constraints = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } };

    const stream = await navigator.mediaDevices.getUserMedia({
      video: constraints,
      audio: false,
    });

    state.stream = stream;
    state.currentDeviceId = deviceId;
    els.video.srcObject = stream;

    await new Promise((resolve, reject) => {
      if (els.video.readyState >= 1) {
        resolve();
        return;
      }
      const timer = setTimeout(() => reject(new Error("Camera switch timed out.")), 10000);
      els.video.onloadedmetadata = () => {
        clearTimeout(timer);
        resolve();
      };
    });

    await els.video.play();

    state.running = true;
    state.consecutiveErrors = 0;
    state.fpsSamples = [];
    state.currentFps = 0;
    state.frameId += 1;
    state.chartMode = "live";
    state.liveResponse = null;
    state.liveMeta = null;

    els.stage.classList.add("is-live");
    els.cameraEmpty.classList.add("hidden");
    els.noDetectionHint.classList.add("hidden");
    els.startCamBtn.disabled = false;
    els.startCamBtn.textContent = "Start Camera";
    els.stopCamBtn.disabled = false;
    els.roiBtn.disabled = false;
    setCameraStatus("live", "CAMERA LIVE");
    els.binaryBadge.textContent = "LIVE";
    updateFpsDisplay();
    scheduleNextInference(0);
    hideCameraAlert();
  } catch (error) {
    // Rollback: try to restore previous camera
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    els.video.srcObject = null;
    els.startCamBtn.disabled = false;
    els.startCamBtn.textContent = "Start Camera";

    let message = "Could not switch camera: " + error.message;
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      setCameraStatus("denied", "CAMERA DENIED");
      message = "Camera permission was denied. Allow camera access in the browser and try again.";
    } else if (error.name === "NotFoundError" || error.name === "OverconstrainedError") {
      setCameraStatus("offline", "NO CAMERA");
      message = "Selected camera not found. It may have been disconnected.";
      // Reset dropdown to auto
      els.cameraSelect.value = "";
      state.currentDeviceId = null;
    } else {
      setCameraStatus("offline", "CAMERA ERROR");
    }
    showCameraAlert(message);
  }
}

/* ── Camera lifecycle ────────────────────────────────────── */
async function startCamera() {
  if (state.running) return;
  hideCameraAlert();
  els.startCamBtn.disabled = true;
  els.startCamBtn.textContent = "Starting...";
  setCameraStatus("checking", "CAMERA REQUESTING");

  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("This browser does not support getUserMedia.");
    }

    const deviceId = els.cameraSelect?.value;
    const constraints = deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } };

    const stream = await navigator.mediaDevices.getUserMedia({
      video: constraints,
      audio: false,
    });
    state.stream = stream;
    state.currentDeviceId = deviceId;
    els.video.srcObject = stream;

    // Populate camera list AFTER permission granted (labels become available)
    await populateCameraSelect();

    await new Promise((resolve, reject) => {
      if (els.video.readyState >= 1) {
        resolve();
        return;
      }
      const timer = setTimeout(() => reject(new Error("Camera startup timed out.")), 10000);
      els.video.onloadedmetadata = () => {
        clearTimeout(timer);
        resolve();
      };
    });

    await els.video.play();

    state.running = true;
    state.consecutiveErrors = 0;
    state.fpsSamples = [];
    state.currentFps = 0;
    state.frameId += 1;
    state.chartMode = "live";

    els.stage.classList.add("is-live");
    els.cameraEmpty.classList.add("hidden");
    els.noDetectionHint.classList.add("hidden");
    els.startCamBtn.disabled = true;
    els.startCamBtn.textContent = "Start Camera";
    els.stopCamBtn.disabled = false;
    els.roiBtn.disabled = false;
    setCameraStatus("live", "CAMERA LIVE");
    els.binaryBadge.textContent = "LIVE";
    updateFpsDisplay();
    scheduleNextInference(0);
  } catch (error) {
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    els.video.srcObject = null;
    els.startCamBtn.disabled = false;
    els.startCamBtn.textContent = "Start Camera";

    let message = "Could not start the camera: " + error.message;
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      setCameraStatus("denied", "CAMERA DENIED");
      message = "Camera permission was denied. Allow camera access in the browser and try again.";
    } else if (error.name === "NotFoundError" || error.name === "OverconstrainedError") {
      setCameraStatus("offline", "NO CAMERA");
      message = "No usable camera was found. Connect a camera and try again.";
    } else {
      setCameraStatus("offline", "CAMERA ERROR");
    }
    showCameraAlert(message);
  }
}

function stopCamera() {
  state.running = false;
  clearTimeout(state.inferenceTimer);
  state.inferenceTimer = null;

  if (state.abortController) {
    state.abortController.abort();
    state.abortController = null;
  }
  state.inFlight = false;

  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }

  els.video.srcObject = null;
  els.video.onloadedmetadata = null;
  els.stage.classList.remove("is-live", "roi-mode");
  els.cameraEmpty.classList.remove("hidden");
  els.noDetectionHint.classList.add("hidden");
  els.roiHint.classList.add("hidden");
  els.startCamBtn.disabled = false;
  els.startCamBtn.textContent = "Start Camera";
  els.stopCamBtn.disabled = true;
  els.roiBtn.disabled = true;
  els.roiBtn.textContent = "ROI: CENTER";
  els.resetRoiBtn.disabled = true;
  els.binaryBadge.textContent = "IDLE";
  els.binaryImageA.classList.add("hidden");
  els.binaryImageB.classList.add("hidden");
  els.binaryEmpty.classList.remove("hidden");
  els.thresholdValue.textContent = "-";
  els.frameId.textContent = "-";
  els.detectionCount.textContent = "0 detections";
  setCameraStatus("offline", "CAMERA OFFLINE");

  // Reset camera selector to default
  state.currentDeviceId = null;
  if (els.cameraSelect) {
    els.cameraSelect.value = "";
    els.cameraSelect.disabled = true;
    els.cameraSelect.title = "Start camera first to enumerate devices";
  }

  state.roi = null;
  state.roiMode = false;
  state.dragStart = null;
  state.dragCurrent = null;
  state.liveResponse = null;
  state.liveMeta = null;
  state.fpsSamples = [];
  state.currentFps = 0;
  updateFpsDisplay();
  clearOverlay();
  hideCameraAlert();
}

function updateFpsDisplay() {
  const text = `${state.currentFps.toFixed(1)} FPS`;
  els.fpsChip.textContent = text;
  els.processingValue.textContent = text;
}

/* ── Upload mode ─────────────────────────────────────────── */
function setUploadFile(file) {
  if (!file) return;
  state.uploadFile = file;
  if (state.uploadPreviewUrl) URL.revokeObjectURL(state.uploadPreviewUrl);
  state.uploadPreviewUrl = URL.createObjectURL(file);
  els.uploadPreview.src = state.uploadPreviewUrl;
  els.uploadPreview.classList.remove("hidden");
  els.uploadBadge.textContent = "READY";
  hideUploadAlert();
}

els.fileInput.addEventListener("change", (event) => {
  setUploadFile(event.target.files[0]);
});

els.dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  els.dropzone.classList.add("dragging");
});

els.dropzone.addEventListener("dragleave", () => {
  els.dropzone.classList.remove("dragging");
});

els.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  els.dropzone.classList.remove("dragging");
  const file = event.dataTransfer.files && event.dataTransfer.files[0];
  if (file) {
    els.fileInput.files = event.dataTransfer.files;
    setUploadFile(file);
  }
});

async function analyzeUpload() {
  if (!state.uploadFile) {
    showUploadAlert("Please choose an image first.", true);
    return;
  }

  hideUploadAlert();
  els.uploadBadge.textContent = "PROCESSING";
  els.analyzeUploadBtn.disabled = true;

  const form = new FormData();
  form.append("file", state.uploadFile, state.uploadFile.name || "upload.png");
  const params = getParams();
  Object.entries(params).forEach(([key, value]) => form.append(key, String(value)));
  form.append("frame_id", String(state.frameId));
  state.frameId += 1;
  form.append("include_annotated", "true");
  form.append("include_canonical", "true");
  form.append("include_gray", "false");

  try {
    const res = await fetch(API_BASE + "/predict", { method: "POST", body: form });
    if (!res.ok) throw new Error(await responseError(res));
    const data = await res.json();
    renderUploadResult(data);
    els.uploadBadge.textContent = "DONE";
    setApiStatus("ready", "API READY");
  } catch (error) {
    els.uploadBadge.textContent = "ERROR";
    showUploadAlert("Image analysis failed: " + error.message);
    setApiStatus("error", "API ERROR");
  } finally {
    els.analyzeUploadBtn.disabled = false;
  }
}

function renderUploadResult(data) {
  els.uploadResult.classList.remove("hidden");
  els.uploadResultNumber.textContent = data.full_number || "-";

  if (data.annotated_image) els.annotatedImage.src = data.annotated_image;
  if (data.binary_image) els.binaryImageStatic.src = data.binary_image;

  els.digitsContainer.innerHTML = "";
  if (!data.digits || !data.digits.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No digits were detected in this image.";
    els.digitsContainer.appendChild(empty);
  } else {
    data.digits.forEach((detection) => {
      const tile = document.createElement("div");
      tile.className = "digit-tile";
      tile.innerHTML = `
        <div class="digit">${detection.digit}</div>
        <div class="conf">Confidence
          <span class="${confidenceClass(detection.confidence)}">
            ${(detection.confidence * 100).toFixed(1)}%
          </span>
        </div>
        ${detection.canonical_image
          ? `<img class="canonical" src="${detection.canonical_image}" alt="28x28 digit" />`
          : ""}
        <div>${probBars(detection.probabilities, detection.digit)}</div>
      `;
      els.digitsContainer.appendChild(tile);
    });
  }

  state.chartData = data;
  state.chartMode = "upload";
  state.selectedIndex = 0;
  updateDigitSelector(data);
  updateChart(data);
  addHistory({
    frame_id: data.frame_id,
    full_number: data.full_number,
    confidence: data.average_confidence,
    latency: data.processing_time_ms,
    source: "upload",
    ts: Date.now(),
  });
}

function confidenceClass(confidence) {
  if (confidence >= 0.9) return "conf-good";
  if (confidence >= 0.7) return "conf-mid";
  return "conf-low";
}

function probBars(probabilities, bestDigit) {
  return (probabilities || [])
    .map((probability, index) => {
      const percent = (Number(probability) * 100).toFixed(1);
      const weight = index === bestDigit ? "font-weight:800;" : "";
      return `
        <div class="prob-row">
          <span style="width:14px;text-align:right;${weight}">${index}</span>
          <div class="prob-track"><div class="prob-fill" style="width:${percent}%;background:${BAR_COLORS[index]}"></div></div>
          <span class="pct">${percent}%</span>
        </div>`;
    })
    .join("");
}

els.analyzeUploadBtn.addEventListener("click", analyzeUpload);

/* ── Global lifecycle ────────────────────────────────────── */
window.addEventListener("beforeunload", () => {
  state.running = false;
  clearTimeout(state.inferenceTimer);
  if (state.abortController) state.abortController.abort();
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
});

window.addEventListener("pagehide", () => {
  state.running = false;
  clearTimeout(state.inferenceTimer);
  if (state.abortController) state.abortController.abort();
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
});

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    clearTimeout(state.inferenceTimer);
    state.inferenceTimer = null;
  } else if (state.running) {
    scheduleNextInference(0);
  }
});

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(drawSparkline, 120);
});

/* ── Init ────────────────────────────────────────────────── */
els.startCamBtn.addEventListener("click", startCamera);
els.stopCamBtn.addEventListener("click", stopCamera);

els.cameraSelect.addEventListener("change", () => {
  if (state.running) {
    switchCamera(els.cameraSelect.value || null);
  }
});

// Listen for device changes (plug/unplug)
if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    if (state.running) {
      populateCameraSelect();
    }
  });
}

initChart();
updateFpsDisplay();
checkHealth();
setInterval(checkHealth, 10000);
requestAnimationFrame(drawOverlayLoop);
drawHistory();
