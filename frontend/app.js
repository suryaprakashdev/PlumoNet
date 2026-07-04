/**
 * PulmoNet — Frontend (rich display + chunked upload)
 * ======================================================
 * Display logic ported from the working repo frontend:
 *   - Nodule 0N naming, prob-colored (red/amber/green), centroid coords
 *   - Per-nodule navigation: selecting a nodule jumps ALL 3 planes to that
 *     nodule's centroid slice (Z/Y/X), so the red marker is always in view
 *   - Nearest-slice fallback so a marker shows even if the exact centroid
 *     index wasn't among the rendered active_slices
 *   - Patient banner, zoom/pan modal, progress ring + stage dots
 * Upload path kept as the chunked /api/upload/* flow (works with the
 * ordered-commit service_azure.py).
 */

const API_BASE = "/api";
let currentData = null;
let selectedIdx = 0;

const CHUNK_SIZE = 5 * 1024 * 1024;
const MAX_PARALLEL_CHUNKS = 3;

// ─── CLOCK ───────────────────────────────────────────────────
setInterval(() => {
  const el = document.getElementById("hdr-clock");
  if (el) el.textContent = new Date().toLocaleTimeString("en-GB");
}, 1000);

// ─── STUDY BANNER (real DICOM metadata only — no fabricated identity) ─

// DICOM StudyDate is "YYYYMMDD"; render as "YYYY-MM-DD" if present and well-formed.
function formatStudyDate(raw) {
  if (!raw || raw.length !== 8) return raw || '—';
  return `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`;
}

function setPatientInfo(metadata, score) {
  const m = metadata || {};
  document.getElementById('pat-id').textContent = m.patient_id || 'UNKNOWN';
  document.getElementById('pat-date').textContent = formatStudyDate(m.study_date);
  document.getElementById('pat-series').textContent = m.series_uid || '—';
  document.getElementById('pat-volume').textContent = (m.volume_shape || []).join('×') || '—';
  document.getElementById('patient-banner').style.display = 'block';

  const pill = document.getElementById('pat-verdict');
  const isMal = score >= 0.5;
  pill.textContent = isMal ? '⚠ MALIGNANT' : '✓ BENIGN';
  pill.className = 'verdict-pill ' + (isMal ? 'verdict-mal' : 'verdict-ben');
  pill.style.display = 'block';

  adjustWorkspaceHeight();
}

function adjustWorkspaceHeight() {
  const header = document.querySelector('.header').offsetHeight;
  const banner = document.querySelector('.patient-banner').offsetHeight;
  document.getElementById('workspace').style.height =
    `calc(100vh - ${header + banner + 2}px)`;
}

// ─── PROGRESS VISUALIZER ──────────────────────────────────────
// Driven by real SSE progress events from modal_inference.py (Pattern 5),
// not a wall-clock guess — see mapInferenceEvent() below for the stage→%
// mapping used once the upload phase (0-60%) hands off to inference.
let _timerInterval = null;
let _inferenceStart = null;
const STAGE_THRESHOLDS = [0, 45, 80, 93, 100];

function _updateStages(pct) {
  for (let i = 0; i < 4; i++) {
    const el = document.getElementById('stage-' + i);
    if (!el) continue;
    if (pct >= STAGE_THRESHOLDS[i + 1]) el.className = 'stage-dot done';
    else if (pct >= STAGE_THRESHOLDS[i]) el.className = 'stage-dot active';
    else el.className = 'stage-dot';
  }
}

function _setRing(pct) {
  const circ = 2 * Math.PI * 30;
  const ring = document.getElementById('ring-circle');
  if (ring) {
    ring.style.strokeDashoffset = circ * (1 - pct / 100);
    ring.style.stroke = pct >= 95 ? '#10b981' : '#0ea5e9';
  }
  const lbl = document.getElementById('ring-pct');
  if (lbl) lbl.textContent = Math.round(pct) + '%';
}

// Plain running clock — no longer used to interpolate progress, since real
// per-stage events drive the percentage now.
function _startElapsedTimer() {
  _inferenceStart = Date.now();
  if (_timerInterval) clearInterval(_timerInterval);
  _timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - _inferenceStart) / 1000);
    const el = document.getElementById('elapsed-time');
    if (el) el.textContent = elapsed + 's';
  }, 1000);
}

function setProgress(pct, label) {
  const bar = document.getElementById('prog-bar');
  if (bar) bar.style.width = pct + '%';
  const vis = document.getElementById('proc-visualizer');
  if (vis) vis.style.display = 'block';
  const st = document.getElementById('prog-status');
  if (st) st.textContent = label;
  const fill = document.getElementById('prog-fill');
  if (fill) fill.style.width = pct + '%';
  _setRing(pct);
  _updateStages(pct);
  if (pct >= 65 && !_timerInterval) _startElapsedTimer();
}

// Maps one real SSE event from modal_inference.py to a (pct, label) pair.
// Returns null for events that don't move the bar (so the caller keeps the
// previous state) and { done: true, result } / { error: true, message } for
// terminal events.
function mapInferenceEvent(evt) {
  switch (evt.stage) {
    case 'downloaded':
      return { pct: 60, label: 'Scan received — starting analysis...' };
    case 'preprocessing':
      return evt.status === 'started'
        ? { pct: 62, label: 'Preprocessing volume...' }
        : { pct: 68, label: 'Preprocessing complete' };
    case 'segmentation':
      return evt.status === 'started'
        ? { pct: 70, label: 'Running 3D segmentation...' }
        : { pct: 80, label: 'Segmentation complete' };
    case 'classifying': {
      const frac = evt.total ? evt.current / evt.total : 1;
      return { pct: 80 + frac * 10, label: `Classifying candidate ${evt.current}/${evt.total}...` };
    }
    case 'rendering': {
      const frac = evt.total ? evt.current / evt.total : 1;
      return { pct: 90 + frac * 7, label: `Rendering views ${evt.current}/${evt.total}...` };
    }
    case 'done':
      return { done: true, result: evt.result };
    case 'error':
      return { error: true, message: evt.message || 'Inference failed' };
    default:
      return null;
  }
}

function hideProgress() {
  const bar = document.getElementById('prog-bar');
  if (bar) bar.style.width = '0%';
  if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
  const vis = document.getElementById('proc-visualizer');
  if (vis) vis.style.display = 'none';
  _setRing(100);
  _updateStages(100);
  const rem = document.getElementById('remain-time');
  if (rem) rem.textContent = '—';
}

function showError(msg) {
  const t = document.getElementById('err-toast');
  document.getElementById('err-msg').textContent = msg;
  t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 7000);
  if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
  const st = document.getElementById('prog-status');
  if (st) st.textContent = 'Error — see notification';
  console.error("[Error]", msg);
}

// ─── UPLOAD ENTRY POINTS ──────────────────────────────────────
async function handleZip(file) {
  if (!file) return;
  setProgress(10, 'Reading ZIP archive...');
  await sendInference(file);
}

async function handleFolder(files) {
  if (!files || !files.length) return;
  setProgress(10, `Compressing ${files.length} DICOM files...`);
  try {
    const zip = new JSZip();
    for (const f of files) zip.file(f.webkitRelativePath || f.name, f);
    setProgress(35, 'Finalizing archive...');
    const blob = await zip.generateAsync({
      type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 1 }
    });
    setProgress(15, `Archive ready (${(blob.size / 1024 / 1024).toFixed(1)} MB) — starting upload...`);
    await sendInference(blob);
  } catch (e) {
    showError('Compression failed: ' + e.message);
  }
}

function handleDrop(event) {
  event.preventDefault();
  event.stopPropagation();
  document.getElementById('upload-zone').classList.remove('drag-over');
  const files = event.dataTransfer.files;
  if (files.length === 1 && files[0].name.endsWith('.zip')) handleZip(files[0]);
  else if (files.length > 1) handleFolder(files);
  else showError('Drop a .zip file or a DICOM folder.');
}

// Permanent, curated sample scan (copied once into Blob Storage) — lets
// "Run Sample" skip the upload step entirely and jump straight to inference.
const SAMPLE_BLOB_NAME = 'samples/lidc-idri-sample.zip';

async function runSample() {
  const btn = document.getElementById('sample-btn');
  if (btn) btn.disabled = true;
  try {
    _updateStages(0);
    setProgress(58, 'Loading sample scan (LIDC-IDRI)...');
    await runPrediction(SAMPLE_BLOB_NAME);
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ─── INFERENCE (chunked upload → streaming predict) ────────────
// Note: deliberately not using EventSource — it's GET-only (we need to POST
// blob_name) and it auto-reconnects on any network hiccup, which here would
// risk silently re-triggering a second, expensive GPU run.
const STREAM_INACTIVITY_MS = 90000; // no SSE frame for this long = stalled

async function sendInference(blob) {
  _updateStages(0);
  let blobName;
  try {
    blobName = await uploadFileChunked(blob);
  } catch (e) {
    showError('Upload failed: ' + e.message);
    return;
  }
  if (!blobName) { showError('Upload failed'); return; }

  setProgress(58, 'Starting 3D segmentation + classification...');
  await runPrediction(blobName);
}

// Streams /api/predict for an already-uploaded blob_name and renders the
// result. Shared by sendInference() (after upload) and runSample() (which
// skips upload entirely, using a pre-existing blob).
async function runPrediction(blobName) {
  const controller = new AbortController();
  const overallTimeout = setTimeout(() => controller.abort(), 600000);
  let inactivityTimer = null;
  const armInactivityWatchdog = () => {
    if (inactivityTimer) clearTimeout(inactivityTimer);
    inactivityTimer = setTimeout(() => controller.abort(), STREAM_INACTIVITY_MS);
  };

  try {
    const resp = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ blob_name: blobName }),
      signal: controller.signal,
    });
    if (!resp.ok || !resp.body) {
      const txt = await resp.text().catch(() => '');
      throw new Error(`HTTP ${resp.status}: ${txt.slice(0, 160)}`);
    }

    armInactivityWatchdog();
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finished = false;

    while (!finished) {
      const { done, value } = await reader.read();
      if (done) break;
      armInactivityWatchdog();

      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split('\n\n');
      buffer = frames.pop(); // keep the last, possibly-partial frame

      for (const frame of frames) {
        if (!frame.startsWith('data: ')) continue;
        let evt;
        try {
          evt = JSON.parse(frame.slice(6));
        } catch {
          continue; // malformed frame — skip rather than crash the stream
        }
        const mapped = mapInferenceEvent(evt);
        if (!mapped) continue;

        if (mapped.error) {
          showError('Inference error: ' + mapped.message);
          finished = true;
          break;
        }
        if (mapped.done) {
          hideProgress();
          currentData = mapped.result;
          renderResults(mapped.result);
          finished = true;
          break;
        }
        setProgress(mapped.pct, mapped.label);
      }
    }

    if (!finished) {
      showError('Inference stream ended unexpectedly — please retry.');
    }
  } catch (e) {
    if (e.name === 'AbortError') {
      showError('Inference stream stalled or timed out — please retry.');
    } else {
      showError('Inference error: ' + e.message);
    }
  } finally {
    clearTimeout(overallTimeout);
    if (inactivityTimer) clearTimeout(inactivityTimer);
  }
}

// ─── CHUNKED UPLOAD ───────────────────────────────────────────
async function uploadFileChunked(file) {
  const fileName = file.name || `study-${Date.now()}.zip`;
  const fileSize = file.size;
  const totalChunks = Math.ceil(fileSize / CHUNK_SIZE);

  setProgress(20, 'Preparing upload...');
  const initRes = await fetch(`${API_BASE}/upload/initiate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileName, fileSize }),
  });
  if (!initRes.ok) throw new Error('Failed to initiate upload');
  const { sessionId } = await initRes.json();

  const chunks = [];
  for (let i = 0; i < totalChunks; i++) {
    const start = i * CHUNK_SIZE;
    chunks.push({ index: i, blob: file.slice(start, Math.min(start + CHUNK_SIZE, fileSize)) });
  }

  let done = 0;
  for (let i = 0; i < chunks.length; i += MAX_PARALLEL_CHUNKS) {
    const batch = chunks.slice(i, i + MAX_PARALLEL_CHUNKS);
    await Promise.all(batch.map(async (chunk) => {
      const fd = new FormData();
      fd.append('sessionId', sessionId);
      fd.append('chunkIndex', chunk.index);
      fd.append('totalChunks', totalChunks);
      fd.append('chunk', chunk.blob);
      const r = await fetch(`${API_BASE}/upload/chunk`, { method: 'POST', body: fd });
      if (!r.ok) throw new Error(`Chunk ${chunk.index} failed`);
      done++;
      setProgress(20 + (done / totalChunks) * 35,
        `Uploading… ${Math.round((done / totalChunks) * 100)}%`);
    }));
  }

  const finRes = await fetch(`${API_BASE}/upload/finalize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId }),
  });
  if (!finRes.ok) {
    const err = await finRes.json().catch(() => ({}));
    throw new Error(err.error || 'Failed to finalize upload');
  }
  const { blobName } = await finRes.json();
  return blobName;
}

// ─── RENDER RESULTS ───────────────────────────────────────────
function probStyle(p) {
  if (p >= 0.5) return { color: 'var(--red)', shape: 'shape-mal', label: 'MALIGNANT' };
  if (p >= 0.3) return { color: 'var(--amber)', shape: 'shape-bor', label: 'BORDERLINE' };
  return { color: 'var(--green)', shape: 'shape-ben', label: 'BENIGN' };
}

function renderResults(d) {
  const m = d.metadata || {};
  document.getElementById('upload-overlay').style.display = 'none';
  setPatientInfo(m, d.patient_score || 0);

  const cands = d.top_candidates || [];
  document.getElementById('cand-count').textContent = cands.length + ' found';
  document.getElementById('cand-list').innerHTML = cands.map((c, i) => {
    const s = probStyle(c.prob);
    const pct = (c.prob * 100).toFixed(1);
    return `<div class="cand-item${i === 0 ? ' active' : ''}" id="cand-${i}" onclick="selectCandidate(${i})">
  <div class="cand-header">
    <div class="cand-title"><span class="${s.shape}"></span> Nodule 0${i + 1}</div>
    <div class="cand-score" style="color:${s.color}">${pct}%</div>
  </div>
  <div class="cand-bar"><div class="cand-bar-fill" style="width:${pct}%;background:${s.color}"></div></div>
  <div class="cand-footer">
    <span class="cand-label" style="color:${s.color}">${s.label}</span>
    <span class="cand-coords">[${(c.centroid || []).map(v => Math.round(v)).join(', ')}]</span>
  </div>
</div>`;
  }).join('');

  document.getElementById('st-seg').textContent = (m.seg_time_s || '?') + 's';
  document.getElementById('st-cls').textContent = (m.cls_time_s || '?') + 's';
  document.getElementById('st-vol').textContent = (m.volume_shape || []).join('×') || '—';

  if (cands.length) selectCandidate(0);
}

// Find the rendered slice nearest to a target index (so a marker always shows
// even if the exact centroid slice wasn't among active_slices).
function nearestSlice(planeViews, targetIdx) {
  if (!planeViews) return null;
  const keys = Object.keys(planeViews);
  if (!keys.length) return null;
  if (planeViews[targetIdx]) return { idx: targetIdx, data: planeViews[targetIdx] };
  let best = null, bestDist = Infinity;
  for (const k of keys) {
    const dist = Math.abs(parseInt(k, 10) - targetIdx);
    if (dist < bestDist) { bestDist = dist; best = k; }
  }
  return best === null ? null : { idx: parseInt(best, 10), data: planeViews[best] };
}

function selectCandidate(idx) {
  if (!currentData) return;
  const cands = currentData.top_candidates || [];
  if (idx < 0 || idx >= cands.length) return;

  selectedIdx = idx;
  document.querySelectorAll('.cand-item').forEach(el => el.classList.remove('active'));
  const el = document.getElementById('cand-' + idx);
  if (el) { el.classList.add('active'); el.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); }

  const cand = cands[idx];
  const views = currentData.candidate_views || {};
  const [zc, yc, xc] = (cand.centroid || [0, 0, 0]).map(v => Math.round(v));

  const ax = nearestSlice(views.axial, zc);
  const cor = nearestSlice(views.coronal, yc);
  const sag = nearestSlice(views.sagittal, xc);

  document.getElementById('ax-tag').textContent = `AXIAL — Z: ${ax ? ax.idx : zc}`;
  document.getElementById('cor-tag').textContent = `CORONAL — Y: ${cor ? cor.idx : yc}`;
  document.getElementById('sag-tag').textContent = `SAGITTAL — X: ${sag ? sag.idx : xc}`;

  showSlice('ax', ax ? ax.data.image : null);
  showSlice('cor', cor ? cor.data.image : null);
  showSlice('sag', sag ? sag.data.image : null);
}

function showSlice(prefix, b64) {
  const img = document.getElementById(prefix + '-img');
  const empty = document.getElementById(prefix + '-empty');
  if (!b64) {
    img.style.display = 'none';
    empty.style.display = 'flex';
    empty.innerHTML = '<span class="mpr-empty-icon">⚠</span>No slice at this level';
    return;
  }
  img.src = 'data:image/png;base64,' + b64;
  img.style.display = 'block';
  empty.style.display = 'none';
}

function setActiveTool(btn) {
  document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

function cycleCandidate(dir) {
  if (!currentData) return;
  const n = (currentData.top_candidates || []).length;
  if (!n) return;
  selectCandidate((selectedIdx + dir + n) % n);
}

function resetStudy() {
  currentData = null; selectedIdx = 0;
  document.getElementById('upload-overlay').style.display = 'flex';
  document.getElementById('patient-banner').style.display = 'none';
  document.getElementById('pat-verdict').style.display = 'none';
  document.getElementById('cand-list').innerHTML =
    '<div style="color:var(--dim);font-size:11px;text-align:center;padding:30px 16px">Run inference to see detected nodules</div>';
  ['ax', 'cor', 'sag'].forEach(p => {
    document.getElementById(p + '-empty').style.display = 'flex';
    document.getElementById(p + '-empty').innerHTML = '<span class="mpr-empty-icon">🫁</span>Awaiting scan';
    document.getElementById(p + '-img').style.display = 'none';
  });
  ['ax-tag', 'cor-tag', 'sag-tag'].forEach((id, i) =>
    document.getElementById(id).textContent = ['AXIAL', 'CORONAL', 'SAGITTAL'][i]);
  ['st-seg', 'st-cls', 'st-vol', 'cand-count'].forEach(id => document.getElementById(id).textContent = '—');
  document.getElementById('workspace').style.height = 'calc(100vh - 54px)';
  document.getElementById('zip-input').value = '';
  document.getElementById('folder-input').value = '';
  const vis = document.getElementById('proc-visualizer');
  if (vis) vis.style.display = 'none';
  _setRing(0); _updateStages(0);
  if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
  const et = document.getElementById('elapsed-time'); if (et) et.textContent = '0s';
  const rt = document.getElementById('remain-time'); if (rt) rt.textContent = '—';
}

// ─── ZOOM/PAN MODAL ───────────────────────────────────────────
let modalScale = 1, modalPanning = false, modalPointX = 0, modalPointY = 0;
let modalStart = { x: 0, y: 0 };
const modalImg = document.getElementById('modal-img');

function setModalTransform() {
  modalImg.style.transform = `translate(${modalPointX}px, ${modalPointY}px) scale(${modalScale})`;
}
modalImg.onmousedown = function (e) {
  e.preventDefault();
  modalStart = { x: e.clientX - modalPointX, y: e.clientY - modalPointY };
  modalPanning = true;
};
window.onmouseup = () => { modalPanning = false; };
window.onmousemove = function (e) {
  if (!modalPanning) return;
  e.preventDefault();
  modalPointX = e.clientX - modalStart.x;
  modalPointY = e.clientY - modalStart.y;
  setModalTransform();
};
document.getElementById('modal').onwheel = function (e) {
  e.preventDefault();
  const xs = (e.clientX - modalPointX) / modalScale;
  const ys = (e.clientY - modalPointY) / modalScale;
  const delta = e.wheelDelta ? e.wheelDelta : -e.deltaY;
  modalScale *= delta > 0 ? 1.1 : (1 / 1.1);
  modalScale = Math.min(Math.max(0.5, modalScale), 10);
  modalPointX = e.clientX - xs * modalScale;
  modalPointY = e.clientY - ys * modalScale;
  setModalTransform();
};
function openModal(src) {
  modalScale = 1; modalPointX = 0; modalPointY = 0;
  setModalTransform();
  document.getElementById('modal-img').src = src;
  document.getElementById('modal').classList.add('open');
}
function closeModal() { document.getElementById('modal').classList.remove('open'); }

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop', e => e.preventDefault());