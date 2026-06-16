const IS_LOCAL = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
document.getElementById('api-status').textContent = IS_LOCAL ? 'BentoML :3000 Connected' : 'BentoML Cloud Run';

let currentData = null;
let selectedIdx = 0;

// ─── CLOCK ───────────────────────────────────────────────────
setInterval(() => {
    document.getElementById('hdr-clock').textContent = new Date().toLocaleTimeString('en-GB');
}, 1000);

// ─── PATIENT DATA ─────────────────────────────────────────────
const NAMES = ['Anderson, James R.', 'Chen, Robert W.', 'Martinez, David L.', 'Thompson, Michael K.', 'Williams, Patricia A.'];
const PHYSICIANS = ['Dr. Sarah Chen, MD', 'Dr. Michael Torres, MD', 'Dr. Emily Watson, MD', 'Dr. Raj Patel, MD'];
const DOBS = ['1958-03-14', '1962-07-22', '1955-11-08', '1949-04-30', '1967-09-15'];
const AGES = [68, 64, 71, 77, 59];

function setPatientInfo(patientId, score) {
    const h = patientId.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
    const name = NAMES[h % NAMES.length];
    document.getElementById('pat-initials').textContent = name.split(',')[0].split(' ').map(w => w[0]).join('').slice(0, 2);
    document.getElementById('pat-name').textContent = name;
    document.getElementById('pat-mrn').textContent = 'MRN-' + String(Math.abs(h * 1337 % 99999)).padStart(5, '0');
    document.getElementById('pat-dob').textContent = DOBS[h % DOBS.length];
    document.getElementById('pat-age').textContent = AGES[h % AGES.length] + ' yrs';
    document.getElementById('pat-physician').textContent = PHYSICIANS[h % PHYSICIANS.length];
    document.getElementById('pat-date').textContent = new Date().toISOString().slice(0, 10);
    document.getElementById('pat-series').textContent = '...' + patientId.slice(-14);
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
    const ws = document.getElementById('workspace');
    ws.style.height = `calc(100vh - ${header + banner + 2}px)`;
}

// ─── PROGRESS VISUALIZER ──────────────────────────────────────
let _timerInterval = null;
let _inferenceStart = null;
const EXPECTED_INFERENCE_S = 150;
const STAGE_THRESHOLDS = [0, 45, 80, 93, 100];

function _updateStages(pct) {
    for (let i = 0; i < 4; i++) {
        const el = document.getElementById('stage-' + i);
        if (!el) continue;
        if (pct >= STAGE_THRESHOLDS[i + 1]) {
            el.className = 'stage-dot done';
        } else if (pct >= STAGE_THRESHOLDS[i]) {
            el.className = 'stage-dot active';
        } else {
            el.className = 'stage-dot';
        }
    }
}

function _setRing(pct) {
    const circ = 2 * Math.PI * 30;
    const offset = circ * (1 - pct / 100);
    const ring = document.getElementById('ring-circle');
    if (ring) {
        ring.style.strokeDashoffset = offset;
        ring.style.stroke = pct >= 95 ? '#10b981' : '#0ea5e9';
    }
    const lbl = document.getElementById('ring-pct');
    if (lbl) lbl.textContent = Math.round(pct) + '%';
}

function _startElapsedTimer() {
    _inferenceStart = Date.now();
    if (_timerInterval) clearInterval(_timerInterval);
    _timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - _inferenceStart) / 1000);
        const el = document.getElementById('elapsed-time');
        if (el) el.textContent = elapsed + 's';

        const barPct = parseFloat(document.getElementById('prog-fill').style.width || '0');
        if (barPct >= 65 && barPct < 95) {
            const inferElapsed = Math.floor((Date.now() - _inferenceStart) / 1000);
            const timePct = Math.min(inferElapsed / EXPECTED_INFERENCE_S, 1);
            const animated = 65 + timePct * (93 - 65);
            _setRing(Math.min(animated, 93));
            const rem = document.getElementById('remain-time');
            if (rem) {
                const secsLeft = Math.max(0, EXPECTED_INFERENCE_S - inferElapsed);
                rem.textContent = secsLeft > 0 ? secsLeft + 's' : 'almost done…';
            }
        }
    }, 1000);
}

function setProgress(pct, label) {
    document.getElementById('prog-bar').style.width = pct + '%';
    const vis = document.getElementById('proc-visualizer');
    if (vis) vis.style.display = 'block';
    const st = document.getElementById('prog-status');
    if (st) st.textContent = label;
    const fill = document.getElementById('prog-fill');
    if (fill) fill.style.width = pct + '%';
    const barPct = parseFloat(fill ? fill.style.width : '0');
    if (barPct < 65 || barPct >= 93) _setRing(pct);
    _updateStages(pct);
    if (pct >= 65 && !_timerInterval) _startElapsedTimer();
}

function hideProgress() {
    document.getElementById('prog-bar').style.width = '0%';
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
}

// ─── SAMPLE RUN ───────────────────────────────────────────────
const SAMPLE_URL = 'https://media.githubusercontent.com/media/suryaprakashdev/PlumoNet/main/frontend/dicoms.zip';

async function runSample() {
    const btn = document.getElementById('sample-btn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Fetching sample DICOM…';
    try {
        setProgress(15, 'Downloading sample DICOM from cloud storage...');
        const resp = await fetch(SAMPLE_URL, { signal: AbortSignal.timeout(120000) });
        if (!resp.ok) throw new Error(`Download failed: HTTP ${resp.status}`);
        setProgress(45, 'Sample downloaded — sending to inference server...');
        const blob = await resp.blob();
        await sendInference(blob);
    } catch (e) {
        showError('Sample run failed: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🔬 Run Sample DICOM Study <span style="font-size:10px;color:var(--muted);font-family:var(--mono)">(LIDC-IDRI)</span>';
    }
}

// ─── UPLOAD ───────────────────────────────────────────────────
async function handleZip(file) {
    if (!file) return;
    setProgress(25, 'Reading ZIP archive...');
    await sendInference(file);
}

async function handleFolder(files) {
    if (!files || !files.length) return;
    setProgress(10, `Compressing ${files.length} DICOM files...`);
    try {
        const zip = new JSZip();
        for (const f of files) zip.file(f.webkitRelativePath || f.name, f);
        setProgress(35, 'Finalizing archive...');
        const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 1 } });
        setProgress(15, `Archive ready (${(blob.size / 1024 / 1024).toFixed(1)} MB) — starting upload...`);
        await sendInference(blob);
    } catch (e) { showError('Compression failed: ' + e.message); }
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('upload-zone').classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length === 1 && files[0].name.endsWith('.zip')) handleZip(files[0]);
    else if (files.length > 1) handleFolder(files);
    else showError('Drop a .zip file or a DICOM folder.');
}

function _endpoint(path) {
    return IS_LOCAL ? `http://${location.hostname}:3000/${path}` : `/${path}`;
}

async function sendInference(blob) {
    try {
        setProgress(20, 'Preparing secure upload...');
        const urlResp = await fetch(_endpoint('get_upload_url'), { method: 'POST' });
        if (!urlResp.ok) throw new Error(`Could not get upload URL: HTTP ${urlResp.status}`);
        const { sas_url, blob_name } = await urlResp.json();

        await uploadToBlob(sas_url, blob);

        setProgress(60, 'Running 3D segmentation + classification (this takes ~2–3 min)...');
        const resp = await fetch(_endpoint('predict'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ blob_name }),
            signal: AbortSignal.timeout(600000)
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${(await resp.text()).slice(0, 160)}`);
        setProgress(95, 'Parsing results...');
        const data = await resp.json();
        hideProgress();
        currentData = data;
        renderResults(data);
    } catch (e) {
        showError('Inference error: ' + e.message);
    }
}

function uploadToBlob(sasUrl, blob) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open('PUT', sasUrl);
        xhr.setRequestHeader('x-ms-blob-type', 'BlockBlob');
        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const pct = 25 + Math.round((e.loaded / e.total) * 30);
                setProgress(pct, `Uploading ${(e.loaded/1024/1024).toFixed(1)} / ${(e.total/1024/1024).toFixed(1)} MB...`);
            }
        };
        xhr.onload = () => xhr.status === 201
            ? resolve()
            : reject(new Error(`Blob upload failed: ${xhr.status}`));
        xhr.onerror = () => reject(new Error('Blob upload network error'));
        xhr.send(blob);
    });
}

// ─── RENDER ───────────────────────────────────────────────────
function probStyle(p) {
    if (p >= 0.5) return { color: 'var(--red)',   shape: 'shape-mal', label: 'MALIGNANT' };
    if (p >= 0.3) return { color: 'var(--amber)', shape: 'shape-bor', label: 'BORDERLINE' };
    return             { color: 'var(--green)',   shape: 'shape-ben', label: 'BENIGN' };
}

function renderResults(d) {
    const m = d.metadata || {};

    document.getElementById('upload-overlay').style.display = 'none';
    setPatientInfo(m.patient_id || 'UNKNOWN', d.patient_score || 0);

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

    selectCandidate(0);
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

    ['ax', 'cor', 'sag'].forEach(p => {
        document.getElementById(p + '-empty').style.display = 'flex';
        document.getElementById(p + '-img').style.display = 'none';
    });

    if (views) {
        const z = Math.round(cand.centroid[0]);
        const y = Math.round(cand.centroid[1]);
        const x = Math.round(cand.centroid[2]);

        document.getElementById('ax-tag').textContent = `AXIAL — Z: ${z}`;
        document.getElementById('cor-tag').textContent = `CORONAL — Y: ${y}`;
        document.getElementById('sag-tag').textContent = `SAGITTAL — X: ${x}`;

        showSlice('ax', views.axial[z] ? views.axial[z].image : null);
        showSlice('cor', views.coronal[y] ? views.coronal[y].image : null);
        showSlice('sag', views.sagittal[x] ? views.sagittal[x].image : null);
    } else {
        ['ax', 'cor', 'sag'].forEach(p => {
            document.getElementById(p + '-empty').innerHTML =
                '<span class="mpr-empty-icon">⚠</span>No slice data available';
        });
    }
}

function showSlice(prefix, b64) {
    if (!b64) return;
    const img = document.getElementById(prefix + '-img');
    img.src = 'data:image/png;base64,' + b64;
    img.style.display = 'block';
    document.getElementById(prefix + '-empty').style.display = 'none';
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
    ['st-seg', 'st-cls', 'st-vol', 'cand-count'].forEach(id => document.getElementById(id).textContent = '—');
    document.getElementById('workspace').style.height = 'calc(100vh - 54px)';
    document.getElementById('zip-input').value = '';
    document.getElementById('folder-input').value = '';
    const vis = document.getElementById('proc-visualizer');
    if (vis) vis.style.display = 'none';
    _setRing(0);
    _updateStages(0);
    if (_timerInterval) { clearInterval(_timerInterval); _timerInterval = null; }
    const et = document.getElementById('elapsed-time'); if (et) et.textContent = '0s';
    const rt = document.getElementById('remain-time'); if (rt) rt.textContent = '—';
}

// ─── MODAL ────────────────────────────────────────────────────
let modalScale = 1;
let modalPanning = false;
let modalPointX = 0;
let modalPointY = 0;
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

window.onmouseup = function () { modalPanning = false; };

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
    if (delta > 0) { modalScale *= 1.1; } else { modalScale /= 1.1; }
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
