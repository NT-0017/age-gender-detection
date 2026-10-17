'use strict';

/* ── Config ──────────────────────────────────────────────────────────────── */
const API_DETECT = 'https://age-gender-detection-hg6n.onrender.com/api/detect';
const MAX_BYTES      = 10 * 1024 * 1024;          // 10 MB (matches backend)
const MAX_CANVAS_PX  = 1600;                       // cap canvas resolution for performance
const ACCEPTED_TYPES = new Set([
  'image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/webp',
]);
// Distinct colours for each detected face (box + card accent)
const FACE_COLORS = [
  '#6366f1', '#f59e0b', '#10b981',
  '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899',
];

/* ── DOM refs ────────────────────────────────────────────────────────────── */
const dropZone          = document.getElementById('dropZone');
const fileInput         = document.getElementById('fileInput');
const browseBtn         = document.getElementById('browseBtn');
const previewBlock      = document.getElementById('previewBlock');
const previewName       = document.getElementById('previewName');
const clearBtn          = document.getElementById('clearBtn');
const canvas            = document.getElementById('previewCanvas');
const ctx               = canvas.getContext('2d');
const detectBtn         = document.getElementById('detectBtn');
const detectLabel       = document.getElementById('detectLabel');
const btnSpinner        = document.getElementById('btnSpinner');
const detectIcon        = detectBtn.querySelector('.detect-icon');
const resultsPlaceholder = document.getElementById('resultsPlaceholder');
const resultsList       = document.getElementById('resultsList');
// Webcam elements
const webcamVideo = document.getElementById('webcamVideo');
const webcamCanvas = document.getElementById('webcamCanvas');
const webcamStatus = document.getElementById('webcamStatus');

const startWebcamBtn = document.getElementById('startWebcamBtn');
const captureWebcamBtn = document.getElementById('captureWebcamBtn');
const stopWebcamBtn = document.getElementById('stopWebcamBtn');

let webcamStream = null;

/* ── State ───────────────────────────────────────────────────────────────── */
let currentFile  = null;   // File selected by the user
let currentImage = null;   // HTMLImageElement (decoded)
let canvasScale  = 1;      // ratio: canvas logical px / original image px

/* ══════════════════════════════════════════════════════════════════════════
   DRAG & DROP / FILE INPUT
══════════════════════════════════════════════════════════════════════════ */
dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// Click anywhere on the drop zone to open file picker
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
});

// "click to browse" link inside the drop zone
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

clearBtn.addEventListener('click', resetAll);
detectBtn.addEventListener('click', runDetection);

/* ══════════════════════════════════════════════════════════════════════════
   FILE VALIDATION & PREVIEW
══════════════════════════════════════════════════════════════════════════ */
function handleFile(file) {
  // Validate type
  const ext = file.name.split('.').pop().toLowerCase();
  const extOk  = ['jpg','jpeg','png','bmp','webp'].includes(ext);
  const typeOk = ACCEPTED_TYPES.has((file.type || '').toLowerCase());
  if (!typeOk && !extOk) {
    showMessage('error', '⚠ Unsupported file. Please upload a JPEG, PNG, BMP, or WebP image.');
    return;
  }
  // Validate size
  if (file.size > MAX_BYTES) {
    showMessage('error', '⚠ File is too large. Maximum allowed size is 10 MB.');
    return;
  }

  currentFile = file;
  previewName.textContent = file.name;
  previewName.title       = file.name;

  // Switch UI
  dropZone.classList.add('hidden');
  previewBlock.classList.remove('hidden');
  resetResults();
  loadToCanvas(file);
}

function loadToCanvas(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();

  img.onload = () => {
    currentImage = img;

    // Scale down if the image is very large (keeps drawing fast)
    const scale = Math.min(1, MAX_CANVAS_PX / Math.max(img.naturalWidth, img.naturalHeight));
    canvas.width  = Math.round(img.naturalWidth  * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    canvasScale   = scale;

    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(url);
    detectBtn.disabled = false;
  };

  img.onerror = () => {
    URL.revokeObjectURL(url);
    showMessage('error', '⚠ Could not decode the image. The file may be corrupt.');
    resetAll();
  };

  img.src = url;
}

function resetAll() {
  currentFile  = null;
  currentImage = null;
  canvasScale  = 1;
  fileInput.value = '';

  dropZone.classList.remove('hidden');
  previewBlock.classList.add('hidden');
  detectBtn.disabled = true;
  canvas.width  = 0;
  canvas.height = 0;

  resetResults();
}

function resetResults() {
  resultsPlaceholder.classList.remove('hidden');
  resultsList.classList.add('hidden');
  resultsList.innerHTML = '';
}

/* ══════════════════════════════════════════════════════════════════════════
   DETECTION — POST to backend
══════════════════════════════════════════════════════════════════════════ */
async function runDetection() {
  if (!currentFile) return;

  setLoading(true);
  resetResults();

  const form = new FormData();
  form.append('file', currentFile);   // field name must be "file" (FastAPI expects this)

  try {
    const res = await fetch(API_DETECT, { method: 'POST', body: form });

    if (!res.ok) {
      let detail = `Server error (HTTP ${res.status})`;
      try { const j = await res.json(); detail = j.detail || detail; } catch (_) {}
      throw new Error(detail);
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    // TypeError is thrown by fetch when the server is unreachable
    if (err instanceof TypeError) {
      showMessage(
        'error',
        '🔌 Could not reach the backend. Make sure the server is running:\n' +
        '.venv\\Scripts\\python.exe -m uvicorn backend.app.main:app --reload',
      );
    } else {
      showMessage('error', `⚠ ${err.message}`);
    }
    redrawImage();   // clear any partial overlay

  } finally {
    setLoading(false);
  }
}

function setLoading(on) {
  detectBtn.disabled    = on;
  detectLabel.textContent = on ? 'Detecting…' : 'Detect Age & Gender';
  btnSpinner.classList.toggle('hidden', !on);
  detectIcon.classList.toggle('hidden', on);
}

/* ══════════════════════════════════════════════════════════════════════════
   RENDER RESULTS
══════════════════════════════════════════════════════════════════════════ */
function renderResults(data) {
  const faces = Array.isArray(data.faces) ? data.faces : [];

  if (faces.length === 0) {
    redrawImage();
    showMessage('warn', '🔍 No faces were detected in this image. Try a clearer photo with visible, forward-facing faces.');
    return;
  }

  // 1. Draw bounding boxes on canvas
  redrawImage();
  faces.forEach((face, i) => drawBox(face.box, i + 1, FACE_COLORS[i % FACE_COLORS.length]));

  // 2. Build result cards
  resultsPlaceholder.classList.add('hidden');
  resultsList.classList.remove('hidden');

  const summary = document.createElement('p');
  summary.className = 'results-summary';
  summary.textContent =
    `${faces.length} face${faces.length !== 1 ? 's' : ''} detected`;
  resultsList.appendChild(summary);

  faces.forEach((face, i) => {
    const color = FACE_COLORS[i % FACE_COLORS.length];
    const card  = buildFaceCard(face, i, color);
    card.style.animationDelay = `${i * 0.07}s`;
    resultsList.appendChild(card);
  });
}

/* ══════════════════════════════════════════════════════════════════════════
   CANVAS: draw image + bounding boxes
══════════════════════════════════════════════════════════════════════════ */
function redrawImage() {
  if (!currentImage) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
}

/**
 * Draw a labelled bounding box.
 * API box coords are in original image pixels; scale by canvasScale first.
 */
function drawBox(box, label, color) {
  const s  = canvasScale;
  const x  = Math.round(box.x1 * s);
  const y  = Math.round(box.y1 * s);
  const w  = Math.round((box.x2 - box.x1) * s);
  const h  = Math.round((box.y2 - box.y1) * s);
  const lw = Math.max(2, Math.round(canvas.width / 280));   // line width scaled to canvas

  // Box outline
  ctx.strokeStyle = color;
  ctx.lineWidth   = lw;
  ctx.strokeRect(x, y, w, h);

  // Label badge above the box
  const fontSize = Math.max(11, Math.round(canvas.width / 42));
  ctx.font = `bold ${fontSize}px 'Segoe UI', sans-serif`;
  const pad    = lw * 2;
  const text   = `#${label}`;
  const tw     = ctx.measureText(text).width;
  const badgeH = fontSize + pad * 2;
  const badgeW = tw + pad * 2;
  const badgeY = Math.max(0, y - badgeH);

  ctx.fillStyle = color;
  ctx.fillRect(x, badgeY, badgeW, badgeH);
  ctx.fillStyle = '#fff';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, x + pad, badgeY + badgeH / 2);
  ctx.textBaseline = 'alphabetic';   // reset
}

/* ══════════════════════════════════════════════════════════════════════════
   BUILD A FACE RESULT CARD
══════════════════════════════════════════════════════════════════════════ */
function buildFaceCard(face, index, color) {
  const genderPct = (face.gender_confidence * 100).toFixed(1);
  const facePct   = (face.face_confidence   * 100).toFixed(1);
  const genderSym = face.gender === 'Female' ? '♀' : '♂';

  const card = document.createElement('article');
  card.className = 'face-card';

  card.innerHTML = `
    <div class="face-card-header">
      <div class="face-badge"
           style="background:${color}22;color:${color};border-color:${color}">
        ${index + 1}
      </div>
      <div>
        <div class="face-card-name">${genderSym} ${face.gender}, ${face.age} yrs</div>
        <div class="face-card-label">Face #${index + 1}</div>
      </div>
    </div>

    <div class="stat-grid">

      <!-- Gender -->
      <div class="stat-box">
        <div class="stat-label">Gender</div>
        <div class="stat-value" style="color:${color}">${face.gender}</div>
        <div class="conf-track">
          <div class="conf-fill" style="width:${genderPct}%"></div>
        </div>
        <div class="conf-pct">${genderPct}% confidence</div>
      </div>

      <!-- Age -->
      <div class="stat-box">
        <div class="stat-label">Estimated Age</div>
        <div class="stat-value">
          ${face.age}<span class="stat-unit">yrs</span>
        </div>
      </div>

      <!-- Face detection score (full-width) -->
      <div class="stat-box stat-full">
        <div class="stat-label">Face Detection Score</div>
        <div class="conf-track">
          <div class="conf-fill" style="width:${facePct}%"></div>
        </div>
        <div class="conf-pct">${facePct}%</div>
      </div>

    </div>
  `;

  return card;
}

/* ══════════════════════════════════════════════════════════════════════════
   STATUS MESSAGES (error / warn / info)
══════════════════════════════════════════════════════════════════════════ */
function showMessage(type, text) {
  resultsPlaceholder.classList.add('hidden');
  resultsList.classList.remove('hidden');
  resultsList.innerHTML = '';

  const msg = document.createElement('div');
  msg.className = `status-msg ${type}`;
  // Preserve \n as line breaks
  text.split('\n').forEach((line, i) => {
    if (i > 0) msg.appendChild(document.createElement('br'));
    msg.appendChild(document.createTextNode(line));
  });
  resultsList.appendChild(msg);
}


startWebcamBtn.addEventListener('click', async () => {
  try {
    webcamStatus.textContent = 'Requesting camera permission...';

    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    webcamVideo.srcObject = webcamStream;

    startWebcamBtn.disabled = true;
    captureWebcamBtn.disabled = false;
    stopWebcamBtn.disabled = false;

    webcamStatus.textContent = 'Webcam is running. Capture a photo to detect age and gender.';
  } catch (error) {
    console.error('Webcam error:', error);
    webcamStatus.textContent =
      'Unable to access webcam. Please allow camera permission and try again.';
  }
});

stopWebcamBtn.addEventListener('click', () => {
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }

  webcamVideo.srcObject = null;

  startWebcamBtn.disabled = false;
  captureWebcamBtn.disabled = true;
  stopWebcamBtn.disabled = true;

  webcamStatus.textContent = 'Webcam stopped.';
});


captureWebcamBtn.addEventListener('click', async () => {
  if (!webcamStream || !webcamVideo.videoWidth) {
    webcamStatus.textContent = 'Please start the webcam first.';
    return;
  }

  webcamCanvas.width = webcamVideo.videoWidth;
  webcamCanvas.height = webcamVideo.videoHeight;

  const context = webcamCanvas.getContext('2d');
  context.drawImage(
    webcamVideo,
    0,
    0,
    webcamCanvas.width,
    webcamCanvas.height
  );

  webcamStatus.textContent = 'Capturing photo...';

  webcamCanvas.toBlob(async (blob) => {
    if (!blob) {
      webcamStatus.textContent = 'Could not capture the photo. Try again.';
      return;
    }

    const file = new File([blob], 'webcam_photo.jpg', {
      type: 'image/jpeg'
    });

    // Reuse the existing image-upload workflow
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);

    fileInput.files = dataTransfer.files;
    fileInput.dispatchEvent(new Event('change', { bubbles: true }));

    webcamStatus.textContent =
      'Photo captured. Click Detect Age & Gender to run prediction.';
  }, 'image/jpeg', 0.95);
});
