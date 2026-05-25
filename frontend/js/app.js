/* ═══════════════════════════════════════════════════════════
   NEUROSCAN — APP LOGIC
   ═══════════════════════════════════════════════════════════ */

const API = 'http://localhost:8000';
const $ = id => document.getElementById(id);

const el = {
    stateUpload: $('stateUpload'),
    stateLoading: $('stateLoading'),
    stateResults: $('stateResults'),
    dropZone: $('dropZone'),
    dropEmpty: $('dropEmpty'),
    dropPreview: $('dropPreview'),
    fileInput: $('fileInput'),
    browseBtn: $('browseBtn'),
    previewImg: $('previewImg'),
    removeBtn: $('removeBtn'),
    analyzeBtn: $('analyzeBtn'),
    resetBtn: $('resetBtn'),
    verdictClass: $('verdictClass'),
    verdictDesc: $('verdictDesc'),
    verdictScore: $('verdictScore'),
    barsContainer: $('barsContainer'),
    heatmapsContainer: $('heatmapsContainer'),
    toastBox: $('toastBox'),
    themeToggle: $('themeToggle'),
};

let selectedFile = null;
let previewUrl = null;

/* ═══════════════════════════════════════════════════════════
   THEME
   ═══════════════════════════════════════════════════════════ */
function getTheme() {
    const saved = localStorage.getItem('ns-theme');
    if (saved) return saved;
    return 'light'; // default to light mode
}

function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('ns-theme', theme);
}

setTheme(getTheme());

el.themeToggle.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme');
    setTheme(current === 'dark' ? 'light' : 'dark');
});

/* ═══════════════════════════════════════════════════════════
   STATE SWITCHING
   ═══════════════════════════════════════════════════════════ */
function showState(name) {
    el.stateUpload.style.display = name === 'upload' ? '' : 'none';
    el.stateLoading.style.display = name === 'loading' ? '' : 'none';
    el.stateResults.style.display = name === 'results' ? '' : 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

/* ═══════════════════════════════════════════════════════════
   TOASTS
   ═══════════════════════════════════════════════════════════ */
function toast(msg, type = '') {
    const t = document.createElement('div');
    t.className = 'toast' + (type ? ' ' + type : '');
    t.textContent = msg;
    el.toastBox.appendChild(t);
    setTimeout(() => {
        t.style.opacity = '0';
        t.style.transition = 'opacity 0.25s';
        setTimeout(() => t.remove(), 300);
    }, 4000);
}

/* ═══════════════════════════════════════════════════════════
   FILE HANDLING
   ═══════════════════════════════════════════════════════════ */
function pickFile(file) {
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
        toast('Only JPEG and PNG are supported', 'err');
        return;
    }
    if (file.size > 10 * 1024 * 1024) {
        toast('File must be under 10 MB', 'err');
        return;
    }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = e => {
        previewUrl = e.target.result;
        el.previewImg.src = previewUrl;
        el.dropEmpty.style.display = 'none';
        el.dropPreview.style.display = '';
        el.analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function clearFile() {
    selectedFile = null;
    previewUrl = null;
    el.fileInput.value = '';
    el.previewImg.src = '';
    el.dropEmpty.style.display = '';
    el.dropPreview.style.display = 'none';
    el.analyzeBtn.disabled = true;
}

/* ═══════════════════════════════════════════════════════════
   API
   ═══════════════════════════════════════════════════════════ */
async function predict(file) {
    const fd = new FormData();
    fd.append('file', file);
    const res = await fetch(`${API}/api/predict`, { method: 'POST', body: fd });
    if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || 'Prediction failed');
    }
    return res.json();
}

/* ═══════════════════════════════════════════════════════════
   RENDER RESULTS
   ═══════════════════════════════════════════════════════════ */
const LABELS = {
    NonDemented: 'Non Demented',
    VeryMildDemented: 'Very Mild Dementia',
    MildDemented: 'Mild Dementia',
    ModerateDemented: 'Moderate Dementia',
};


function render(data) {
    const cls = data.predicted_class;

    el.verdictClass.textContent = LABELS[cls] || cls;
    el.verdictDesc.textContent = data.description;
    el.verdictScore.textContent = (data.confidence * 100).toFixed(1) + '%';

    // Bars
    el.barsContainer.innerHTML = '';
    const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    entries.forEach(([name, prob], i) => {
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.innerHTML =
            '<div class="bar-header">' +
            '<span class="bar-label">' + (LABELS[name] || name) + '</span>' +
            '<span class="bar-pct">' + (prob * 100).toFixed(1) + '%</span>' +
            '</div>' +
            '<div class="bar-track"><div class="bar-fill' + (name === cls ? ' bar-fill--top' : '') + '" id="bf' + i + '"></div></div>';
        el.barsContainer.appendChild(bar);
        requestAnimationFrame(() => requestAnimationFrame(() => {
            const fill = document.getElementById('bf' + i);
            if (fill) fill.style.width = Math.max(prob * 100, 1.5) + '%';
        }));
    });

    // Heatmaps
    el.heatmapsContainer.innerHTML = '';
    addHM(previewUrl, 'Original Scan');
    if (data.gradcam_images) {
        if (data.gradcam_images.overlay) {
            addHM('data:image/jpeg;base64,' + data.gradcam_images.overlay, 'Attention Map');
        }
    }

    showState('results');
}

function addHM(src, cap) {
    const div = document.createElement('div');
    div.className = 'hm';
    div.innerHTML = '<div class="hm-img"><img src="' + src + '" alt="' + cap + '"></div><div class="hm-cap">' + cap + '</div>';
    el.heatmapsContainer.appendChild(div);
}

/* ═══════════════════════════════════════════════════════════
   RUN ANALYSIS
   ═══════════════════════════════════════════════════════════ */
async function runAnalysis() {
    if (!selectedFile) return;
    showState('loading');
    try {
        const data = await predict(selectedFile);
        render(data);
    } catch (err) {
        console.error(err);
        toast(err.message || 'Something went wrong', 'err');
        showState('upload');
    }
}

/* ═══════════════════════════════════════════════════════════
   EVENTS
   ═══════════════════════════════════════════════════════════ */
el.browseBtn.addEventListener('click', e => { e.stopPropagation(); el.fileInput.click(); });
el.dropZone.addEventListener('click', () => { if (!selectedFile) el.fileInput.click(); });
el.fileInput.addEventListener('change', e => { if (e.target.files[0]) pickFile(e.target.files[0]); });

el.dropZone.addEventListener('dragover', e => { e.preventDefault(); el.dropZone.classList.add('dragging'); });
el.dropZone.addEventListener('dragleave', () => el.dropZone.classList.remove('dragging'));
el.dropZone.addEventListener('drop', e => {
    e.preventDefault();
    el.dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) pickFile(e.dataTransfer.files[0]);
});

el.removeBtn.addEventListener('click', e => { e.stopPropagation(); clearFile(); });
el.analyzeBtn.addEventListener('click', runAnalysis);
el.resetBtn.addEventListener('click', () => { clearFile(); showState('upload'); });

document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => e.preventDefault());
