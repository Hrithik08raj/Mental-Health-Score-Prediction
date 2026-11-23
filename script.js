// MindScore – Powered by FastAPI Backend

const API_BASE = (window.location.protocol === 'file:' || (window.location.port !== '8000' && window.location.port !== ''))
  ? 'http://127.0.0.1:8000'
  : '';

const RANGES = {
  Age: [18, 24],
  Avg_Daily_Usage_Hours: [1, 8.8],
  Daily_Unlocks: [62, 273],
  Study_Hours: [0.3, 8.3],
  Physical_Activity_Hours: [0, 4.1],
  Sleep_Hours_Per_Night: [3.6, 9.9]
};

const CATEGORIES = {
  Most_Used_Platform: [
    'Facebook', 'Instagram', 'KakaoTalk', 'LINE', 'LinkedIn',
    'Snapchat', 'TikTok', 'Twitter', 'VKontakte', 'WeChat', 'WhatsApp', 'YouTube'
  ],
  Purpose_Of_Use: ['Education', 'Entertainment', 'Networking', 'News'],
  Grouped_country: [
    'Australia', 'Canada', 'France', 'Germany', 'India', 'Mexico',
    'Other', 'Spain', 'Turkey', 'UK', 'USA'
  ]
};

const DATASET_MEAN = 6.23;
const NUMBERS = Object.keys(RANGES);
const $ = id => document.getElementById(id);
const form = document.getElementById('form');
const statusPill = $('backend-status');

let currentAbortController = null;
let debounceTimer = null;
let isBackendHealthy = false;

// Populate dropdowns with defaults
const defaultSelections = {
  Most_Used_Platform: 'Instagram',
  Purpose_Of_Use: 'Entertainment',
  Grouped_country: 'India'
};

function populateDropdowns(cats = CATEGORIES) {
  for (const k of Object.keys(defaultSelections)) {
    const el = $(k);
    if (!el || !cats[k]) continue;
    const currentVal = el.value || defaultSelections[k];
    el.innerHTML = cats[k]
      .map(c => `<option value="${c}"${c === currentVal ? ' selected' : ''}>${c}</option>`)
      .join('');
  }
}

function setBackendStatus(state, message) {
  if (!statusPill) return;
  statusPill.className = `status-pill status-${state}`;
  const textEl = statusPill.querySelector('.status-text');
  if (textEl) {
    textEl.textContent = message;
  }
}

function clamp(key, v) {
  const [lo, hi] = RANGES[key];
  return Math.min(Math.max(v, lo), hi);
}

function readForm() {
  const d = {};
  new FormData(form).forEach((v, k) => {
    d[k] = NUMBERS.includes(k) ? Number(v) : v;
  });
  NUMBERS.forEach(k => {
    d[k] = clamp(k, d[k]);
  });
  return d;
}

function syncSliders() {
  form.querySelectorAll('input[type=range]').forEach(r => {
    r.style.setProperty('--fill', ((r.value - r.min) / (r.max - r.min) * 100) + '%');
    const out = form.querySelector(`[data-out="${r.name}"]`);
    if (out) {
      out.textContent = (r.step === '1' ? r.value : Number(r.value).toFixed(1)) + (r.dataset.unit || '');
    }
  });
}

function bandFor(s) {
  if (s < 5) return 'Under pressure';
  if (s < 6.5) return 'Holding steady';
  if (s < 8) return 'Doing well';
  return 'Thriving';
}

function renderUIFromResponse(data) {
  const score = Number(data.score);
  const diff = Number(data.diff_from_mean ?? (score - DATASET_MEAN));
  const band = data.band || bandFor(score);

  // Update circular gauge and main readout
  $('fill').style.strokeDashoffset = 100 - score * 10;
  $('score').textContent = score.toFixed(1);
  $('band').textContent = band;

  // Survey comparison readout
  if (Math.abs(diff) < 0.05) {
    $('vs').innerHTML = 'Right at the survey average of 6.2.';
  } else {
    const isAbove = diff > 0;
    $('vs').innerHTML = `<b class="${isAbove ? 'up' : 'down'}">${Math.abs(diff).toFixed(1)} ${isAbove ? 'above' : 'below'}</b> the survey average of 6.2.`;
  }

  // Counterfactual What-If changes list
  const changesEl = $('changes');
  if (data.what_if && data.what_if.length > 0) {
    changesEl.innerHTML = data.what_if.map(c => {
      const k = c.direction || (c.delta > 0.04 ? 'pos' : c.delta < -0.04 ? 'neg' : 'zero');
      const sign = c.delta > 0.004 ? '+' : c.delta < -0.004 ? '−' : '';
      return `<li><span>${c.label}</span><span class="${k}">${sign}${Math.abs(c.delta).toFixed(2)}</span></li>`;
    }).join('');
  }
}

// Fallback in case backend is loading / unreachable
function clientSideFallback(r) {
  if (window.MODEL) {
    const M = window.MODEL;
    const STRESS = ['Low', 'Medium', 'High', 'Very High'];
    const NUM_ORDER = ['Age', 'Avg_Daily_Usage_Hours', 'Daily_Unlocks', 'Physical_Activity_Hours', 'Sleep_Hours_Per_Night'];
    const CAT_ORDER = ['Gender', 'Academic_Level', 'Most_Used_Platform', 'Purpose_Of_Use', 'Grouped_country'];

    function toFeatures(row) {
      const f = [(Math.log1p(row.Study_Hours) - M.study.mean) / M.study.scale];
      NUM_ORDER.forEach((k, i) => f.push((row[k] - M.num.mean[i]) / M.num.scale[i]));
      f.push(STRESS.indexOf(row.Stress_Level));
      CAT_ORDER.forEach(k => M.cats[k].forEach(c => f.push(row[k] === c ? 1 : 0)));
      return f;
    }

    function predictSingle(row) {
      const x = toFeatures(row);
      let sum = 0;
      for (const t of M.trees) {
        let n = 0;
        while (t[n * 4] !== -1) n = x[t[n * 4]] <= t[n * 4 + 1] ? t[n * 4 + 2] : t[n * 4 + 3];
        sum += t[n * 4 + 1];
      }
      return Math.min(Math.max(sum / M.trees.length, 0), 10);
    }

    const score = predictSingle(r);
    const tweaks = [
      ['Sleep one more hour', 'Sleep_Hours_Per_Night', +1],
      ['Use social media one hour less', 'Avg_Daily_Usage_Hours', -1],
      ['Move half an hour more a day', 'Physical_Activity_Hours', +0.5],
      ['Unlock your phone 30 fewer times', 'Daily_Unlocks', -30]
    ];
    const what_if = tweaks.map(([label, key, d]) => {
      const v = clamp(key, r[key] + d);
      if (v === r[key]) return null;
      const delta = predictSingle({ ...r, [key]: v }) - score;
      const direction = delta > 0.04 ? 'pos' : delta < -0.04 ? 'neg' : 'zero';
      return { label, delta: Number(delta.toFixed(2)), direction };
    }).filter(Boolean).sort((a, b) => b.delta - a.delta);

    renderUIFromResponse({
      score: score.toFixed(1),
      band: bandFor(score),
      diff_from_mean: (score - DATASET_MEAN).toFixed(1),
      what_if
    });
  }
}

async function requestPrediction(formData) {
  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();

  try {
    const response = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData),
      signal: currentAbortController.signal
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    renderUIFromResponse(data);
    isBackendHealthy = true;
    setBackendStatus('online', 'FastAPI Connected');
  } catch (err) {
    if (err.name === 'AbortError') {
      return; // Request was aborted due to newer user input
    }
    console.warn('FastAPI request failed:', err);
    isBackendHealthy = false;
    setBackendStatus('offline', 'FastAPI Offline (Fallback)');
    clientSideFallback(formData);
  }
}

function handleInput() {
  syncSliders();
  const formData = readForm();

  // Debounce API calls for responsive 60fps sliding
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    requestPrediction(formData);
  }, 40);
}

async function init() {
  populateDropdowns();
  syncSliders();

  // Initial health check & fetch metadata from FastAPI
  try {
    const healthRes = await fetch(`${API_BASE}/api/health`);
    if (healthRes.ok) {
      const healthData = await healthRes.json();
      if (healthData.model_loaded) {
        isBackendHealthy = true;
        setBackendStatus('online', 'FastAPI Connected');
      }
    }

    const metaRes = await fetch(`${API_BASE}/api/metadata`);
    if (metaRes.ok) {
      const meta = await metaRes.json();
      if (meta.categories) {
        populateDropdowns(meta.categories);
      }
    }
  } catch (err) {
    console.warn('Could not reach FastAPI on startup:', err);
    setBackendStatus('offline', 'FastAPI Offline');
  }

  // Trigger initial prediction
  requestPrediction(readForm());
  form.addEventListener('input', handleInput);
}

// Start application
init();
