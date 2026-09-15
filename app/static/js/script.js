/**
 * =============================================================
 * Campus Placement Prediction System — Frontend Client Engine
 * Features:
 *   - Slider ↔ display badge synchronization + gradient fill
 *   - Preset profile chip one-click fill (sliders, selects, radios)
 *   - AJAX prediction via /api/predict with loading overlay
 *   - Animated circular SVG probability gauge
 *   - Smooth counter animation for gauge percentage
 *   - Dynamic benchmark bars, XAI factors, recommendation cards
 *   - Mobile nav toggle
 * =============================================================
 */

'use strict';

/* ── Entry Point ─────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  initSliderSync();
  initSliderFill();
  initPresetChips();
  initPredictionForm();
  initExistingGauge();
  initMobileNav();
});

/* ── 1. Slider ↔ Badge Synchronization ───────────────────────────────────── */
function initSliderSync() {
  document.querySelectorAll('input[type="range"]').forEach(slider => {
    const targetId = slider.getAttribute('data-sync');
    if (!targetId) return;

    const display = document.getElementById(targetId);
    if (!display) return;

    // Initialize badge value
    display.textContent = parseFloat(slider.value).toFixed(getDecimalPlaces(slider));

    slider.addEventListener('input', (e) => {
      display.textContent = parseFloat(e.target.value).toFixed(getDecimalPlaces(slider));
      updateSliderFill(slider);
    });
  });
}

function getDecimalPlaces(slider) {
  const step = parseFloat(slider.step);
  if (!step || step >= 1) return 0;
  return (step.toString().split('.')[1] || '').length;
}

/* ── 2. Slider Range-Fill Visual Gradient ────────────────────────────────── */
function initSliderFill() {
  document.querySelectorAll('input[type="range"]').forEach(slider => {
    updateSliderFill(slider);
  });
}

function updateSliderFill(slider) {
  const min = parseFloat(slider.min);
  const max = parseFloat(slider.max);
  const val = parseFloat(slider.value);
  const pct = ((val - min) / (max - min)) * 100;
  // Matches --clr-primary (#4f6ef7) and --surface-raised (#171c2b)
  slider.style.background = `linear-gradient(90deg, #4f6ef7 ${pct}%, #171c2b ${pct}%)`;
}

/* ── 3. Preset Profile Chip Handling ─────────────────────────────────────── */
function initPresetChips() {
  const chips = document.querySelectorAll('.preset-chip');
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      chips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');

      const presetDataRaw = chip.getAttribute('data-preset');
      if (!presetDataRaw) return;

      try {
        const profile = JSON.parse(presetDataRaw);
        populateForm(profile);
        submitPrediction();
      } catch (err) {
        console.error('[Preset] Failed to parse preset JSON:', err);
      }
    });
  });
}

/**
 * Populate all form controls from a flat key→value dictionary.
 * Handles: range sliders, number inputs, selects, and radio buttons.
 */
function populateForm(data) {
  for (const [key, value] of Object.entries(data)) {
    // 1. Try a direct element match by id
    const directEl = document.getElementById(key);
    if (directEl) {
      directEl.value = value;
      // Sync badge display
      const displayId = directEl.getAttribute('data-sync');
      if (displayId) {
        const display = document.getElementById(displayId);
        if (display) {
          display.textContent = parseFloat(value).toFixed(getDecimalPlaces(directEl));
        }
      }
      // Update slider fill if applicable
      if (directEl.type === 'range') updateSliderFill(directEl);
      continue;
    }

    // 2. Handle radio groups (name="gender", name="volunteer_experience", etc.)
    const radios = document.querySelectorAll(`input[type="radio"][name="${key}"]`);
    if (radios.length > 0) {
      radios.forEach(radio => {
        radio.checked = String(radio.value) === String(value);
      });
    }
  }
}

/* ── 4. Form Intercept → AJAX ────────────────────────────────────────────── */
function initPredictionForm() {
  const form = document.getElementById('predictionForm');
  if (!form) return;

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    submitPrediction();
  });
}

/**
 * Collect all form inputs, call /api/predict, and animate results.
 */
async function submitPrediction() {
  const form = document.getElementById('predictionForm');
  const btn  = document.getElementById('predictBtn');
  if (!form) return;

  showLoadingOverlay(true);
  if (btn) {
    btn.disabled = true;
    btn.textContent = '⏳ Analysing Profile…';
  }

  // Collect form data into a plain object (handles all input types)
  const formData = new FormData(form);
  const payload  = {};
  formData.forEach((val, key) => { payload[key] = val; });

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const json = await res.json();

    if (json.success) {
      updateResultDashboard(json.prediction);
    } else {
      showInlineError('Prediction failed: ' + (json.error || 'Unknown error.'));
    }

  } catch (networkErr) {
    console.error('[API] Network error:', networkErr);
    // Graceful fallback: normal form submit
    form.submit();
  } finally {
    showLoadingOverlay(false);
    if (btn) {
      btn.disabled = false;
      btn.textContent = '⚡ Execute Ensemble Prediction';
    }
  }
}

/* ── 5. Result Dashboard Updater ─────────────────────────────────────────── */
/**
 * Animates all result card components from a JSON prediction payload.
 * Targets the new rc-* result card structure.
 */
function updateResultDashboard(pred) {
  const prob     = parseFloat(pred.probability);
  const isPlaced = Boolean(pred.is_placed);

  /* ── Status Banner ── */
  const banner = document.getElementById('statusBanner');
  const icon   = document.getElementById('statusIcon');
  const label  = document.getElementById('statusLabel');
  if (banner) {
    banner.className = 'rc-status-banner ' + (isPlaced ? 'rc-banner-placed' : 'rc-banner-risk');
  }
  if (icon)  icon.textContent  = isPlaced ? '✓' : '⚠';
  if (label) label.textContent = isPlaced ? 'Placement Likely' : 'Placement At Risk';

  /* ── Probability Number ── */
  const probEl = document.getElementById('gaugePercentText');
  if (probEl) {
    // Rebuild to preserve the <span class="rc-prob-unit"> child
    probEl.innerHTML = prob.toFixed(1) + '<span class="rc-prob-unit">%</span>';
  }

  /* ── Probability Bar ── */
  const probBar = document.getElementById('probBarFill');
  if (probBar) {
    probBar.style.width = prob + '%';
    probBar.className   = 'rc-prob-bar-fill ' + (isPlaced ? 'rc-bar-placed' : 'rc-bar-risk');
  }

  /* ── Hidden SVG circle (kept for initExistingGauge compat) ── */
  const gaugeCircle = document.getElementById('gaugeCircle');
  if (gaugeCircle) {
    gaugeCircle.setAttribute('data-probability', prob);
    gaugeCircle.setAttribute('class',
      'gauge-progress ' + (isPlaced ? 'gauge-placed' : 'gauge-not-placed'));
  }

  /* ── Confidence Pill ── */
  const pill = document.getElementById('statusPill');
  if (pill) {
    pill.textContent = pred.risk_level || (isPlaced ? 'High Confidence' : 'Action Needed');
    pill.className = 'rc-conf-pill rc-conf-' + (pred.risk_badge || 'warning');
  }

  /* ── Salary ── */
  const salaryVal   = document.getElementById('salaryValue');
  const salaryRange = document.getElementById('salaryRange');
  if (salaryVal)   salaryVal.textContent   = '₹ ' + pred.salary_lpa + ' LPA';
  if (salaryRange) salaryRange.textContent = 'Estimated Range \u00a0₹ ' + pred.salary_range;

  /* ── Factor Bars (benchmark items) ── */
  if (pred.benchmarks && pred.benchmarks.length) {
    pred.benchmarks.forEach(item => {
      const safeId  = item.label.replace(/[^a-zA-Z0-9]/g, '');
      const bar     = document.getElementById('bench-bar-' + safeId);
      const valText = document.getElementById('bench-val-' + safeId);
      if (bar)     bar.style.width     = item.user_pct + '%';
      if (valText) valText.textContent = item.user_value;
    });
  }

  /* ── XAI Signals ── */
  const signalsContainer = document.getElementById('signalsList');
  if (signalsContainer && pred.factors) {
    signalsContainer.innerHTML = '';
    pred.factors.forEach(f => {
      const isPos = f.impact === 'positive';
      const div   = document.createElement('div');
      div.className = 'rc-signal ' + (isPos ? 'rc-signal-pos' : 'rc-signal-neg');
      div.innerHTML = `
        <span class="rc-signal-dot">${isPos ? '▲' : '▼'}</span>
        <div class="rc-signal-body">
          <div class="rc-signal-name">${escapeHtml(f.name)}
            <span class="rc-signal-badge">${escapeHtml(f.badge)}</span>
          </div>
          <div class="rc-signal-desc">${escapeHtml(f.desc)}</div>
        </div>`;
      signalsContainer.appendChild(div);
    });
  }

  /* ── Recommendations ── */
  const recsContainer = document.getElementById('recommendationsList');
  if (recsContainer && pred.recommendations) {
    recsContainer.innerHTML = '';
    pred.recommendations.forEach(rec => {
      const card = document.createElement('div');
      card.className = `rec-card rec-${rec.type}`;
      card.innerHTML = `
        <div class="rec-title">${escapeHtml(rec.title)}</div>
        <div class="rec-desc">${escapeHtml(rec.description)}</div>
      `;
      recsContainer.appendChild(card);
    });
  }

  /* 5h. Scroll result panel into view on mobile */
  const resultPanel = document.getElementById('resultDashboard');
  if (resultPanel && window.innerWidth <= 1024) {
    resultPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

/* ── 6. Circular SVG Gauge Animation ─────────────────────────────────────── */
// Circumference = 2π × 80 ≈ 502.65  (must match stroke-dasharray in CSS)
const CIRCUMFERENCE = 2 * Math.PI * 80;

function animateGauge(probability, isPlaced) {
  const circle = document.getElementById('gaugeCircle');
  if (!circle) return;

  const offset = CIRCUMFERENCE - (CIRCUMFERENCE * probability / 100);
  circle.style.strokeDashoffset = offset;
  circle.setAttribute('class',
    'gauge-progress ' + (isPlaced ? 'gauge-placed' : 'gauge-not-placed')
  );
}

/**
 * Initialize gauge from server-rendered data attribute on page load.
 */
function initExistingGauge() {
  const circle = document.getElementById('gaugeCircle');
  if (!circle) return;

  const probAttr = circle.getAttribute('data-probability');
  if (!probAttr) return;

  const prob = parseFloat(probAttr);
  // Defer slightly to allow CSS transition to play
  requestAnimationFrame(() => {
    setTimeout(() => animateGauge(prob, circle.classList.contains('gauge-placed')), 180);
  });
}

/* ── 7. Smooth Counter Animation ─────────────────────────────────────────── */
/**
 * @param {HTMLElement} el      - Element whose textContent will be animated
 * @param {number}      target  - Final numeric value
 * @param {string}      suffix  - Text appended after value (e.g. '%')
 * @param {number}      decimals - Number of decimal places to display
 */
function animateCount(el, target, suffix = '', decimals = 1) {
  const duration  = 900;
  const startTime = performance.now();
  const startVal  = parseFloat(el.textContent) || 0;

  function step(now) {
    const elapsed  = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const eased   = 1 - Math.pow(1 - progress, 3);
    const current = startVal + (target - startVal) * eased;
    el.textContent = current.toFixed(decimals) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }

  requestAnimationFrame(step);
}

/* ── 8. Loading Overlay ──────────────────────────────────────────────────── */
function showLoadingOverlay(visible) {
  let overlay = document.getElementById('loadingOverlay');

  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id = 'loadingOverlay';
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
      <div class="spinner"></div>
      <div class="loading-label">Running Ensemble Inference…</div>
    `;
    document.body.appendChild(overlay);
  }

  if (visible) {
    overlay.classList.add('active');
  } else {
    overlay.classList.remove('active');
  }
}

/* ── 9. Error Display ────────────────────────────────────────────────────── */
function showInlineError(message) {
  // Attempt to surface error in the form panel
  const formPanel = document.getElementById('formPanel');
  if (!formPanel) { alert(message); return; }

  let errEl = document.getElementById('inlineError');
  if (!errEl) {
    errEl = document.createElement('div');
    errEl.id = 'inlineError';
    errEl.className = 'rec-card rec-danger';
    errEl.style.marginBottom = '1rem';
    // Insert before form
    const form = document.getElementById('predictionForm');
    if (form) formPanel.insertBefore(errEl, form);
    else formPanel.prepend(errEl);
  }

  errEl.innerHTML = `<div class="rec-title">⚠ Prediction Error</div><div class="rec-desc">${escapeHtml(message)}</div>`;
  errEl.style.display = 'block';

  // Auto-hide after 8 seconds
  setTimeout(() => { if (errEl) errEl.style.display = 'none'; }, 8000);
}

/* ── 10. XSS-safe HTML escaping ─────────────────────────────────────────── */
function escapeHtml(str) {
  if (typeof str !== 'string') return String(str);
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/* ── 11. Mobile Navigation Toggle ───────────────────────────────────────── */
function initMobileNav() {
  const toggle = document.querySelector('.nav-toggle');
  const links  = document.querySelector('.nav-links');
  if (!toggle || !links) return;

  toggle.addEventListener('click', () => {
    const isOpen = links.classList.toggle('open');
    toggle.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  });

  // Close on outside click
  document.addEventListener('click', (e) => {
    if (!toggle.contains(e.target) && !links.contains(e.target)) {
      links.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    }
  });

  // Close when a link is clicked
  links.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => {
      links.classList.remove('open');
      toggle.setAttribute('aria-expanded', 'false');
    });
  });
}
