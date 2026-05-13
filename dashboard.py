"""
dashboard.py
============
Flask + SocketIO IoT dashboard for the Smart Waste Segregation System.

Routes:
    GET  /           → Live dashboard UI
    GET  /api/status → Latest classification + sensor JSON
    POST /api/classify → Trigger classification from uploaded image

Real-time push: every classification result is broadcast via WebSocket
to all connected browser clients.

Author : Smart Waste Segregation System
License: MIT
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy Flask / SocketIO imports (keeps module importable without flask)
# ---------------------------------------------------------------------------

_flask_available = False
try:
    from flask import Flask, render_template_string, request, jsonify
    from flask_socketio import SocketIO, emit
    _flask_available = True
except ImportError:
    logger.warning("Flask / flask-socketio not installed. Dashboard unavailable.")


# ---------------------------------------------------------------------------
# HTML Template (self-contained, no external files needed)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Smart Waste Segregation — Dashboard</title>
  <script src="https://cdn.socket.io/4.6.1/socket.io.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0f172a; --card: #1e293b; --border: #334155;
      --text: #e2e8f0; --muted: #94a3b8;
      --wet: #22c55e; --rec: #3b82f6; --haz: #ef4444;
      --accent: #6366f1;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; }
    header {
      background: linear-gradient(135deg, var(--accent) 0%, #0ea5e9 100%);
      padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem;
    }
    header h1 { font-size: 1.4rem; font-weight: 700; }
    header span { font-size: 0.85rem; opacity: 0.85; }
    .live-dot { width:10px; height:10px; border-radius:50%; background:#22c55e;
      animation: pulse 1.5s infinite; margin-left:auto; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

    main { padding: 1.5rem 2rem; display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.2rem; }

    .card { background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 1.2rem; }
    .card h2 { font-size: 0.75rem; text-transform: uppercase; letter-spacing:.08em;
      color: var(--muted); margin-bottom: .8rem; }

    /* Category badge */
    .badge { display:inline-block; padding:.3rem .9rem; border-radius:999px;
      font-weight:700; font-size:1rem; }
    .badge.Wet        { background:rgba(34,197,94,.2);  color:var(--wet); }
    .badge.Recyclable { background:rgba(59,130,246,.2); color:var(--rec); }
    .badge.Hazardous  { background:rgba(239,68,68,.2);  color:var(--haz); }
    .badge.default    { background:rgba(148,163,184,.15); color:var(--muted); }

    #category-label { font-size: 2rem; font-weight:800; margin:.4rem 0; }
    #confidence-bar-wrap { height:8px; background:var(--border); border-radius:4px; overflow:hidden; }
    #confidence-bar { height:100%; border-radius:4px; background:var(--accent);
      transition: width .5s ease; width:0%; }
    #confidence-text { font-size:.85rem; color:var(--muted); margin-top:.4rem; }

    /* Servo */
    .servo-visual { text-align:center; padding:.5rem 0; }
    #servo-angle-text { font-size:2.5rem; font-weight:800; color:var(--accent); }
    #servo-desc { font-size:.85rem; color:var(--muted); margin-top:.2rem; }
    canvas#servoCanvas { display:block; margin:.8rem auto 0; }

    /* Sensor grid */
    .sensor-grid { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; }
    .sensor-item label { font-size:.7rem; text-transform:uppercase; letter-spacing:.06em;
      color:var(--muted); display:block; margin-bottom:.2rem; }
    .sensor-item .val { font-size:1.3rem; font-weight:700; }

    /* Bin fill */
    .bin-row { display:flex; align-items:center; gap:.8rem; margin-bottom:.7rem; }
    .bin-row .bin-name { width:90px; font-size:.8rem; color:var(--muted); }
    .bin-bar-wrap { flex:1; height:14px; background:var(--border); border-radius:7px; overflow:hidden; }
    .bin-bar { height:100%; border-radius:7px; transition:width .6s ease; }
    .bin-bar.wet  { background:var(--wet); }
    .bin-bar.rec  { background:var(--rec); }
    .bin-bar.haz  { background:var(--haz); }
    .bin-pct { width:36px; font-size:.8rem; text-align:right; }

    /* Reasoning */
    #reasoning-list { list-style:none; }
    #reasoning-list li { font-size:.82rem; color:var(--muted); padding:.25rem 0;
      border-bottom:1px solid var(--border); }
    #reasoning-list li:last-child { border:none; }

    /* History chart */
    .chart-wrap { position:relative; height:180px; }

    /* Upload */
    #upload-area { border:2px dashed var(--border); border-radius:10px;
      padding:1.5rem; text-align:center; cursor:pointer; transition:border-color .2s; }
    #upload-area:hover { border-color:var(--accent); }
    #upload-area p { font-size:.85rem; color:var(--muted); }
    #file-input { display:none; }
    #preview-img { max-width:100%; border-radius:8px; margin-top:.8rem;
      display:none; border:1px solid var(--border); }

    footer { text-align:center; padding:1rem; color:var(--muted); font-size:.78rem; }
  </style>
</head>
<body>
<header>
  <div>♻️</div>
  <div>
    <h1>Smart Waste Segregation System</h1>
    <span>AI-Powered Sorting Dashboard — Real-time IoT Monitor</span>
  </div>
  <div class="live-dot" title="Live"></div>
</header>

<main>

  <!-- Classification Result -->
  <div class="card">
    <h2>🔍 Classification Result</h2>
    <div id="category-label" class="badge default">—</div>
    <div id="confidence-bar-wrap"><div id="confidence-bar"></div></div>
    <div id="confidence-text">Awaiting classification…</div>
    <div style="margin-top:.8rem; font-size:.8rem; color:var(--muted);">
      ImageNet label: <span id="imagenet-label">—</span>
    </div>
  </div>

  <!-- Servo Control -->
  <div class="card">
    <h2>⚙️ Servo Position</h2>
    <div class="servo-visual">
      <div id="servo-angle-text">90°</div>
      <div id="servo-desc">Neutral / Standby</div>
      <canvas id="servoCanvas" width="160" height="90"></canvas>
    </div>
  </div>

  <!-- Sensor Readings -->
  <div class="card">
    <h2>📡 Sensor Readings</h2>
    <div class="sensor-grid">
      <div class="sensor-item">
        <label>💧 Moisture</label>
        <div class="val" id="s-moisture">—</div>
      </div>
      <div class="sensor-item">
        <label>💨 Gas (ppm)</label>
        <div class="val" id="s-gas">—</div>
      </div>
      <div class="sensor-item">
        <label>⚖️ Weight (g)</label>
        <div class="val" id="s-weight">—</div>
      </div>
      <div class="sensor-item">
        <label>🕐 Last Update</label>
        <div class="val" style="font-size:.9rem;" id="s-time">—</div>
      </div>
    </div>
  </div>

  <!-- Bin Fill Levels -->
  <div class="card">
    <h2>🗑️ Bin Fill Levels</h2>
    <div class="bin-row">
      <div class="bin-name">🟢 Wet</div>
      <div class="bin-bar-wrap"><div class="bin-bar wet" id="bar-wet" style="width:0%"></div></div>
      <div class="bin-pct" id="pct-wet">0%</div>
    </div>
    <div class="bin-row">
      <div class="bin-name">🔵 Recyclable</div>
      <div class="bin-bar-wrap"><div class="bin-bar rec" id="bar-rec" style="width:0%"></div></div>
      <div class="bin-pct" id="pct-rec">0%</div>
    </div>
    <div class="bin-row">
      <div class="bin-name">🔴 Hazardous</div>
      <div class="bin-bar-wrap"><div class="bin-bar haz" id="bar-haz" style="width:0%"></div></div>
      <div class="bin-pct" id="pct-haz">0%</div>
    </div>
    <div id="bin-alert" style="font-size:.8rem;color:#f59e0b;margin-top:.5rem;display:none;">
      ⚠ One or more bins approaching capacity!
    </div>
  </div>

  <!-- Reasoning -->
  <div class="card">
    <h2>🧠 Fusion Reasoning</h2>
    <ul id="reasoning-list">
      <li style="color:var(--muted)">Run a classification to see reasoning…</li>
    </ul>
  </div>

  <!-- Category History Chart -->
  <div class="card">
    <h2>📊 Category History (last 20)</h2>
    <div class="chart-wrap">
      <canvas id="historyChart"></canvas>
    </div>
  </div>

  <!-- Image Upload / Classify -->
  <div class="card">
    <h2>📷 Classify Image</h2>
    <div id="upload-area" onclick="document.getElementById('file-input').click()">
      <p>Click to upload an image for classification</p>
      <input type="file" id="file-input" accept="image/*">
      <img id="preview-img" src="" alt="preview">
    </div>
    <button onclick="classifyUploaded()" style="
      margin-top:.8rem; width:100%; padding:.6rem; border:none;
      background:var(--accent); color:#fff; border-radius:8px;
      font-weight:600; cursor:pointer; font-size:.9rem;">
      ▶ Classify
    </button>
    <div id="classify-status" style="font-size:.8rem;color:var(--muted);margin-top:.4rem;"></div>
  </div>

</main>

<footer>Smart Waste Segregation System · MIT License · AI + IoT + Embedded Systems</footer>

<script>
const socket = io();
let historyData = { labels: [], wet: [], rec: [], haz: [] };

// --- Servo canvas ---
function drawServo(angle) {
  const canvas = document.getElementById('servoCanvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, 160, 90);
  // Arc
  ctx.beginPath();
  ctx.arc(80, 80, 60, Math.PI, 2 * Math.PI);
  ctx.strokeStyle = '#334155'; ctx.lineWidth = 10; ctx.stroke();
  // Needle
  const rad = Math.PI + (angle / 180) * Math.PI;
  ctx.beginPath();
  ctx.moveTo(80, 80);
  ctx.lineTo(80 + 55 * Math.cos(rad), 80 + 55 * Math.sin(rad));
  ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 3; ctx.stroke();
  // Labels
  ctx.fillStyle = '#94a3b8'; ctx.font = '10px sans-serif';
  ctx.fillText('0°', 12, 82); ctx.fillText('90°', 74, 18); ctx.fillText('180°', 128, 82);
}
drawServo(90);

// --- Chart ---
const chartCtx = document.getElementById('historyChart').getContext('2d');
const histChart = new Chart(chartCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Wet',        data: [], borderColor: '#22c55e', tension: .4, pointRadius: 3 },
      { label: 'Recyclable', data: [], borderColor: '#3b82f6', tension: .4, pointRadius: 3 },
      { label: 'Hazardous',  data: [], borderColor: '#ef4444', tension: .4, pointRadius: 3 },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
    scales: {
      x: { ticks: { color: '#94a3b8', maxTicksLimit: 6 }, grid: { color: '#1e293b' } },
      y: { min: 0, max: 1, ticks: { color: '#94a3b8' }, grid: { color: '#334155' } }
    }
  }
});

function updateChart(scores) {
  const t = new Date().toLocaleTimeString();
  histChart.data.labels.push(t);
  histChart.data.datasets[0].data.push(scores.Wet || 0);
  histChart.data.datasets[1].data.push(scores.Recyclable || 0);
  histChart.data.datasets[2].data.push(scores.Hazardous || 0);
  if (histChart.data.labels.length > 20) {
    histChart.data.labels.shift();
    histChart.data.datasets.forEach(d => d.shift && d.data.shift());
  }
  histChart.update();
}

// --- SocketIO event handler ---
socket.on('classification_update', function(data) {
  const cat = data.final_category;
  const conf = data.final_confidence;
  const sensors = data.sensor_readings;
  const scores = data.fused_scores || {};

  // Badge
  const lbl = document.getElementById('category-label');
  lbl.textContent = cat;
  lbl.className = 'badge ' + cat;

  // Confidence bar
  document.getElementById('confidence-bar').style.width = (conf * 100).toFixed(1) + '%';
  document.getElementById('confidence-text').textContent =
    `Confidence: ${(conf * 100).toFixed(1)}%  (vision: ${(data.vision_confidence * 100).toFixed(1)}%)`;
  document.getElementById('imagenet-label').textContent = data.imagenet_label || '—';

  // Servo
  const angleMap = { Wet: 0, Recyclable: 90, Hazardous: 180 };
  const descMap  = { Wet: '← Left bin (Wet)', Recyclable: '↓ Centre bin (Recyclable)', Hazardous: '→ Right bin (Hazardous)' };
  const angle = angleMap[cat] ?? 90;
  document.getElementById('servo-angle-text').textContent = angle + '°';
  document.getElementById('servo-desc').textContent = descMap[cat] || 'Neutral';
  drawServo(angle);

  // Sensors
  if (sensors) {
    document.getElementById('s-moisture').textContent = sensors.moisture_pct?.toFixed(1) + '%';
    document.getElementById('s-gas').textContent = sensors.gas_ppm?.toFixed(0);
    document.getElementById('s-weight').textContent = sensors.weight_g?.toFixed(0);
    document.getElementById('s-time').textContent = new Date().toLocaleTimeString();

    // Bins
    const bins = [
      ['wet', sensors.fill_wet_pct],
      ['rec', sensors.fill_recyclable_pct],
      ['haz', sensors.fill_hazardous_pct],
    ];
    let anyFull = false;
    bins.forEach(([id, val]) => {
      const pct = (val || 0).toFixed(1);
      document.getElementById('bar-' + id).style.width = pct + '%';
      document.getElementById('pct-' + id).textContent = pct + '%';
      if (val >= 80) anyFull = true;
    });
    document.getElementById('bin-alert').style.display = anyFull ? 'block' : 'none';
  }

  // Reasoning
  const ul = document.getElementById('reasoning-list');
  ul.innerHTML = '';
  (data.reasoning || []).forEach(r => {
    const li = document.createElement('li'); li.textContent = r; ul.appendChild(li);
  });

  // Chart
  updateChart(scores);
});

// --- Image upload ---
document.getElementById('file-input').addEventListener('change', function() {
  const file = this.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    const img = document.getElementById('preview-img');
    img.src = e.target.result; img.style.display = 'block';
  };
  reader.readAsDataURL(file);
});

async function classifyUploaded() {
  const file = document.getElementById('file-input').files[0];
  const status = document.getElementById('classify-status');
  if (!file) { status.textContent = 'Please select an image first.'; return; }
  status.textContent = 'Classifying…';
  const formData = new FormData();
  formData.append('image', file);
  try {
    const res = await fetch('/api/classify', { method: 'POST', body: formData });
    const data = await res.json();
    status.textContent = data.error ? '❌ ' + data.error : '✅ Done — ' + data.final_category;
  } catch(e) {
    status.textContent = '❌ Request failed.';
  }
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Dashboard application
# ---------------------------------------------------------------------------

class WasteDashboard:
    """
    Flask + SocketIO dashboard server.

    Parameters
    ----------
    classifier   : WasteClassifier instance
    fusion_engine: SensorFusionEngine instance
    controller   : ServoController instance
    host         : bind address
    port         : HTTP port
    debug        : Flask debug mode
    """

    def __init__(
        self,
        classifier=None,
        fusion_engine=None,
        controller=None,
        host: str = "0.0.0.0",
        port: int = 5000,
        debug: bool = False,
    ) -> None:
        if not _flask_available:
            raise RuntimeError("Flask is required. Run: pip install flask flask-socketio eventlet")

        self.classifier = classifier
        self.fusion_engine = fusion_engine
        self.controller = controller
        self.host = host
        self.port = port
        self.debug = debug

        self._latest: dict = {}
        self._app = Flask(__name__)
        self._app.config["SECRET_KEY"] = "waste-seg-secret"
        self._socketio = SocketIO(self._app, cors_allowed_origins="*", async_mode="eventlet")

        self._register_routes()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        app = self._app
        sio = self._socketio

        @app.route("/")
        def index():
            return render_template_string(DASHBOARD_HTML)

        @app.route("/api/status")
        def status():
            return jsonify(self._latest)

        @app.route("/api/classify", methods=["POST"])
        def api_classify():
            if "image" not in request.files:
                return jsonify({"error": "No image uploaded"}), 400
            file = request.files["image"]
            try:
                from PIL import Image as PILImage
                img = PILImage.open(io.BytesIO(file.read()))
                result = self._run_pipeline(img)
                return jsonify(result)
            except Exception as exc:
                logger.exception("Classification failed")
                return jsonify({"error": str(exc)}), 500

        @sio.on("connect")
        def on_connect():
            if self._latest:
                emit("classification_update", self._latest)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(self, source) -> dict:
        """Run classifier → fusion → servo and return combined result dict."""
        # Classification
        if self.classifier:
            vision = self.classifier.classify_with_fallback(source)
        else:
            from src.classifier import _simulated_classification
            vision = _simulated_classification(source)

        # Fusion
        if self.fusion_engine:
            fusion = self.fusion_engine.fuse(vision)
        else:
            # Minimal shim
            class _FakeResult:
                final_category = vision["waste_category"]
                final_confidence = vision["confidence"]
                vision_category = vision["waste_category"]
                vision_confidence = vision["confidence"]
                sensor_boost = 0.0
                fused_scores = vision.get("category_scores", {})
                reasoning = ["Vision-only result (no fusion engine attached)."]
                class sensor_readings:
                    moisture_pct = 0.0; gas_ppm = 0.0; weight_g = 0.0
                    fill_wet_pct = 0.0; fill_recyclable_pct = 0.0; fill_hazardous_pct = 0.0
                    def to_dict(self): return {}
            fusion = _FakeResult()

        # Servo
        if self.controller:
            self.controller.sort(fusion.final_category)

        result = {
            "final_category": fusion.final_category,
            "final_confidence": round(fusion.final_confidence, 4),
            "vision_category": fusion.vision_category,
            "vision_confidence": round(fusion.vision_confidence, 4),
            "sensor_boost": round(fusion.sensor_boost, 4),
            "fused_scores": fusion.fused_scores,
            "reasoning": fusion.reasoning,
            "imagenet_label": vision.get("imagenet_label", "—"),
            "sensor_readings": fusion.sensor_readings.to_dict(),
        }
        self._latest = result
        self._socketio.emit("classification_update", result)
        return result

    # ------------------------------------------------------------------
    # Background auto-classify (demo mode)
    # ------------------------------------------------------------------

    def start_demo_loop(self, sample_dir: str = "data/sample_images", interval: float = 8.0) -> None:
        """Continuously classify sample images to demo the dashboard."""
        import glob, random

        def _loop():
            images = glob.glob(f"{sample_dir}/**/*.jpg", recursive=True) + \
                     glob.glob(f"{sample_dir}/**/*.png", recursive=True)
            if not images:
                # Generate synthetic demo events
                while True:
                    from src.sensor_fusion import SensorSimulator
                    from src.classifier import _simulated_classification
                    import random
                    cats = ["Wet", "Recyclable", "Hazardous"]
                    fake_source = f"data/sample_images/{random.choice(cats).lower()}_item.jpg"
                    vision = _simulated_classification(fake_source)
                    if self.fusion_engine:
                        fusion = self.fusion_engine.fuse(vision)
                        result = {
                            "final_category": fusion.final_category,
                            "final_confidence": round(fusion.final_confidence, 4),
                            "vision_category": fusion.vision_category,
                            "vision_confidence": round(fusion.vision_confidence, 4),
                            "sensor_boost": round(fusion.sensor_boost, 4),
                            "fused_scores": fusion.fused_scores,
                            "reasoning": fusion.reasoning,
                            "imagenet_label": vision.get("imagenet_label", "[demo]"),
                            "sensor_readings": fusion.sensor_readings.to_dict(),
                        }
                    else:
                        result = {
                            "final_category": vision["waste_category"],
                            "final_confidence": vision["confidence"],
                            "vision_category": vision["waste_category"],
                            "vision_confidence": vision["confidence"],
                            "sensor_boost": 0.0,
                            "fused_scores": vision.get("category_scores", {}),
                            "reasoning": ["[Demo mode — no real image]"],
                            "imagenet_label": "[demo]",
                            "sensor_readings": {},
                        }
                    if self.controller:
                        self.controller.sort(result["final_category"])
                    self._latest = result
                    self._socketio.emit("classification_update", result)
                    time.sleep(interval)
            else:
                while True:
                    img_path = random.choice(images)
                    self._run_pipeline(img_path)
                    time.sleep(interval)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        logger.info("Demo loop started (interval=%.1fs)", interval)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, demo_loop: bool = True) -> None:
        """Start the dashboard server."""
        if demo_loop:
            self.start_demo_loop()
        logger.info("Dashboard starting at http://%s:%d", self.host, self.port)
        self._socketio.run(
            self._app,
            host=self.host,
            port=self.port,
            debug=self.debug,
            use_reloader=False,
            log_output=False,
        )


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Import siblings
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.classifier import WasteClassifier
    from src.sensor_fusion import SensorSimulator, SensorFusionEngine
    from src.sorting_controller import ServoController

    clf = WasteClassifier()
    sim = SensorSimulator()
    engine = SensorFusionEngine(simulator=sim)
    ctrl = ServoController(simulate=True)

    dash = WasteDashboard(
        classifier=clf,
        fusion_engine=engine,
        controller=ctrl,
        port=5000,
        debug=False,
    )
    dash.run(demo_loop=True)
