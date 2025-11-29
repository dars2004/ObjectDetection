import base64
import io
from flask import Flask, request, redirect
from flask import Response
from flask import jsonify
from PIL import Image
import numpy as np

# Ultralytics YOLOv11
from ultralytics import YOLO

app = Flask(__name__)

# Load model at startup; this will auto-download weights on first run
# You can change to other variants like 'yolo11s.pt', 'yolo11m.pt', etc.
model = YOLO("yolo11x.pt")


def html_page(content: str) -> str:
    tmpl = """
    <!doctype html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>YOLOv11 Image Detection</title>
      <style>
        :root {
          --bg: linear-gradient(110deg, #e63946, #f77f00, #fcbf49, #4568dc, #b06ab3);
          --glass: rgba(255,255,255,0.06);
          --glass-2: rgba(255,255,255,0.08);
          --text: #e9edf5;
          --muted: #a6b0c3;
          --accent: #7dd3fc;
          --accent-2: #34d399;
          --border: rgba(255,255,255,0.15);
          --shadow: 0 10px 30px rgba(0,0,0,0.35);
        }
        * { box-sizing: border-box; }
        html, body { height:100%; }
        body { 
          margin:0; 
          background: var(--bg); 
          background-size: 300% 300%;
          animation: gradient-animation 18s ease infinite;
          color: var(--text); 
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; 
        }
        .wrap { max-width: 1100px; margin: 40px auto; padding: 0 16px; }
        .hero-section {
          display: flex;
          justify-content: center;
          align-items: center;
          padding: 40px;
          text-align: center;
          margin-bottom: 40px;
        }
        @keyframes gradient-animation {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        .badge { padding:6px 10px; border-radius:999px; border:1px solid var(--border); background: var(--glass); backdrop-filter: blur(10px); font-size:12px; color:var(--muted); }
        .card { position:relative; background: var(--glass); border: 1px solid var(--border); border-radius: 18px; padding: 18px; box-shadow: var(--shadow); backdrop-filter: blur(14px); }
        h1 { margin: 0 0 6px; font-size: 28px; letter-spacing: .2px; }
        p { color: var(--muted); margin:0; }
        .grid { display:grid; grid-template-columns: 1.2fr 1fr; gap: 18px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .panel { background: var(--glass-2); border:1px solid var(--border); border-radius:14px; padding:14px; }
        .controls { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
        button { padding:10px 14px; border-radius:12px; border:1px solid transparent; background:linear-gradient(90deg, var(--accent), var(--accent-2)); color:#0b101b; font-weight:700; cursor:pointer; }
        button.ghost { background:transparent; border:1px solid var(--border); color:var(--text); }
        input[type=file] { padding: 9px; border: 1px dashed var(--border); border-radius: 10px; background: rgba(255,255,255,0.03); color: var(--muted); }
        .cam-wrap { position:relative; width:100%; aspect-ratio:16/9; border-radius:12px; overflow:hidden; border:1px solid var(--border); background:#05070d; }
        video, canvas { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; }
        .legend { margin-top:10px; display:flex; gap:10px; flex-wrap:wrap; color:var(--muted); font-size:12px; }
        .footer { margin-top: 22px; font-size: 12px; color: var(--muted); text-align:center; }
        .notice { background: var(--glass-2); border: 1px solid var(--border); padding: 10px 12px; border-radius: 10px; color: var(--muted); }
        .tip { font-size: 12px; color: var(--muted); margin-top: 6px; }
        .glow-char {
          display: inline-block;
          animation: letter-glow 2s ease-in-out infinite;
          color: rgba(255,255,255,0.9);
        }
        @keyframes letter-glow {
          0%, 100% { 
            text-shadow: 0 0 4px rgba(255, 255, 255, 0.2);
            transform: translateY(0);
          }
          50% { 
            text-shadow: 0 0 25px rgba(255, 255, 255, 0.9), 0 0 10px rgba(255, 255, 255, 0.7);
            transform: translateY(-4px);
            color: #fff;
          }
        }
      </style>
    </head>
    <body>
      <div class="hero-section">
        <div>
          <span class="badge" style="background:rgba(255,255,255,0.2);color:white;border-color:rgba(255,255,255,0.4);">Ultralytics YOLOv11</span>
          <h1 id="mainTitle" style="font-size:48px;margin-top:16px;">Image Detection</h1>
          <p style="color:rgba(255,255,255,0.9);font-size:18px;">Real-time object detection with state-of-the-art accuracy</p>
        </div>
      </div>
      <div class="wrap">
        <div class="card">
          <h1>Detect Objects</h1>
          <p>Use your camera for live detection or upload an image. Boxes show class and confidence percentage.</p>
          <div class="grid" style="margin-top:16px;">
            <div class="panel">
              <div class="controls">
                <button id="startBtn">Start Camera</button>
                <button id="stopBtn" class="ghost">Stop</button>
                <form id="uploadForm" method="POST" action="/predict" enctype="multipart/form-data" style="display:flex;gap:8px;align-items:center;">
                  <input type="file" name="image" accept="image/*" required />
                  <button type="submit" class="ghost">Detect Image</button>
                </form>
              </div>
              <div class="cam-wrap">
                <video id="cam" autoplay playsinline muted></video>
                <canvas id="overlay"></canvas>
              </div>
              <div class="legend" id="legend"></div>
              <div class="tip">Model: <code>Yolo11x</code></div>
            </div>
            <div class="panel" id="resultPanel">
              %%CONTENT%%
            </div>
          </div>
        </div>
        <div class=\"footer\">Powered by <a href=\"https://github.com/ultralytics/ultralytics\" style=\"color:var(--accent)\">Ultralytics</a></div>
      </div>

      <script>
        const cam = document.getElementById('cam');
        const overlay = document.getElementById('overlay');
        const ctx = overlay.getContext('2d');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const legend = document.getElementById('legend');
        const uploadForm = document.getElementById('uploadForm');
        const resultPanel = document.getElementById('resultPanel');
        let stream = null;
        let running = false;

        uploadForm.addEventListener('submit', async (e) => {
          e.preventDefault();
          const formData = new FormData(uploadForm);
          resultPanel.innerHTML = '<div class="notice">Processing...</div>';
          
          try {
            const res = await fetch('/predict', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.image) {
               resultPanel.innerHTML = `<div class="preview"><img src="data:image/png;base64,${data.image}" alt="Detections" style="width:100%;border-radius:8px;" /></div>`;
            } else if (data.error) {
               resultPanel.innerHTML = `<div class="notice">${data.error}</div>`;
            }
          } catch (err) {
            resultPanel.innerHTML = `<div class="notice">Error: ${err}</div>`;
          }
        });

        function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

        async function startCamera() {
          try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
            cam.srcObject = stream;
            await cam.play();
            resizeCanvas();
            running = true;
            detectLoop();
          } catch (e) {
            alert('Camera access failed: ' + e);
          }
        }

        function stopCamera() {
          running = false;
          if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
          ctx.clearRect(0, 0, overlay.width, overlay.height);
        }

        function resizeCanvas() {
          const w = cam.videoWidth || 1280;
          const h = cam.videoHeight || 720;
          overlay.width = w;
          overlay.height = h;
        }

        async function detectLoop() {
          while (running) {
            if (!cam.videoWidth) { await sleep(50); continue; }
            resizeCanvas();
            // Draw current frame to an offscreen canvas
            const off = document.createElement('canvas');
            off.width = cam.videoWidth; off.height = cam.videoHeight;
            const octx = off.getContext('2d');
            octx.drawImage(cam, 0, 0, off.width, off.height);
            const blob = await new Promise(r => off.toBlob(r, 'image/jpeg', 0.8));

            const form = new FormData();
            form.append('frame', blob, 'frame.jpg');

            try {
              const res = await fetch('/detect', { method: 'POST', body: form });
              const data = await res.json();
              drawDetections(data);
            } catch (e) {
              console.error('detect error', e);
            }

            await sleep(150); // ~6-7 FPS throttle
          }
        }

        function drawDetections(data) {
          ctx.clearRect(0, 0, overlay.width, overlay.height);
          const boxes = data.boxes || [];
          const colors = {};
          legend.innerHTML = '';
          boxes.forEach((b) => {
            const x1 = b.x1, y1 = b.y1, x2 = b.x2, y2 = b.y2;
            const w = x2 - x1, h = y2 - y1;
            const label = b.label || String(b.cls);
            const confPct = Math.round((b.conf || 0)*100);
            if (!colors[label]) {
              // pseudo-random color by label
              const hash = Array.from(label).reduce((a,c)=>a+c.charCodeAt(0),0);
              const r = 100 + (hash*37)%155, g = 100 + (hash*57)%155, bl = 100 + (hash*97)%155;
              colors[label] = `rgb(${r},${g},${bl})`;
              const chip = document.createElement('span');
              chip.textContent = `${label}`;
              chip.style.cssText = 'padding:4px 8px;border-radius:999px;border:1px solid var(--border);background:var(--glass);backdrop-filter:blur(8px);';
              legend.appendChild(chip);
            }
            ctx.strokeStyle = colors[label];
            ctx.lineWidth = 2;
            ctx.strokeRect(x1, y1, w, h);

            const text = `${label} ${confPct}%`;
            ctx.font = '14px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto';
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            const tw = ctx.measureText(text).width;
            const th = 18;
            ctx.fillRect(x1, Math.max(0,y1-th), tw+8, th);
            ctx.fillStyle = '#fff';
            ctx.fillText(text, x1+4, Math.max(12,y1-4));
          });
        }

        startBtn.addEventListener('click', startCamera);
        stopBtn.addEventListener('click', stopCamera);

        // Sequential Glow Effect for Title
        const title = document.getElementById('mainTitle');
        if (title) {
          const text = title.textContent;
          title.innerHTML = '';
          [...text].forEach((char, i) => {
            const span = document.createElement('span');
            span.textContent = char === ' ' ? '\u00A0' : char;
            span.className = 'glow-char';
            span.style.animationDelay = `${i * 0.1}s`;
            title.appendChild(span);
          });
        }
      </script>
    </body>
    </html>
    """
    return tmpl.replace("%%CONTENT%%", content)


@app.get("/")
def index() -> str:
    return html_page('<div class="notice">No image yet. Choose an image and click Detect.</div>')


@app.post("/predict")
def predict() -> Response:
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify({"error": "Please choose a valid image file."})

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Unsupported image file."})

    # Run YOLO inference
    results = model.predict(img, imgsz=640, conf=0.25)
    r0 = results[0]

    # Render predictions to an array (BGR) and convert to RGB
    plotted_bgr = r0.plot()
    plotted_rgb = plotted_bgr[:, :, ::-1]

    # Encode to PNG in-memory and return as inline base64 image
    out_img = Image.fromarray(plotted_rgb)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    return jsonify({"image": b64})


@app.post("/detect")
def detect():
    file = request.files.get("frame")
    if not file or file.filename == "":
        return jsonify({"error": "no frame"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "bad image"}), 400

    results = model.predict(img, imgsz=640, conf=0.25, verbose=False)
    r0 = results[0]

    boxes_out = []
    names = getattr(r0, 'names', None)
    try:
        xyxy = r0.boxes.xyxy.tolist()
        cls = r0.boxes.cls.tolist()
        conf = r0.boxes.conf.tolist()
    except Exception:
        xyxy, cls, conf = [], [], []

    for b, c, cf in zip(xyxy, cls, conf):
        idx = int(c) if c is not None else -1
        label = names[idx] if names and idx in names else str(idx)
        boxes_out.append({
            "x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
            "cls": idx,
            "label": label,
            "conf": float(cf) if cf is not None else 0.0
        })

    return jsonify({
        "boxes": boxes_out,
        "width": img.width,
        "height": img.height
    })


if __name__ == "__main__":
    # For local development
    app.run(host="127.0.0.1", port=5001, debug=True)
