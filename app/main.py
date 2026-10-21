import csv
import inspect
import io
import os
from datetime import datetime
from pathlib import Path

import gradio as gr
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from PIL import Image, UnidentifiedImageError

from app.advisor import Advisor
from app.predictor import Predictor

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
predictor = Predictor()
advisor = Advisor(cfg["llm_model"])
LOG_DIR = Path(cfg["log_dir"])
(LOG_DIR / "images").mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.csv"


def log_prediction(img, r, source):
    ts = datetime.now()
    name = f"{ts:%Y%m%d_%H%M%S_%f}_{r['class']}.jpg"
    img.convert("RGB").save(LOG_DIR / "images" / name, quality=90)
    new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "source", "image", "class", "confidence", "confident"])
        w.writerow([ts.isoformat(timespec="seconds"), source, name, r["class"], r["confidence"], r["confident"]])
    return ts


def fallback_text(r):
    head = "" if r["confident"] else "**Emin değilim.** "
    return head + "**Ne yapmalı?**\n" + "\n".join(f"- {s}" for s in r["treatment"])


app = FastAPI(title="Plant Disease Detection API")


@app.get("/")
def root():
    return RedirectResponse("/ui")


@app.get("/health")
def health():
    return {"status": "ok", "classes": predictor.classes, "llm": advisor.enabled}


@app.post("/predict")
async def predict(file: UploadFile = File(...), explain: bool = False):
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Geçersiz görüntü dosyası")
    r = predictor.predict(img)
    log_prediction(img, r, "api")
    if explain:
        r["ai_advice"] = advisor.advise(r)
    return r


def result_card(r):
    if not r["confident"]:
        color, badge = "#6b7280", "Emin değil"
    elif r["class"].endswith("healthy"):
        color, badge = "#16a34a", "Sağlıklı"
    else:
        color, badge = "#ea580c", "Hastalık tespit edildi"
    pct = r["confidence"] * 100
    return f"""
<div style="border:1px solid #e5e7eb;border-radius:16px;padding:20px;background:#fff">
  <span style="background:{color}1a;color:{color};font-weight:600;padding:4px 12px;border-radius:999px;font-size:13px">{badge}</span>
  <div style="font-size:28px;font-weight:700;margin:12px 0 4px;color:#111827">{r['name']}</div>
  <div style="color:#6b7280;font-size:14px;margin-bottom:8px">Güven %{pct:.1f}</div>
  <div style="height:10px;background:#f3f4f6;border-radius:999px;overflow:hidden">
    <div style="width:{pct:.1f}%;height:100%;background:{color}"></div>
  </div>
</div>"""


def analyze(img, history):
    if img is None:
        raise gr.Error("Önce bir yaprak fotoğrafı yükleyin.")
    r = predictor.predict(img)
    ts = log_prediction(img, r, "ui")
    text = advisor.advise(r) or fallback_text(r)
    text += "\n\n*Bu öneriler genel bilgidir; kesin teşhis için bir ziraat mühendisine danışın.*"
    thumb = img.convert("RGB").copy()
    thumb.thumbnail((256, 256))
    history = [{"time": f"{ts:%H:%M:%S}", "name": r["name"], "conf": r["confidence"], "img": thumb}] + history
    gallery = [(h["img"], f"{h['name']} · %{h['conf'] * 100:.0f}") for h in history]
    table = [[h["time"], h["name"], f"%{h['conf'] * 100:.1f}"] for h in history]
    labels = {t["name"]: t["confidence"] for t in r["top3"]}
    return result_card(r), labels, text, history, gallery, table


def clear_history():
    return [], [], []


CSS = """
.gradio-container {max-width: 1200px !important; margin: auto}
#header {text-align: center; padding: 8px 0 4px}
#header h1 {font-size: 34px; margin-bottom: 4px}
#analyze {font-size: 18px}
"""
theme = gr.themes.Soft(primary_hue="green", secondary_hue="emerald", radius_size="lg")
style = {"theme": theme, "css": CSS}
blocks_kw = style if "css" in inspect.signature(gr.Blocks.__init__).parameters else {}
mount_params = inspect.signature(gr.mount_gradio_app).parameters
mount_kw = {} if blocks_kw else {k: v for k, v in style.items() if k in mount_params}

examples = []
data_dir = Path(cfg["data_dir"])
if data_dir.exists():
    for c in predictor.classes:
        first = next((data_dir / c).glob("*"), None)
        if first:
            examples.append([str(first)])

with gr.Blocks(title="Bitki Hastalığı Tespiti", **blocks_kw) as demo:
    history = gr.State([])
    gr.HTML(f"""<div id="header"><h1>🌱 Bitki Hastalığı Tespiti</h1>
<p>Domates yaprağının fotoğrafını yükleyin; hastalığı ve ne yapmanız gerektiğini söyleyelim.</p>
<p style="color:#6b7280;font-size:13px">Model: MobileNetV3 (ONNX) · {'AI açıklama: Gemini' if advisor.enabled else 'AI açıklama kapalı (GEMINI_API_KEY yok)'}</p></div>""")
    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            img = gr.Image(type="pil", label="Yaprak fotoğrafı", height=380)
            btn = gr.Button("🔍 Analiz et", variant="primary", elem_id="analyze")
            if examples:
                gr.Examples(examples=examples, inputs=img, label="Örnek fotoğraflar")
        with gr.Column(scale=6):
            card = gr.HTML()
            top3 = gr.Label(num_top_classes=3, label="Olasılıklar")
            advice = gr.Markdown()
    with gr.Accordion("🕘 Analiz geçmişi", open=True):
        gallery = gr.Gallery(label="Bu oturumdaki analizler", columns=6, height=220)
        table = gr.Dataframe(headers=["Saat", "Sonuç", "Güven"], interactive=False)
        clear = gr.Button("Geçmişi temizle", size="sm")

    btn.click(analyze, [img, history], [card, top3, advice, history, gallery, table])
    clear.click(clear_history, None, [history, gallery, table])

app = gr.mount_gradio_app(app, demo, path="/ui", **mount_kw)