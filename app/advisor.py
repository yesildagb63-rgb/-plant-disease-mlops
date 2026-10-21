import os

PROMPT = """Sen ev bahçesi ve küçük çiftçilere yardım eden bir ziraat asistanısın.
Bir görüntü sınıflandırma modeli domates yaprağı fotoğrafını analiz etti.

Model sonucu: {name}
Güven: %{conf:.0f}
Diğer olasılıklar: {others}
Doğrulanmış tedavi notları:
{notes}

Türkçe, sade ve samimi bir dille, en fazla 150 kelimeyle cevap ver:
1. Bu hastalık nedir? (1-2 cümle)
2. Teşhisi doğrulamak için yaprakta neye bakmalı? (1-2 madde)
3. Ne yapmalı? Bugün, bu hafta ve gelecek sezon için kısa adımlar. Sadece yukarıdaki notlara dayan.
4. Ne zaman bir ziraat mühendisine başvurmalı? (1 cümle)

Kurallar: İlaç markası veya doz uydurma. Güven %70'in altındaysa önce bunu belirt ve daha net bir fotoğraf iste.
Yaprak sağlıklıysa sadece kısa koruyucu öneriler ver. Başlık olarak kalın yazı kullan, uzun giriş yapma."""


class Advisor:
    def __init__(self, model):
        self.model = model
        self.client = None
        if os.getenv("GEMINI_API_KEY"):
            try:
                from google import genai
                self.client = genai.Client()
            except Exception as e:
                print(f"UYARI: Gemini baslatilamadi: {e}")

    @property
    def enabled(self):
        return self.client is not None

    def advise(self, r):
        if not self.enabled:
            return None
        prompt = PROMPT.format(
            name=r["name"], conf=r["confidence"] * 100,
            others=", ".join(f"{t['name']} %{t['confidence'] * 100:.0f}" for t in r["top3"][1:]),
            notes="\n".join(f"- {s}" for s in r["treatment"]))
        try:
            try:
                return self.client.interactions.create(model=self.model, input=prompt).output_text
            except AttributeError:
                return self.client.models.generate_content(model=self.model, contents=prompt).text
        except Exception as e:
            print(f"UYARI: Gemini yaniti alinamadi: {e}")
            return None