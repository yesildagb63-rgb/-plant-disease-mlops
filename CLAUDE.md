# Proje
Bitki hastalığı sınıflandırma + MLOps. MobileNetV3 transfer learning (PyTorch/torchvision).
Veri: PlantVillage (eğitim), PlantDoc + kendi fotoğraflarımız (domain shift testi).
Başlangıçta 5 sınıf (domates).
# Yığın
DVC, MLflow, ONNX, FastAPI, Docker, Gradio, GitHub Actions, Evidently.
# Yapı
data/ src/ models/ app/ tests/ .github/workflows/ config.yaml
# Kurallar
- data/ ve models/ klasörlerini ASLA okuma/listeleme (büyük). DVC ile takip edilir.
- Kısa, sade kod yaz. Gereksiz açıklama, yorum ve README şişirme yok.
- Sadece istenen dosyaları oluştur/değiştir. Bitince tek satırda ne yaptığını söyle.
- Tüm parametreler config.yaml'dan gelsin.