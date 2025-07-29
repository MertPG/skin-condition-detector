# 🧠 Skin Condition Detector

Bu proje, yüz görsellerinden aşağıdaki cilt problemlerini tespit etmek için eğitilmiş **FastAI tabanlı derin öğrenme modelini** ve bir **Gradio kullanıcı arayüzünü** içerir:

- 🟣 Akne (var/yok)
- 🔴 Kızarıklık (Redness)
- 👁 Gözaltı torbası (Bags)
- 🧓 Kırışıklık (Wrinkles)

Ayrıca, ayrı bir model ile akne şiddeti (0'dan 3'e kadar) de tahmin edilebilir.

---

## 🔧 Kullanılan Teknolojiler

- Python 3.10+
- [FastAI](https://docs.fast.ai/)
- PyTorch
- Gradio (arayüz için)
- Jupyter Notebook (prototipleme)

---

## 🚀 Nasıl Çalıştırılır

### 1. Gerekli paketleri yükleyin:

```bash
pip install fastai gradio


skin_model.pkl           # Multi-label model (akne, kızarıklık, vs.)
akne_derece_model.pkl    # (İsteğe bağlı) Akne şiddeti tahmin modeli


python gradio_app.py


Tarayıcınızda açılır !!!