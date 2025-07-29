from fastai.vision.all import *
import gradio as gr

# Modeli yükle
learn = load_learner("M:/Skin_Analyz/skin_model.pkl")


# Tahmin fonksiyonu
def classify_skin(img):
    img = PILImage.create(img)
    pred, _, probs = learn.predict(img)

    # Tahmin edilen etiketleri birleştir
    predicted_labels = ", ".join(pred)

    # Olasılık sözlüğü
    result = {learn.dls.vocab[i]: float(f"{probs[i]:.2f}") for i in range(len(probs))}

    return predicted_labels, result


# Gradio arayüzü
iface = gr.Interface(
    fn=classify_skin,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Tahmin Edilen Etiketler"),
        gr.Label(label="Etiket Olasılıkları")
    ],
    title="Cilt Durumu Tahmini",
    description="Bir yüz görseli yükleyin. Model, akne, kırışıklık, gözaltı torbası ve kızarıklık olup olmadığını tahmin eder."
)

iface.launch()
