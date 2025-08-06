from fastai.vision.all import *
import gradio as gr


learn = load_learner("M:/Skin_Analyz/skin_model.pkl")



def classify_skin(img):
    img = PILImage.create(img)
    pred, _, probs = learn.predict(img)


    predicted_labels = ", ".join(pred)

    result = {learn.dls.vocab[i]: float(f"{probs[i]:.2f}") for i in range(len(probs))}

    return predicted_labels, result


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
