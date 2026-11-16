import gradio as gr
from transformers import pipeline

image_classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

def classify_thumbnail(image):
    labels = ["Sports and Golf", "Food and Baking", "Science and Travel Trivia"]
    results = image_classifier(image, candidate_labels=labels)
    return {res['label']: res['score'] for res in results}

app = gr.Interface(
    fn=classify_thumbnail,
    inputs=gr.Image(type="pil", label="Apne Short ka Screenshot upload karein"),
    outputs=gr.Label(num_top_classes=3, label="AI Prediction"),
    title="Tricky Kashi - Shorts Content Classifier"
)
app.launch()
