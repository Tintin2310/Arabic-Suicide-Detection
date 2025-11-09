import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from deep_translator import GoogleTranslator

# Load trained model
model_path = "arabert-suicidal-detector"  # folder where you saved the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

labels = ["Non-Suicidal", "Suicidal"]

def classify_tweet(text):
    # 🔹 Step 1: Translate Arabic → English
    try:
        english_text = GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        english_text = "(Translation failed)"

    # 🔹 Step 2: Model prediction on Arabic text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        suicidal_score = probs[0][1].item()

    # 🔹 Step 3: Decide label
    label = "🟥 Suicidal" if suicidal_score > 0.4 else "🟩 Non-Suicidal"

    return label, f"{suicidal_score:.2f}", text, english_text


# Build a cleaner Gradio UI with multiple output boxes
app = gr.Interface(
    fn=classify_tweet,
    inputs=gr.Textbox(
        lines=3,
        placeholder="✍️ Enter Arabic tweet here... / أدخل التغريدة بالعربية هنا...",
        label="Input Tweet (Arabic)"
    ),
    outputs=[
        gr.Textbox(label="Prediction", interactive=False),
        gr.Textbox(label="Confidence Score", interactive=False),
        gr.Textbox(label="Original Arabic Text", interactive=False),
        gr.Textbox(label="English Translation", interactive=False),
    ],
    title="🌙 Arabic Suicidality Detector",
    description="This app classifies Arabic tweets as **Suicidal** or **Non-Suicidal** and also provides an English translation."
)

app.launch()
