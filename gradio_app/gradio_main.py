import os
import sys
import json
import base64
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image
from groq import Groq
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

from dotenv import load_dotenv
load_dotenv()
# =========================
# Paths
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
KB_PATH = PROJECT_ROOT / "knowledge_base" / "diseases.json"
TRANSLATIONS_PATH = PROJECT_ROOT / "knowledge_base" / "translations.json"
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "data" / "sample_images"
DB_PATH = PROJECT_ROOT / "gradio_app" / "gradio_memory.db"
LOGO_PATH = PROJECT_ROOT / "gradio_app" / "logo.png"

# Make backend/app importable
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.inference import predict_image_pipeline

# =========================
# Load resources
# =========================
with open(KB_PATH, "r", encoding="utf-8") as f:
    DISEASES = json.load(f)

with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
    TRANSLATIONS = json.load(f)

with open(LOGO_PATH, "rb") as f:
    LOGO_BASE64 = base64.b64encode(f.read()).decode("utf-8")

# =========================
# SQLite memory (optional logging only)
# =========================
Base = declarative_base()
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)


class GradioMessage(Base):
    __tablename__ = "gradio_messages"

    id = Column(Integer, primary_key=True)
    username = Column(String, index=True)
    role = Column(String)
    content = Column(Text)
    disease_context = Column(String)


Base.metadata.create_all(bind=engine)

# =========================
# Helpers
# =========================
NON_DISEASE_CONTEXTS = {"healthy", "not_wheat", "uncertain", None}


def tr(language: str, key: str) -> str:
    return TRANSLATIONS.get(language, {}).get(key, key)


def build_header_html(language: str) -> str:
    return f"""
    <div style="
        position: fixed;
        top: 12px;
        left: 18px;
        right: 18px;
        z-index: 1000;
        background: linear-gradient(90deg, #2e7d32, #388e3c);
        border-radius: 16px;
        padding: 14px 22px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.12);
    ">
        <div style="
            display: flex;
            align-items: center;
            gap: 18px;
        ">
            <img
                src="data:image/png;base64,{LOGO_BASE64}"
                style="
                    width: 84px;
                    height: 84px;
                    object-fit: contain;
                    border-radius: 10px;
                    background: rgba(255,255,255,0.10);
                    padding: 6px;
                "
            >
            <div>
                <div style="
                    font-size: 40px;
                    font-weight: 900;
                    color: white;
                    line-height: 1.05;
                    margin-bottom: 6px;
                ">
                    {tr(language, "site_title")}
                </div>
                <div style="
                    font-size: 17px;
                    color: #f1f8e9;
                    font-weight: 500;
                ">
                    {tr(language, "header_subtitle")}
                </div>
            </div>
        </div>
    </div>

    <div style="height: 125px;"></div>
    """


def prettify_label(label: str) -> str:
    return label.replace("_", " ").title()


def save_temp_image(image: Image.Image) -> str:
    """
    Save uploaded PIL image to a temporary file so backend inference
    can read it from disk.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        return tmp.name


def save_message(username: str, role: str, content: str, disease_context: str = None):
    db = SessionLocal()
    db.add(
        GradioMessage(
            username=username,
            role=role,
            content=content,
            disease_context=disease_context,
        )
    )
    db.commit()
    db.close()


def build_disease_markdown(disease_info: dict, predicted_label: str, language: str):
    if language == "de":
        name = disease_info.get("display_name_de", prettify_label(predicted_label))
        overview = disease_info.get("overview_de", "")
        causes = disease_info.get("causes_de", "")
        season = disease_info.get("season_de", "")
        spread = disease_info.get("spread_de", "")
        treatment = disease_info.get("treatment_de", "")
        prevention = disease_info.get("prevention_de", "")
        warning = disease_info.get("warning_de", "")

        return f"""
### {name}

**Überblick**  
{overview}

**Ursachen**  
{causes}

**Saison**  
{season}

**Ausbreitung**  
{spread}

**Behandlung**  
{treatment}

**Vorbeugung**  
{prevention}

**Hinweis**  
{warning}
"""
    else:
        name = disease_info.get("display_name_en", prettify_label(predicted_label))
        overview = disease_info.get("overview_en", "")
        causes = disease_info.get("causes_en", "")
        season = disease_info.get("season_en", "")
        spread = disease_info.get("spread_en", "")
        treatment = disease_info.get("treatment_en", "")
        prevention = disease_info.get("prevention_en", "")
        warning = disease_info.get("warning_en", "")

        return f"""
### {name}

**Overview**  
{overview}

**Causes**  
{causes}

**Season**  
{season}

**Spread**  
{spread}

**Treatment**  
{treatment}

**Prevention**  
{prevention}

**Note**  
{warning}
"""


def ask_llm(message: str, language: str, disease_context: str = None, chat_history=None):
    disease_block = ""

    if disease_context and disease_context not in NON_DISEASE_CONTEXTS and disease_context in DISEASES:
        info = DISEASES[disease_context]
        if language == "de":
            disease_block = (
                f"Krankheit: {info.get('display_name_de', disease_context)}\n"
                f"Überblick: {info.get('overview_de', '')}\n"
                f"Ursachen: {info.get('causes_de', '')}\n"
                f"Behandlung: {info.get('treatment_de', '')}\n"
                f"Vorbeugung: {info.get('prevention_de', '')}\n"
            )
        else:
            disease_block = (
                f"Disease: {info.get('display_name_en', disease_context)}\n"
                f"Overview: {info.get('overview_en', '')}\n"
                f"Causes: {info.get('causes_en', '')}\n"
                f"Treatment: {info.get('treatment_en', '')}\n"
                f"Prevention: {info.get('prevention_en', '')}\n"
            )

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        if language == "de":
            return "Groq API key not found. Bitte setze GROQ_API_KEY."
        return "Groq API key not found. Please set GROQ_API_KEY."

    if language == "de":
        if disease_context == "not_wheat":
            system_prompt = (
                "You are WheatIQ. Reply only in German. "
                "The uploaded image was not identified as wheat. "
                "Explain this clearly and briefly. Ask the user to upload a clearer wheat leaf image. "
                "Do not invent a disease."
            )
        elif disease_context == "healthy":
            system_prompt = (
                "You are WheatIQ. Reply only in German. "
                "The uploaded image was identified as a healthy wheat plant. "
                "Be clear, practical, and short. Mention that the answer is general guidance."
            )
        elif disease_context == "uncertain":
            system_prompt = (
                "You are WheatIQ. Reply only in German. "
                "The uploaded image seems related to wheat, but the disease prediction is uncertain. "
                "Explain this clearly and briefly. Ask the user to upload a clearer image. "
                "Do not invent a disease."
            )
        else:
            system_prompt = (
                "You are WheatIQ, a bilingual wheat disease assistant. "
                "Reply only in German. "
                "Use the internal disease information first. "
                "Be clear, practical, and short. "
                "Always mention that the answer is general guidance. "
                "If the user asks an ambiguous follow-up question without enough context, ask for clarification instead of assuming."
            )
    else:
        if disease_context == "not_wheat":
            system_prompt = (
                "You are WheatIQ. Reply only in English. "
                "The uploaded image was not identified as wheat. "
                "Explain this clearly and briefly. Ask the user to upload a clearer wheat leaf image. "
                "Do not invent a disease."
            )
        elif disease_context == "healthy":
            system_prompt = (
                "You are WheatIQ. Reply only in English. "
                "The uploaded image was identified as a healthy wheat plant. "
                "Be clear, practical, and short. Mention that the answer is general guidance."
            )
        elif disease_context == "uncertain":
            system_prompt = (
                "You are WheatIQ. Reply only in English. "
                "The uploaded image seems related to wheat, but the disease prediction is uncertain. "
                "Explain this clearly and briefly. Ask the user to upload a clearer image. "
                "Do not invent a disease."
            )
        else:
            system_prompt = (
                "You are WheatIQ, a bilingual wheat disease assistant. "
                "Reply only in English. "
                "Use the internal disease information first. "
                "Be clear, practical, and short. "
                "Always mention that the answer is general guidance. "
                "If the user asks an ambiguous follow-up question without enough context, ask for clarification instead of assuming."
            )

    messages = [{"role": "system", "content": system_prompt}]

    if disease_block:
        messages.append({"role": "system", "content": disease_block})

    recent_history = chat_history[-12:] if chat_history else []

    for item in recent_history:
        role = item.get("role")
        content = item.get("content")
        if role and content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content


def classify_and_fill(image, language):
    if image is None:
        return (
            "",
            tr(language, "upload_first"),
            tr(language, "no_analysis"),
            None,
            None,
        )

    temp_path = save_temp_image(image)

    try:
        result = predict_image_pipeline(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if result.get("reason") == "not_wheat":
        if language == "de":
            detected_text = "⚠️ Kein Weizen erkannt"
            summary_text = "Dieses Bild scheint keine Weizenpflanze bzw. kein Weizenblatt zu zeigen."
            disease_md = (
                "### Kein Weizen erkannt\n\n"
                "Bitte laden Sie ein klareres Bild eines Weizenblatts hoch. "
                "Dies ist nur eine allgemeine Einschätzung."
            )
        else:
            detected_text = "⚠️ Not Wheat"
            summary_text = "This image does not appear to show a wheat plant or wheat leaf."
            disease_md = (
                "### Not Wheat Detected\n\n"
                "Please upload a clearer image of a wheat leaf. "
                "This is only general guidance."
            )

        return detected_text, summary_text, disease_md, None, "not_wheat"

    if result.get("predicted_label") == "healthy":
        if language == "de":
            detected_text = "🌾 Gesunder Weizen"
            summary_text = "Das hochgeladene Bild scheint eine gesunde Weizenpflanze zu zeigen."
            disease_md = (
                "### Gesunder Weizen\n\n"
                "Auf diesem Bild wurde keine Krankheit erkannt. "
                "Dies ist nur eine allgemeine Einschätzung."
            )
        else:
            detected_text = "🌾 Healthy Wheat"
            summary_text = "The uploaded image appears to show a healthy wheat plant."
            disease_md = (
                "### Healthy Wheat\n\n"
                "No disease was detected in this image. "
                "This is only general guidance."
            )

        return detected_text, summary_text, disease_md, None, "healthy"

    if result.get("reason") == "uncertain_disease":
        if language == "de":
            detected_text = "⚠️ Unsichere Vorhersage"
            summary_text = "Das Bild scheint Weizen zu zeigen, aber die Krankheitsvorhersage ist unsicher."
            disease_md = (
                "### Unsichere Vorhersage\n\n"
                "Bitte laden Sie ein klareres Bild mit besserem Fokus und guter Beleuchtung hoch. "
                "Dies ist nur eine allgemeine Einschätzung."
            )
        else:
            detected_text = "⚠️ Uncertain Prediction"
            summary_text = "The image seems to show wheat, but the disease prediction is uncertain."
            disease_md = (
                "### Uncertain Prediction\n\n"
                "Please upload a clearer image with better focus and lighting. "
                "This is only general guidance."
            )

        return detected_text, summary_text, disease_md, None, "uncertain"

    predicted_label = result["predicted_label"]
    disease_info = DISEASES.get(predicted_label, {})
    sample_image = disease_info.get("sample_image")
    sample_image_path = str(SAMPLE_IMAGES_DIR / sample_image) if sample_image else None
    pretty_name = prettify_label(predicted_label)

    if language == "de":
        display_name = disease_info.get("display_name_de", pretty_name)
        detected_text = f"🌾 {display_name}"
        summary_text = f"Dieses Weizenbild ist wahrscheinlich von {display_name} betroffen."
    else:
        display_name = disease_info.get("display_name_en", pretty_name)
        detected_text = f"🌾 {display_name}"
        conf = result.get("confidence", 0.0)
        summary_text = f"The system detects {display_name}."

    disease_md = build_disease_markdown(disease_info, predicted_label, language)
    return detected_text, summary_text, disease_md, sample_image_path, predicted_label


def chat_fn(message, language, disease_context, chat_history):
    username = "guest"

    if not message or not message.strip():
        return chat_history if chat_history else [], ""

    if chat_history is None:
        chat_history = []

    answer = ask_llm(
        message=message.strip(),
        language=language,
        disease_context=disease_context,
        chat_history=chat_history,
    )

    save_message(username, "user", message.strip(), disease_context)
    save_message(username, "assistant", answer, disease_context)

    chat_history = list(chat_history)
    chat_history.append({"role": "user", "content": message.strip()})
    chat_history.append({"role": "assistant", "content": answer})

    return chat_history, ""


def clear_chat():
    return [], ""


def update_ui_language(language):
    return (
        gr.update(value=build_header_html(language)),
        gr.update(label=tr(language, "language")),
        gr.update(value=f"## 📷 {tr(language, 'image_analysis_title')}"),
        gr.update(label=tr(language, "upload_title")),
        gr.update(value=f"🔍 {tr(language, 'predict')}"),
        gr.update(label=tr(language, "detected_disease")),
        gr.update(label=tr(language, "diagnosis_summary")),
        gr.update(value=tr(language, "no_analysis")),
        gr.update(value=f"## 🧪 {tr(language, 'reference_title')}"),
        gr.update(label=tr(language, "sample_image")),
        gr.update(value=f"## 💬 {tr(language, 'assistant_title')}"),
        gr.update(label=tr(language, "chat_title")),
        gr.update(label=tr(language, "ask_question"), placeholder=tr(language, "chat_placeholder")),
        gr.update(value=tr(language, "send")),
        gr.update(value=tr(language, "clear_chat")),
    )


# =========================
# Custom CSS
# =========================
CUSTOM_CSS = """
body {
    font-family: Arial, sans-serif;
}

.section-card {
    border-radius: 18px;
    padding: 10px;
}

.main-content {
    margin-top: 10px;
}

button[title="Download"],
button[title="Share"],
button[aria-label="Download"],
button[aria-label="Share"] {
    display: none !important;
}

/* ===== CUSTOM GREEN BUTTONS ===== */

#analyze-btn,
#send-btn {
    background: linear-gradient(90deg, #2e7d32, #388e3c) !important;
    border: 1px solid #2e7d32 !important;
    color: white !important;
    font-weight: 700 !important;
}

#analyze-btn:hover,
#send-btn:hover {
    background: linear-gradient(90deg, #27682a, #2f7d34) !important;
}



"""

# =========================
# UI
# =========================
with gr.Blocks() as demo:
    header_html = gr.HTML(build_header_html("en"))

    with gr.Column(elem_classes="main-content"):
        with gr.Row():
            with gr.Column(scale=2):
                language = gr.Radio(
                    choices=["en", "de"],
                    value="en",
                    label=tr("en", "language")
                )

        with gr.Row():
            with gr.Column(scale=6):
                with gr.Group(elem_classes="section-card"):
                    image_analysis_md = gr.Markdown(f"## 📷 {tr('en', 'image_analysis_title')}")

                    image_input = gr.Image(
                        type="pil",
                        label=tr("en", "upload_title"),
                        
                    )

                    analyze_btn = gr.Button(
                    f"🔍 {tr('en', 'predict')}",
                    variant="primary",
                    elem_id="analyze-btn"
)

                    detected_disease = gr.Textbox(
                        label=tr("en", "detected_disease"),
                        interactive=False
                    )

                    diagnosis_summary = gr.Textbox(
                        label=tr("en", "diagnosis_summary"),
                        interactive=False
                    )

                    disease_info_md = gr.Markdown(tr("en", "no_analysis"))

            with gr.Column(scale=5):
                with gr.Group(elem_classes="section-card"):
                    reference_image_md = gr.Markdown(f"## 🧪 {tr('en', 'reference_title')}")

                    sample_image = gr.Image(
                        type="filepath",
                        label=tr("en", "sample_image"),
                        interactive=False,
                        height=250
                        

                    )

                with gr.Group(elem_classes="section-card"):
                    assistant_md = gr.Markdown(f"## 💬 {tr('en', 'assistant_title')}")

                    chatbot = gr.Chatbot(
                        label=tr("en", "chat_title"),
                        height=420,
                        value=[]
                    )

                    msg = gr.Textbox(
                        label=tr("en", "ask_question"),
                        placeholder=tr("en", "chat_placeholder")
                    )

                    with gr.Row():
                        send_btn = gr.Button(
                        tr("en", "send"),
                        variant="primary",
                        elem_id="send-btn"
)
                        clear_btn = gr.Button(tr("en", "clear_chat"))

    disease_context_state = gr.State(value=None)

    analyze_btn.click(
        fn=classify_and_fill,
        inputs=[image_input, language],
        outputs=[
            detected_disease,
            diagnosis_summary,
            disease_info_md,
            sample_image,
            disease_context_state,
        ],
    )

    send_btn.click(
        fn=chat_fn,
        inputs=[msg, language, disease_context_state, chatbot],
        outputs=[chatbot, msg],
    )

    msg.submit(
        fn=chat_fn,
        inputs=[msg, language, disease_context_state, chatbot],
        outputs=[chatbot, msg],
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot, msg],
    )

    language.change(
        fn=update_ui_language,
        inputs=[language],
        outputs=[
            header_html,
            language,
            image_analysis_md,
            image_input,
            analyze_btn,
            detected_disease,
            diagnosis_summary,
            disease_info_md,
            reference_image_md,
            sample_image,
            assistant_md,
            chatbot,
            msg,
            send_btn,
            clear_btn,
        ],
    )

demo.launch(
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue"),
    css=CUSTOM_CSS
)