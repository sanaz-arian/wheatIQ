import logging
from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.kb_loader import load_diseases_kb

logger = logging.getLogger("uvicorn.error")
logger.warning(">>> chatbot.py module loaded")

DISEASES_KB = load_diseases_kb()


def build_disease_context(disease_key: str, language: str) -> str:
    if not disease_key or disease_key not in DISEASES_KB:
        return ""

    item = DISEASES_KB[disease_key]
    if language == "de":
        return (
            f"Krankheit: {item.get('display_name_de', disease_key)}\n"
            f"Überblick: {item.get('overview_de', '')}\n"
            f"Ursachen: {item.get('causes_de', '')}\n"
            f"Saison/Bedingungen: {item.get('season_de', '')}\n"
            f"Ausbreitung: {item.get('spread_de', '')}\n"
            f"Behandlung: {item.get('treatment_de', '')}\n"
            f"Vorbeugung: {item.get('prevention_de', '')}\n"
            f"Hinweis: {item.get('warning_de', '')}\n"
        )

    return (
        f"Disease: {item.get('display_name_en', disease_key)}\n"
        f"Overview: {item.get('overview_en', '')}\n"
        f"Causes: {item.get('causes_en', '')}\n"
        f"Season/Conditions: {item.get('season_en', '')}\n"
        f"Spread: {item.get('spread_en', '')}\n"
        f"Treatment: {item.get('treatment_en', '')}\n"
        f"Prevention: {item.get('prevention_en', '')}\n"
        f"Note: {item.get('warning_en', '')}\n"
    )


def fallback_answer(user_message: str, disease_context: str, language: str) -> str:
    if language == "de":
        if disease_context:
            return (
                "Ich konnte das Sprachmodell gerade nicht erreichen. "
                "Basierend auf der internen Wissensbasis kann ich sagen:\n\n"
                f"{disease_context}\n"
                "Dies sind allgemeine Hinweise und kein Ersatz für fachliche pflanzenbauliche Beratung."
            )
        return (
            "Ich konnte das Sprachmodell gerade nicht erreichen. "
            "Bitte versuche es erneut. Ich kann weiterhin auf Basis der internen Wissensbasis helfen."
        )

    if disease_context:
        return (
            "I could not reach the language model right now. "
            "Based on the internal knowledge base, here is a useful summary:\n\n"
            f"{disease_context}\n"
            "This is general guidance and not a substitute for expert agronomic advice."
        )

    return (
        "I could not reach the language model right now. "
        "Please try again. I can still help based on the internal knowledge base."
    )


def ask_chatbot(user_message: str, language: str, disease_key: str = None, recent_history=None):
    logger.warning(">>> ask_chatbot function is running")
    logger.warning(f"GROQ_API_KEY exists: {bool(GROQ_API_KEY)}")
    logger.warning(f"GROQ_MODEL: {GROQ_MODEL}")
    logger.warning(f"disease_key: {disease_key}")
    logger.warning(f"language: {language}")

    disease_context = build_disease_context(disease_key, language) if disease_key else ""

    if not GROQ_API_KEY:
        logger.warning(">>> fallback because GROQ_API_KEY is missing")
        return fallback_answer(user_message, disease_context, language)

    system_prompt = (
        "You are WheatIQ, a bilingual wheat disease assistant. "
        "Answer only in the user's selected language. "
        "Use the internal disease context first. "
        "Be practical, concise, and clear. "
        "Do not claim certainty beyond the model result. "
        "Always include that guidance is general and not a substitute for expert agronomic advice."
    )

    if language == "de":
        system_prompt += " The selected language is German."
    else:
        system_prompt += " The selected language is English."

    messages = [{"role": "system", "content": system_prompt}]

    if disease_context:
        messages.append({
            "role": "system",
            "content": f"Internal disease context:\n{disease_context}"
        })

    if recent_history:
        for item in recent_history:
            messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3
        )
        logger.warning(">>> Groq call succeeded")
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f">>> Groq exception: {repr(e)}")
        return fallback_answer(user_message, disease_context, language)