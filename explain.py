"""
explain.py — LLM-based plain-language explanation layer for the WTI price forecast.

Place this file at the repo root, next to predict.py.
It does not touch the LSTM at all — it takes the LSTM's numeric output
(from predict_oil_price) plus a few raw feature values, and asks Gemini
to turn them into a short, plain-English explanation.

Uses Google's Gemini API (free tier — no credit card required).
Get a key at https://aistudio.google.com/api-keys
"""

import os
import json
from google import genai

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def explain_prediction(predicted_price: float, current_price: float, top_features: dict) -> dict:
    """
    Turns a raw model prediction + its top contributing features into a
    plain-language explanation.

    Parameters
    ----------
    predicted_price : float
        The LSTM's predicted WTI price for the next day.
    current_price : float
        Today's actual WTI price (for context/comparison).
    top_features : dict
        A small dict of the most influential features and their values,
        e.g. {"Momentum_7": 1.2, "Momentum_21": -0.4, "Volatility_21": 3.1}

    Returns
    -------
    dict with keys: "summary", "key_drivers" (list of str), "confidence_note"
    Falls back to a safe default if the API call or JSON parsing fails,
    so a bad response never breaks the prediction endpoint.
    """
    fallback = {
        "summary": "Explanation temporarily unavailable.",
        "key_drivers": [],
        "confidence_note": "Model estimate only — not financial advice."
    }

    try:
        direction = "up" if predicted_price > current_price else "down"
        change_pct = abs(predicted_price - current_price) / current_price * 100

        prompt = f"""You are explaining an oil price forecast to a non-technical reader.

Current WTI price: ${current_price:.2f}
Predicted next-day price: ${predicted_price:.2f} ({direction}, {change_pct:.1f}% change)
Top contributing model features: {json.dumps(top_features)}

Respond with ONLY valid JSON, no other text, in exactly this shape:
{{
  "summary": "2 sentence plain-English explanation of the forecast",
  "key_drivers": ["short phrase 1", "short phrase 2", "short phrase 3"],
  "confidence_note": "1 sentence noting this is a model estimate, not financial advice"
}}"""

        response = _get_client().models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )

        raw_text = response.text.strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        parsed = json.loads(raw_text)

        for key in ("summary", "key_drivers", "confidence_note"):
            if key not in parsed:
                return fallback

        return parsed

    except Exception:
        return fallback
