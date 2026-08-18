import os
import sys
from pathlib import Path

# Add repository root to Python's import path so this app can import live_data.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from datetime import datetime
import pandas as pd
import requests
import streamlit as st
from live_data.pipeline import prepare_latest_model_input


DEFAULT_API_URL = os.getenv("PREDICTION_API_URL", "https://oil-price-api-3-0.onrender.com/predict")

st.set_page_config(page_title="Oil Price Prediction", page_icon="📈", layout="wide")

st.markdown("""
<style>
.block-container{max-width:1120px;padding-top:2rem}
.hero{padding:1.5rem 1.7rem;border:1px solid rgba(128,128,128,.25);border-radius:18px;margin-bottom:1.25rem}
.eyebrow{font-size:.82rem;font-weight:700;letter-spacing:.12em;opacity:.65;text-transform:uppercase}
.hero h1{margin:.25rem 0;font-size:2.25rem}
.hero p{margin:0;opacity:.72}
.prediction-card{text-align:center;padding:2rem 1rem;border:1px solid rgba(128,128,128,.28);border-radius:22px;margin:.5rem 0 1.25rem}
.prediction-label{font-size:.9rem;font-weight:700;letter-spacing:.12em;opacity:.65;text-transform:uppercase}
.prediction-price{font-size:clamp(3.8rem,10vw,6.8rem);line-height:1;font-weight:800;margin:.35rem 0}
.prediction-unit{font-size:1rem;opacity:.65}
div.stButton>button{min-height:3.25rem;font-size:1.05rem;font-weight:700;border-radius:12px}
</style>
""", unsafe_allow_html=True)

def prepare_live_input():
    latest_60, metadata = prepare_latest_model_input(history_calendar_days=500)
    if latest_60.shape != (60, 26):
        raise ValueError(f"Expected (60, 26), got {latest_60.shape}.")
    return latest_60, metadata

def predict(latest_60, api_url):
    r = requests.post(api_url, json={"data": latest_60.values.tolist()}, timeout=180)
    r.raise_for_status()
    result = r.json()
    value = result.get("predicted_oil_price", result.get("prediction"))
    if value is None:
        raise KeyError("Prediction field missing from API response.")
    explanation = result.get("explanation")
    return float(value), explanation

st.markdown("""
<div class="hero">
<div class="eyebrow">Applied AI • LSTM Forecasting</div>
<h1>Oil Price Prediction</h1>
<p>Current market data → 26 engineered features → 60-day sequence → LSTM forecast</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("System")
    api_url = st.text_input("Prediction API", DEFAULT_API_URL)
    st.caption("60 time steps × 26 features")
    st.caption("FastAPI + scaler + LSTM")

st.write("Each click downloads the latest available market data, rebuilds the 60 × 26 input in Streamlit, then sends only that matrix to the Docker/FastAPI prediction service.")

if st.button("Generate New Prediction", type="primary", use_container_width=True):
    try:
        with st.status("Preparing current market data…", expanded=True) as status:
            st.write("Downloading configured market series…")
            latest_60, metadata = prepare_live_input()
            st.write(f"Prepared: {latest_60.shape[0]} rows × {latest_60.shape[1]} features")
            st.write(f"Latest usable market date: {metadata['latest_input_date']}")
            st.write("Sending matrix to FastAPI…")
            price, explanation = predict(latest_60, api_url)
            status.update(label="Prediction completed", state="complete", expanded=False)
        st.session_state.update(prediction=price, explanation=explanation, metadata=metadata, latest_60=latest_60.copy(),
                                generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. For local testing, make sure Docker is running on port 8000.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

if "prediction" in st.session_state:
    price = st.session_state["prediction"]
    metadata = st.session_state["metadata"]
    latest_60 = st.session_state["latest_60"]
    st.markdown(f"""
    <div class="prediction-card">
      <div class="prediction-label">Next Oil Price Forecast</div>
      <div class="prediction-price">${price:,.2f}</div>
      <div class="prediction-unit">USD per barrel</div>
    </div>
    """, unsafe_allow_html=True)

    a,b,c = st.columns(3)
    a.metric("Latest market-data date", str(metadata["latest_input_date"]))
    b.metric("Input window", "60 days")
    c.metric("Model features", "26")

    explanation = st.session_state.get("explanation")
    if explanation:
        with st.expander("💡 Why this forecast?", expanded=True):
            st.write(explanation.get("summary", ""))
            drivers = explanation.get("key_drivers") or []
            if drivers:
                st.markdown("**Key drivers:**")
                for driver in drivers:
                    st.markdown(f"- {driver}")
            if explanation.get("confidence_note"):
                st.caption(explanation["confidence_note"])


    # Feature 0 is Oil_Price in the trained 26-feature model.
    hist = pd.DataFrame({"Oil price": pd.to_numeric(latest_60.iloc[:,0], errors="coerce").values})
    hist.index = range(1, len(hist)+1)
    st.subheader("Oil price input window")
    st.line_chart(hist, height=300)
    st.caption("Historical Oil_Price values from the 60-row model input; the large number above is the LSTM forecast.")

    with st.expander("Prediction details"):
        st.write("First input date:", metadata["first_input_date"])
        st.write("Latest input date:", metadata["latest_input_date"])
        st.write("Generated:", st.session_state["generated_at"])
        st.write("API endpoint:", api_url)
        if metadata.get("sources"):
            st.write("Data sources:", metadata["sources"])
else:
    st.info("Click **Generate New Prediction**. Streamlit will fetch current market data and prepare the model input before calling the prediction API.")
