import os
from datetime import datetime

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv(
    "PREDICTION_API_URL",
    "https://oil-price-api-4-0.onrender.com/predict/latest",
)


def latest_endpoint(api_url: str) -> str:
    """Accept either a base API URL, /predict, or /predict/latest."""
    url = api_url.strip().rstrip("/")
    if url.endswith("/predict/latest"):
        return url
    if url.endswith("/predict"):
        return url[: -len("/predict")] + "/predict/latest"
    return url + "/predict/latest"


def get_latest_prediction(api_url: str) -> dict:
    """
    Ask FastAPI to predict from shared_data/latest_input.json.

    Streamlit does NOT download FRED/XAUS data and does NOT create
    the 60 x 26 matrix. That input must already have been prepared
    outside/before the Docker container.
    """
    endpoint = latest_endpoint(api_url)
    response = requests.get(endpoint, timeout=180)
    response.raise_for_status()
    result = response.json()

    if "predicted_oil_price" not in result:
        raise KeyError("predicted_oil_price is missing from the API response.")

    return result


st.set_page_config(
    page_title="Oil Price Prediction",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container{max-width:1120px;padding-top:2rem}
    .hero{padding:1.5rem 1.7rem;border:1px solid rgba(128,128,128,.25);
          border-radius:18px;margin-bottom:1.25rem}
    .eyebrow{font-size:.82rem;font-weight:700;letter-spacing:.12em;
             opacity:.65;text-transform:uppercase}
    .hero h1{margin:.25rem 0;font-size:2.25rem}
    .hero p{margin:0;opacity:.72}
    .prediction-card{text-align:center;padding:2rem 1rem;
                     border:1px solid rgba(128,128,128,.28);
                     border-radius:22px;margin:.5rem 0 1.25rem}
    .prediction-label{font-size:.9rem;font-weight:700;letter-spacing:.12em;
                      opacity:.65;text-transform:uppercase}
    .prediction-price{font-size:clamp(3.8rem,10vw,6.8rem);
                      line-height:1;font-weight:800;margin:.35rem 0}
    .prediction-unit{font-size:1rem;opacity:.65}
    div.stButton>button{min-height:3.25rem;font-size:1.05rem;
                        font-weight:700;border-radius:12px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Applied AI • LSTM Forecasting</div>
      <h1>Oil Price Prediction</h1>
      <p>Prepared 60 × 26 market input → FastAPI → scaler → LSTM forecast</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("System")
    api_url = st.text_input("Prediction API", DEFAULT_API_URL)
    st.caption("60 time steps × 26 features")
    st.caption("FastAPI + scaler + LSTM")

st.write(
    "The market input is prepared outside the Docker container. "
    "This app sends no requests to FRED or XAUS."
)

if st.button("Generate New Prediction", type="primary", use_container_width=True):
    try:
        with st.status("Requesting prediction from FastAPI…", expanded=True) as status:
            endpoint = latest_endpoint(api_url)
            st.write(f"Calling: {endpoint}")
            result = get_latest_prediction(api_url)
            price = float(result["predicted_oil_price"])
            status.update(
                label="Prediction completed",
                state="complete",
                expanded=False,
            )

        st.session_state["prediction"] = price
        st.session_state["result"] = result
        st.session_state["generated_at"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    except requests.exceptions.ConnectionError as exc:
        st.error(f"Could not connect to the FastAPI service: {exc}")
    except requests.exceptions.Timeout:
        st.error("The FastAPI service did not respond before the timeout.")
    except requests.exceptions.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:
            pass
        st.error(
            f"FastAPI returned HTTP {exc.response.status_code}. "
            f"{detail or exc}"
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


if "prediction" in st.session_state:
    price = st.session_state["prediction"]
    result = st.session_state["result"]

    st.markdown(
        f"""
        <div class="prediction-card">
          <div class="prediction-label">Next Oil Price Forecast</div>
          <div class="prediction-price">${price:,.2f}</div>
          <div class="prediction-unit">USD per barrel</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)
    a.metric(
        "Latest market-data date",
        str(result.get("latest_input_date") or "Not provided"),
    )
    b.metric("Input window", "60 days")
    c.metric("Model features", "26")

    with st.expander("Prediction details"):
        st.write("First input date:", result.get("first_input_date"))
        st.write("Latest input date:", result.get("latest_input_date"))
        st.write("Input shape:", result.get("input_shape", [60, 26]))
        st.write(
            "Input generated at (UTC):",
            result.get("generated_at_utc"),
        )
        st.write(
            "Displayed at:",
            st.session_state["generated_at"],
        )
        st.write("API endpoint:", latest_endpoint(api_url))
        if result.get("sources"):
            st.write("Data sources:", result["sources"])
else:
    st.info("Click **Generate New Prediction** to call the deployed FastAPI service.")
