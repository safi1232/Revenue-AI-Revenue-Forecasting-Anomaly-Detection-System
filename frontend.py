"""
Revenue AI — Revenue Forecasting & Anomaly Detection Dashboard
================================================================
A pure Streamlit FRONTEND/CLIENT for an existing FastAPI backend.

This application:
  - collects user input,
  - calls the FastAPI backend over HTTP,
  - displays the backend's response exactly as returned,
  - visualizes returned data,
  - maintains session-level state and prediction history.

This application NEVER:
  - loads or trains a model,
  - imports pickle / joblib / scikit-learn / any ML library,
  - calculates revenue or anomaly scores,
  - defines its own anomaly thresholds,
  - reinterprets or rescales a backend result.

Every prediction and anomaly decision comes directly from FastAPI.

Run with:
    streamlit run app.py
"""

import json
from datetime import datetime, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Revenue AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# USER_INPUT_FIELDS
# IMPORTANT: Replace this configuration with the REAL FastAPI
# UserInput schema before production use. Backend compatibility
# CANNOT be guaranteed until these are swapped for the real
# Pydantic field names, types, and constraints.
#
# Every field consumed by the Revenue Prediction form is driven
# ONLY by this list — nothing is hardcoded elsewhere in the app.
#
# Supported "type" values: "number", "int", "bool", "date",
# "categorical", "slider", "text"
# ============================================================

USER_INPUT_FIELDS = [

    # ========================================================
    # BUSINESS INFORMATION
    # ========================================================

    {
        "name": "Date",
        "label": "Prediction Date",
        "type": "date",
        "group": "Business Information",
        "default": date.today(),
        "help": "Prediction date in YYYY-MM-DD format.",
    },

    {
        "name": "Products",
        "label": "Products Sold",
        "type": "int",
        "group": "Business Information",
        "default": 95,
        "min": 0,
        "help": "Number of unique products sold.",
    },

    {
        "name": "Orders",
        "label": "Orders",
        "type": "int",
        "group": "Business Information",
        "default": 250,
        "min": 0,
        "help": "Number of orders.",
    },

    {
        "name": "Customers",
        "label": "Customers",
        "type": "int",
        "group": "Business Information",
        "default": 180,
        "min": 0,
        "help": "Number of unique customers.",
    },

]
REQUIRED_RESPONSE_FIELDS = [
    "predicted_revenue",
    "is_anomaly",
    "anomaly_status",
    "anomaly_score",
    "model_version",
]

DEFAULT_API_URL = "https://ai-powerd.fastapicloud.dev"
CURRENCY_SYMBOL = "$"  # Change if the backend/project specifies another currency.


# ============================================================
# CUSTOM CSS
# ============================================================

def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }

        #MainMenu, footer { visibility: hidden; }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .app-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1.25rem 1.75rem;
            background: linear-gradient(135deg, #0b1f3a 0%, #14345e 100%);
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 18px rgba(11, 31, 58, 0.18);
        }
        .app-header h1 {
            color: #ffffff;
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0;
        }
        .app-header p {
            color: #c7d6ee;
            margin: 0.15rem 0 0 0;
            font-size: 0.95rem;
        }
        .status-pill {
            padding: 0.4rem 0.9rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
            white-space: nowrap;
        }
        .status-pill.online {
            background: #e3f7ea;
            color: #14804a;
        }
        .status-pill.offline {
            background: #fdeaea;
            color: #b3261e;
        }

        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            color: #0b1f3a;
            margin: 1.75rem 0 0.75rem 0;
            border-left: 4px solid #1a56db;
            padding-left: 0.6rem;
        }

        .kpi-card {
            background: #ffffff;
            border: 1px solid #e7ebf2;
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 10px rgba(16, 30, 54, 0.05);
            height: 100%;
        }
        .kpi-label {
            font-size: 0.8rem;
            font-weight: 600;
            color: #6b7688;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.35rem;
        }
        .kpi-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: #0b1f3a;
        }
        .kpi-value.green { color: #14804a; }
        .kpi-value.red { color: #b3261e; }
        .kpi-sub {
            font-size: 0.8rem;
            color: #8a93a3;
            margin-top: 0.15rem;
        }

        .result-card {
            background: linear-gradient(135deg, #0b1f3a 0%, #163b6b 100%);
            border-radius: 18px;
            padding: 2rem 2.2rem;
            color: #ffffff;
            box-shadow: 0 6px 24px rgba(11, 31, 58, 0.25);
        }
        .result-card .label {
            font-size: 0.95rem;
            color: #b9caea;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .result-card .value {
            font-size: 3rem;
            font-weight: 800;
            margin: 0.2rem 0 0.4rem 0;
        }
        .result-card .meta {
            font-size: 0.85rem;
            color: #9db3d9;
        }

        .anomaly-card {
            border-radius: 16px;
            padding: 1.5rem 1.8rem;
            border: 1px solid transparent;
        }
        .anomaly-card.normal {
            background: #eefaf2;
            border-color: #bfe8cd;
        }
        .anomaly-card.anomaly {
            background: #fdeeed;
            border-color: #f3c3c0;
        }
        .anomaly-title {
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .anomaly-title.normal { color: #14804a; }
        .anomaly-title.anomaly { color: #b3261e; }
        .anomaly-desc {
            font-size: 0.92rem;
            color: #3d4759;
            margin-bottom: 0.5rem;
        }
        .anomaly-score-tag {
            display: inline-block;
            background: rgba(11, 31, 58, 0.06);
            border-radius: 8px;
            padding: 0.25rem 0.6rem;
            font-size: 0.85rem;
            font-weight: 600;
            color: #0b1f3a;
        }

        .info-panel {
            background: #f6f8fb;
            border: 1px solid #e7ebf2;
            border-radius: 14px;
            padding: 1.2rem 1.4rem;
        }

        .cta-panel {
            background: #f0f5ff;
            border: 1px dashed #9db8ef;
            border-radius: 16px;
            padding: 1.6rem 1.8rem;
            text-align: center;
        }
        .cta-panel h3 {
            color: #0b1f3a;
            margin-bottom: 0.3rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SESSION STATE
# ============================================================

def init_session_state() -> None:
    defaults = {
        "api_url": DEFAULT_API_URL,
        "api_status": None,          # None | "online" | "offline"
        "model_loaded": None,
        "model_version": None,
        "last_prediction": None,      # full raw backend response dict
        "last_predicted_revenue": None,
        "last_anomaly_status": None,
        "last_anomaly_score": None,
        "last_is_anomaly": None,
        "prediction_history": [],     # list of dicts
        "nav_page": "📊 Dashboard",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# API LAYER
# ============================================================

def normalize_url(url: str) -> str:
    return url.strip().rstrip("/")


def check_api_health(api_url: str, timeout: int = 8) -> dict:
    """
    Calls GET /health and returns a structured result.
    Never raises — always returns a dict describing the outcome.
    """
    url = normalize_url(api_url)
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        if response.status_code == 200:
            try:
                data = response.json()
            except ValueError:
                data = {}
            return {
                "ok": True,
                "status_code": response.status_code,
                "data": data,
                "error": None,
            }
        return {
            "ok": False,
            "status_code": response.status_code,
            "data": None,
            "error": f"Health check returned HTTP {response.status_code}.",
        }
    except requests.exceptions.ConnectionError:
        return {"ok": False, "status_code": None, "data": None,
                "error": "connection_error"}
    except requests.exceptions.Timeout:
        return {"ok": False, "status_code": None, "data": None,
                "error": "timeout"}
    except requests.exceptions.RequestException as exc:
        return {"ok": False, "status_code": None, "data": None, "error": str(exc)}


def predict_revenue(api_url: str, payload: dict, timeout: int = 30) -> dict:
    """
    Calls POST /predict and returns a structured result:
        {"ok": bool, "status_code": int|None, "data": dict|None,
         "error_type": str|None, "raw_text": str|None}
    Never raises — all exceptions are converted into error_type values.
    """
    url = normalize_url(api_url)
    try:
        response = requests.post(f"{url}/predict", json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return {"ok": False, "status_code": None, "data": None,
                "error_type": "connection_error", "raw_text": None}
    except requests.exceptions.Timeout:
        return {"ok": False, "status_code": None, "data": None,
                "error_type": "timeout", "raw_text": None}
    except requests.exceptions.RequestException as exc:
        return {"ok": False, "status_code": None, "data": None,
                "error_type": "request_exception", "raw_text": str(exc)}

    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            return {"ok": False, "status_code": 200, "data": None,
                    "error_type": "invalid_json", "raw_text": response.text}
        return {"ok": True, "status_code": 200, "data": data,
                "error_type": None, "raw_text": None}

    if response.status_code == 422:
        error_type = "validation_error"
    elif response.status_code == 500:
        error_type = "server_error"
    else:
        error_type = "unexpected_status"

    return {
        "ok": False,
        "status_code": response.status_code,
        "data": None,
        "error_type": error_type,
        "raw_text": response.text,
    }


def handle_api_error(result: dict) -> str:
    """
    Maps a predict_revenue()/check_api_health() error result to the
    exact user-facing message required by the spec.
    """
    error_type = result.get("error_type") or result.get("error")
    messages = {
        "connection_error": "🔴 Unable to connect to the FastAPI server.",
        "timeout": "⏱️ The prediction request timed out. Please try again.",
        "validation_error": "⚠️ Invalid input. Please check the prediction fields.",
        "server_error": "❌ Prediction service encountered an internal error.",
    }
    return messages.get(error_type, "⚠️ The API returned an unexpected error.")


def validate_prediction_response(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    return all(field in data for field in REQUIRED_RESPONSE_FIELDS)


# ============================================================
# FORMATTING HELPERS
# ============================================================

def format_currency(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{CURRENCY_SYMBOL}{float(value):,.2f}"
    except (TypeError, ValueError):
        return "—"


def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# RENDER HELPERS
# ============================================================

def render_kpi_card(label: str, value: str, sub: str = "", color: str = "") -> str:
    color_class = f" {color}" if color else ""
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value{color_class}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
    """


def render_anomaly_card(is_anomaly: bool, anomaly_score) -> str:
    state = "anomaly" if is_anomaly else "normal"
    title = "🔴 ANOMALY DETECTED" if is_anomaly else "🟢 NORMAL"
    desc = (
        "Revenue significantly differs from the expected pattern."
        if is_anomaly
        else "Revenue is within the expected range."
    )
    score_display = anomaly_score if anomaly_score is not None else "—"
    return f"""
        <div class="anomaly-card {state}">
            <div class="anomaly-title {state}">{title}</div>
            <div class="anomaly-desc">{desc}</div>
            <span class="anomaly-score-tag">Anomaly Score: {score_display}</span>
        </div>
    """


def render_app_header(title: str, subtitle: str) -> None:
    status = st.session_state.get("api_status")
    if status == "online":
        pill_class, pill_text = "online", "🟢 API Connected"
    elif status == "offline":
        pill_class, pill_text = "offline", "🔴 API Offline"
    else:
        pill_class, pill_text = "offline", "⚪ API Status Unknown"

    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
            <div class="status-pill {pill_class}">{pill_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHART / VISUALIZATION FUNCTIONS
# ============================================================

def create_revenue_chart(history: list) -> go.Figure:
    """
    Builds a Plotly line chart of predicted revenue over time using
    ONLY real session prediction history. Anomalous points are
    visually distinguished from normal points.
    """
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["predicted_revenue"],
            mode="lines+markers",
            name="Predicted Revenue",
            line=dict(color="#1a56db", width=2.5),
            marker=dict(size=7, color="#1a56db"),
            hovertemplate=(
                "Timestamp: %{x}<br>"
                "Predicted Revenue: %{y:$,.2f}<br>"
                "<extra></extra>"
            ),
        )
    )

    anomalies = df[df["is_anomaly"] == True]  # noqa: E712
    if not anomalies.empty:
        fig.add_trace(
            go.Scatter(
                x=anomalies["timestamp"],
                y=anomalies["predicted_revenue"],
                mode="markers",
                name="Anomaly",
                marker=dict(size=13, color="#b3261e", symbol="x", line=dict(width=2)),
                customdata=anomalies[["anomaly_status", "anomaly_score"]],
                hovertemplate=(
                    "Timestamp: %{x}<br>"
                    "Predicted Revenue: %{y:$,.2f}<br>"
                    "Status: %{customdata[0]}<br>"
                    "Anomaly Score: %{customdata[1]}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Revenue Prediction History",
        xaxis_title="Timestamp",
        yaxis_title="Predicted Revenue",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color="#0b1f3a"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#eef1f6"),
        hovermode="closest",
    )
    return fig


def create_anomaly_gauge(anomaly_score) -> go.Figure:
    """
    Visualizes the raw backend anomaly score. Does NOT assume a
    0-1 or 0-100 range unless one is known, and never rescales the
    backend's value. Purely a visualization — makes no independent
    anomaly decision.
    """
    try:
        score = float(anomaly_score)
    except (TypeError, ValueError):
        score = 0.0

    # Dynamic, sensible visualization range since the backend does not
    # define a normalized scale. This is display-only and never alters
    # the raw score shown to the user.
    axis_max = max(abs(score) * 1.5, 1.0)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".4f"},
            title={"text": "Raw Anomaly Score"},
            gauge={
                "axis": {"range": [0, axis_max]},
                "bar": {"color": "#b3261e" if score >= axis_max / 2 else "#1a56db"},
                "bgcolor": "#ffffff",
                "borderwidth": 1,
                "bordercolor": "#e7ebf2",
                "steps": [
                    {"range": [0, axis_max / 2], "color": "#eefaf2"},
                    {"range": [axis_max / 2, axis_max], "color": "#fdeeed"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        font=dict(color="#0b1f3a"),
        margin=dict(l=20, r=20, t=60, b=10),
        height=320,
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <div style="font-size:1.4rem; font-weight:800; color:#0b1f3a;">📈 Revenue AI</div>
                <div style="font-size:0.85rem; color:#6b7688;">Revenue Forecasting &amp; Anomaly Detection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Backend Configuration**")
        api_url_input = st.text_input(
            "API URL",
            value=st.session_state["api_url"],
            help="Base URL of the FastAPI backend.",
        )
        st.session_state["api_url"] = normalize_url(api_url_input) if api_url_input else DEFAULT_API_URL

        if st.button("🔌 Check API Connection", use_container_width=True):
            result = check_api_health(st.session_state["api_url"])
            if result["ok"]:
                st.session_state["api_status"] = "online"
                data = result["data"] or {}
                st.session_state["model_loaded"] = data.get("model_loaded", data.get("model_status"))
                st.session_state["model_version"] = data.get("model_version")
            else:
                st.session_state["api_status"] = "offline"
                st.session_state["model_loaded"] = None
                st.session_state["_last_health_error"] = result

        status = st.session_state.get("api_status")
        if status == "online":
            st.success("🟢 Connected")
            if st.session_state.get("model_loaded") is not None:
                st.caption(f"Model loaded: {st.session_state['model_loaded']}")
            if st.session_state.get("model_version"):
                st.caption(f"Model version: {st.session_state['model_version']}")
        elif status == "offline":
            st.error("🔴 Offline")
            with st.expander("Technical Details"):
                st.write(st.session_state.get("_last_health_error"))

        st.markdown("---")
        st.markdown("**Navigation**")
        pages = [
            "📊 Dashboard",
            "📈 Revenue Prediction",
            "🚨 Anomaly Detection",
            "🔌 API Status",
            "ℹ️ About",
        ]
        st.session_state["nav_page"] = st.radio(
            "Go to",
            pages,
            index=pages.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in pages else 0,
            label_visibility="collapsed",
        )


# ============================================================
# DYNAMIC FORM RENDERING (driven by USER_INPUT_FIELDS)
# ============================================================

def render_input_field(field: dict):
    ftype = field["type"]
    label = field["label"]
    help_text = field.get("help", "")

    if ftype == "number":
        return st.number_input(
            label, value=float(field.get("default", 0.0)),
            min_value=field.get("min"), help=help_text,
        )
    if ftype == "int":
        return st.number_input(
            label, value=int(field.get("default", 0)),
            min_value=field.get("min"), step=1, help=help_text,
        )
    if ftype == "bool":
        return st.checkbox(label, value=bool(field.get("default", False)), help=help_text)
    if ftype == "date":
        return st.date_input(label, value=field.get("default", date.today()), help=help_text)
    if ftype == "categorical":
        options = field.get("options", [])
        default = field.get("default")
        index = options.index(default) if default in options else 0
        return st.selectbox(label, options=options, index=index, help=help_text)
    if ftype == "slider":
        return st.slider(
            label,
            min_value=field.get("min", 0),
            max_value=field.get("max", 100),
            value=field.get("default", field.get("min", 0)),
            help=help_text,
        )
    if ftype == "text":
        return st.text_input(label, value=str(field.get("default", "")), help=help_text)

    # Fallback — should not occur if USER_INPUT_FIELDS is well-formed.
    return st.text_input(label, value=str(field.get("default", "")), help=help_text)


def build_prediction_form() -> dict | None:
    groups: dict[str, list[dict]] = {}
    for field in USER_INPUT_FIELDS:
        groups.setdefault(field["group"], []).append(field)

    values: dict = {}

    with st.form("prediction_form", clear_on_submit=False):
        for group_name, fields in groups.items():
            st.markdown(f'<div class="section-header">{group_name}</div>', unsafe_allow_html=True)
            cols = st.columns(min(len(fields), 3) or 1)
            for i, field in enumerate(fields):
                with cols[i % len(cols)]:
                    raw_value = render_input_field(field)
                    if isinstance(raw_value, date):
                        values[field["name"]] = raw_value.isoformat()
                    else:
                        values[field["name"]] = raw_value

        submitted = st.form_submit_button("Generate Forecast", use_container_width=True)

    return values if submitted else None


# ============================================================
# PAGE: DASHBOARD
# ============================================================

def page_dashboard() -> None:
    render_app_header(
        "Revenue Forecasting & Anomaly Detection",
        "AI-powered revenue prediction and anomaly monitoring",
    )

    # --- KPI cards ---
    last = st.session_state["last_prediction"]
    revenue_val = format_currency(st.session_state["last_predicted_revenue"])
    anomaly_status = st.session_state["last_anomaly_status"]
    is_anomaly = st.session_state["last_is_anomaly"]
    anomaly_score = st.session_state["last_anomaly_score"]
    model_version = last.get("model_version") if last else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            render_kpi_card(
                "Predicted Revenue",
                revenue_val if last else "—",
                "AI Forecast" if last else "No prediction available",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        if last is None:
            st.markdown(render_kpi_card("Anomaly Status", "—", "No prediction available"), unsafe_allow_html=True)
        else:
            label = "🔴 ANOMALY" if is_anomaly else "🟢 NORMAL"
            color = "red" if is_anomaly else "green"
            st.markdown(
                render_kpi_card("Anomaly Status", label, anomaly_status or "", color),
                unsafe_allow_html=True,
            )
    with col3:
        score_display = anomaly_score if last else "—"
        st.markdown(render_kpi_card("Anomaly Score", str(score_display), "Raw backend value"), unsafe_allow_html=True)
    with col4:
        st.markdown(
            render_kpi_card("Model Version", model_version if last else "—", "Backend model"),
            unsafe_allow_html=True,
        )

    # --- Quick prediction / latest summary ---
    st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)
    if last is None:
        st.markdown(
            """
            <div class="cta-panel">
                <h3>Generate your first revenue forecast</h3>
                <p style="color:#4a5568;">Use the Revenue Prediction page to send business data to the AI backend.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Generate Revenue Forecast →", use_container_width=True):
            st.session_state["nav_page"] = "📈 Revenue Prediction"
            st.rerun()
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Latest Predicted Revenue</div>
                    <div class="value">{revenue_val}</div>
                    <div class="meta">Model Version: {model_version}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(render_anomaly_card(bool(is_anomaly), anomaly_score), unsafe_allow_html=True)

    # --- Revenue history chart ---
    st.markdown('<div class="section-header">Revenue Prediction History</div>', unsafe_allow_html=True)
    history = st.session_state["prediction_history"]
    if history:
        st.plotly_chart(create_revenue_chart(history), use_container_width=True)
    else:
        st.markdown(
            '<div class="info-panel">No prediction history yet for this session.</div>',
            unsafe_allow_html=True,
        )

    # --- History table ---
    st.markdown('<div class="section-header">Prediction History Table</div>', unsafe_allow_html=True)
    if history:
        df = pd.DataFrame(history).sort_values("timestamp", ascending=False)
        display_df = df.copy()
        display_df["predicted_revenue"] = display_df["predicted_revenue"].apply(format_currency)
        display_df["anomaly_status"] = display_df["anomaly_status"]
        display_df = display_df[
            ["timestamp", "predicted_revenue", "anomaly_status", "anomaly_score", "model_version"]
        ]
        display_df.columns = ["Timestamp", "Predicted Revenue", "Status", "Anomaly Score", "Model Version"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv_bytes = pd.DataFrame(history).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download History CSV",
            data=csv_bytes,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.markdown(
            '<div class="info-panel">No history to display yet.</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE: REVENUE PREDICTION
# ============================================================

def page_revenue_prediction() -> None:
    render_app_header("Revenue Prediction", "Enter business information to generate an AI-powered revenue forecast.")

    payload = build_prediction_form()

    if payload is not None:
        with st.spinner("Running AI revenue forecast..."):
            result = predict_revenue(st.session_state["api_url"], payload)

        if not result["ok"]:
            st.error(handle_api_error(result))
            with st.expander("Technical Details"):
                st.write(
                    {
                        "status_code": result.get("status_code"),
                        "error_type": result.get("error_type"),
                        "raw_text": result.get("raw_text"),
                    }
                )
        else:
            data = result["data"]
            if not validate_prediction_response(data):
                st.warning("⚠️ The API returned an unexpected response format.")
                with st.expander("Technical Details"):
                    st.write(data)
            else:
                st.success("✅ Prediction completed successfully")

                # Store exact backend response — values are never altered.
                st.session_state["last_prediction"] = data
                st.session_state["last_predicted_revenue"] = data["predicted_revenue"]
                st.session_state["last_anomaly_status"] = data["anomaly_status"]
                st.session_state["last_anomaly_score"] = data["anomaly_score"]
                st.session_state["last_is_anomaly"] = data["is_anomaly"]

                history_entry = {
                    "timestamp": format_timestamp(datetime.now()),
                    "predicted_revenue": data["predicted_revenue"],
                    "is_anomaly": data["is_anomaly"],
                    "anomaly_status": data["anomaly_status"],
                    "anomaly_score": data["anomaly_score"],
                    "model_version": data["model_version"],
                }
                st.session_state["prediction_history"].append(history_entry)

    # --- Result section (uses latest stored prediction, if any) ---
    last = st.session_state["last_prediction"]
    if last:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Predicted Revenue</div>
                    <div class="value">{format_currency(last["predicted_revenue"])}</div>
                    <div class="meta">Model Version: {last["model_version"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_anomaly_card(bool(last["is_anomaly"]), last["anomaly_score"]),
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-header">Anomaly Score Visualization</div>', unsafe_allow_html=True)
        st.plotly_chart(create_anomaly_gauge(last["anomaly_score"]), use_container_width=True)

        st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
        export_row = {
            "timestamp": format_timestamp(datetime.now()),
            "predicted_revenue": last["predicted_revenue"],
            "is_anomaly": last["is_anomaly"],
            "anomaly_status": last["anomaly_status"],
            "anomaly_score": last["anomaly_score"],
            "model_version": last["model_version"],
        }
        csv_bytes = pd.DataFrame([export_row]).to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Prediction",
            data=csv_bytes,
            file_name="prediction_result.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# PAGE: ANOMALY DETECTION
# ============================================================

def page_anomaly_detection() -> None:
    render_app_header("Anomaly Detection", "Monitoring revenue behavior for unusual patterns")

    st.markdown(
        """
        <div class="info-panel">
        The FastAPI backend evaluates the revenue prediction and determines whether the
        result represents normal or unusual revenue behavior.
        </div>
        """,
        unsafe_allow_html=True,
    )

    last = st.session_state["last_prediction"]
    st.markdown('<div class="section-header">Current Status</div>', unsafe_allow_html=True)

    if last is None:
        st.markdown(
            '<div class="info-panel">No prediction available yet. Generate a forecast on the Revenue Prediction page.</div>',
            unsafe_allow_html=True,
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_kpi_card("Predicted Revenue", format_currency(last["predicted_revenue"])), unsafe_allow_html=True)
    with c2:
        label = "🔴 ANOMALY DETECTED" if last["is_anomaly"] else "🟢 NORMAL"
        color = "red" if last["is_anomaly"] else "green"
        st.markdown(render_kpi_card("Anomaly Status", label, last["anomaly_status"], color), unsafe_allow_html=True)
    with c3:
        st.markdown(render_kpi_card("Anomaly Score", str(last["anomaly_score"])), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Model Version", last["model_version"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Risk Indicator</div>', unsafe_allow_html=True)
    st.markdown(render_anomaly_card(bool(last["is_anomaly"]), last["anomaly_score"]), unsafe_allow_html=True)

    st.markdown('<div class="section-header">Anomaly Score Visualization</div>', unsafe_allow_html=True)
    st.plotly_chart(create_anomaly_gauge(last["anomaly_score"]), use_container_width=True)


# ============================================================
# PAGE: API STATUS
# ============================================================

def page_api_status() -> None:
    render_app_header("API Status", "Live status of the FastAPI backend")

    if st.button("🔄 Refresh Status", use_container_width=True):
        result = check_api_health(st.session_state["api_url"])
        if result["ok"]:
            st.session_state["api_status"] = "online"
            data = result["data"] or {}
            st.session_state["model_loaded"] = data.get("model_loaded", data.get("model_status"))
            st.session_state["model_version"] = data.get("model_version")
            st.session_state["_last_health_response"] = data
        else:
            st.session_state["api_status"] = "offline"
            st.session_state["model_loaded"] = None
            st.session_state["_last_health_error"] = result

    status = st.session_state.get("api_status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            render_kpi_card("API Status", "🟢 ONLINE" if status == "online" else "🔴 OFFLINE" if status == "offline" else "⚪ UNKNOWN"),
            unsafe_allow_html=True,
        )
    with c2:
        loaded = st.session_state.get("model_loaded")
        st.markdown(render_kpi_card("Model Loaded", str(loaded) if loaded is not None else "—"), unsafe_allow_html=True)
    with c3:
        version = st.session_state.get("model_version")
        st.markdown(render_kpi_card("Model Version", version if version else "—"), unsafe_allow_html=True)
    with c4:
        st.markdown(render_kpi_card("Backend URL", st.session_state["api_url"]), unsafe_allow_html=True)

    if status == "offline":
        st.error(handle_api_error(st.session_state.get("_last_health_error", {})))
        with st.expander("Technical Details"):
            st.write(st.session_state.get("_last_health_error"))
    elif status == "online":
        with st.expander("Raw Health Response"):
            st.write(st.session_state.get("_last_health_response"))


# ============================================================
# PAGE: ABOUT
# ============================================================

def page_about() -> None:
    render_app_header("About Revenue AI", "How this system works")

    st.markdown(
        """
        <div class="info-panel">
        <p><strong>Revenue AI</strong> is an AI-powered revenue forecasting and anomaly
        detection system. This dashboard is a Streamlit <em>frontend only</em> — every
        prediction and anomaly decision is produced by the FastAPI backend and displayed
        here exactly as returned.</p>

        <p><strong>What this application does:</strong></p>
        <ul>
            <li>Revenue forecasting via the FastAPI backend</li>
            <li>Anomaly detection via the FastAPI backend</li>
            <li>Interactive Plotly visualization of predictions and anomaly scores</li>
            <li>Session-level prediction history with CSV export</li>
            <li>Live API health monitoring</li>
        </ul>

        <p><strong>What this application does not do:</strong> it does not load, train,
        or run any machine-learning model, and it does not calculate, threshold, or
        reinterpret anomaly scores. All modeling details (e.g. specific algorithms,
        feature engineering techniques) are the responsibility of the backend and are
        only described here if confirmed by that backend.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    init_session_state()
    inject_css()
    render_sidebar()

    page = st.session_state["nav_page"]
    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "📈 Revenue Prediction":
        page_revenue_prediction()
    elif page == "🚨 Anomaly Detection":
        page_anomaly_detection()
    elif page == "🔌 API Status":
        page_api_status()
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
