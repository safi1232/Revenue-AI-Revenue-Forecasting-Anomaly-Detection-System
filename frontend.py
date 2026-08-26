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

import html
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
# Matches the real FastAPI UserInput schema:
#   { "Date": "YYYY-MM-DD", "Orders": int, "Customers": int,
#     "Products": int, "Actual_Revenue": float (OPTIONAL) }
#
# Every field consumed by the Revenue Prediction form is driven
# ONLY by this list — nothing is hardcoded elsewhere in the app.
#
# Supported "type" values: "number", "int", "bool", "date",
# "categorical", "slider", "text", "optional_number"
#
# "optional_number" fields are OMITTED from the payload entirely
# if left blank — they are never sent as 0 or null.
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

    {
        "name": "Actual_Revenue",
        "label": "Actual Revenue (optional)",
        "type": "optional_number",
        "group": "Business Information",
        "default": None,
        "help": "Optional. If you already know the actual revenue for this date, "
                "enter it here for comparison. Leave blank to skip — it will not "
                "be sent to the backend if empty.",
    },

]

# Fields the backend guarantees on every successful /predict response.
REQUIRED_RESPONSE_FIELDS = [
    "predicted_revenue",
    "is_anomaly",
    "anomaly_status",
    "anomaly_score",
    "model_version",
]

# Additional fields the backend may return (e.g. when Actual_Revenue was
# supplied). These are optional — never required for validation — but are
# preserved and carried through history/export when present.
OPTIONAL_RESPONSE_FIELDS = [
    "actual_revenue",
    "residual",
    "absolute_error",
    "forecast_anomaly",
    "isolation_anomaly",
]

DEFAULT_API_URL = "https://ai-powerd.fastapicloud.dev"
CURRENCY_SYMBOL = "$"  # Keep the backend/project currency unchanged.


# ============================================================
# THEME CONFIGURATION
# ============================================================

THEMES = {
    "light": {
        "page_bg": "#f5f8fc",
        "primary_bg": "#ffffff",
        "card_bg": "#ffffff",
        "sidebar_bg": "#f8fafc",
        "text": "#0b1f3a",
        "muted_text": "#68758a",
        "border": "#dfe6ef",
        "primary": "#1a73e8",
        "success": "#16834f",
        "danger": "#c0392b",
        "warning": "#a66a00",
        "chart_bg": "#ffffff",
        "grid": "#e8edf4",
        "header_bg": "#0b1f3a",
        "header_bg_2": "#14345e",
        "header_text": "#ffffff",
        "input_bg": "#ffffff",
        "button_bg": "#1a73e8",
        "button_text": "#ffffff",
        "table_header": "#eef3f9",
        "surface_alt": "#f6f8fb",
        "success_bg": "#eaf8f0",
        "success_border": "#bfe8cd",
        "danger_bg": "#fdf0ef",
        "danger_border": "#f0c8c4",
        "warning_bg": "#fff7e6",
        "warning_border": "#f0d79a",
        "shadow": "rgba(16, 30, 54, 0.08)",
    },
    "dark": {
        "page_bg": "#07111f",
        "primary_bg": "#0a1626",
        "card_bg": "#0d2035",
        "sidebar_bg": "#091725",
        "text": "#f4f7fb",
        "muted_text": "#9aabc0",
        "border": "#1d3853",
        "primary": "#3b9cff",
        "success": "#35c878",
        "danger": "#ff6b61",
        "warning": "#f0b84b",
        "chart_bg": "#0a1626",
        "grid": "#20374f",
        "header_bg": "#08182a",
        "header_bg_2": "#0d2a48",
        "header_text": "#f7fbff",
        "input_bg": "#0d2035",
        "button_bg": "#1677e8",
        "button_text": "#ffffff",
        "table_header": "#102941",
        "surface_alt": "#0b1b2c",
        "success_bg": "#0d3022",
        "success_border": "#1d6946",
        "danger_bg": "#351b1d",
        "danger_border": "#71302d",
        "warning_bg": "#332914",
        "warning_border": "#70551f",
        "shadow": "rgba(0, 0, 0, 0.28)",
    },
}


def get_theme() -> dict:
    """Return the active theme from session state."""
    return THEMES.get(st.session_state.get("theme", "light"), THEMES["light"])


# ============================================================
# CUSTOM CSS
# ============================================================

def inject_css() -> None:
    """Inject CSS generated from the active theme."""
    t = get_theme()

    css = f"""
    <style>
    :root {{
        --page-bg: {t['page_bg']};
        --primary-bg: {t['primary_bg']};
        --card-bg: {t['card_bg']};
        --sidebar-bg: {t['sidebar_bg']};
        --text: {t['text']};
        --muted-text: {t['muted_text']};
        --border: {t['border']};
        --primary: {t['primary']};
        --success: {t['success']};
        --danger: {t['danger']};
        --warning: {t['warning']};
        --chart-bg: {t['chart_bg']};
        --grid: {t['grid']};
        --header-bg: {t['header_bg']};
        --header-bg-2: {t['header_bg_2']};
        --header-text: {t['header_text']};
        --input-bg: {t['input_bg']};
        --button-bg: {t['button_bg']};
        --button-text: {t['button_text']};
        --table-header: {t['table_header']};
        --surface-alt: {t['surface_alt']};
        --success-bg: {t['success_bg']};
        --success-border: {t['success_border']};
        --danger-bg: {t['danger_bg']};
        --danger-border: {t['danger_border']};
        --warning-bg: {t['warning_bg']};
        --warning-border: {t['warning_border']};
        --shadow: {t['shadow']};
    }}

    html, body, [class*="css"] {{
        font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    }}

    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background: var(--page-bg) !important;
        color: var(--text) !important;
    }}

    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--border);
    }}

    [data-testid="stSidebar"] * {{
        color: var(--text);
    }}

    #MainMenu, footer {{ visibility: hidden; }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1500px;
    }}

    .app-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        padding: 1.35rem 1.75rem;
        background: linear-gradient(135deg, var(--header-bg) 0%, var(--header-bg-2) 100%);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 5px 20px var(--shadow);
    }}
    .app-header h1 {{ color: var(--header-text); font-size: 1.65rem; font-weight: 700; margin: 0; }}
    .app-header p {{ color: #c7d6ee; margin: 0.15rem 0 0 0; font-size: 0.95rem; }}

    .status-pill {{ padding: 0.45rem 0.9rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem; white-space: nowrap; }}
    .status-pill.online {{ background: var(--success-bg); color: var(--success); border: 1px solid var(--success-border); }}
    .status-pill.offline {{ background: var(--danger-bg); color: var(--danger); border: 1px solid var(--danger-border); }}

    .section-header {{ font-size: 1.1rem; font-weight: 700; color: var(--text); margin: 1.6rem 0 0.75rem 0; border-left: 4px solid var(--primary); padding-left: 0.6rem; }}

    .kpi-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem 1.25rem; box-shadow: 0 2px 10px var(--shadow); height: 100%; min-height: 122px; }}
    .kpi-label {{ font-size: 0.78rem; font-weight: 700; color: var(--muted-text); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.35rem; }}
    .kpi-value {{ font-size: 1.55rem; font-weight: 800; color: var(--text); overflow-wrap: anywhere; }}
    .kpi-value.green {{ color: var(--success); }}
    .kpi-value.red {{ color: var(--danger); }}
    .kpi-sub {{ font-size: 0.8rem; color: var(--muted-text); margin-top: 0.2rem; }}

    .result-card {{ background: linear-gradient(135deg, var(--header-bg) 0%, var(--header-bg-2) 100%); border-radius: 18px; padding: 2rem 2.2rem; color: var(--header-text); box-shadow: 0 6px 24px var(--shadow); }}
    .result-card .label {{ font-size: 0.95rem; color: #b9caea; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
    .result-card .value {{ font-size: clamp(2rem, 4vw, 3rem); font-weight: 800; margin: 0.2rem 0 0.4rem 0; overflow-wrap: anywhere; }}
    .result-card .meta {{ font-size: 0.85rem; color: #9db3d9; overflow-wrap: anywhere; }}

    .anomaly-card {{ border-radius: 16px; padding: 1.5rem 1.8rem; border: 1px solid transparent; height: 100%; box-sizing: border-box; }}
    .anomaly-card.normal {{ background: var(--success-bg); border-color: var(--success-border); }}
    .anomaly-card.anomaly {{ background: var(--danger-bg); border-color: var(--danger-border); }}
    .anomaly-title {{ font-size: 1.25rem; font-weight: 800; margin-bottom: 0.25rem; }}
    .anomaly-title.normal {{ color: var(--success); }}
    .anomaly-title.anomaly {{ color: var(--danger); }}
    .anomaly-desc {{ font-size: 0.92rem; color: var(--text); margin-bottom: 0.6rem; opacity: 0.88; }}
    .anomaly-score-tag {{ display: inline-block; background: rgba(128, 148, 170, 0.16); border: 1px solid var(--border); border-radius: 8px; padding: 0.25rem 0.6rem; font-size: 0.85rem; font-weight: 600; color: var(--text); overflow-wrap: anywhere; }}

    .info-panel {{ background: var(--surface-alt); border: 1px solid var(--border); border-radius: 14px; padding: 1.2rem 1.4rem; color: var(--text); }}
    .cta-panel {{ background: color-mix(in srgb, var(--primary) 9%, var(--card-bg)); border: 1px dashed var(--primary); border-radius: 16px; padding: 1.6rem 1.8rem; text-align: center; color: var(--text); }}
    .cta-panel h3 {{ color: var(--text); margin-bottom: 0.3rem; }}

    .sidebar-title {{ color: var(--text); font-size: 1.4rem; font-weight: 800; }}
    .sidebar-subtitle {{ color: var(--muted-text); font-size: 0.85rem; }}
    .theme-status {{ color: var(--muted-text); font-size: 0.82rem; margin-top: 0.35rem; }}
    .footer {{ margin-top: 2.5rem; padding: 1rem 0 0.25rem; text-align: center; color: var(--muted-text); font-size: 0.78rem; border-top: 1px solid var(--border); }}

    /* Streamlit native controls */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input {{
        background: var(--input-bg) !important; color: var(--text) !important; border-color: var(--border) !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {{
        background: var(--input-bg) !important; color: var(--text) !important; border-color: var(--border) !important;
    }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{ background: var(--card-bg) !important; color: var(--text) !important; }}
    [role="option"] {{ background: var(--card-bg) !important; color: var(--text) !important; }}
    [role="option"]:hover {{ background: var(--surface-alt) !important; }}
    [data-testid="stSlider"] [role="slider"] {{ background: var(--primary) !important; }}
    [data-testid="stCheckbox"] label, [data-testid="stRadio"] label {{ color: var(--text) !important; }}
    [data-testid="stButton"] > button, [data-testid="stDownloadButton"] > button, [data-testid="stFormSubmitButton"] > button {{
        background: var(--button-bg) !important; color: var(--button-text) !important; border: 1px solid var(--button-bg) !important;
        border-radius: 9px !important; font-weight: 650 !important; min-height: 2.5rem;
    }}
    [data-testid="stButton"] > button:hover, [data-testid="stDownloadButton"] > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{ filter: brightness(1.08); }}
    [data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
    [data-testid="stExpander"] {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; }}
    [data-testid="stAlert"] {{ border-radius: 10px; }}
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {{ color: var(--text); }}

    @media (max-width: 800px) {{
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
        .app-header {{ align-items: flex-start; flex-direction: column; }}
        .status-pill {{ align-self: flex-start; }}
        .result-card {{ padding: 1.4rem; }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


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
        # Keep the sidebar radio widget's own state key in sync with
        # "nav_page" so programmatic navigation (e.g. the dashboard CTA
        # button) actually moves the visible page, not just the tracking
        # variable.
        "navigation_radio": "📊 Dashboard",
        "theme": "light",
        "reset_confirm": False,
        "_last_health_response": None,
        "_last_health_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to_page(page_name: str) -> None:
    """
    Programmatically navigate to a page.

    Streamlit's radio widget (key="navigation_radio" in the sidebar) owns
    its own entry in session_state once created; simply changing "nav_page"
    does not move the widget's visible selection on rerun. Both keys must
    be updated together for navigation triggered from buttons (e.g. the
    dashboard "Generate Revenue Forecast" CTA) to actually work.
    """
    st.session_state["nav_page"] = page_name
    st.session_state["navigation_radio"] = page_name


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
    return messages.get(error_type, "⚠️ The API returned an unexpected response format.")


def validate_prediction_response(data: dict) -> bool:
    """
    Only the fields the backend always guarantees are required.
    Optional fields (actual_revenue, residual, absolute_error,
    forecast_anomaly, isolation_anomaly) are accepted if present but never
    required — a response without them (e.g. no Actual_Revenue was sent)
    is still valid.
    """
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
    color_class = f" {color}" if color in {"green", "red"} else ""
    return f"""
        <div class="kpi-card">
            <div class="kpi-label">{html.escape(str(label))}</div>
            <div class="kpi-value{color_class}">{html.escape(str(value))}</div>
            <div class="kpi-sub">{html.escape(str(sub))}</div>
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
    score_display = html.escape(str(score_display))
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

    safe_title = html.escape(str(title))
    safe_subtitle = html.escape(str(subtitle))
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <h1>{safe_title}</h1>
                <p>{safe_subtitle}</p>
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
    """Build the existing revenue-history visualization using theme colors."""
    t = get_theme()
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
            line=dict(color=t["primary"], width=2.5),
            marker=dict(size=7, color=t["primary"]),
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
                marker=dict(size=13, color=t["danger"], symbol="x", line=dict(width=2)),
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
        plot_bgcolor=t["chart_bg"],
        paper_bgcolor=t["chart_bg"],
        font=dict(color=t["text"]),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, color=t["text"], linecolor=t["border"]),
        yaxis=dict(showgrid=True, gridcolor=t["grid"], color=t["text"], linecolor=t["border"]),
        hovermode="closest",
    )
    return fig


def create_anomaly_gauge(anomaly_score) -> go.Figure:
    """Visualize the raw backend anomaly score; styling only is theme-aware."""
    t = get_theme()
    try:
        score = float(anomaly_score)
    except (TypeError, ValueError):
        score = 0.0

    # Display range retained from the original visualization; it does not alter
    # the raw backend score or make an anomaly decision.
    axis_max = max(abs(score) * 1.5, 1.0)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".4f", "font": {"color": t["text"]}},
            title={"text": "Raw Anomaly Score", "font": {"color": t["text"]}},
            gauge={
                "axis": {"range": [0, axis_max], "tickcolor": t["text"], "tickfont": {"color": t["text"]}},
                "bar": {"color": t["danger"] if score >= axis_max / 2 else t["primary"]},
                "bgcolor": t["surface_alt"],
                "borderwidth": 1,
                "bordercolor": t["border"],
                "steps": [
                    {"range": [0, axis_max / 2], "color": t["success_bg"]},
                    {"range": [axis_max / 2, axis_max], "color": t["danger_bg"]},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=t["chart_bg"],
        plot_bgcolor=t["chart_bg"],
        font=dict(color=t["text"]),
        margin=dict(l=20, r=20, t=60, b=10),
        height=320,
    )
    return fig


# ============================================================
# SIDEBAR
# ============================================================

def _apply_health_result(result: dict) -> None:
    """Store the explicit health-check result without making another request."""
    if result.get("ok"):
        st.session_state["api_status"] = "online"
        data = result.get("data") or {}
        st.session_state["model_loaded"] = data.get("model_loaded", data.get("model_status"))
        st.session_state["model_version"] = data.get("model_version")
        st.session_state["_last_health_response"] = data
        st.session_state["_last_health_error"] = None
    else:
        st.session_state["api_status"] = "offline"
        st.session_state["model_loaded"] = None
        st.session_state["_last_health_error"] = result


def check_api_status_from_sidebar() -> None:
    with st.spinner("Checking FastAPI /health..."):
        result = check_api_health(st.session_state["api_url"])
    _apply_health_result(result)


def reset_session_data() -> None:
    """Clear prediction/API session data while preserving theme and API URL."""
    api_url = st.session_state.get("api_url", DEFAULT_API_URL)
    theme = st.session_state.get("theme", "light")
    nav_page = st.session_state.get("nav_page", "📊 Dashboard")
    st.session_state.update({
        "api_url": api_url,
        "theme": theme,
        "nav_page": nav_page,
        "navigation_radio": nav_page,
        "api_status": None,
        "model_loaded": None,
        "model_version": None,
        "last_prediction": None,
        "last_predicted_revenue": None,
        "last_anomaly_status": None,
        "last_anomaly_score": None,
        "last_is_anomaly": None,
        "prediction_history": [],
        "_last_health_response": None,
        "_last_health_error": None,
        "reset_confirm": False,
    })


def render_sidebar() -> None:
    with st.sidebar:
        theme = get_theme()
        current_theme = st.session_state.get("theme", "light")
        current_label = "☀️ Light Mode" if current_theme == "light" else "🌙 Dark Mode"
        next_label = "🌙 Dark Mode" if current_theme == "light" else "☀️ Light Mode"

        st.markdown(
            '<div style="padding:0.45rem 0 1rem 0;">'
            '<div class="sidebar-title">📈 Revenue AI</div>'
            '<div class="sidebar-subtitle">Revenue Forecasting &amp; Anomaly Detection</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Backend Configuration")
        api_url_input = st.text_input(
            "API URL",
            value=st.session_state["api_url"],
            help="Base URL of the FastAPI backend.",
            key="api_url_input",
        )
        normalized = normalize_url(api_url_input) if api_url_input else DEFAULT_API_URL
        st.session_state["api_url"] = normalized

        if st.button("🔌 Check API Connection", use_container_width=True, key="sidebar_check_api"):
            check_api_status_from_sidebar()

        status = st.session_state.get("api_status")
        if status == "online":
            st.success("🟢 Connected")
            loaded = st.session_state.get("model_loaded")
            version = st.session_state.get("model_version")
            if loaded is not None:
                st.caption(f"Model loaded: {loaded}")
            if version:
                st.caption(f"Model version: {version}")
        elif status == "offline":
            st.error("🔴 Offline")
            with st.expander("Technical Details"):
                st.write(st.session_state.get("_last_health_error"))
        else:
            st.caption("API status has not been checked.")

        st.markdown("---")
        st.markdown("### Appearance")
        if st.button(next_label, use_container_width=True, key="theme_toggle"):
            st.session_state["theme"] = "dark" if current_theme == "light" else "light"
            st.rerun()
        st.markdown(f'<div class="theme-status">Current: {html.escape(current_label)}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Navigation")
        pages = [
            "📊 Dashboard",
            "📈 Revenue Prediction",
            "🚨 Anomaly Detection",
            "🔌 API Status",
            "ℹ️ About",
        ]
        selected_page = st.radio(
            "Go to",
            pages,
            index=pages.index(st.session_state["nav_page"]) if st.session_state["nav_page"] in pages else 0,
            label_visibility="collapsed",
            key="navigation_radio",
        )
        st.session_state["nav_page"] = selected_page

        st.markdown("---")
        st.markdown("### Session")
        if not st.session_state.get("reset_confirm"):
            if st.button("🗑️ Reset Session", use_container_width=True, key="reset_session"):
                st.session_state["reset_confirm"] = True
                st.rerun()
        else:
            st.warning("Are you sure you want to clear prediction history and the latest prediction?")
            yes_col, no_col = st.columns(2)
            with yes_col:
                if st.button("Yes, Reset", use_container_width=True, key="confirm_reset"):
                    reset_session_data()
                    st.rerun()
            with no_col:
                if st.button("Cancel", use_container_width=True, key="cancel_reset"):
                    st.session_state["reset_confirm"] = False
                    st.rerun()


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
    if ftype == "optional_number":
        # Rendered as a plain text field so it can be left blank. Blank
        # means "not provided" and is dropped from the payload entirely —
        # it is never coerced to 0 or sent as null.
        default_str = "" if field.get("default") is None else str(field.get("default"))
        return st.text_input(label, value=default_str, help=help_text, placeholder="Leave blank if unknown")
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
    validation_error: str | None = None

    with st.form("prediction_form", clear_on_submit=False):
        for group_name, fields in groups.items():
            st.markdown(f'<div class="section-header">{group_name}</div>', unsafe_allow_html=True)
            cols = st.columns(min(len(fields), 3) or 1)
            for i, field in enumerate(fields):
                with cols[i % len(cols)]:
                    raw_value = render_input_field(field)
                    ftype = field["type"]
                    name = field["name"]

                    if ftype == "date":
                        values[name] = raw_value.isoformat() if isinstance(raw_value, date) else raw_value
                    elif ftype == "optional_number":
                        text_val = (raw_value or "").strip()
                        if text_val == "":
                            # Truly optional: omit the key so the backend
                            # never receives Actual_Revenue at all.
                            continue
                        try:
                            values[name] = float(text_val)
                        except ValueError:
                            validation_error = f"'{field['label']}' must be a number if provided."
                    else:
                        values[name] = raw_value

        submitted = st.form_submit_button("Generate Forecast", use_container_width=True)

    if submitted and validation_error:
        st.error(f"⚠️ {validation_error}")
        return None

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
                <p>Use the Revenue Prediction page to send business data to the AI backend.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Generate Revenue Forecast →", use_container_width=True, key="cta_generate_forecast"):
            # Update both the tracking variable AND the radio widget's own
            # state key so the sidebar actually reflects the navigation
            # (see go_to_page docstring for why both are needed).
            go_to_page("📈 Revenue Prediction")
            st.rerun()
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="label">Latest Predicted Revenue</div>
                    <div class="value">{revenue_val}</div>
                    <div class="meta">Model Version: {html.escape(str(model_version))}</div>
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

        # Full history (including optional backend fields like
        # actual_revenue/residual/absolute_error when present) goes into
        # the CSV export, even though the on-screen table above stays
        # exactly as designed.
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
                # This already includes any optional fields the backend
                # returned (actual_revenue, residual, absolute_error,
                # forecast_anomaly, isolation_anomaly) since we keep the
                # full dict as-is.
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
                # Carry through optional backend fields when present,
                # without altering any value returned by the backend.
                for opt_field in OPTIONAL_RESPONSE_FIELDS:
                    if opt_field in data:
                        history_entry[opt_field] = data[opt_field]

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
                    <div class="meta">Model Version: {html.escape(str(last["model_version"]))}</div>
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
        for opt_field in OPTIONAL_RESPONSE_FIELDS:
            if opt_field in last:
                export_row[opt_field] = last[opt_field]

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

    if st.button("🔄 Refresh Status", use_container_width=True, key="refresh_api_status"):
        with st.spinner("Checking FastAPI /health..."):
            result = check_api_health(st.session_state["api_url"])
        _apply_health_result(result)

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

    st.markdown(
        '<div class="footer">Revenue AI • FastAPI Backend • Streamlit Frontend</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
