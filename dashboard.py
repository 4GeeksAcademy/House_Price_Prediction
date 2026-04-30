import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# Set page config
st.set_page_config(
    page_title="Housing Price Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Keyframe animations */
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    /* Main page background */
    .stMain {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        animation: fadeIn 0.8s ease-in;
    }
    
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        animation: slideInDown 0.6s ease-out;
    }
    
    /* Sidebar text color */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Metric cards with shadow and gradient */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        transition: all 0.3s ease;
        animation: slideInDown 0.5s ease-out backwards;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.3);
        animation: pulse 0.6s ease-in-out;
    }
    
    /* Metric labels */
    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 600;
        color: #555;
    }
    
    /* Metric values */
    [data-testid="stMetricValue"] {
        font-size: 18px !important;
        font-weight: 700;
        color: #667eea;
        animation: slideInDown 0.6s ease-out;
    }
    
    /* Section headings */
    h1 {
        color: #667eea !important;
        font-weight: 800 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.05);
        animation: slideInDown 0.6s ease-out;
    }
    
    h2 {
        color: #764ba2 !important;
        font-weight: 700 !important;
        animation: slideInDown 0.6s ease-out;
    }
    
    h3 {
        margin: 0.5rem 0 !important;
        font-size: 1.1rem !important;
        color: #555 !important;
        font-weight: 600;
        animation: slideInDown 0.5s ease-out;
    }
    
    h4 {
        margin: 0.2rem 0 !important;
        font-size: 0.95rem !important;
        color: #667eea !important;
    }
    
    /* Containers with nice borders */
    [data-testid="stContainer"] {
        border-radius: 12px;
        animation: fadeIn 0.7s ease-in;
    }
    
    /* Form inputs styling */
    input, textarea, select {
        border: 2px solid #e0e0e0 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
        transition: all 0.3s ease;
    }
    
    input:focus, textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 8px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Buttons styling */
    button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.3s ease !important;
    }
    
    button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Info/Success/Warning/Error boxes */
    .stAlert {
        border-radius: 8px !important;
        border-left: 4px solid !important;
    }
    
    /* Markdown text */
    p {
        color: #333 !important;
        line-height: 1.6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data using the trained model's data pipeline
@st.cache_data
def load_data():
    import os
    import sys
    from pathlib import Path
    from src.house_price_prediction.data import load_dataset
    from src.house_price_prediction.config import load_settings
    
    # Load using the same settings as training
    settings = load_settings()
    df = load_dataset(settings.raw_data_path)
    return df

# Load trained model from pickle/joblib file
@st.cache_resource
def load_trained_model():
    """Load the trained model artifact from the pkl file."""
    import os
    from pathlib import Path
    from src.house_price_prediction.model import load_model_artifact
    
    model_path = Path(__file__).parent / "models" / "house_price_model.pkl"
    
    if not model_path.exists():
        st.warning(f"⚠️ Model file not found at {model_path}")
        return None
    
    try:
        artifact = load_model_artifact(model_path)
        return artifact
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

df = load_data()
model_artifact = load_trained_model()


def call_api(method: str, base_url: str, path: str, payload: dict | None = None, params: dict | None = None):
    """Execute a backend API request and normalize response handling for UI rendering."""
    target_url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.request(
            method=method,
            url=target_url,
            json=payload,
            params=params,
            timeout=20,
        )
        try:
            body = response.json()
        except ValueError:
            body = {"raw": response.text}
        return response.status_code, body, target_url
    except requests.RequestException as exc:
        return None, {"error": str(exc)}, target_url


def calculate_market_metrics(dataset: pd.DataFrame) -> dict:
    """Calculate 5 key metrics from the housing dataset."""
    if dataset.empty:
        return {}
    
    metrics = {}
    
    # 1. Neighborhood Stats
    metrics["neighborhood_stats"] = {
        "avg_price": dataset["price"].mean(),
        "median_price": dataset["price"].median(),
        "price_range": (dataset["price"].min(), dataset["price"].max()),
        "total_properties": len(dataset),
    }
    
    # 2. Property Characteristics
    metrics["property_characteristics"] = {
        "avg_lot_area": dataset["sqft_lot"].mean(),
        "avg_grade": dataset["grade"].mean(),
        "avg_condition": dataset["condition"].mean(),
        "avg_living_area": dataset["sqft_living"].mean(),
        "avg_bathrooms": dataset["bathrooms"].mean(),
        "avg_bedrooms": dataset["bedrooms"].mean(),
        "avg_floors": dataset["floors"].mean(),
    }
    
    # 3. Market Data (Price per sqft)
    dataset_copy = dataset.copy()
    dataset_copy["price_per_sqft"] = dataset_copy["price"] / dataset_copy["sqft_living"]
    dataset_copy = dataset_copy[dataset_copy["price_per_sqft"] > 0]
    metrics["market_data"] = {
        "avg_price_per_sqft": dataset_copy["price_per_sqft"].mean(),
        "median_price_per_sqft": dataset_copy["price_per_sqft"].median(),
    }
    
    # 4. Age Estimate (Average Year Built and Age)
    metrics["age_estimate"] = {
        "avg_year_built": dataset["yr_built"].mean(),
        "avg_age": (2026 - dataset["yr_built"].mean()),
        "oldest_year": dataset["yr_built"].min(),
        "newest_year": dataset["yr_built"].max(),
    }
    
    # 5. Tax Estimate (Approximate based on average price)
    # Typical property tax rate ~1.2% of property value per year in US
    tax_rate = 0.012
    metrics["tax_estimate"] = {
        "avg_annual_tax": (dataset["price"].mean() * tax_rate),
        "annual_tax_rate": f"{tax_rate * 100:.1f}%",
    }
    
    return metrics


def lookup_state_key(slot_index: int, name: str) -> str:
    return f"lookup_{slot_index}_{name}"


def render_prediction_result(slot_index: int) -> None:
    prediction = st.session_state.get(lookup_state_key(slot_index, "prediction"))
    prediction_error = st.session_state.get(lookup_state_key(slot_index, "prediction_error"))

    if prediction:
        st.success("✅ Prediction Complete!")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            price = prediction.get("predicted_price")
            if price is not None:
                st.metric("Predicted Price", f"${price:,.0f}")

        with res_col2:
            completeness = prediction.get("feature_snapshot", {}).get("completeness_score")
            if completeness is not None:
                st.metric("Data Completeness", f"{completeness:.1%}")

        with res_col3:
            request_id = str(prediction.get("request_id", "N/A"))
            st.metric("Request ID", request_id[:8] + "..." if len(request_id) > 8 else request_id)

        key_features = prediction.get("feature_snapshot", {}).get("features", {})
        if key_features:
            st.markdown("**Key Property Features:**")
            feat_cols = st.columns(min(len(key_features), 4))
            for idx, (feat_name, feat_val) in enumerate(list(key_features.items())[:4]):
                with feat_cols[idx % 4]:
                    st.metric(feat_name, feat_val)
    elif prediction_error:
        st.error(prediction_error["message"])
        detail = prediction_error.get("detail")
        if detail:
            if isinstance(detail, str):
                st.error(f"Error: {detail}")
            else:
                st.json(detail)


def render_market_metrics(metrics: dict) -> None:
    """Display the 5 calculated market metrics."""
    if not metrics:
        return
    
    st.markdown("---")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; text-align: center; animation: slideInDown 0.7s ease-out; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);">
        <h2 style="color: white; margin: 0; font-size: 2.2em; text-align: center; letter-spacing: 0.5px; animation: slideInDown 0.8s ease-out;">🏠 House Price Prediction with Socioeconomic Indicators</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Neighborhood Stats
    if "neighborhood_stats" in metrics:
        st.markdown("**Neighborhood Statistics**")
        ns = metrics["neighborhood_stats"]
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Avg Price", f"${ns['avg_price']:,.0f}")
        with col2:
            st.metric("Median Price", f"${ns['median_price']:,.0f}")
        with col3:
            st.metric("Min Price", f"${ns['price_range'][0]:,.0f}")
        with col4:
            st.metric("Max Price", f"${ns['price_range'][1]:,.0f}")
        with col5:
            st.metric("# Properties", f"{ns['total_properties']}")
    
    # 2. Property Characteristics
    if "property_characteristics" in metrics:
        st.markdown("**Average Property Characteristics**")
        pc = metrics["property_characteristics"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Lot Area (sqft)", f"{pc['avg_lot_area']:,.0f}")
            st.metric("Grade (1-13)", f"{pc['avg_grade']:.1f}")
            st.metric("Living Area (sqft)", f"{pc['avg_living_area']:,.0f}")
        with col2:
            st.metric("Condition (1-5)", f"{pc['avg_condition']:.1f}")
            st.metric("Bathrooms", f"{pc['avg_bathrooms']:.1f}")
            st.metric("Bedrooms", f"{pc['avg_bedrooms']:.1f}")
        with col3:
            st.metric("Floors", f"{pc['avg_floors']:.1f}")
    
    # 3. Market Data
    if "market_data" in metrics:
        st.markdown("**Market Data**")
        md = metrics["market_data"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Price/sqft", f"${md['avg_price_per_sqft']:.2f}")
        with col2:
            st.metric("Median Price/sqft", f"${md['median_price_per_sqft']:.2f}")
    
    # 4. Age Estimate
    if "age_estimate" in metrics:
        st.markdown("**Property Age**")
        ae = metrics["age_estimate"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Year Built", f"{ae['avg_year_built']:.0f}")
        with col2:
            st.metric("Avg Age", f"{ae['avg_age']:.1f} years")
        with col3:
            st.metric("Age Range", f"{ae['oldest_year']:.0f} - {ae['newest_year']:.0f}")
    
    # 5. Tax Estimate
    if "tax_estimate" in metrics:
        st.markdown("**Tax Estimate**")
        te = metrics["tax_estimate"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Annual Tax", f"${te['avg_annual_tax']:,.0f}")
        with col2:
            st.metric("Tax Rate", te['annual_tax_rate'])


def render_lookup_slot(slot_index: int, api_base_url: str) -> dict | None:
    normalized_key = lookup_state_key(slot_index, "normalized")
    prediction_key = lookup_state_key(slot_index, "prediction")
    prediction_error_key = lookup_state_key(slot_index, "prediction_error")

    if normalized_key not in st.session_state:
        st.session_state[normalized_key] = None
    if prediction_key not in st.session_state:
        st.session_state[prediction_key] = None
    if prediction_error_key not in st.session_state:
        st.session_state[prediction_error_key] = None

    slot_label = f"Search Address {slot_index + 1}"
    with st.container(border=True):
        st.markdown(f"### {slot_label}")

        with st.form(f"address_lookup_form_{slot_index}"):
            # Row 1: Address Line 1 and Address Line 2
            col1, col2 = st.columns([1, 1], gap="medium")
            with col1:
                st.text_input(
                    "Address Line 1 *",
                    placeholder="123 Main Street",
                    key=lookup_state_key(slot_index, "line1"),
                )
            with col2:
                st.text_input(
                    "Address Line 2",
                    placeholder="Apt 456",
                    key=lookup_state_key(slot_index, "line2"),
                )

            # Row 2: City and Postal Code
            col3, col4 = st.columns([1, 1], gap="medium")
            with col3:
                st.text_input(
                    "City *",
                    placeholder="Miami",
                    key=lookup_state_key(slot_index, "city"),
                )
            with col4:
                st.text_input(
                    "Postal Code *",
                    placeholder="33101",
                    key=lookup_state_key(slot_index, "postal"),
                )

            # Row 3: State and Country
            col5, col6 = st.columns([1, 1], gap="medium")
            with col5:
                st.text_input(
                    "State *",
                    placeholder="FL",
                    key=lookup_state_key(slot_index, "state"),
                )
            with col6:
                st.text_input(
                    "Country",
                    key=lookup_state_key(slot_index, "country"),
                    value=st.session_state.get(lookup_state_key(slot_index, "country"), "US"),
                )

            search_submitted = st.form_submit_button("🔍 Search Address", use_container_width=True)

        if search_submitted:
            lookup_line1 = st.session_state.get(lookup_state_key(slot_index, "line1"), "")
            lookup_city = st.session_state.get(lookup_state_key(slot_index, "city"), "")
            lookup_state = st.session_state.get(lookup_state_key(slot_index, "state"), "")
            lookup_line2 = st.session_state.get(lookup_state_key(slot_index, "line2"), "")
            lookup_postal = st.session_state.get(lookup_state_key(slot_index, "postal"), "")
            lookup_country = st.session_state.get(lookup_state_key(slot_index, "country"), "US")

            if not lookup_line1.strip() or not lookup_city.strip() or not lookup_state.strip() or not lookup_postal.strip():
                st.error("❌ Please fill in all required fields (marked with *).")
            else:
                lookup_payload = {
                    "address_line_1": lookup_line1.strip(),
                    "city": lookup_city.strip(),
                    "state": lookup_state.strip(),
                    "postal_code": lookup_postal.strip(),
                    "country": lookup_country.strip(),
                }
                if lookup_line2.strip():
                    lookup_payload["address_line_2"] = lookup_line2.strip()

                with st.spinner(f"🌍 Looking up address for {slot_label.lower()}..."):
                    sc, body, _url = call_api("POST", api_base_url, "/v1/properties/normalize", payload=lookup_payload)

                if sc == 200 and isinstance(body, dict):
                    st.session_state[normalized_key] = body
                    st.session_state[prediction_key] = None
                    st.session_state[prediction_error_key] = None
                    st.success("✅ Address found!")
                else:
                    st.session_state[normalized_key] = None
                    st.error(f"❌ Address lookup failed (Status {sc})")
                    st.json(body)

        normalized = st.session_state.get(normalized_key)
        if not normalized:
            st.info("Enter an address above and click Search Address.")
            return None

        col_addr, col_coords = st.columns([2, 1])

        with col_addr:
            st.subheader("📬 Normalized Address")
            addr_display = f"{normalized.get('address_line_1', '')}"
            if normalized.get("address_line_2"):
                addr_display += f", {normalized.get('address_line_2')}"
            addr_display += (
                f"\n{normalized.get('city', '')}, {normalized.get('state', '')} "
                f"{normalized.get('postal_code', '')}"
            )
            if normalized.get("country"):
                addr_display += f"\n{normalized.get('country', '')}"
            st.info(addr_display)

            if normalized.get("formatted_address"):
                st.caption(f"Formatted: {normalized['formatted_address']}")

        with col_coords:
            latitude = normalized.get("latitude")
            longitude = normalized.get("longitude")
            if latitude is not None and longitude is not None:
                st.metric("Latitude", f"{latitude:.4f}")
                st.metric("Longitude", f"{longitude:.4f}")
                if normalized.get("geocoding_source"):
                    st.caption(f"📡 Source: {normalized['geocoding_source']}")

        # Display market metrics
        metrics = calculate_market_metrics(df)
        render_market_metrics(metrics)

        st.markdown("---")
        st.subheader("💰 Predict Price")
        st.write("Use this address to get an instant price prediction.")

        st.text_input(
            "Your Email (optional)",
            placeholder="you@example.com",
            key=lookup_state_key(slot_index, "requested_by"),
        )

        if st.button("🔮 Predict House Price", use_container_width=True, key=lookup_state_key(slot_index, "predict")):
            pred_payload = {
                "address_line_1": normalized.get("address_line_1"),
                "city": normalized.get("city"),
                "state": normalized.get("state"),
                "postal_code": normalized.get("postal_code"),
                "country": normalized.get("country"),
            }
            if normalized.get("address_line_2"):
                pred_payload["address_line_2"] = normalized.get("address_line_2")

            requested_by = st.session_state.get(lookup_state_key(slot_index, "requested_by"), "")
            if requested_by.strip():
                pred_payload["requested_by"] = requested_by.strip()

            with st.spinner(f"🧠 Analyzing property for {slot_label.lower()}..."):
                pred_sc, pred_body, _pred_url = call_api(
                    "POST",
                    api_base_url,
                    "/v1/predictions",
                    payload=pred_payload,
                )

            if pred_sc == 201 and isinstance(pred_body, dict):
                st.session_state[prediction_key] = pred_body
                st.session_state[prediction_error_key] = None
                st.session_state["last_prediction_id"] = str(pred_body.get("prediction_id", ""))
            else:
                st.session_state[prediction_key] = None
                st.session_state[prediction_error_key] = {
                    "message": f"❌ Prediction failed (Status {pred_sc})",
                    "detail": pred_body.get("detail") if isinstance(pred_body, dict) else pred_body,
                }

        render_prediction_result(slot_index)
        return normalized

# Sidebar
st.sidebar.title("🏠 Housing Dashboard")
st.sidebar.markdown("---")

# Get API URL from environment or default to localhost
import os
default_api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

api_base_url = st.sidebar.text_input(
    "Live API Base URL",
    value=st.session_state.get("api_base_url", default_api_url),
    help="Use the backend URL you want the UI to test against.",
)
st.session_state["api_base_url"] = api_base_url

# Show environment info
if "onrender.com" in api_base_url:
    st.sidebar.success("🚀 Connected to Render deployment")
elif "localhost" in api_base_url or "127.0.0.1" in api_base_url:
    st.sidebar.info("🏠 Connected to local development server")
else:
    st.sidebar.info(f"📡 Custom API: {api_base_url}")

if st.sidebar.button("Check API Runtime", use_container_width=True):
    sc, body, url = call_api("GET", api_base_url, "/v1/health")
    st.session_state["api_runtime_health"] = {
        "status_code": sc,
        "body": body,
        "url": url,
    }

runtime_health = st.session_state.get("api_runtime_health")
if runtime_health is not None:
    status_code = runtime_health.get("status_code")
    health_body = runtime_health.get("body")
    if status_code == 200 and isinstance(health_body, dict):
        st.sidebar.success("API reachable")
        st.sidebar.caption(
            f"Providers: {health_body.get('geocoding_provider', 'unknown')} / "
            f"{health_body.get('property_data_provider', 'unknown')}"
        )
        st.sidebar.caption(
            f"Mock predictor: {health_body.get('mock_predictor_enabled', 'unknown')}"
        )
        geocoding_provider = str(health_body.get("geocoding_provider", "")).strip().lower()
        property_provider = str(health_body.get("property_data_provider", "")).strip().lower()
        mock_enabled = bool(health_body.get("mock_predictor_enabled", False))

        if mock_enabled:
            st.sidebar.warning(
                "Predictions are currently using mock mode. Set ENABLE_MOCK_PREDICTOR=false "
                "for live ML model inference."
            )
        if geocoding_provider == "fake" or property_provider == "fake":
            st.sidebar.warning(
                "At least one provider is fake. Use free-fallback providers for live address enrichment."
            )
    else:
        st.sidebar.error(f"Health check failed ({status_code}): {health_body}")

# Page selection
page = st.sidebar.radio(
    "Select a page:",
    ["Overview", "Data Analysis", "Feature Exploration", "Statistics", "Address Lookup", "Live API Tester"]
)

# ==================== PAGE: OVERVIEW ====================
if page == "Overview":
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #667eea; margin-bottom: 0.5rem;">🏠 Housing Price Prediction & Economic Indicators</h1>
        <p style="color: #666; font-size: 1.1rem;">Explore the housing dataset with interactive visualizations</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Status
    if model_artifact:
        st.success("✅ **Trained Model Loaded** from `models/house_price_model.pkl`")
        col_meta1, col_meta2, col_meta3 = st.columns(3)
        with col_meta1:
            st.metric("Model Name", model_artifact.metadata.model_name or "Random Forest")
        with col_meta2:
            st.metric("Features", len(model_artifact.metadata.feature_columns))
        with col_meta3:
            st.metric("Version", model_artifact.metadata.model_version or "1.0")
    else:
        st.warning("⚠️ No trained model found. Train a model using `scripts/train.py`")
    
    st.markdown("---")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Properties", len(df), delta=None)
    
    with col2:
        st.metric("Avg Price", f"${df['price'].mean():,.0f}", delta=None)
    
    with col3:
        st.metric("Price Range", f"${df['price'].min():,.0f} - ${df['price'].max():,.0f}", delta=None)
    
    with col4:
        st.metric("Avg Bedrooms", f"{df['bedrooms'].mean():.1f}", delta=None)
    
    st.markdown("---")
    
    # Price distribution
    st.subheader("📊 Price Distribution")
    
    fig = px.histogram(
        df, 
        x='price', 
        nbins=50,
        labels={'price': 'Price ($)'},
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(
        height=450, 
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig.add_annotation(
        text="<b>Price Distribution</b>",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=16, color="#667eea"),
        xanchor="center"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    fig = px.box(
        df, 
        y='price',
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(
        height=450, 
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig.add_annotation(
        text="<b>Price Box Plot</b>",
        xref="paper", yref="paper",
        x=0.5, y=1.08,
        showarrow=False,
        font=dict(size=16, color="#667eea"),
        xanchor="center"
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ==================== PAGE: DATA ANALYSIS ====================
elif page == "Data Analysis":
    st.title("📈 Data Analysis")
    
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_price = st.slider(
            "Min Price ($)",
            int(df['price'].min()),
            int(df['price'].max()),
            int(df['price'].min()),
            step=10000
        )
    
    with col2:
        max_price = st.slider(
            "Max Price ($)",
            int(df['price'].min()),
            int(df['price'].max()),
            int(df['price'].max()),
            step=10000
        )
    
    with col3:
        bedrooms = st.multiselect(
            "Bedrooms",
            sorted(df['bedrooms'].unique()),
            default=sorted(df['bedrooms'].unique())
        )
    
    # Apply filters
    filtered_df = df[
        (df['price'] >= min_price) & 
        (df['price'] <= max_price) &
        (df['bedrooms'].isin(bedrooms))
    ]
    
    st.markdown(f"**Showing {len(filtered_df)} properties** (out of {len(df)})")
    
    st.markdown("---")
    
    # Correlation Analysis
    st.subheader("🔗 Correlation with Price")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    correlation = numeric_df.corr()['price'].sort_values(ascending=False)
    correlation_df = correlation.reset_index()
    correlation_df.columns = ['feature', 'correlation']
    
    fig = px.bar(
        correlation_df,
        x='correlation',
        y='feature',
        orientation='h',
        title='Feature Correlation with Price',
        labels={'x': 'Correlation Coefficient'},
        color='correlation',
        color_continuous_scale='RdBu'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Scatter plots
    st.subheader("📍 Price vs Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            filtered_df,
            x='sqft_living',
            y='price',
            title='Price vs Living Area',
            trendline='ols',
            labels={'sqft_living': 'Sq Ft Living', 'price': 'Price ($)'},
            color='price',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            filtered_df,
            x='grade',
            y='price',
            title='Price vs Grade',
            trendline='ols',
            labels={'grade': 'Grade', 'price': 'Price ($)'},
            color='price',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: FEATURE EXPLORATION ====================
elif page == "Feature Exploration":
    st.title("🔬 Feature Exploration")
    
    st.markdown("---")
    
    # Feature selector
    st.subheader("Select a Feature to Explore")
    
    features = [col for col in df.columns if col not in ['id', 'date', 'lat', 'long']]
    selected_feature = st.selectbox("Choose a feature:", features)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution
        fig = px.histogram(
            df,
            x=selected_feature,
            title=f'Distribution of {selected_feature}',
            nbins=30,
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # vs Price
        if selected_feature != 'price':
            fig = px.scatter(
                df,
                x=selected_feature,
                y='price',
                title=f'{selected_feature} vs Price',
                trendline='ols',
                color='price',
                color_continuous_scale='Viridis'
            )
        else:
            fig = px.box(
                df,
                y='price',
                title=f'{selected_feature} Distribution',
                color_discrete_sequence=['#636EFA']
            )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Statistics
    st.subheader(f"📊 Statistics for {selected_feature}")
    
    stats = {
        'Count': df[selected_feature].count(),
        'Mean': df[selected_feature].mean(),
        'Median': df[selected_feature].median(),
        'Std Dev': df[selected_feature].std(),
        'Min': df[selected_feature].min(),
        'Max': df[selected_feature].max(),
        'Unique Values': df[selected_feature].nunique()
    }
    
    stats_df = pd.DataFrame(stats.items(), columns=['Metric', 'Value'])
    st.dataframe(stats_df, use_container_width=True)

# ==================== PAGE: STATISTICS ====================
elif page == "Statistics":
    st.title("📊 Detailed Statistics")
    
    st.markdown("---")
    
    # Summary statistics
    st.subheader("Summary Statistics")
    
    summary_stats = df.describe().T
    st.dataframe(summary_stats, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.subheader("🔗 Correlation Matrix")
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(width=900, height=900)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Data info
    st.subheader("Data Info")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Rows", len(df))
        st.metric("Total Columns", len(df.columns))
    
    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())
        st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    
    st.markdown("---")
    
    # Column details
    st.subheader("Column Details")
    
    col_info = []
    for col in df.columns:
        col_info.append({
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Missing': df[col].isnull().sum(),
            'Unique': df[col].nunique()
        })
    
    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df, use_container_width=True)

# ==================== PAGE: ADDRESS LOOKUP ====================
elif page == "Address Lookup":
    st.title("🔍 Address Lookup & Price Comparison")
    st.markdown(
        "Search for multiple property addresses side by side, compare their normalized "
        "results, and run instant price predictions."
    )

    st.caption("Each panel keeps its own search result and prediction so you can compare addresses side by side.")
    st.markdown("---")

    slot_count = 2

    # Initialize session state for lookup
    if "lookup_predictions" not in st.session_state:
        st.session_state["lookup_predictions"] = []
    if "lookup_normalized" not in st.session_state:
        st.session_state["lookup_normalized"] = None
    
    # Initialize address slots (up to 2 side-by-side)
    num_slots = 2
    
    st.subheader("📍 Enter Property Addresses (Side-by-Side)")
    st.write("Fill in one or more addresses and search them individually")
    
    # Create columns for address forms
    form_cols = st.columns(num_slots)
    
    # Process each address slot
    for slot_idx in range(num_slots):
        with form_cols[slot_idx]:
            st.markdown(f"### Property {slot_idx + 1}")
            
            # Use unique keys for each form
            line1_key = f"lookup_line1_{slot_idx}"
            city_key = f"lookup_city_{slot_idx}"
            state_key = f"lookup_state_{slot_idx}"
            line2_key = f"lookup_line2_{slot_idx}"
            postal_key = f"lookup_postal_{slot_idx}"
            country_key = f"lookup_country_{slot_idx}"
            
            # Input fields
            line1 = st.text_input(
                f"Address Line 1 *",
                placeholder="123 Main Street",
                value=st.session_state.get(line1_key, ""),
                key=f"input_{line1_key}"
            )
            
            city = st.text_input(
                f"City *",
                placeholder="Miami",
                value=st.session_state.get(city_key, ""),
                key=f"input_{city_key}"
            )
            
            state = st.text_input(
                f"State *",
                placeholder="FL",
                value=st.session_state.get(state_key, ""),
                key=f"input_{state_key}"
            )
            
            line2 = st.text_input(
                f"Address Line 2",
                placeholder="Apt 456",
                value=st.session_state.get(line2_key, ""),
                key=f"input_{line2_key}"
            )
            
            postal = st.text_input(
                f"Postal Code *",
                placeholder="33101",
                value=st.session_state.get(postal_key, ""),
                key=f"input_{postal_key}"
            )
            
            country = st.text_input(
                f"Country",
                value=st.session_state.get(country_key, "US"),
                key=f"input_{country_key}"
            )
            
            # Search button for this slot
            if st.button(f"🔍 Search Property {slot_idx + 1}", use_container_width=True, key=f"search_btn_{slot_idx}"):
                # Save inputs
                st.session_state[line1_key] = line1
                st.session_state[city_key] = city
                st.session_state[state_key] = state
                st.session_state[line2_key] = line2
                st.session_state[postal_key] = postal
                st.session_state[country_key] = country
                
                # Validate
                if not line1.strip() or not city.strip() or not state.strip() or not postal.strip():
                    st.error("❌ Fill all required fields (*)")
                else:
                    # Build payload
                    lookup_payload = {
                        "address_line_1": line1.strip(),
                        "city": city.strip(),
                        "state": state.strip(),
                        "postal_code": postal.strip(),
                        "country": country.strip(),
                    }
                    if line2.strip():
                        lookup_payload["address_line_2"] = line2.strip()
                    
                    with st.spinner("🌍 Looking up address..."):
                        sc, body, url = call_api("POST", api_base_url, "/v1/properties/normalize", payload=lookup_payload)
                    
                    if sc == 200 and isinstance(body, dict):
                        # Store geocode result for this slot
                        st.session_state[f"geocode_{slot_idx}"] = body
                        # Store in predictions with auto-prediction
                        pred_payload = dict(lookup_payload)
                        with st.spinner("🧠 Predicting price..."):
                            pred_sc, pred_body, _ = call_api("POST", api_base_url, "/v1/predictions", payload=pred_payload)
                        
                        if pred_sc == 201 and isinstance(pred_body, dict):
                            prediction_entry = {
                                "address": f"{line1}, {city}, {state}",
                                "formatted_address": body.get('formatted_address', ''),
                                "latitude": body.get('latitude'),
                                "longitude": body.get('longitude'),
                                "predicted_price": pred_body.get('predicted_price'),
                                "completeness_score": pred_body.get('completeness_score'),
                                "features": pred_body.get('feature_snapshot', {}).get('features', {}),
                                "prediction_id": pred_body.get('prediction_id'),
                            }
                            st.session_state["lookup_predictions"].append(prediction_entry)
                            st.session_state[f"prediction_{slot_idx}"] = pred_body
                            st.success(f"✅ Property {slot_idx + 1} found!")
                        else:
                            st.session_state.pop(f"prediction_{slot_idx}", None)
                            st.error(f"❌ Prediction failed")
                    else:
                        st.session_state.pop(f"geocode_{slot_idx}", None)
                        st.error(f"❌ Address lookup failed")

            # Show geocoded result and prediction for this slot
            geocode = st.session_state.get(f"geocode_{slot_idx}")
            prediction = st.session_state.get(f"prediction_{slot_idx}")
            if geocode:
                st.markdown("---")
                st.markdown("**📍 Geocoded Location**")
                st.caption(geocode.get("formatted_address", ""))
                lat = geocode.get("latitude")
                lon = geocode.get("longitude")
                if lat and lon:
                    st.caption(f"🌐 Lat: `{lat:.5f}`, Lon: `{lon:.5f}`")
                    slot_map_df = pd.DataFrame([{"lat": lat, "lon": lon}])
                    st.map(slot_map_df, latitude="lat", longitude="lon", zoom=13, size=50)
            if prediction:
                price = prediction.get("predicted_price")
                completeness = prediction.get("feature_snapshot", {}).get("completeness_score")
                features = prediction.get("feature_snapshot", {}).get("features", {})
                if price is not None:
                    st.markdown("**🏠 Price Prediction**")
                    st.metric("Predicted Price", f"${price:,.0f}")
                if completeness is not None:
                    st.metric("Data Completeness", f"{completeness:.1%}")
                if features:
                    st.markdown("**Key Features:**")
                    feat_cols = st.columns(2)
                    for i, (k, v) in enumerate(list(features.items())[:4]):
                        with feat_cols[i % 2]:
                            st.metric(k, round(v, 2) if isinstance(v, float) else v)
    
    st.markdown("---")
    
    # Display results
    normalized = st.session_state.get("lookup_normalized")
    
    st.markdown("---")
    
    # ========== COMPARISON SECTION ==========
    predictions = st.session_state.get("lookup_predictions", [])
    
    if predictions:
        st.subheader(f"📊 Comparison ({len(predictions)} properties)")
        
        # Comparison table
        comparison_data = []
        for idx, pred in enumerate(predictions):
            comparison_data.append({
                "Property": pred.get("address", "Unknown"),
                "Predicted Price": f"${pred.get('predicted_price', 0):,.0f}",
                "Completeness": f"{pred.get('completeness_score', 0):.1%}",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Price comparison chart
        if len(predictions) > 1:
            st.markdown("###")
            st.subheader("💵 Price Comparison Chart")
            
            chart_data = pd.DataFrame([
                {
                    "Address": p.get("address", "Unknown").split(",")[0],  # Just street
                    "Price": p.get("predicted_price", 0)
                }
                for p in predictions
            ])
            
            fig = px.batestsr(
                chart_data,
                x="Address",
                y="Price",
                title="Predicted Prices Comparison",
                labels={"Price": "Price ($)", "Address": "Property"},
                color="Price",
                color_continuous_scale="Viridis"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            prices = [p.get("predicted_price", 0) for p in predictions]
            
            with col1:
                st.metric("Highest Price", f"${max(prices):,.0f}")
            with col2:
                st.metric("Lowest Price", f"${min(prices):,.0f}")
            with col3:
                st.metric("Average Price", f"${sum(prices)/len(prices):,.0f}")
            with col4:
                st.metric("Price Range", f"${max(prices) - min(prices):,.0f}")
        
        # Map with all properties
        if any(p.get('latitude') and p.get('longitude') for p in predictions):
            st.markdown("###")
            st.subheader("🗺️ All Properties Map")
            
            try:
                import folium
                from streamlit_folium import st_folium
                
                # Calculate center of all properties
                lats = [p['latitude'] for p in predictions if p.get('latitude')]
                lons = [p['longitude'] for p in predictions if p.get('longitude')]
                
                if lats and lons:
                    center_lat = sum(lats) / len(lats)
                    center_lon = sum(lons) / len(lons)
                    
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=12,
                        tiles="OpenStreetMap"
                    )
                    
                    # Color map for markers
                    colors = ["red", "blue", "green", "purple", "orange", "darkred", "darkblue", "darkgreen"]
                    
                    # Add markers for each property
                    for idx, pred in enumerate(predictions):
                        if pred.get('latitude') and pred.get('longitude'):
                            color = colors[idx % len(colors)]
                            price = pred.get('predicted_price', 0)
                            
                            folium.Marker(
                                location=[pred['latitude'], pred['longitude']],
                                popup=f"{pred.get('address', 'Unknown')}<br>${price:,.0f}",
                                tooltip=f"${price:,.0f}",
                                icon=folium.Icon(color=color, icon="home", prefix="fa")
                            ).add_to(m)
                    
                    st_folium(m, width=700, height=500)
            except Exception as e:
                st.warning(f"⚠️ Could not display map: {str(e)}")
        
        # Remove individual predictions
        st.markdown("###")
        st.subheader("🗑️ Manage Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Clear All", use_container_width=True):
                st.session_state["lookup_predictions"] = []
                st.rerun()
        
        with col2:
            selected_to_remove = st.selectbox(
                "Remove a property:",
                options=[f"{i+1}. {p['address'].split(',')[0]}" for i, p in enumerate(predictions)],
                key="remove_select"
            )
            if st.button("❌ Remove Selected", use_container_width=True):
                remove_idx = int(selected_to_remove.split(".")[0]) - 1
                predictions.pop(remove_idx)
                st.session_state["lookup_predictions"] = predictions
                st.rerun()
    else:
        if st.session_state.get("lookup_normalized"):
            st.info("💡 Predict an address price using the button above to start comparing properties!")


# ==================== PAGE: LIVE API TESTER ====================
elif page == "Live API Tester":
    st.title("🧪 Live API Tester")
    st.markdown(
        "Automated scenario pipeline runner — tests the full contract chain "
        "(UI → FastAPI → Orchestration → DB) without manual address entry."
    )
    st.caption(f"API target: **{api_base_url}**")

    if "last_prediction_id" not in st.session_state:
        st.session_state["last_prediction_id"] = ""

    def render_response(title: str, status_code, body: dict, target_url: str) -> None:
        st.subheader(title)
        st.write(f"URL: `{target_url}`")
        if status_code is None:
            st.error("Request failed before receiving a response.")
        elif 200 <= status_code < 300:
            st.success(f"Status: {status_code}")
        else:
            st.warning(f"Status: {status_code}")
        st.json(body)

    # ------------------------------------------------------------------ #
    # SECTION 1 — Quick Actions                                           #
    # ------------------------------------------------------------------ #
    st.subheader("Quick Actions")
    qa_col1, qa_col2 = st.columns(2)
    with qa_col1:
        if st.button("Check Health", use_container_width=True):
            sc, body, url = call_api("GET", api_base_url, "/v1/health")
            render_response("Health Response", sc, body, url)
    with qa_col2:
        if st.button("List Recent Predictions", use_container_width=True):
            sc, body, url = call_api("GET", api_base_url, "/v1/predictions", params={"limit": 10})
            render_response("Predictions List Response", sc, body, url)

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # SECTION 2 — Live Validation Runner                                #
    # ------------------------------------------------------------------ #
    st.subheader("Live Validation Runner")
    st.caption(
        "One-click real-address flow: health → normalize → baseline → prediction → detail. "
        "This path is gated by API live readiness by default."
    )

    with st.form("live_validation_form"):
        lv_col1, lv_col2 = st.columns(2)
        with lv_col1:
            lv_line1 = st.text_input("Address Line 1", value="1600 Pennsylvania Ave NW", key="lv_line1")
            lv_city = st.text_input("City", value="Washington", key="lv_city")
            lv_state = st.text_input("State", value="DC", key="lv_state")
        with lv_col2:
            lv_line2 = st.text_input("Address Line 2 (optional)", value="", key="lv_line2")
            lv_postal = st.text_input("Postal Code", value="20500", key="lv_postal")
            lv_country = st.text_input("Country", value="US", key="lv_country")

        lv_requested_by = st.text_input("Requested By", value="live-ui@local", key="lv_requested_by")
        allow_not_ready = st.checkbox(
            "Allow run even when API is not live-ready",
            value=False,
            key="lv_allow_not_ready",
            help="Keep off for strict live-mode validation.",
        )
        lv_submit = st.form_submit_button("Run Live Validation", use_container_width=True)

    if lv_submit:
        live_payload: dict = {
            "address_line_1": lv_line1,
            "city": lv_city,
            "state": lv_state,
            "postal_code": lv_postal,
            "country": lv_country,
        }
        if lv_line2.strip():
            live_payload["address_line_2"] = lv_line2.strip()

        with st.spinner("Running live validation workflow…"):
            health_sc, health_body, health_url = call_api("GET", api_base_url, "/v1/health")

            if health_sc != 200 or not isinstance(health_body, dict):
                render_response("Live Validation • Health", health_sc, health_body, health_url)
            else:
                render_response("Live Validation • Health", health_sc, health_body, health_url)
                live_mode_ready = bool(health_body.get("live_mode_ready", False))
                live_mode_issues = health_body.get("live_mode_issues", [])
                if (not live_mode_ready) and (not allow_not_ready):
                    st.error(
                        "API is not live-ready. Resolve runtime issues first or enable override to continue."
                    )
                    if isinstance(live_mode_issues, list) and live_mode_issues:
                        st.warning("\n".join(f"- {issue}" for issue in live_mode_issues))
                else:
                    norm_sc, norm_body, norm_url = call_api(
                        "POST",
                        api_base_url,
                        "/v1/properties/normalize",
                        payload=live_payload,
                    )
                    render_response("Live Validation • Normalize", norm_sc, norm_body, norm_url)

                    baseline_sc, baseline_body, baseline_url = call_api(
                        "POST",
                        api_base_url,
                        "/v1/validation/address-baseline",
                        payload=live_payload,
                    )
                    render_response("Live Validation • Baseline", baseline_sc, baseline_body, baseline_url)

                    prediction_payload = dict(live_payload)
                    if lv_requested_by.strip():
                        prediction_payload["requested_by"] = lv_requested_by.strip()

                    pred_sc, pred_body, pred_url = call_api(
                        "POST",
                        api_base_url,
                        "/v1/predictions",
                        payload=prediction_payload,
                    )
                    render_response("Live Validation • Create Prediction", pred_sc, pred_body, pred_url)

                    prediction_id = pred_body.get("prediction_id") if isinstance(pred_body, dict) else None
                    if prediction_id:
                        st.session_state["last_prediction_id"] = prediction_id
                        detail_sc, detail_body, detail_url = call_api(
                            "GET",
                            api_base_url,
                            f"/v1/predictions/{prediction_id}",
                        )
                        render_response("Live Validation • Prediction Detail", detail_sc, detail_body, detail_url)

                        if detail_sc == 200 and isinstance(detail_body, dict):
                            provider_responses = detail_body.get("provider_responses", [])
                            if provider_responses:
                                latest_provider = provider_responses[-1]
                                st.info(
                                    "Latest provider response: "
                                    f"{latest_provider.get('provider_name', 'unknown')} "
                                    f"(feature_source={latest_provider.get('feature_source', 'unknown')})"
                                )

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # SECTION 3 — Automated Scenario Pipeline Runner                     #
    # ------------------------------------------------------------------ #
    st.subheader("Automated Scenario Pipeline Runner")
    st.write(
        "Fetch the registered scenario catalog, choose which scenarios to execute, "
        "and run the full audit pipeline for each one in a single request."
    )

    load_col, _spacer = st.columns([1, 3])
    with load_col:
        if st.button("Load Scenarios", use_container_width=True):
            sc, body, _url = call_api("GET", api_base_url, "/v1/validation/scenarios")
            if sc == 200 and isinstance(body, dict):
                st.session_state["scenarios"] = body.get("scenarios", [])
                st.session_state["batch_result"] = None  # clear stale result on reload
                st.success(f"Loaded {len(st.session_state['scenarios'])} scenario(s).")
            else:
                st.error(f"Failed to load scenarios (status {sc}).")
                st.json(body)

    scenarios: list[dict] = st.session_state.get("scenarios", [])

    if not scenarios:
        st.info("Click **Load Scenarios** to fetch the registered catalog from the API.")
    else:
        # Build label → scenario_id map (only registry scenarios; skip live-derived)
        registry_scenarios = [
            s for s in scenarios if not s.get("scenario_id", "").startswith("live-")
        ]
        live_scenarios = [
            s for s in scenarios if s.get("scenario_id", "").startswith("live-")
        ]

        label_to_id: dict[str, str] = {s["label"]: s["scenario_id"] for s in registry_scenarios}

        selected_labels: list[str] = st.multiselect(
            "Select registered scenarios to run",
            options=list(label_to_id.keys()),
            default=list(label_to_id.keys()),
            help="All registered scenarios are selected by default. De-select to skip individual ones.",
        )

        if live_scenarios:
            st.caption(
                f"{len(live_scenarios)} live-derived scenario(s) also available (from recent traffic) "
                "— not shown in multiselect; they run without pre-configured expectations."
            )

        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            run_selected = st.button(
                "Run Selected Scenarios",
                use_container_width=True,
                disabled=not selected_labels,
            )
        with btn_col2:
            run_all = st.button("Run All", use_container_width=True)

        if run_selected or run_all:
            if run_all:
                batch_payload: dict = {}
            else:
                batch_payload = {"scenario_ids": [label_to_id[l] for l in selected_labels]}

            with st.spinner("Running scenario pipeline batch…"):
                sc, body, _url = call_api(
                    "POST",
                    api_base_url,
                    "/v1/validation/run-scenario-batch",
                    payload=batch_payload,
                )
            st.session_state["batch_result"] = (sc, body)

    # Display batch result (persists across reruns)
    batch_result = st.session_state.get("batch_result")
    if batch_result is not None:
        sc, body = batch_result
        if sc == 200 and isinstance(body, dict):
            total = body.get("total", 0)
            passed = body.get("passed", 0)
            failed = body.get("failed", 0)
            errors = body.get("errors", 0)

            st.markdown("#### Batch Run Summary")
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric("Total", total)
            sm2.metric("Passed", passed, delta=None)
            sm3.metric("Failed", failed)
            sm4.metric("Errors", errors)

            results: list[dict] = body.get("results", [])
            if results:
                rows = []
                for r in results:
                    status_icon = {"pass": "✅ PASS", "fail": "❌ FAIL", "error": "⚠️ ERROR"}.get(
                        r.get("pipeline_status", ""), r.get("pipeline_status", "")
                    )
                    rows.append({
                        "Scenario": r.get("label", r.get("scenario_id")),
                        "Category": r.get("category", ""),
                        "Status": status_icon,
                        "Completeness": (
                            f"{r['completeness_score']:.1%}"
                            if r.get("completeness_score") is not None
                            else "—"
                        ),
                        "Price (USD)": (
                            f"${r['predicted_price']:,.0f}"
                            if r.get("predicted_price") is not None
                            else "—"
                        ),
                        "Issues": "; ".join(r.get("issues", [])) or "none",
                        "Error": r.get("error_message") or "",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # Per-scenario key feature surfacing
                feature_results = [r for r in results if r.get("key_feature_values")]
                if feature_results:
                    with st.expander("Key Feature Values per Scenario", expanded=False):
                        for r in feature_results:
                            kfv: dict = r.get("key_feature_values", {})
                            st.markdown(f"**{r.get('label', r['scenario_id'])}**")
                            kv_cols = st.columns(min(len(kfv), 6) or 1)
                            for idx, (feat, val) in enumerate(kfv.items()):
                                kv_cols[idx % 6].metric(feat, val)

                with st.expander("Raw Batch Response", expanded=False):
                    st.json(body)
        else:
            st.error(f"Batch run failed (status {sc}).")
            st.json(body)

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # SECTION 4 — Ad-Hoc Address Testing                                 #
    # ------------------------------------------------------------------ #
    with st.expander("Ad-Hoc Address & Prediction Testing", expanded=False):
        st.caption(
            "Enter any address to test the pipeline individually. "
            "Use the registered scenarios above for systematic, repeatable testing."
        )

        with st.form("adhoc_address_form"):
            addr_col1, addr_col2 = st.columns(2)
            with addr_col1:
                adhoc_line1 = st.text_input("Address Line 1", value="123 Main St")
                adhoc_city = st.text_input("City", value="Miami")
                adhoc_state = st.text_input("State", value="FL")
            with addr_col2:
                adhoc_line2 = st.text_input("Address Line 2 (optional)", value="")
                adhoc_postal = st.text_input("Postal Code", value="33101")
                adhoc_country = st.text_input("Country", value="US")

            adhoc_requested_by = st.text_input("Requested By (optional)", value="tester@example.com")
            save_btn = st.form_submit_button("Save Payload")

        adhoc_payload: dict = {
            "address_line_1": adhoc_line1,
            "city": adhoc_city,
            "state": adhoc_state,
            "postal_code": adhoc_postal,
            "country": adhoc_country,
        }
        if adhoc_line2.strip():
            adhoc_payload["address_line_2"] = adhoc_line2.strip()
        if save_btn:
            st.info("Ad-hoc payload saved.")
            st.json(adhoc_payload)

        st.markdown("##### Baseline Expectations")
        use_exp = st.checkbox("Apply expectation checks", value=True, key="adhoc_use_exp")
        min_comp = st.slider("Min Completeness", 0.0, 1.0, 0.85, 0.01, disabled=not use_exp)
        req_feats_csv = st.text_input(
            "Required Features (comma-separated)",
            value="BedroomAbvGr,TotRmsAbvGrd,GrLivArea,LotArea",
            disabled=not use_exp,
        )
        use_bounds = st.checkbox("Enforce default feature bounds", value=True, disabled=not use_exp)

        adhoc_expectations = None
        if use_exp:
            req_feats = [f.strip() for f in req_feats_csv.split(",") if f.strip()]
            adhoc_expectations = {
                "min_completeness_score": min_comp,
                "required_features": req_feats,
                "feature_bounds": (
                    {
                        "BedroomAbvGr": {"minimum": 1, "maximum": 8},
                        "TotRmsAbvGrd": {"minimum": 2, "maximum": 14},
                        "GrLivArea": {"minimum": 500, "maximum": 6000},
                        "LotArea": {"minimum": 1000, "maximum": 50000},
                    }
                    if use_bounds
                    else {}
                ),
            }

        adhoc_flow_col1, adhoc_flow_col2 = st.columns(2)
        with adhoc_flow_col1:
            if st.button("Normalize Address", use_container_width=True, key="adhoc_normalize"):
                sc, body, url = call_api("POST", api_base_url, "/v1/properties/normalize", payload=adhoc_payload)
                render_response("Normalize Response", sc, body, url)
        with adhoc_flow_col2:
            if st.button("Create Prediction", use_container_width=True, key="adhoc_predict"):
                pred_payload = dict(adhoc_payload)
                if adhoc_requested_by.strip():
                    pred_payload["requested_by"] = adhoc_requested_by.strip()
                sc, body, url = call_api("POST", api_base_url, "/v1/predictions", payload=pred_payload)
                if isinstance(body, dict) and body.get("prediction_id"):
                    st.session_state["last_prediction_id"] = body["prediction_id"]
                render_response("Create Prediction Response", sc, body, url)

        if st.button("Generate Address Baseline", use_container_width=True, key="adhoc_baseline"):
            bl_payload = dict(adhoc_payload)
            if adhoc_expectations:
                bl_payload["expectations"] = adhoc_expectations
            sc, body, url = call_api("POST", api_base_url, "/v1/validation/address-baseline", payload=bl_payload)
            render_response("Address Baseline Response", sc, body, url)
            if sc == 200 and isinstance(body, dict):
                kfv = body.get("features", {}).get("key_feature_values", {})
                if isinstance(kfv, dict) and kfv:
                    st.subheader("Key Property Features")
                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("Bedrooms", kfv.get("BedroomAbvGr", "n/a"))
                    fc2.metric("Total Rooms", kfv.get("TotRmsAbvGrd", "n/a"))
                    fc3.metric("Living SqFt", kfv.get("GrLivArea", "n/a"))
                    fc4, fc5, fc6 = st.columns(3)
                    fc4.metric("Lot Area", kfv.get("LotArea", "n/a"))
                    fc5.metric("Full Baths", kfv.get("FullBath", "n/a"))
                    fc6.metric("Half Baths", kfv.get("HalfBath", "n/a"))
                assessment = body.get("assessment", {})
                checks = assessment.get("checks", []) if isinstance(assessment, dict) else []
                if checks:
                    st.write(f"Overall: **{assessment.get('overall_status', 'unknown')}**")
                    st.dataframe(pd.DataFrame(checks), use_container_width=True)

        if st.button("Run Full Live Audit", use_container_width=True, key="adhoc_full_audit"):
            fa_payload = dict(adhoc_payload)
            if adhoc_requested_by.strip():
                fa_payload["requested_by"] = adhoc_requested_by.strip()
            if adhoc_expectations:
                fa_payload["expectations"] = adhoc_expectations
            sc, body, url = call_api("POST", api_base_url, "/v1/validation/full-audit", payload=fa_payload)
            render_response("Full Live Audit Response", sc, body, url)
            if sc == 200 and isinstance(body, dict):
                issues = body.get("issues", [])
                if issues:
                    st.warning("\n".join(f"- {i}" for i in issues))
                else:
                    st.success("No audit issues detected.")

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # SECTION 5 — Policy Simulation                                      #
    # ------------------------------------------------------------------ #
    st.subheader("Policy Simulation")
    st.caption("Training-only utility for policy experiments; not part of live production prediction flow.")

    enable_training_sim = st.checkbox(
        "Enable training simulations",
        value=False,
        help="Keep this off during live-address validation; enable only when exploring policy behavior.",
    )

    if not enable_training_sim:
        st.info(
            "Simulation tools are disabled by default. "
            "Use Normalize/Create Prediction/Full Audit above for live-address validation."
        )
    else:
        with st.form("policy_sim_form"):
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_line1 = st.text_input("Address Line 1", value="413 Duff Ave", key="sim_line1")
                sim_city = st.text_input("City", value="Ames", key="sim_city")
                sim_state = st.text_input("State", value="IA", key="sim_state")
            with sim_col2:
                sim_postal = st.text_input("Postal Code", value="50010", key="sim_postal")
                sim_country = st.text_input("Country", value="US", key="sim_country")
                sim_policies_csv = st.text_input(
                    "Policy Names (comma-separated)",
                    value="balanced-v1,quality-first-v1",
                    key="sim_policies",
                )
            sim_submitted = st.form_submit_button("Simulate Policies")

        if sim_submitted:
            sim_payload: dict = {
                "address_line_1": sim_line1,
                "city": sim_city,
                "state": sim_state,
                "postal_code": sim_postal,
                "country": sim_country,
            }
            policy_names = [p.strip() for p in sim_policies_csv.split(",") if p.strip()]
            if policy_names:
                sim_payload["policy_names"] = policy_names
            sc, body, url = call_api("POST", api_base_url, "/v1/policies/feature/simulate", payload=sim_payload)
            render_response("Policy Simulation Response", sc, body, url)
            if sc == 200 and isinstance(body, dict):
                sims = body.get("simulations", [])
                if sims:
                    st.dataframe(
                        pd.DataFrame([
                            {
                                "Policy": s["policy_name"],
                                "Version": s["policy_version"],
                                "Price ($)": f"${s['predicted_price']:,.0f}",
                                "Completeness": f"{s['completeness_score']:.1%}",
                            }
                            for s in sims
                        ]),
                        use_container_width=True,
                    )

    st.markdown("---")

    # ------------------------------------------------------------------ #
    # SECTION 6 — Prediction Drill-Down                                  #
    # ------------------------------------------------------------------ #
    st.subheader("Prediction Drill-Down")
    prediction_id_input = st.text_input(
        "Prediction ID",
        value=st.session_state.get("last_prediction_id", ""),
        help="Auto-populated after Create Prediction succeeds.",
    )

    drill_col1, drill_col2, drill_col3 = st.columns(3)
    with drill_col1:
        if st.button("Get Detail", use_container_width=True, disabled=not prediction_id_input.strip()):
            sc, body, url = call_api("GET", api_base_url, f"/v1/predictions/{prediction_id_input.strip()}")
            render_response("Prediction Detail", sc, body, url)
    with drill_col2:
        if st.button("Get Trace", use_container_width=True, disabled=not prediction_id_input.strip()):
            sc, body, url = call_api("GET", api_base_url, f"/v1/predictions/{prediction_id_input.strip()}/trace")
            render_response("Prediction Trace", sc, body, url)
    with drill_col3:
        if st.button("Get Events", use_container_width=True, disabled=not prediction_id_input.strip()):
            sc, body, url = call_api(
                "GET",
                api_base_url,
                f"/v1/predictions/{prediction_id_input.strip()}/events",
                params={"limit": 50, "offset": 0, "sort": "desc"},
            )
            render_response("Prediction Events", sc, body, url)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Use the sidebar to navigate between different sections of the dashboard.")
