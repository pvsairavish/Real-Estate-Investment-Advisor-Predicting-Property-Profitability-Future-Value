"""
PropVision AI — Real Estate Investment Advisor
High-Performance Interactive UI/UX with Dual-Engine Machine Learning
(Investment Classifier & 5-Year Scenario Regressor)
"""

import io
import json
import sys
import warnings
from collections import UserList
from pathlib import Path

# Suppress serialization warnings from legacy XGBoost / scikit-learn builds
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Compatibility shim for scikit-learn ColumnTransformer
# ---------------------------------------------------------
try:
    import sklearn.compose._column_transformer as _ct

    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(UserList):
            """Compatibility fallback for deserializing models across scikit-learn versions."""
            def __init__(self, columns=None, *, future_dtype=None, warning_was_emitted=False, warning_enabled=True):
                super().__init__(columns if columns is not None else [])
                self.future_dtype = future_dtype
                self.warning_was_emitted = warning_was_emitted
                self.warning_enabled = warning_enabled

        _ct._RemainderColsList = _RemainderColsList
        sys.modules["sklearn.compose._column_transformer"]._RemainderColsList = _RemainderColsList
except Exception:
    pass

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="PropVision AI • Real Estate Investment Advisor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
CLASSIFICATION_MODEL_PATH = MODEL_DIR / "best_classification_pipeline.joblib"
REGRESSION_MODEL_PATH = MODEL_DIR / "best_regression_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
LOCATIONS_PATH = BASE_DIR / "locations.json"
DATASET_PATH = BASE_DIR / "india_housing_prices.csv"

# ---------------------------------------------------------
# Custom Modern UI / UX Stylesheet
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero Header Card */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #0F172A 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(0,0,0,0) 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 60%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        max-width: 820px;
        line-height: 1.55;
    }

    /* Modern Metric Cards */
    .stat-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.25s ease;
    }
    .stat-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .stat-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-bottom: 0.35rem;
    }
    .stat-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .stat-delta {
        font-size: 0.88rem;
        font-weight: 600;
        margin-top: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }
    .stat-delta.positive { color: #10B981; }
    .stat-delta.neutral { color: #F59E0B; }
    .stat-delta.negative { color: #EF4444; }

    /* Verdict Banners */
    .verdict-good {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.16) 0%, rgba(5, 150, 105, 0.06) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-radius: 16px;
        padding: 1.5rem;
        color: #10B981;
    }
    .verdict-caution {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.16) 0%, rgba(217, 119, 6, 0.06) 100%);
        border: 1px solid rgba(245, 158, 11, 0.45);
        border-radius: 16px;
        padding: 1.5rem;
        color: #F59E0B;
    }

    /* Section Cards */
    .section-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        font-size: 0.75rem;
        font-weight: 700;
        border-radius: 9999px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-primary {
        background: rgba(99, 102, 241, 0.18);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.35);
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0 1.25rem;
        color: #94A3B8;
        border: none;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.22) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(99, 102, 241, 0.45) !important;
    }

    /* Streamlit Buttons */
    div.stButton > button:first-child {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px -4px rgba(99, 102, 241, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Resource Loading & Caching
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models_and_metadata():
    classification_model = joblib.load(CLASSIFICATION_MODEL_PATH)
    regression_model = joblib.load(REGRESSION_MODEL_PATH)

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

    locations = {}
    if LOCATIONS_PATH.exists():
        with open(LOCATIONS_PATH, "r", encoding="utf-8") as file:
            locations = json.load(file)

    return classification_model, regression_model, metadata, locations


@st.cache_data(show_spinner=False)
def load_market_sample():
    if not DATASET_PATH.exists():
        return pd.DataFrame()
    try:
        cols = [
            "State", "City", "Property_Type", "BHK", "Size_in_SqFt",
            "Price_in_Lakhs", "Price_per_SqFt", "Furnished_Status",
            "Public_Transport_Accessibility", "Availability_Status"
        ]
        df = pd.read_csv(DATASET_PATH, nrows=25000, usecols=cols)
        return df
    except Exception:
        return pd.DataFrame()


# Validate model artifacts
missing_artifacts = [str(p.name) for p in [CLASSIFICATION_MODEL_PATH, REGRESSION_MODEL_PATH] if not p.exists()]
if missing_artifacts:
    st.error(f"⚠️ Missing required model artifacts: {', '.join(missing_artifacts)}")
    st.stop()

try:
    classification_model, regression_model, metadata, locations_data = load_models_and_metadata()
except Exception as exc:
    st.error(f"❌ Failed to load predictive models: {exc}")
    st.stop()

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def format_inr(lakhs: float) -> str:
    """Format value in Indian Rupees (Lakhs / Crores)."""
    if lakhs >= 100:
        crores = lakhs / 100.0
        return f"₹ {crores:.2f} Cr"
    return f"₹ {lakhs:,.2f} L"


def calculate_cagr(start_val: float, end_val: float, periods: int = 5) -> float:
    """Calculate Compound Annual Growth Rate percentage."""
    if start_val <= 0 or end_val <= 0 or periods <= 0:
        return 0.0
    return ((end_val / start_val) ** (1.0 / periods) - 1.0) * 100.0


def count_amenities(value) -> int:
    if value is None or str(value).strip() == "":
        return 0
    return len([item for item in str(value).split(",") if item.strip()])


def build_model_input(
    state: str,
    city: str,
    property_type: str,
    bhk: int,
    size_sqft: float,
    price_lakhs: float,
    year_built: int,
    age_of_property: int,
    furnished_status: str,
    floor_no: int,
    total_floors: int,
    nearby_schools: int,
    nearby_hospitals: int,
    public_transport: str,
    parking_space: str,
    security: str,
    amenities: str,
    facing: str,
    owner_type: str,
    availability_status: str,
) -> pd.DataFrame:
    """Construct precisely formatted feature row matching training schema."""
    price_per_sqft = (price_lakhs / size_sqft) if size_sqft > 0 else np.nan
    has_parking = 1 if parking_space == "Yes" else 0
    has_security = 1 if security == "Yes" else 0
    is_ready_to_move = 1 if availability_status == "Ready_to_Move" else 0

    floor_ratio = (np.clip(floor_no / total_floors, 0, 1)) if total_floors > 0 else np.nan
    amenities_count = count_amenities(amenities)
    school_hospital_access = nearby_schools + nearby_hospitals

    if age_of_property <= 5:
        property_age_band = "New"
    elif age_of_property <= 15:
        property_age_band = "Moderate"
    elif age_of_property <= 25:
        property_age_band = "Old"
    else:
        property_age_band = "Very_Old"

    return pd.DataFrame(
        [
            {
                "State": state,
                "City": city,
                "Property_Type": property_type,
                "BHK": bhk,
                "Size_in_SqFt": size_sqft,
                "Price_in_Lakhs": price_lakhs,
                "Price_per_SqFt": price_per_sqft,
                "Year_Built": year_built,
                "Furnished_Status": furnished_status,
                "Floor_No": floor_no,
                "Total_Floors": total_floors,
                "Age_of_Property": age_of_property,
                "Nearby_Schools": nearby_schools,
                "Nearby_Hospitals": nearby_hospitals,
                "Public_Transport_Accessibility": public_transport,
                "Parking_Space": parking_space,
                "Security": security,
                "Amenities": amenities,
                "Facing": facing,
                "Owner_Type": owner_type,
                "Availability_Status": availability_status,
                "Amenities_Count": amenities_count,
                "Has_Parking": has_parking,
                "Has_Security": has_security,
                "Is_Ready_To_Move": is_ready_to_move,
                "Floor_Ratio": floor_ratio,
                "School_Hospital_Access": school_hospital_access,
                "Property_Age_Band": property_age_band,
            }
        ]
    )


# ---------------------------------------------------------
# Curated Preset Scenarios for 1-Click Interactive Demo
# ---------------------------------------------------------
PRESETS = {
    "urban_metro": {
        "name": "🌆 Urban Metro High-Growth 3BHK",
        "desc": "Prime Tech Corridor • Hyderabad • Ready to Move",
        "data": {
            "state": "Telangana",
            "city": "Hyderabad",
            "property_type": "Apartment",
            "bhk": 3,
            "size_sqft": 1650.0,
            "price_lakhs": 125.0,
            "year_built": 2022,
            "age_of_property": 4,
            "furnished_status": "Semi-furnished",
            "floor_no": 7,
            "total_floors": 15,
            "nearby_schools": 4,
            "nearby_hospitals": 3,
            "public_transport": "High",
            "parking_space": "Yes",
            "security": "Yes",
            "amenities": ["Gym", "Pool", "Clubhouse", "Playground"],
            "facing": "East",
            "owner_type": "Owner",
            "availability_status": "Ready_to_Move",
        },
    },
    "luxury_villa": {
        "name": "🏡 Premium Gated Luxury Villa",
        "desc": "Exclusive Suburb • Bangalore • High Cap Gains",
        "data": {
            "state": "Karnataka",
            "city": "Bangalore",
            "property_type": "Villa",
            "bhk": 4,
            "size_sqft": 3200.0,
            "price_lakhs": 295.0,
            "year_built": 2023,
            "age_of_property": 3,
            "furnished_status": "Furnished",
            "floor_no": 2,
            "total_floors": 2,
            "nearby_schools": 5,
            "nearby_hospitals": 4,
            "public_transport": "Medium",
            "parking_space": "Yes",
            "security": "Yes",
            "amenities": ["Clubhouse", "Garden", "Gym", "Playground", "Pool"],
            "facing": "North",
            "owner_type": "Builder",
            "availability_status": "Ready_to_Move",
        },
    },
    "sky_penthouse": {
        "name": "🏢 Sea-View High-Rise Penthouse",
        "desc": "Luxury Living • Mumbai • Elite Neighborhood",
        "data": {
            "state": "Maharashtra",
            "city": "Mumbai",
            "property_type": "Apartment",
            "bhk": 4,
            "size_sqft": 2600.0,
            "price_lakhs": 480.0,
            "year_built": 2021,
            "age_of_property": 5,
            "furnished_status": "Furnished",
            "floor_no": 28,
            "total_floors": 35,
            "nearby_schools": 6,
            "nearby_hospitals": 5,
            "public_transport": "High",
            "parking_space": "Yes",
            "security": "Yes",
            "amenities": ["Clubhouse", "Gym", "Pool"],
            "facing": "West",
            "owner_type": "Owner",
            "availability_status": "Ready_to_Move",
        },
    },
    "starter_home": {
        "name": "🌱 Budget Suburban Starter Home",
        "desc": "High Growth Potential • Ahmedabad • High Rental ROI",
        "data": {
            "state": "Gujarat",
            "city": "Ahmedabad",
            "property_type": "Apartment",
            "bhk": 2,
            "size_sqft": 1050.0,
            "price_lakhs": 52.0,
            "year_built": 2024,
            "age_of_property": 2,
            "furnished_status": "Semi-furnished",
            "floor_no": 3,
            "total_floors": 9,
            "nearby_schools": 3,
            "nearby_hospitals": 2,
            "public_transport": "Medium",
            "parking_space": "Yes",
            "security": "Yes",
            "amenities": ["Garden", "Playground"],
            "facing": "East",
            "owner_type": "Builder",
            "availability_status": "Under_Construction",
        },
    },
}

# ---------------------------------------------------------
# Synchronized Session State Management
# ---------------------------------------------------------
def apply_preset(preset_key: str):
    """Update widget-bound session state variables to reflect selected preset."""
    d = PRESETS[preset_key]["data"]
    st.session_state["w_state"] = d["state"]
    st.session_state["w_city"] = d["city"]
    st.session_state["w_prop_type"] = d["property_type"]
    st.session_state["w_bhk"] = d["bhk"]
    st.session_state["w_size"] = d["size_sqft"]
    st.session_state["w_price"] = d["price_lakhs"]
    st.session_state["w_year"] = d["year_built"]
    st.session_state["w_age"] = d["age_of_property"]
    st.session_state["w_furnished"] = d["furnished_status"]
    st.session_state["w_floor"] = d["floor_no"]
    st.session_state["w_total_floors"] = d["total_floors"]
    st.session_state["w_schools"] = d["nearby_schools"]
    st.session_state["w_hospitals"] = d["nearby_hospitals"]
    st.session_state["w_transport"] = d["public_transport"]
    st.session_state["w_parking"] = d["parking_space"]
    st.session_state["w_security"] = d["security"]
    st.session_state["w_amenities"] = d["amenities"]
    st.session_state["w_facing"] = d["facing"]
    st.session_state["w_owner"] = d["owner_type"]
    st.session_state["w_avail"] = d["availability_status"]

# Initialize session defaults if not already present
if "w_state" not in st.session_state:
    apply_preset("urban_metro")

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; padding: 0.5rem 0 1.5rem 0;">
            <div style="font-size: 2.8rem; line-height: 1;">🏢</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC; margin-top: 0.5rem;">PropVision AI</div>
            <div style="font-size: 0.75rem; color: #818CF8; letter-spacing: 0.1em; font-weight: 700;">INVESTMENT INTELLIGENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚡ Quick Presets")
    st.caption("Click to load pre-configured real estate scenarios:")

    for p_key, p_val in PRESETS.items():
        if st.button(f"{p_val['name']}", key=f"btn_{p_key}", use_container_width=True):
            apply_preset(p_key)
            st.rerun()

    st.markdown("---")
    st.markdown("### 🤖 ML Engine Specs")
    st.markdown(
        f"""
        - **Classifier:** `{metadata.get('classification_model', 'Tuned XGBoost Classifier')}`
        - **5Y Regressor:** `{metadata.get('regression_model', 'Tuned XGBoost Regressor')}`
        - **Training Set:** `{metadata.get('dataset_rows', 250000):,}` properties
        - **Dimensions:** 28 engineered features
        """
    )

    st.info(
        "💡 **Scenario Disclaimer:** The 5-year price is an algorithmic scenario projection modeled on historical trends. Always perform physical due diligence."
    )

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <span class="badge badge-primary">PropVision Intelligence Engine</span>
        <h1 class="hero-title">Real Estate Investment Advisor</h1>
        <p class="hero-subtitle">
            Harness machine learning models trained on 250,000+ Indian property records to evaluate
            deal viability, predict 5-year capital appreciation, and stress-test mortgage ROI.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab_advisor, tab_market, tab_roi, tab_compare, tab_dossier = st.tabs(
    [
        "🔮 AI Investment Advisor",
        "📊 Market Trends & Analytics",
        "💰 Mortgage & ROI Planner",
        "⚖️ What-If Comparison",
        "📋 Investment Dossier",
    ]
)

# =========================================================
# TAB 1: AI INVESTMENT ADVISOR
# =========================================================
with tab_advisor:
    st.markdown("### 🏢 Property Input & Specification")
    st.caption("Customize property parameters or select a preset from the sidebar.")

    with st.container():
        col_loc, col_spec, col_amen = st.columns([1, 1, 1], gap="medium")

        # --- Column 1: Location & Category ---
        with col_loc:
            st.markdown("##### 📍 Location & Category")
            available_states = sorted(list(locations_data.keys())) if locations_data else [st.session_state["w_state"]]
            
            def_st_idx = available_states.index(st.session_state["w_state"]) if st.session_state["w_state"] in available_states else 0
            selected_state = st.selectbox("State", available_states, index=def_st_idx, key="w_state")

            # Cascading City selection
            available_cities = locations_data.get(selected_state, [st.session_state.get("w_city", "Hyderabad")])
            curr_city = st.session_state.get("w_city", available_cities[0])
            if curr_city not in available_cities:
                curr_city = available_cities[0]
            
            def_city_idx = available_cities.index(curr_city)
            selected_city = st.selectbox("City", available_cities, index=def_city_idx, key="w_city")

            prop_types = ["Apartment", "Independent House", "Villa"]
            property_type = st.selectbox(
                "Property Type",
                prop_types,
                index=prop_types.index(st.session_state.get("w_prop_type", "Apartment")),
                key="w_prop_type",
            )

            facings = ["East", "North", "South", "West"]
            facing = st.selectbox(
                "Facing Direction",
                facings,
                index=facings.index(st.session_state.get("w_facing", "East")),
                key="w_facing",
            )

            owners = ["Owner", "Broker", "Builder"]
            owner_type = st.selectbox(
                "Seller / Owner Type",
                owners,
                index=owners.index(st.session_state.get("w_owner", "Owner")),
                key="w_owner",
            )

            avails = ["Ready_to_Move", "Under_Construction"]
            availability_status = st.selectbox(
                "Availability Status",
                avails,
                index=avails.index(st.session_state.get("w_avail", "Ready_to_Move")),
                format_func=lambda x: "Ready to Move" if x == "Ready_to_Move" else "Under Construction",
                key="w_avail",
            )

        # --- Column 2: Dimensions & Pricing ---
        with col_spec:
            st.markdown("##### 📐 Pricing & Dimensions")
            
            bhk = st.slider("Bedrooms (BHK)", min_value=1, max_value=8, value=int(st.session_state.get("w_bhk", 3)), step=1, key="w_bhk")
            
            size_sqft = st.number_input(
                "Carpet / Super Area (SqFt)",
                min_value=300.0,
                max_value=15000.0,
                value=float(st.session_state.get("w_size", 1650.0)),
                step=50.0,
                key="w_size",
                help="Approx. area in square feet",
            )

            price_lakhs = st.number_input(
                "Current Purchase Price (₹ Lakhs)",
                min_value=5.0,
                max_value=5000.0,
                value=float(st.session_state.get("w_price", 125.0)),
                step=1.0,
                key="w_price",
                help="100 Lakhs = 1 Crore",
            )

            calc_rate = (price_lakhs * 100000.0) / size_sqft if size_sqft > 0 else 0
            st.markdown(
                f"""
                <div style="background: rgba(99, 102, 241, 0.1); border: 1px dashed rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 0.6rem 0.9rem; margin-bottom: 0.8rem; font-size: 0.85rem;">
                    💡 <strong>Valuation Rate:</strong> ₹ {calc_rate:,.0f} / SqFt &nbsp;|&nbsp; <strong>Total:</strong> {format_inr(price_lakhs)}
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_age1, col_age2 = st.columns(2)
            with col_age1:
                year_built = st.number_input(
                    "Year Built",
                    min_value=1950,
                    max_value=2026,
                    value=int(st.session_state.get("w_year", 2022)),
                    step=1,
                    key="w_year",
                )
            with col_age2:
                age_of_property = st.number_input(
                    "Age (Years)",
                    min_value=0,
                    max_value=60,
                    value=int(st.session_state.get("w_age", 4)),
                    step=1,
                    key="w_age",
                )

            col_fl1, col_fl2 = st.columns(2)
            with col_fl1:
                floor_no = st.number_input(
                    "Floor Number",
                    min_value=0,
                    max_value=100,
                    value=int(st.session_state.get("w_floor", 7)),
                    step=1,
                    key="w_floor",
                )
            with col_fl2:
                total_floors = st.number_input(
                    "Total Floors",
                    min_value=1,
                    max_value=100,
                    value=int(st.session_state.get("w_total_floors", 15)),
                    step=1,
                    key="w_total_floors",
                )

        # --- Column 3: Amenities & Neighborhood ---
        with col_amen:
            st.markdown("##### 🌟 Amenities & Infrastructure")

            furn_options = ["Furnished", "Semi-furnished", "Unfurnished"]
            furnished_status = st.selectbox(
                "Furnishing Condition",
                furn_options,
                index=furn_options.index(st.session_state.get("w_furnished", "Semi-furnished")),
                key="w_furnished",
            )

            all_amenity_options = ["Clubhouse", "Garden", "Gym", "Playground", "Pool"]
            cur_amen = st.session_state.get("w_amenities", ["Gym", "Pool"])
            if isinstance(cur_amen, str):
                cur_amen = [x.strip() for x in cur_amen.split(",") if x.strip()]

            selected_amenities = st.multiselect(
                "Included Amenities",
                all_amenity_options,
                default=[a for a in cur_amen if a in all_amenity_options],
                key="w_amenities",
                help="Select all society amenities available",
            )
            amenities_str = ", ".join(sorted(selected_amenities))

            public_transport = st.select_slider(
                "Public Transit Connectivity",
                options=["Low", "Medium", "High"],
                value=st.session_state.get("w_transport", "High"),
                key="w_transport",
            )

            col_sc, col_hp = st.columns(2)
            with col_sc:
                nearby_schools = st.number_input("Nearby Schools", min_value=0, max_value=30, value=int(st.session_state.get("w_schools", 4)), key="w_schools")
            with col_hp:
                nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=30, value=int(st.session_state.get("w_hospitals", 3)), key="w_hospitals")

            col_pk, col_sec = st.columns(2)
            with col_pk:
                parking_space = st.radio("Dedicated Parking", ["Yes", "No"], horizontal=True, index=0 if st.session_state.get("w_parking") == "Yes" else 1, key="w_parking")
            with col_sec:
                security = st.radio("24/7 Security", ["Yes", "No"], horizontal=True, index=0 if st.session_state.get("w_security") == "Yes" else 1, key="w_security")

    # Validation
    if floor_no > total_floors:
        st.warning("⚠️ Floor Number cannot exceed Total Floors.")

    # -----------------------------------------------------
    # Model Inference Execution
    # -----------------------------------------------------
    input_df = build_model_input(
        state=selected_state,
        city=selected_city,
        property_type=property_type,
        bhk=bhk,
        size_sqft=size_sqft,
        price_lakhs=price_lakhs,
        year_built=year_built,
        age_of_property=age_of_property,
        furnished_status=furnished_status,
        floor_no=floor_no,
        total_floors=total_floors,
        nearby_schools=nearby_schools,
        nearby_hospitals=nearby_hospitals,
        public_transport=public_transport,
        parking_space=parking_space,
        security=security,
        amenities=amenities_str,
        facing=facing,
        owner_type=owner_type,
        availability_status=availability_status,
    )

    try:
        class_pred = classification_model.predict(input_df)[0]
        future_price_pred = float(regression_model.predict(input_df)[0])
        future_price_pred = max(future_price_pred, price_lakhs * 0.5)

        prob_good = 0.5
        if hasattr(classification_model, "predict_proba"):
            probs = classification_model.predict_proba(input_df)[0]
            if len(probs) > 1:
                prob_good = float(probs[1])
    except Exception as err:
        st.error(f"Inference computation error: {err}")
        st.stop()

    is_good_investment = int(class_pred) == 1
    abs_gain = future_price_pred - price_lakhs
    pct_gain = (abs_gain / price_lakhs) * 100.0 if price_lakhs > 0 else 0.0
    cagr = calculate_cagr(price_lakhs, future_price_pred, 5)

    # -----------------------------------------------------
    # Interactive Prediction Results Dashboard
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### 🎯 Predictive Intelligence Dashboard")

    col_verdict, col_gauge = st.columns([1.3, 1], gap="large")

    with col_verdict:
        if is_good_investment:
            st.markdown(
                """
                <div class="verdict-good">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 2.2rem;">🌟</span>
                        <div>
                            <div style="font-size: 1.45rem; font-weight: 800; color: #10B981;">STRONG BUY / HIGH POTENTIAL</div>
                            <div style="font-size: 0.95rem; color: #D1FAE5; margin-top: 0.2rem;">
                                This property exhibits favorable valuation metrics, high civic infrastructure density, and above-average projected capital growth.
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="verdict-caution">
                    <div style="display: flex; align-items: center; gap: 0.75rem;">
                        <span style="font-size: 2.2rem;">⚠️</span>
                        <div>
                            <div style="font-size: 1.45rem; font-weight: 800; color: #F59E0B;">HOLD / CAUTION ADVISED</div>
                            <div style="font-size: 0.95rem; color: #FEF3C7; margin-top: 0.2rem;">
                                Model indicators suggest a sub-optimal appreciation rate or high entry multiple relative to comparable assets.
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        stat1, stat2 = st.columns(2)
        with stat1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Estimated 5-Year Value</div>
                    <div class="stat-value">{format_inr(future_price_pred)}</div>
                    <div class="stat-delta {'positive' if abs_gain >= 0 else 'negative'}">
                        {'▲' if abs_gain >= 0 else '▼'} {format_inr(abs(abs_gain))} ({pct_gain:+.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Projected Annual CAGR</div>
                    <div class="stat-value">{cagr:.2f}% <span style="font-size: 1rem; font-weight: 500; color: #94A3B8;">/ yr</span></div>
                    <div class="stat-delta {'positive' if cagr >= 8 else 'neutral'}">
                        Benchmark: ~7.5% real estate index
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        stat3, stat4 = st.columns(2)
        with stat3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Entry Rate / SqFt</div>
                    <div class="stat-value">₹ {calc_rate:,.0f}</div>
                    <div class="stat-delta neutral">Base Price: {format_inr(price_lakhs)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with stat4:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Civic Access Score</div>
                    <div class="stat-value">{nearby_schools + nearby_hospitals} <span style="font-size: 1rem; color: #94A3B8;">points</span></div>
                    <div class="stat-delta positive">
                        {nearby_schools} Schools • {nearby_hospitals} Hospitals
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_gauge:
        gauge_color = "#10B981" if prob_good >= 0.65 else "#F59E0B" if prob_good >= 0.45 else "#EF4444"
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob_good * 100,
                number={"suffix": "%", "font": {"size": 36, "color": "#F8FAFC", "family": "Plus Jakarta Sans"}},
                title={"text": "AI Investment Confidence Score", "font": {"size": 16, "color": "#94A3B8"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                    "bar": {"color": gauge_color, "thickness": 0.28},
                    "bgcolor": "rgba(255, 255, 255, 0.05)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 45], "color": "rgba(239, 68, 68, 0.15)"},
                        {"range": [45, 65], "color": "rgba(245, 158, 11, 0.15)"},
                        {"range": [65, 100], "color": "rgba(16, 185, 129, 0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#FFFFFF", "width": 3},
                        "thickness": 0.8,
                        "value": prob_good * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.caption(
            f"Model predicts a **{prob_good*100:.1f}% probability** that this asset outperforms regional median inflation benchmarks over 5 years."
        )

    # -----------------------------------------------------
    # Interactive Visualizations: Growth Trajectory & Radar Scorecard
    # -----------------------------------------------------
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    col_chart1, col_chart2 = st.columns([1.4, 1], gap="medium")

    with col_chart1:
        st.markdown("##### 📈 5-Year Capital Appreciation Trajectory")
        
        years = ["Today", "Year 1", "Year 2", "Year 3", "Year 4", "Year 5"]
        r = cagr / 100.0 if cagr > -50 else 0.05
        projected_curve = [price_lakhs * ((1 + r) ** i) for i in range(6)]
        benchmark_curve = [price_lakhs * ((1 + 0.07) ** i) for i in range(6)]

        df_trajectory = pd.DataFrame({
            "Timeline": years,
            "PropVision Forecast": projected_curve,
            "7% Inflation Benchmark": benchmark_curve,
        })

        fig_traj = go.Figure()

        fig_traj.add_trace(
            go.Scatter(
                x=df_trajectory["Timeline"],
                y=df_trajectory["PropVision Forecast"],
                mode="lines+markers",
                name="PropVision Forecast",
                line=dict(color="#6366F1", width=3.5),
                marker=dict(size=8, color="#818CF8"),
                fill="tozeroy",
                fillcolor="rgba(99, 102, 241, 0.12)",
                hovertemplate="<b>%{x}</b><br>Valuation: ₹ %{y:.2f} Lakhs<extra></extra>",
            )
        )

        fig_traj.add_trace(
            go.Scatter(
                x=df_trajectory["Timeline"],
                y=df_trajectory["7% Inflation Benchmark"],
                mode="lines",
                name="7% Inflation Benchmark",
                line=dict(color="#94A3B8", width=2, dash="dot"),
                hovertemplate="<b>%{x}</b><br>Benchmark: ₹ %{y:.2f} Lakhs<extra></extra>",
            )
        )

        fig_traj.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(
                title="Valuation (₹ Lakhs)",
                gridcolor="rgba(255, 255, 255, 0.08)",
                zeroline=False,
            ),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.04)"),
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with col_chart2:
        st.markdown("##### 🧭 5-Pillar Asset Health Radar")

        transit_score = 90 if public_transport == "High" else 60 if public_transport == "Medium" else 30
        civic_score = min(100, int(((nearby_schools + nearby_hospitals) / 10.0) * 100))
        amenity_score = min(100, int((len(selected_amenities) / 5.0) * 100))
        safety_score = (50 if parking_space == "Yes" else 0) + (50 if security == "Yes" else 0)
        modernity_score = 95 if age_of_property <= 3 else 75 if age_of_property <= 8 else 50 if age_of_property <= 15 else 30

        categories = [
            "Transit Access",
            "Civic Density",
            "Amenities",
            "Parking & Security",
            "Asset Modernity",
        ]
        values = [transit_score, civic_score, amenity_score, safety_score, modernity_score]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(16, 185, 129, 0.2)",
                line=dict(color="#10B981", width=2.5),
                name="Property Profile",
            )
        )
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            height=340,
            margin=dict(l=30, r=30, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Key Catalysts & Risk Vectors
    st.markdown("##### 🔍 Qualitative Investment Catalysts & Risk Vectors")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10B981; padding: 1rem; border-radius: 8px;">
                <strong style="color: #34D399;">✅ Key Value Drivers:</strong>
                <ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem; color: #E2E8F0; font-size: 0.9rem;">
                    <li><strong>Connectivity:</strong> {public_transport} transit accessibility enhances rental tenant retention.</li>
                    <li><strong>Civic Infrastructure:</strong> {nearby_schools} schools and {nearby_hospitals} hospitals support sustained end-user resale demand.</li>
                    <li><strong>Modernity:</strong> Property age ({age_of_property} yrs) falls in <em>{input_df['Property_Age_Band'].iloc[0]}</em> bracket with minimal immediate CAPEX.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style="background: rgba(245, 158, 11, 0.08); border-left: 4px solid #F59E0B; padding: 1rem; border-radius: 8px;">
                <strong style="color: #FBBF24;">⚠️ Risk Assessment & Watchouts:</strong>
                <ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem; color: #E2E8F0; font-size: 0.9rem;">
                    <li><strong>Floor Ratio:</strong> Level {floor_no} of {total_floors} (Ratio: {input_df['Floor_Ratio'].iloc[0]:.2f}) impacts view premium and resale velocity.</li>
                    <li><strong>Execution Risk:</strong> {'Ready to Move minimizes completion delay' if availability_status == 'Ready_to_Move' else 'Under-construction carries milestone delivery risks'}.</li>
                    <li><strong>Entry Valuation:</strong> Buying at ₹ {calc_rate:,.0f}/sqft requires matching rental yields above 3.0% to beat cost of capital.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# TAB 2: MARKET TRENDS & ANALYTICS
# =========================================================
with tab_market:
    st.markdown("### 📊 Macro Real Estate Intelligence (250,000+ Records)")
    st.caption("Interactive benchmarking against verified transactional housing data across top Indian metro regions.")

    market_df = load_market_sample()

    if market_df.empty:
        st.info("Market dataset file `india_housing_prices.csv` is being loaded or not available.")
    else:
        col_m_f1, col_m_f2 = st.columns([1, 1])
        with col_m_f1:
            state_filter = st.multiselect(
                "Filter by State(s)",
                sorted(market_df["State"].unique().tolist()),
                default=[selected_state] if selected_state in market_df["State"].unique() else market_df["State"].unique()[:4],
            )
        with col_m_f2:
            prop_type_filter = st.multiselect(
                "Filter by Property Type",
                market_df["Property_Type"].unique().tolist(),
                default=market_df["Property_Type"].unique().tolist(),
            )

        filtered_market = market_df[
            (market_df["State"].isin(state_filter)) &
            (market_df["Property_Type"].isin(prop_type_filter))
        ]

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Analyzed Transactions", f"{len(filtered_market):,}")
        with m2:
            avg_price = filtered_market["Price_in_Lakhs"].median()
            st.metric("Median Property Price", format_inr(avg_price))
        with m3:
            avg_sqft = (filtered_market["Price_per_SqFt"] * 100000).mean()
            st.metric("Avg Rate / SqFt", f"₹ {avg_sqft:,.0f}")
        with m4:
            rtm_share = (filtered_market["Availability_Status"] == "Ready_to_Move").mean() * 100
            st.metric("Ready to Move Share", f"{rtm_share:.1f}%")

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("##### 🏙️ Average Price per SqFt by City")
            city_avg = (
                filtered_market.groupby("City")["Price_per_SqFt"]
                .mean()
                .reset_index()
                .sort_values(by="Price_per_SqFt", ascending=False)
                .head(12)
            )
            city_avg["Rate_per_SqFt"] = city_avg["Price_per_SqFt"] * 100000

            fig_city = px.bar(
                city_avg,
                x="City",
                y="Rate_per_SqFt",
                color="Rate_per_SqFt",
                color_continuous_scale="Viridis",
                labels={"Rate_per_SqFt": "Avg Rate (₹/SqFt)"},
            )
            fig_city.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_city, use_container_width=True)

        with col_m2:
            st.markdown("##### 📦 Price Distribution by Property Type")
            fig_box = px.box(
                filtered_market,
                x="Property_Type",
                y="Price_in_Lakhs",
                color="Property_Type",
                labels={"Price_in_Lakhs": "Price (₹ Lakhs)", "Property_Type": "Type"},
            )
            fig_box.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_box, use_container_width=True)


# =========================================================
# TAB 3: MORTGAGE & ROI PLANNER
# =========================================================
with tab_roi:
    st.markdown("### 💰 Financial Structuring & Mortgage ROI Engine")
    st.caption("Model your loan schedule, down payment cash outlay, and net investor return over the 5-year investment window.")

    col_loan1, col_loan2 = st.columns([1, 1.2], gap="large")

    with col_loan1:
        st.markdown("##### 🏦 Mortgage Financing Parameters")
        
        down_payment_pct = st.slider("Down Payment Percentage", min_value=10, max_value=50, value=20, step=5)
        interest_rate = st.slider("Annual Interest Rate (%)", min_value=6.5, max_value=13.0, value=8.5, step=0.25)
        tenure_years = st.slider("Loan Tenure (Years)", min_value=5, max_value=30, value=20, step=1)
        expected_monthly_rent = st.number_input(
            "Expected Monthly Rental Income (₹)",
            min_value=5000,
            max_value=500000,
            value=int(price_lakhs * 100000 * 0.0028),
            step=1000,
            help="Typical gross residential yield in India ranges from 2.5% to 3.8% annually",
        )

        down_payment_amt = price_lakhs * (down_payment_pct / 100.0)
        loan_amt_lakhs = price_lakhs - down_payment_amt
        loan_amt_rupees = loan_amt_lakhs * 100000.0

        r_month = (interest_rate / 100.0) / 12.0
        n_months = tenure_years * 12
        if r_month > 0:
            monthly_emi = (loan_amt_rupees * r_month * ((1 + r_month) ** n_months)) / (((1 + r_month) ** n_months) - 1)
        else:
            monthly_emi = loan_amt_rupees / n_months

        total_loan_repayment = monthly_emi * n_months
        total_interest_payable = total_loan_repayment - loan_amt_rupees

    with col_loan2:
        st.markdown("##### 📊 Cash Outlay & Loan Breakdown")
        
        em1, em2, em3 = st.columns(3)
        with em1:
            st.metric("Monthly EMI", f"₹ {monthly_emi:,.0f}")
        with em2:
            st.metric("Down Payment", format_inr(down_payment_amt))
        with em3:
            st.metric("Loan Principal", format_inr(loan_amt_lakhs))

        fig_donut = go.Figure(
            data=[
                go.Pie(
                    labels=["Down Payment (Equity)", "Loan Principal (Debt)", "Total Bank Interest"],
                    values=[down_payment_amt * 100000, loan_amt_rupees, total_interest_payable],
                    hole=0.55,
                    marker=dict(colors=["#10B981", "#6366F1", "#F59E0B"]),
                )
            ]
        )
        fig_donut.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")
    st.markdown("##### 📈 5-Year Net Investor Profit & Yield Simulation")
    
    interest_paid_5y = (total_interest_payable / tenure_years) * 5.0
    total_rental_5y = expected_monthly_rent * 12 * 5
    capital_gain_rupees = abs_gain * 100000.0

    net_profit_5y = capital_gain_rupees + total_rental_5y - interest_paid_5y
    roi_on_equity = (net_profit_5y / (down_payment_amt * 100000.0)) * 100.0 if down_payment_amt > 0 else 0

    cf1, cf2, cf3, cf4 = st.columns(4)
    with cf1:
        st.metric("5Y Capital Appreciation", f"₹ {capital_gain_rupees:,.0f}")
    with cf2:
        st.metric("5Y Gross Rental Income", f"₹ {total_rental_5y:,.0f}")
    with cf3:
        st.metric("5Y Mortgage Interest Outflow", f"- ₹ {interest_paid_5y:,.0f}")
    with cf4:
        st.metric("5Y Net Investor Gain", f"₹ {net_profit_5y:,.0f}", f"{roi_on_equity:+.1f}% on Equity")


# =========================================================
# TAB 4: WHAT-IF SENSITIVITY COMPARISON
# =========================================================
with tab_compare:
    st.markdown("### ⚖️ 'What-If' Property Sensitivity Simulator")
    st.caption("Test how changes in negotiations, amenities, or floor height transform investment desirability and valuation.")

    col_base, col_alt = st.columns(2, gap="large")

    with col_base:
        st.markdown(
            f"""
            <div class="section-card">
                <span class="badge badge-primary">Base Asset</span>
                <h4 style="margin: 0.5rem 0 0.2rem 0;">{selected_city}, {selected_state}</h4>
                <div style="color: #94A3B8; font-size: 0.9rem;">{bhk} BHK • {size_sqft:,.0f} SqFt • {property_type}</div>
                <hr style="border-color: rgba(255,255,255,0.08); margin: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between; font-size: 0.92rem; margin-bottom: 0.4rem;">
                    <span>Current Price:</span> <strong>{format_inr(price_lakhs)}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.92rem; margin-bottom: 0.4rem;">
                    <span>5Y Projected Price:</span> <strong>{format_inr(future_price_pred)}</strong>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.92rem; margin-bottom: 0.4rem;">
                    <span>Investment Probability:</span> <strong>{prob_good*100:.1f}%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.92rem;">
                    <span>Classification:</span> <strong style="color: {'#10B981' if is_good_investment else '#F59E0B'};">{'Good' if is_good_investment else 'Caution'}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_alt:
        st.markdown(
            """
            <div class="section-card">
                <span class="badge badge-success">Simulated Scenario</span>
                <h4 style="margin: 0.5rem 0 0.2rem 0;">Custom Adjusted Parameters</h4>
            """,
            unsafe_allow_html=True,
        )

        sim_price_delta = st.slider("Price Discount / Premium (%)", -25, 25, -10, step=5, help="Negotiated purchase discount")
        sim_price = price_lakhs * (1 + sim_price_delta / 100.0)

        sim_avail = st.radio("Availability", ["Ready_to_Move", "Under_Construction"], horizontal=True, index=0)
        sim_floor = st.slider("Simulated Floor Level", 0, int(total_floors), min(int(floor_no) + 3, int(total_floors)))
        
        st.markdown("</div>", unsafe_allow_html=True)

    alt_df = build_model_input(
        state=selected_state,
        city=selected_city,
        property_type=property_type,
        bhk=bhk,
        size_sqft=size_sqft,
        price_lakhs=sim_price,
        year_built=year_built,
        age_of_property=age_of_property,
        furnished_status=furnished_status,
        floor_no=sim_floor,
        total_floors=total_floors,
        nearby_schools=nearby_schools,
        nearby_hospitals=nearby_hospitals,
        public_transport=public_transport,
        parking_space=parking_space,
        security=security,
        amenities=amenities_str,
        facing=facing,
        owner_type=owner_type,
        availability_status=sim_avail,
    )

    alt_class = classification_model.predict(alt_df)[0]
    alt_future_price = float(regression_model.predict(alt_df)[0])
    alt_prob = 0.5
    if hasattr(classification_model, "predict_proba"):
        alt_probs = classification_model.predict_proba(alt_df)[0]
        if len(alt_probs) > 1:
            alt_prob = float(alt_probs[1])

    st.markdown("##### 📊 Scenario Comparison Analysis")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        delta_p = sim_price - price_lakhs
        st.metric("Purchase Price", format_inr(sim_price), f"{delta_p:+.2f} L ({sim_price_delta:+d}%)", delta_color="inverse")
    with sc2:
        delta_fp = alt_future_price - future_price_pred
        st.metric("Simulated 5Y Price", format_inr(alt_future_price), f"{delta_fp:+.2f} L")
    with sc3:
        delta_prob = (alt_prob - prob_good) * 100.0
        st.metric("Confidence Score", f"{alt_prob*100:.1f}%", f"{delta_prob:+.1f}%")

    fig_comp = go.Figure(data=[
        go.Bar(name="Base Property", x=["Current Price (L)", "5Y Future Value (L)"], y=[price_lakhs, future_price_pred], marker_color="#6366F1"),
        go.Bar(name="Simulated Scenario", x=["Current Price (L)", "5Y Future Value (L)"], y=[sim_price, alt_future_price], marker_color="#10B981")
    ])
    fig_comp.update_layout(
        barmode="group",
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="₹ Lakhs", gridcolor="rgba(255,255,255,0.08)"),
    )
    st.plotly_chart(fig_comp, use_container_width=True)


# =========================================================
# TAB 5: INVESTMENT DOSSIER & EXPORT
# =========================================================
with tab_dossier:
    st.markdown("### 📋 Executive Investment Dossier & Report Export")
    st.caption("Generate institutional-grade investment summaries for record keeping, partner presentation, or banking.")

    clean_avail = availability_status.replace("_", " ")
    amenities_display = ", ".join(selected_amenities) if selected_amenities else "None specified"
    age_band = input_df["Property_Age_Band"].iloc[0]

    dossier_data = {
        "Attribute": [
            "Project Location",
            "Property Category & Typology",
            "Super Built-up Area",
            "Initial Acquisition Cost",
            "Acquisition Rate / SqFt",
            "Construction Year & Age",
            "Floor Elevation",
            "Civic Infrastructure Score",
            "Transit Accessibility",
            "Amenities Included",
            "Model Classification",
            "Confidence Probability",
            "Projected 5-Year Valuation",
            "Projected 5-Year Absolute Gain",
            "Estimated 5-Year CAGR",
        ],
        "Assessment Value": [
            f"{selected_city}, {selected_state}",
            f"{bhk} BHK {property_type} ({clean_avail})",
            f"{size_sqft:,.0f} SqFt",
            format_inr(price_lakhs),
            f"₹ {calc_rate:,.0f} / SqFt",
            f"Built {year_built} ({age_of_property} years old, {age_band})",
            f"Floor {floor_no} of {total_floors}",
            f"{nearby_schools} Schools, {nearby_hospitals} Hospitals",
            f"{public_transport}",
            amenities_display,
            "✅ Good Investment" if is_good_investment else "⚠️ Caution Advised",
            f"{prob_good*100:.2f}%",
            format_inr(future_price_pred),
            f"+ {format_inr(abs_gain)} ({pct_gain:+.2f}%)",
            f"{cagr:.2f}% per annum",
        ],
    }

    df_dossier = pd.DataFrame(dossier_data)
    st.dataframe(df_dossier, use_container_width=True, hide_index=True)

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_buffer = io.StringIO()
        df_dossier.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Dossier (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"PropVision_{selected_city}_{bhk}BHK_Report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col_exp2:
        json_report = {
            "meta": {
                "generated_by": "PropVision AI",
                "classification_model": metadata.get("classification_model", "XGBoost Classifier"),
                "regression_model": metadata.get("regression_model", "XGBoost Regressor"),
            },
            "property": {
                "state": selected_state,
                "city": selected_city,
                "property_type": property_type,
                "bhk": bhk,
                "size_sqft": size_sqft,
                "price_lakhs": price_lakhs,
                "year_built": year_built,
                "age_of_property": age_of_property,
                "furnished_status": furnished_status,
                "floor_no": floor_no,
                "total_floors": total_floors,
                "amenities": selected_amenities,
                "availability_status": availability_status,
            },
            "prediction": {
                "is_good_investment": bool(is_good_investment),
                "probability_good": round(prob_good, 4),
                "estimated_5y_price_lakhs": round(future_price_pred, 2),
                "cagr_percent": round(cagr, 2),
                "total_gain_lakhs": round(abs_gain, 2),
            },
        }
        st.download_button(
            label="📥 Export Full Audit Dossier (JSON)",
            data=json.dumps(json_report, indent=2),
            file_name=f"PropVision_{selected_city}_{bhk}BHK_Audit.json",
            mime="application/json",
            use_container_width=True,
        )

# ---------------------------------------------------------
# Application Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #64748B; font-size: 0.82rem; padding: 1rem 0;">
        PropVision AI • Dual-Model Real Estate Intelligence Architecture • Engineered for Strategic Property Investors
    </div>
    """,
    unsafe_allow_html=True,
)
