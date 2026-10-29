# 🏢 PropVision AI — Real Estate Investment Advisor

> **Interactive Machine Learning & PropTech Analytics Platform for Residential Property Evaluation in India**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Powered-orange.svg)](https://xgboost.ai/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-informational.svg)](https://plotly.com/)

---


**Try the live application here:**

👉 [(https://2z42grgcd7qg5fwbqsdrso.streamlit.app/)](https://2z42grgcd7qg5fwbqsdrso.streamlit.app/)]

## 🌟 Overview

**PropVision AI** is an interactive, data-driven Real Estate Investment Advisor powered by dual machine-learning engines trained on **250,000+ verified Indian residential housing records** across 20 states and 42 tier-1 & tier-2 metropolitan hubs.

The platform provides institutional-grade property appraisal, deal attractiveness classification, 5-year scenario appreciation projections, mortgage financial modeling, and sensitivity benchmarking.

---

## 🚀 Key Interactive UI / UX Features

### 1. 🔮 Dual-Engine AI Deal Valuation
- **Deal Attractiveness Classifier**: Evaluates whether a property is a **Strong Buy** or **Hold/Caution** using a tuned **XGBoost Classifier**.
- **Scenario 5-Year Price Regressor**: Predicts estimated future asset valuation in Indian Lakhs/Crores and computes the 5-year CAGR (Compound Annual Growth Rate).
- **Investment Confidence Gauge**: Real-time Plotly radial gauge visualizing investment likelihood (0% to 100%) with tiered safety zones.
- **5-Year Capital Appreciation Trajectory**: Interactive Plotly projection curve benchmarking estimated growth against a 7.0% annual inflation index.
- **5-Pillar Health Radar**: Radar scorecard comparing Transit Accessibility, Civic Density, Amenities Richness, Security & Parking, and Asset Modernity.

### 2. ⚡ 1-Click Curated Presets
Instantly simulate diverse real estate investment archetypes with one click:
- 🌆 **Urban Metro High-Growth 3BHK** (Hyderabad • 1,650 SqFt • ₹1.25 Cr)
- 🏡 **Premium Gated Luxury Villa** (Bangalore • 3,200 SqFt • ₹2.95 Cr)
- 🏢 **Sea-View High-Rise Penthouse** (Mumbai • 2,600 SqFt • ₹4.80 Cr)
- 🌱 **Budget Suburban Starter Home** (Ahmedabad • 1,050 SqFt • ₹52 L)

### 3. 📍 Dynamic Cascading Geography & Specs
- **State & City Filtering**: Cascading dropdowns dynamically populate cities based on the selected state (Telangana → Hyderabad/Warangal, Maharashtra → Mumbai/Pune/Nagpur, Karnataka → Bangalore/Mysore/Mangalore, etc.).
- **Live Valuation Pill**: Dynamically computes and displays the rate per square foot (`₹ / SqFt`) and formatted total value (`₹ L` / `₹ Cr`) in real time as sliders move.
- **Interactive Amenities Multiselect**: Society amenities chip selection (Clubhouse, Gym, Pool, Garden, Playground) updating the model's feature vectors.

### 4. 📊 Macro Market Trends & Analytics
- Live macro analytics over sample transactions from `india_housing_prices.csv`.
- Filter by state and property typology to observe median transaction prices, ready-to-move inventory ratios, and city rate rankings.
- Visual price distribution box plots and average price per sqft bar charts.

### 5. 💰 Mortgage Structuring & Net ROI Planner
- Interactive sliders for Down Payment (10% to 50%), Annual Interest Rate (6.5% to 13%), and Loan Tenure (5 to 30 years).
- Real-time monthly EMI computation and donut chart detailing Equity Down Payment vs Principal Loan Debt vs Total Bank Interest.
- 5-year Net Investor Return simulation factoring in projected capital gain, gross rental income, and cumulative loan interest outflow.

### 6. ⚖️ "What-If" Sensitivity Simulator
- Stress-test deal negotiations with interactive discount/premium sliders (-25% to +25%).
- Compare baseline asset against hypothetical scenarios (e.g. Under-Construction vs Ready-to-Move, Higher Floor elevation) with grouped visual delta charts.

### 7. 📋 Institutional Investment Dossier & Export
- Formats a comprehensive executive appraisal sheet.
- One-click export to **CSV** or audited **JSON** reports for documentation or client presentation.

---

## 🛠️ Project Structure

```
Real Estate Investment Advisor/
│
├── app.py                     # Main interactive Streamlit & Plotly application
├── locations.json             # State-to-city categorical hierarchy (20 states, 42 cities)
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive documentation
│
├── models/
│   ├── best_classification_pipeline.joblib  # Trained XGBoost deal classifier pipeline
│   ├── best_regression_pipeline.joblib      # Trained XGBoost 5Y price regressor pipeline
│   └── model_metadata.json                  # Model hyperparams & training metadata
│
└── india_housing_prices.csv   # Source dataset (250,000+ Indian property records)
```

---

## 💻 Quickstart Installation & Execution

### 1. Environment Setup
```bash
# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Model Specifications & Preprocessing

- **Classification Model:** Tuned XGBoost Classifier with Scikit-Learn `ColumnTransformer` (StandardScaler + SimpleImputer for numeric; OneHotEncoder for categorical features).
- **Regression Model:** Tuned XGBoost Regressor with matching feature pipelines.
- **Engineered Features:** 28 input variables including `Price_per_SqFt`, `Floor_Ratio`, `School_Hospital_Access`, `Amenities_Count`, `Property_Age_Band`, `Has_Parking`, `Has_Security`, and `Is_Ready_To_Move`.

---

*Educational & Analytical Disclaimer: Model projections are scenario-based algorithmic estimates and do not constitute formal financial advice.*

