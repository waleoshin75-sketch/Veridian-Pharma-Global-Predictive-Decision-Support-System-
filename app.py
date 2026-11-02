# =============================================================================
# VERIDIAN PHARMA GLOBAL: PHARMACEUTICAL BUSINESS OPERATIONS PREDICTION SYSTEM
# A Streamlit app that uses the five saved models (model, scaler and features .joblib files).
#
#   Page            Business problem                 Model
#   Batch Quality         Will this batch be rejected?     Logistic Regression
#   Demand Planning       How many units will we sell?     Linear Regression (Ridge)
#   Cold-Chain Protection Will this shipment get too hot?  Logistic Regression
#   Drug Safety           Is this report a serious event?  Logistic Regression
#   Pricing Control       Is this invoice priced oddly?    Elliptic Envelope
#
# HOW TO RUN (in the VS Code terminal)
#   1. pip install streamlit pandas numpy scikit-learn joblib
#   2. Put this file (app.py) in the same folder as your "models" folder
#      (or your unzipped "deployment/veridian_pharma_models" folder).
#   3. streamlit run app.py          (it opens in your browser)
#
# HOW THE FILE IS ORGANISED (read it top to bottom)
#   Part 1  Find and load the saved files
#   Part 2  Scoring helpers (turn one case into a prediction)
#   Part 3  The five objectives: the questions to ask, and how answers become model inputs
#   Part 4  Drawing the pages
# =============================================================================
from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="Pharma Operations Prediction System", page_icon="💊", layout="wide")

# =============================================================================
# PART 1. FIND AND LOAD THE SAVED FILES
# =============================================================================
HERE = Path(__file__).parent
FOLDERS = {"batch_quality": "batch_rejection", "demand_planning": "demand_forecast", "cold_chain": "cold_chain",
           "drug_safety": "adverse_events", "pricing_control": "pricing_anomalies"}

def find_model_root():
    """Look in the usual places for the folder that holds batch_rejection/, cold_chain/ and so on."""
    for place in [HERE / "models", HERE / "deployment" / "veridian_pharma_models", HERE / "veridian_pharma_models", HERE]:
        if (place / "batch_rejection" / "model.joblib").exists():
            return place
    return None

ROOT = find_model_root()
if ROOT is None:
    st.error("The saved model files were not found. Put this app.py next to your 'models' folder "
             "(or next to the unzipped 'veridian_pharma_models' folder) and run it again.")
    st.stop()

@st.cache_resource
def load_files(folder):
    """Load the three saved files of one objective (cached, so they load only once)."""
    return (joblib.load(ROOT / folder / "model.joblib"), joblib.load(ROOT / folder / "scaler.joblib"),
            joblib.load(ROOT / folder / "features.joblib"))

# =============================================================================
# PART 2. SCORING HELPERS
# =============================================================================
def prepare(raw_rows, info, scaler):
    """Raw input columns -> exactly the table the model was trained on (same columns, same order, scaled)."""
    X = raw_rows[info["numeric_inputs"] + info["categorical_inputs"]].copy()
    for col in info["categorical_inputs"]:                     # blank or unseen text becomes "Unknown", as in training
        X[col] = X[col].fillna("Unknown")
        X[col] = X[col].where(X[col].isin(info["categories_seen"][col]), "Unknown")
    X = pd.get_dummies(X, columns=info["categorical_inputs"], dtype=int)
    X = X.reindex(columns=info["feature_columns"], fill_value=0)
    X = X.fillna(info["fill_values"])                           # blanks get the value used in training
    X[info["scaled_columns"]] = scaler.transform(X[info["scaled_columns"]])
    return X

def risk_tier(p, info):
    """High / Medium / Low from the cut-offs saved with the model."""
    if p >= info["risk_tiers"]["high_from"]:
        return "High"
    return "Medium" if p >= info["risk_tiers"]["medium_from"] else "Low"

def nice_name(column, info):
    """'formulation_Injection' -> 'formulation = Injection'; 'max_temp_deviation_c' -> 'max temp deviation c'."""
    for cat in info["categorical_inputs"]:
        if column.startswith(cat + "_"):
            return f"{cat.replace('_', ' ')} = {column[len(cat) + 1:]}"
    return column.replace("_", " ")

def main_factors(model, X, info, n=3):
    """The inputs pushing this prediction the most (coefficient x scaled value). Works for the linear models."""
    if not hasattr(model, "coef_"):
        return []
    pull = pd.Series(np.ravel(model.coef_) * X.iloc[0].to_numpy(), index=X.columns)
    top = pull.reindex(pull.abs().sort_values(ascending=False).index).head(n)
    return [(nice_name(name, info), "raises" if value > 0 else "lowers") for name, value in top.items()]

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
def season_of(month):
    return "Winter" if month in (12, 1, 2) else "Spring" if month in (3, 4, 5) else "Summer" if month in (6, 7, 8) else "Autumn"

# =============================================================================
# PART 3. THE FIVE OBJECTIVES
# Each objective has: a list of questions (fields) and a function that turns the answers into the model's inputs.
# Field helpers: num = a number, pick = a drop-down, yesno = Yes/No (stored as 1/0), when = a date.
# =============================================================================
def num(name, label, default, low, high, step=1, help=""):
    return {"name": name, "label": label, "kind": "num", "default": default, "low": low, "high": high, "step": step, "help": help}
def pick(name, label, options, default=None, help=""):
    return {"name": name, "label": label, "kind": "pick", "options": options, "default": default or options[0], "help": help}
def yesno(name, label, default=False, help=""):
    return {"name": name, "label": label, "kind": "yesno", "default": default, "help": help}
def when(name, label, default, help=""):
    return {"name": name, "label": label, "kind": "when", "default": default, "help": help}

AREAS = ["Anti-infective", "Cardiovascular", "Diabetes", "Oncology", "Respiratory", "Vaccines"]
FORMS = ["Capsule", "Injection", "Syrup", "Tablet"]
CUSTOMERS = ["Government Tender", "Hospital", "Retail Pharmacy", "Wholesale Distributor"]
COUNTRIES = ["Brazil", "Germany", "India", "Nigeria", "Singapore", "South Africa", "Switzerland", "UAE", "United Kingdom", "United States"]

# ---------- Objective 1: Batch Quality ----------
batch_fields = [
    pick("formulation", "Product form", FORMS, "Tablet"), pick("therapeutic_area", "Therapeutic area", AREAS),
    num("batch_size_units", "Batch size (units)", 100000, 5000, 200000, 1000),
    num("raw_material_purity_pct", "Raw material purity (%)", 99.0, 90.0, 100.0, 0.1, "Lower purity raises rejection risk."),
    num("max_temp_deviation_c", "Largest temperature deviation (°C)", 1.0, 0.0, 10.0, 0.1, "How far the temperature drifted during manufacture."),
    num("equipment_age_years", "Equipment age (years)", 15, 0, 50),
    num("operator_experience_years", "Operator experience (years)", 12, 0, 40),
    yesno("night_shift", "Made on a night shift?"),
    pick("plant_automation", "Plant automation level", ["Low", "Medium", "High"], "Medium"),
    num("plant_age_years", "Plant age (years)", 13, 0, 150),
    yesno("plant_gmp_certified", "Plant GMP certified?", True),
    num("mfg_month", "Manufacturing month (1-12)", 6, 1, 12),
    num("prod_dosage_mg", "Dosage (mg)", 50, 1, 1000), num("prod_shelf_life_months", "Shelf life (months)", 24, 6, 60)]
def batch_row(v):
    return {"raw_material_purity_pct": v["raw_material_purity_pct"], "max_temp_deviation_c": v["max_temp_deviation_c"],
            "equipment_age_years": v["equipment_age_years"], "night_shift": v["night_shift"],
            "operator_experience_years": v["operator_experience_years"], "log_batch_size": math.log1p(v["batch_size_units"]),
            "plant_automation": {"Low": 1, "Medium": 2, "High": 3}[v["plant_automation"]], "plant_age_years": v["plant_age_years"],
            "plant_gmp_certified": v["plant_gmp_certified"], "mfg_month": v["mfg_month"], "prod_dosage_mg": v["prod_dosage_mg"],
            "prod_shelf_life_months": v["prod_shelf_life_months"], "formulation": v["formulation"], "therapeutic_area": v["therapeutic_area"]}

# ---------- Objective 2: Demand Planning ----------
demand_fields = [
    when("sale_date", "Sale date", pd.Timestamp("2025-10-15").date(), "The model was trained on sales from March 2022 to September 2025."),
    pick("therapeutic_area", "Therapeutic area", AREAS), pick("formulation", "Product form", FORMS, "Tablet"),
    pick("customer_type", "Customer type", CUSTOMERS, "Hospital"), pick("sales_channel", "Sales channel", ["Direct", "Distributor", "Online Pharmacy", "Tender"]),
    pick("customer_country", "Customer country", COUNTRIES, "Nigeria"),
    num("unit_price_usd", "Unit price (USD)", 33.0, 0.5, 7000.0, 0.5), num("discount_pct", "Discount (%)", 10, 0, 20),
    num("customer_credit_score", "Customer credit score (0-8)", 5, 0, 8), num("customer_tenure_years", "Customer tenure (years)", 4.0, 0.0, 10.0, 0.1),
    yesno("is_unknown_customer", "Customer not in our records?"),
    num("prod_dosage_mg", "Dosage (mg)", 50, 1, 1000), num("prod_shelf_life_months", "Shelf life (months)", 24, 6, 60),
    num("prod_margin_pct", "Product margin (%)", 66.0, 30.0, 90.0, 0.5)]
def demand_row(v):
    month = v["sale_date"].month
    return {"discount_pct": v["discount_pct"], "sale_month_sin": math.sin(2 * math.pi * month / 12), "sale_month_cos": math.cos(2 * math.pi * month / 12),
            "months_since_start": (v["sale_date"].year - 2022) * 12 + month - 3,       # months since March 2022, the first month of data
            "customer_credit_score": v["customer_credit_score"], "customer_tenure_years": v["customer_tenure_years"],
            "prod_dosage_mg": v["prod_dosage_mg"], "prod_shelf_life_months": v["prod_shelf_life_months"], "prod_margin_pct": v["prod_margin_pct"],
            "unit_price_usd": v["unit_price_usd"], "is_unknown_customer": v["is_unknown_customer"], "therapeutic_area": v["therapeutic_area"],
            "customer_type": v["customer_type"], "sales_channel": v["sales_channel"], "formulation": v["formulation"], "customer_country": v["customer_country"]}

# ---------- Objective 3: Cold-Chain Protection ----------
cold_fields = [
    pick("transport_mode", "Transport mode", ["Air", "Road", "Sea"], "Road"), pick("packaging_type", "Packaging", ["Standard", "Insulated", "Active Cooling"], "Standard"),
    yesno("cold_chain_required", "Product needs cold chain?", True),
    num("transit_days", "Transit time (days)", 4, 1, 60), num("avg_ambient_temp_c", "Average outside temperature (°C)", 22.0, -5.0, 45.0, 0.5),
    num("ship_month", "Shipping month (1-12)", 6, 1, 12), pick("destination_country", "Destination", COUNTRIES, "Nigeria"),
    pick("customer_type", "Customer type", CUSTOMERS, "Hospital"), pick("formulation", "Product form", FORMS, "Tablet"),
    num("prod_shelf_life_months", "Shelf life (months)", 24, 6, 60)]
def cold_row(v):
    protected = v["packaging_type"] != "Standard"
    return {"transit_days": v["transit_days"], "avg_ambient_temp_c": v["avg_ambient_temp_c"], "heat_exposure": v["transit_days"] * v["avg_ambient_temp_c"],
            "cold_chain_required": v["cold_chain_required"], "protective_packaging": int(protected), "is_air": int(v["transport_mode"] == "Air"),
            "is_sea": int(v["transport_mode"] == "Sea"), "ship_month": v["ship_month"],
            "unprotected_cold_chain": int(v["cold_chain_required"] == 1 and not protected), "product_requires_cold_chain": v["cold_chain_required"],
            "prod_shelf_life_months": v["prod_shelf_life_months"], "packaging_type": v["packaging_type"], "transport_mode": v["transport_mode"],
            "ship_season": season_of(v["ship_month"]), "destination_country": v["destination_country"], "customer_type": v["customer_type"],
            "formulation": v["formulation"]}

# ---------- Objective 4: Drug Safety ----------
safety_fields = [
    num("patient_age", "Patient age", 50, 0, 100), pick("patient_sex", "Patient sex", ["Female", "Male"]),
    num("concomitant_medications", "Other medicines the patient takes", 2, 0, 15), yesno("hepatic_impairment", "Liver (hepatic) impairment?"),
    pick("therapeutic_area", "Therapeutic area", AREAS), pick("formulation", "Product form", FORMS, "Tablet"),
    num("dosage_mg", "Dosage (mg)", 50, 1, 1000), pick("reporter_type", "Who reported it?", ["Nurse", "Patient", "Pharmacist", "Physician"], "Physician"),
    yesno("onset_known", "Is the time to onset known?", True), num("days_to_onset", "Days from first dose to onset", 8, 0, 365),
    num("batch_to_report_days", "Days from batch manufacture to report", 434, 1, 1500), num("report_month", "Report month (1-12)", 6, 1, 12),
    num("prod_shelf_life_months", "Shelf life (months)", 24, 6, 60)]
def safety_row(v):
    known = v["onset_known"] == 1
    return {"patient_age": v["patient_age"], "concomitant_medications": v["concomitant_medications"], "hepatic_impairment": v["hepatic_impairment"],
            "days_to_onset": v["days_to_onset"] if known else np.nan, "dosage_mg": v["dosage_mg"], "batch_to_report_days": v["batch_to_report_days"],
            "report_month": v["report_month"], "prod_shelf_life_months": v["prod_shelf_life_months"], "onset_unknown": int(not known),
            "therapeutic_area": v["therapeutic_area"], "formulation": v["formulation"], "reporter_type": v["reporter_type"], "patient_sex": v["patient_sex"]}

# ---------- Objective 5: Pricing Control ----------
pricing_fields = [
    num("units_sold", "Units sold", 100, 1, 1000000), num("unit_price_usd", "Unit price charged (USD)", 10.0, 0.01, 100000.0, 0.5),
    num("list_price_usd", "List price (USD)", 10.0, 0.01, 100000.0, 0.5), num("unit_cost_usd", "Unit cost (USD)", 4.0, 0.01, 100000.0, 0.5),
    num("discount_pct", "Discount (%)", 0, 0, 100), num("total_amount_usd", "Invoice total (USD)", 1000.0, 0.0, 100000000.0, 10.0,
                                                          "Normally units sold x unit price charged.")]
def pricing_row(v):
    expected = v["units_sold"] * v["unit_price_usd"]
    return {"price_to_list": v["unit_price_usd"] / v["list_price_usd"], "price_to_cost": v["unit_price_usd"] / v["unit_cost_usd"],
            "total_to_expected": v["total_amount_usd"] / expected if expected else np.nan, "log_total_amount": math.log1p(v["total_amount_usd"]),
            "discount_pct": v["discount_pct"], "log_units": math.log1p(v["units_sold"]), "log_price": math.log1p(v["unit_price_usd"])}

# ---------- One entry per page ----------
PAGES = {
    "Batch Quality": {"folder": "batch_rejection", "fields": batch_fields, "row": batch_row, "kind": "risk", "what": "Chance this batch is rejected at quality control",
        "intro": "Describe a batch before release. The model estimates the chance that it will be rejected, so QA can decide how closely to inspect it.",
        "actions": {"High": "Inspect first. Do a full inspection before release.", "Medium": "Inspect if capacity allows.", "Low": "Release as normal."}},
    "Demand Planning": {"folder": "demand_forecast", "fields": demand_fields, "row": demand_row, "kind": "forecast", "what": "Expected units for this sale",
        "intro": "Describe a planned sale. The model forecasts the number of units, to help plan production and stock. Forecasts are most reliable when added up across many sales."},
    "Cold-Chain Protection": {"folder": "cold_chain", "fields": cold_fields, "row": cold_row, "kind": "risk", "what": "Chance of a temperature excursion in transit",
        "intro": "Describe a shipment. The model estimates the chance of a temperature excursion, so logistics can decide where to upgrade packaging or routes.",
        "actions": {"High": "Protect first: upgrade the packaging or route before it ships.", "Medium": "Protect if budget allows.", "Low": "Ship as normal."}},
    "Drug Safety": {"folder": "adverse_events", "fields": safety_fields, "row": safety_row, "kind": "risk", "what": "Chance this report describes a serious event",
        "intro": "Describe an adverse event report. The model estimates the chance it is serious, so the safety team can decide which reports to review first.",
        "actions": {"High": "Review first. Send to the safety team now.", "Medium": "Review next.", "Low": "Standard queue."}},
    "Pricing Control": {"folder": "pricing_anomalies", "fields": pricing_fields, "row": pricing_row, "kind": "anomaly", "what": "Is this invoice priced unusually?",
        "intro": "Enter an invoice. The model and a simple price rule check whether it looks unusual, so finance can decide which invoices to review."}}

# =============================================================================
# PART 4. DRAWING THE PAGES
# =============================================================================
def draw_field(field, key):
    """Show one question and return the answer."""
    if field["kind"] == "num":
        is_int = all(isinstance(x, int) for x in (field["default"], field["low"], field["high"], field["step"]))
        cast = int if is_int else float
        return st.number_input(field["label"], cast(field["low"]), cast(field["high"]), cast(field["default"]), cast(field["step"]), help=field["help"] or None, key=key)
    if field["kind"] == "pick":
        return st.selectbox(field["label"], field["options"], index=field["options"].index(field["default"]), help=field["help"] or None, key=key)
    if field["kind"] == "yesno":
        return int(st.radio(field["label"], ["No", "Yes"], index=int(field["default"]), horizontal=True, help=field["help"] or None, key=key) == "Yes")
    return st.date_input(field["label"], field["default"], help=field["help"] or None, key=key)

def show_result(page, raw_row):
    """Score one case and show the answer."""
    model, scaler, info = load_files(page["folder"])
    X = prepare(pd.DataFrame([raw_row]), info, scaler)

    if page["kind"] == "risk":
        p = float(model.predict_proba(X)[0, 1]); tier = risk_tier(p, info)
        left, right = st.columns([1, 2])
        left.metric(page["what"], f"{p:.0%}"); left.progress(min(p, 1.0))
        message = f"**{tier} risk.** {page['actions'][tier]}"
        if tier == "High":
            right.error(message)
        elif tier == "Medium":
            right.warning(message)
        else:
            right.success(message)
        right.caption(f"High risk starts at {info['risk_tiers']['high_from']:.0%} and Medium at {info['risk_tiers']['medium_from']:.0%}. "
                      "These cut-offs were set when the model was evaluated.")

    elif page["kind"] == "forecast":
        units = float(np.expm1(model.predict(X))[0]); typical = info["metrics"].get("error % of units (WAPE)", 21.0) / 100
        left, right = st.columns([1, 2])
        left.metric(page["what"], f"{units:,.0f} units")
        right.info(f"Typical error on a single sale is about ±{typical:.0%}, so expect roughly {units * (1 - typical):,.0f} to {units * (1 + typical):,.0f} units. "
                   "Adding up forecasts across a product area and month is more accurate (about ±10% in testing).")

    else:                                                       # pricing control
        score, flagged = float(-model.score_samples(X)[0]), model.predict(X)[0] == -1
        price_to_list, price_to_cost, total_ratio = raw_row["price_to_list"], raw_row["price_to_cost"], raw_row["total_to_expected"]
        reasons = []
        if price_to_list < 0.5 or price_to_list > 1.5:
            reasons.append(f"price is {price_to_list:.2f}x the list price (normal is between 0.5x and 1.5x)")
        if price_to_cost < 1:
            reasons.append(f"price is below cost ({price_to_cost:.2f}x cost)")
        if abs(total_ratio - 1) > 0.05:
            reasons.append(f"invoice total is {total_ratio:.2f}x units x price (should be about 1.00x)")
        if flagged:
            reasons.append("the statistical model finds this combination unusual")
        left, right = st.columns([1, 2])
        left.metric("Anomaly score (higher = more unusual)", f"{score:.2f}")
        if reasons or flagged:
            right.error("**Send to finance for review.** " + "; ".join(reasons).capitalize() + ".")
        else:
            right.success("**Looks normal.** Price, cost and total are all in the usual range.")

    factors = main_factors(model, X, info)
    if factors:
        st.write("**Main factors in this result:** " + "; ".join(f"{name} ({effect} the result)" for name, effect in factors))

def single_case_tab(title, page):
    """A form with all the questions, and a Predict button."""
    with st.form(f"form_{title}"):
        columns = st.columns(3)
        answers = {}
        for i, field in enumerate(page["fields"]):
            with columns[i % 3]:
                answers[field["name"]] = draw_field(field, f"{title}_{field['name']}")
        submitted = st.form_submit_button("Predict", type="primary")
    if submitted:
        show_result(page, page["row"](answers))

def many_cases_tab(title, page):
    """Score a whole CSV file: one row per case, same questions as the form."""
    names = [f["name"] for f in page["fields"]]
    template = pd.DataFrame([{f["name"]: (f["default"] if f["kind"] != "yesno" else int(f["default"])) for f in page["fields"]}])
    st.write("1. Download the template, fill in one row per case (Yes/No questions use 1 for Yes and 0 for No).")
    st.download_button("Download CSV template", template.to_csv(index=False), f"{title.lower().replace(' ', '_')}_template.csv", "text/csv", key=f"tpl_{title}")
    upload = st.file_uploader("2. Upload your filled CSV", type="csv", key=f"upload_{title}")
    if upload is None:
        return
    data = pd.read_csv(upload)
    missing = [n for n in names if n not in data.columns]
    if missing:
        st.error("These columns are missing from your file: " + ", ".join(missing))
        return
    for field in page["fields"]:                                  # tidy the types
        if field["kind"] == "when":
            data[field["name"]] = pd.to_datetime(data[field["name"]]).dt.date
        elif field["kind"] == "yesno":
            data[field["name"]] = data[field["name"]].replace({"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}).astype(int)
    model, scaler, info = load_files(page["folder"])
    raw = pd.DataFrame([page["row"](row) for row in data.to_dict("records")])
    X = prepare(raw, info, scaler)
    result = data.copy()
    if page["kind"] == "risk":
        p = model.predict_proba(X)[:, 1]
        result["risk"] = (p * 100).round(1).astype(str) + "%"; result["tier"] = [risk_tier(x, info) for x in p]
        result["recommended action"] = result["tier"].map(page["actions"])
    elif page["kind"] == "forecast":
        result["forecast_units"] = np.expm1(model.predict(X)).round(0)
    else:
        rule = (raw["price_to_list"] < 0.5) | (raw["price_to_list"] > 1.5) | (raw["price_to_cost"] < 1) | ((raw["total_to_expected"] - 1).abs() > 0.05)
        result["anomaly_score"] = (-model.score_samples(X)).round(3)
        result["send_to_finance"] = np.where(rule | (model.predict(X) == -1), "Yes", "No")
    st.dataframe(result)
    st.download_button("Download results", result.to_csv(index=False), "results.csv", "text/csv", key=f"download_{title}")

def home_page():
    st.title("Pharmaceutical Business Operations Prediction System")
    st.caption("Veridian Pharma Global | five predictive models that support day-to-day decisions")
    st.write("Pick a page on the left. Each page answers one business question. You can score a single case with a form, or many cases at once by uploading a CSV file.")
    rows = []
    for title, page in PAGES.items():
        _, _, info = load_files(page["folder"])
        m = info["metrics"]
        if page["kind"] == "risk":
            result = f"ROC AUC {m['roc_auc']:.2f} on unseen data (0.5 = guessing, 1.0 = perfect)"
        elif page["kind"] == "forecast":
            result = f"Average error {m['avg error (units)']:.0f} units per sale ({m['error % of units (WAPE)']:.0f}%)"
        else:
            result = f"Catches {m['recall_%']:.0f}% of anomalies, {m['precision_%']:.0f}% of flags are right"
        rows.append({"Page": title, "Question": page["what"], "Model": info["model_name"], "Test result": result})
    st.table(pd.DataFrame(rows).set_index("Page"))
    st.warning("These models support decisions; they do not replace people. Use the result to decide where to look first, and keep a person in the loop for every flagged case.")
    _, _, info = load_files("batch_rejection")
    if info["sklearn_version"] != sklearn.__version__:
        st.warning(f"The models were trained with scikit-learn {info['sklearn_version']}, but this app is running {sklearn.__version__}. "
                   "Install the training version if you see errors or odd results.")
    st.caption(f"Model files loaded from: {ROOT}")

def objective_page(title):
    page = PAGES[title]
    _, _, info = load_files(page["folder"])
    st.title(title)
    st.write(page["intro"])
    st.caption(f"Model: {info['model_name']} | trained on {info['trained_on']}")
    one, many = st.tabs(["Single case", "Many cases (CSV file)"])
    with one:
        single_case_tab(title, page)
    with many:
        many_cases_tab(title, page)

# ---------- The menu ----------
st.sidebar.title("Pharma Operations Prediction System")
choice = st.sidebar.radio("Go to", ["Home"] + list(PAGES))
if choice == "Home":
    home_page()
else:
    objective_page(choice)