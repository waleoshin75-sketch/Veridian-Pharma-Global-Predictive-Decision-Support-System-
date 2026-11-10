# Pharmaceutical Business Operations Prediction System

**Veridian Pharma Global | Business Operations Analytics**

**Python | pandas | scikit-learn | Streamlit** | Seven messy tables turned into five working prediction models and a live web app

[Live Streamlit App](YOUR-STREAMLIT-LINK) | [GitHub Repository](YOUR-GITHUB-LINK)

---

## At a glance

I took seven dirty operational tables, cleaned and modelled them, then built one prediction model for each of five business problems. All five run inside a Streamlit app where a user fills in a form (or uploads a CSV) and gets a risk score and a recommended action.

| # | Business problem | Best model | Result on unseen data |
|---|---|---|---|
| 1 | Which batches will be rejected? | Logistic Regression | Inspecting 47% of batches catches 86% of rejections |
| 2 | How many units will we sell? | Linear Regression (Ridge) | 21% error per sale, 10% by product area and month, against 337 average error units for no model |
| 3 | Which shipments will overheat? | Logistic Regression | Protecting 71% of shipments prevents up to 97% of excursions |
| 4 | Which safety reports are serious? | Logistic Regression | Reviewing 54% of reports first reaches 99% of serious cases |
| 5 | Which invoices are priced wrongly? | Elliptic Envelope | 36 of 40 anomalies caught, 4 false alarms. A simple price rule caught all 40. |

The data is fictional and was built with realistic quality problems, so every step of a real workflow is shown. Money figures use example costs that you can change.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Dictionary](#3-data-dictionary)
4. [Data Quality Audit](#4-data-quality-audit)
5. [Data Cleaning and Error Checking](#5-data-cleaning-and-error-checking)
6. [Feature Engineering](#6-feature-engineering)
7. [Create Relationship Table](#7-create-relationship-table)
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)
9. [Analytical Base Table](#9-analytical-base-table)
10. [Encode Categorical Columns](#10-encode-categorical-columns)
11. [Separate X and y](#11-separate-x-and-y)
12. [Train Test Split](#12-train-test-split)
13. [Feature Scaling of Numerical Columns](#13-feature-scaling-of-numerical-columns)
14. [Train Model](#14-train-model)
15. [Predict X_test](#15-predict-x_test)
16. [Compare X and y](#16-compare-x-and-y)
17. [Evaluate Model](#17-evaluate-model)
18. [Business Analysis](#18-business-analysis)
19. [Save Model, Scaler and Features](#19-save-model-scaler-and-features)
20. [Download Model, Scaler and Features](#20-download-model-scaler-and-features)
21. [Key Findings and Insights](#21-key-findings-and-insights)
22. [Recommendations](#22-recommendations)
23. [Impact](#23-impact)
24. [Conclusion](#24-conclusion)

---

---

## 1. Project Overview

Veridian Pharma Global is a fictional pharmaceutical company. Its manufacturing, sales, shipping and safety data sits in seven separate, untidy tables. Five decisions depend on that data, and each one becomes a model:

| # | Business objective | ML task | Target |
|---|---|---|---|
| 1 | Reduce batch rejections | Classification | Batch rejected at quality control (15.2% of batches) |
| 2 | Plan production from demand | Regression | Units sold per sale (modelled on a log scale) |
| 3 | Prevent cold-chain failures | Classification | Shipment had a temperature excursion (15.8%) |
| 4 | Prioritise serious adverse events | Classification | Report is severe or life-threatening (20.0%) |
| 5 | Find pricing and invoice anomalies | Unsupervised anomaly detection | None. An answer key is used only to score the models. |

**The workflow:** data quality audit, cleaning, feature engineering, relationship tables, EDA, then for each objective an analytical base table, encoding, X/y split, train/test split, scaling, training four models, prediction, evaluation, business analysis, saving and deployment in a Streamlit app.

**Rules I held myself to**
- At least four models per objective, with the winner chosen on training evidence and confirmed on unseen data.
- No leakage: columns that reveal the answer (release status, severity, outcome and others) never enter a model.
- Every result is turned into a decision, with a cost-based view of what acting on it is worth.

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib, Streamlit.

### Repository structure

```
├── README.md
├── app.py                          the Streamlit app (section 20)
├── images/                         every chart shown in this README
├── pharma_raw/                     the seven raw (dirty) tables and data_dictionary.csv
├── pharma_model/                   the five modelling tables, one per objective
├── models/                         model.joblib, scaler.joblib, features.joblib for each objective
├── deployment/                     veridian_pharma_models.zip
└── notebook                        all the code shown below, one cell per section
```

---

## 2. Dataset Overview

Seven tables arrive dirty. After cleaning, 177 duplicate rows are gone.

| Table | Role | Rows (raw) | Rows (clean) | What it holds |
|---|---|---|---|---|
| `dim_customers` | Dimension | 308 | 300 | Customer, country, type, credit rating |
| `dim_plants` | Dimension | 11 | 10 | Plant, country, GMP status, automation level |
| `dim_products` | Dimension | 43 | 40 | Product, therapeutic area, form, dose, cost, list price |
| `fact_batches` | Fact | 4,050 | 4,000 | Manufacturing batches and quality outcome |
| `fact_sales` | Fact | 6,060 | 6,000 | Orders: units, price, discount, channel |
| `fact_shipments` | Fact | 3,530 | 3,500 | Delivery, packaging, temperature outcome |
| `fact_adverse_events` | Fact | 1,825 | 1,800 | Patient safety reports and severity |

After the relationship phase there is also a `dim_date` table (1,449 days, 2022 to 2025) and five modelling tables, one per objective.

---

## 3. Data Dictionary

67 columns across the seven tables are documented in [`data_dictionary.csv`](pharma_raw/data_dictionary.csv), each with its meaning and known data problems. The columns that matter most:

| Table | Key columns |
|---|---|
| `dim_products` | `product_id`, `therapeutic_area`, `formulation` (Tablet, Capsule, Injection, Syrup), `dosage_mg`, `shelf_life_months`, `requires_cold_chain`, `unit_cost_usd`, `list_price_usd` |
| `dim_plants` | `plant_id`, `plant_name`, `country`, `gmp_certified`, `year_built`, `automation_level` |
| `dim_customers` | `customer_id`, `country`, `customer_type` (Hospital, Retail Pharmacy, Wholesale Distributor, Government Tender), `credit_rating` |
| `fact_batches` | `batch_id`, `manufacture_date`, `batch_size_units`, `raw_material_purity_pct`, `max_temp_deviation_c`, `equipment_age_years`, `night_shift`, `operator_experience_years`, `release_status` (**target 1**) |
| `fact_sales` | `sale_id`, `sale_date`, `units_sold` (**target 2**), `unit_price_usd`, `discount_pct`, `total_amount_usd`, `sales_channel` |
| `fact_shipments` | `shipment_id`, `transport_mode`, `packaging_type`, `cold_chain_required`, `transit_days`, `avg_ambient_temp_c`, `temperature_excursion` (**target 3**) |
| `fact_adverse_events` | `ae_id`, `patient_age`, `patient_sex`, `concomitant_medications`, `hepatic_impairment`, `days_to_onset`, `reporter_type`, `severity` (**target 4**) |

**Leakage columns** exist in the data on purpose and are kept out of every model: `rejection_reason` (objective 1), `product_written_off` (3), and `hospitalised` and `outcome` (4).

---

## 4. Data Quality Audit

The audit changes nothing. It counts every problem first, one reusable function per table.

| Problem | What the audit found |
|---|---|
| Duplicate rows | 177 across the seven tables (batches 50, sales 60, shipments 30, adverse events 25, customers 8, products 3, plants 1) |
| Missing values | Ambient temperature 4.9%, onset time 5.1%, temperature deviation 4.0%, credit rating 4.9%, customer on sales 0.5% |
| Yes/No chaos | Ten spellings of the same two answers (`Yes`, `Y`, `TRUE`, `1`, `yes`, `No`, `N`, `FALSE`, `0`, `no`) in eight columns |
| Country spellings | More than 30 spellings of 10 countries (`NG`, `Deutschland`, `U.A.E`, `Brasil`, `Great Britain`) |
| Dates | Six date formats plus junk values (`??`), in every date column |
| Impossible dates | 95 shipments delivered before they shipped; 32 ship dates and 31 delivery dates unreadable |
| Numbers as text | Purity as `99%`, transit as `3 days`, prices with currency symbols |
| Impossible values | Negative or zero units, purity of 150, equipment aged 250, ambient temperature of 999 |
| Case and spacing | `prd-126` vs `PRD-126`, trailing spaces, mixed case categories |

### Full code

```python
# %% CELL 1 - Imports and the seven tables
import re
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATA_DIR = "pharma_raw"

customers      = pd.read_csv(f"{DATA_DIR}/dim_customers.csv")
plants         = pd.read_csv(f"{DATA_DIR}/dim_plants.csv")
products       = pd.read_csv(f"{DATA_DIR}/dim_products.csv")
batches        = pd.read_csv(f"{DATA_DIR}/fact_batches.csv")
sales          = pd.read_csv(f"{DATA_DIR}/fact_sales.csv")
shipments      = pd.read_csv(f"{DATA_DIR}/fact_shipments.csv")
adverse_events = pd.read_csv(f"{DATA_DIR}/fact_adverse_events.csv")


# %% CELL 2 - Reusable audit function (run once, used for every table)
def audit_table(df, name, id_col):
    """Print shape, missing values, duplicate rows and duplicate IDs for one table."""
    print("=" * 75)
    print(f"{name.upper()}   rows={df.shape[0]}   columns={df.shape[1]}")
    print("=" * 75)

    # 1. Missing values
    miss = df.isna().sum()
    miss_tbl = pd.DataFrame({"missing": miss, "missing_%": (miss / len(df) * 100).round(2)})
    miss_tbl = miss_tbl[miss_tbl["missing"] > 0]
    print("\n[1] Missing values")
    print(miss_tbl if len(miss_tbl) else "    none")

    # 2. Duplicates
    print("\n[2] Duplicates")
    print(f"    fully duplicated rows : {df.duplicated().sum()}")
    print(f"    repeated {id_col:<12}: {df[id_col].duplicated().sum()}")

    # 3. Data types
    print("\n[3] Data types")
    print(df.dtypes.astype(str).to_string())
    print()


# %% CELL 3 - dim_customers
audit_table(customers, "dim_customers", "customer_id")
customers.head(10)


# %% CELL 4 - dim_plants
audit_table(plants, "dim_plants", "plant_id")
plants.head(10)


# %% CELL 5 - dim_products
audit_table(products, "dim_products", "product_id")
products.head(10)


# %% CELL 6 - fact_batches
audit_table(batches, "fact_batches", "batch_id")
batches.head(10)


# %% CELL 7 - fact_sales
audit_table(sales, "fact_sales", "sale_id")
sales.head(10)


# %% CELL 8 - fact_shipments
audit_table(shipments, "fact_shipments", "shipment_id")
shipments.head(10)


# %% CELL 9 - fact_adverse_events
audit_table(adverse_events, "fact_adverse_events", "ae_id")
adverse_events.head(10)


# %% CELL 10 - Numbers stored as text (units, %, $, "days" glued to the value)
def text_in_numeric(df, col):
    """Return the values in a column that cannot be read as a plain number."""
    s = df[col]
    bad = s[pd.to_numeric(s, errors="coerce").isna() & s.notna()]
    return bad

numeric_checks = [
    ("dim_products",        products,       "dosage_mg"),
    ("dim_products",        products,       "unit_cost_usd"),
    ("dim_products",        products,       "list_price_usd"),
    ("fact_batches",        batches,        "raw_material_purity_pct"),
    ("fact_sales",          sales,          "unit_price_usd"),
    ("fact_sales",          sales,          "discount_pct"),
    ("fact_sales",          sales,          "total_amount_usd"),
    ("fact_shipments",      shipments,      "transit_days"),
    ("fact_adverse_events", adverse_events, "patient_age"),
]

rows = []
for tname, df, col in numeric_checks:
    bad = text_in_numeric(df, col)
    rows.append({
        "table": tname, "column": col, "dtype": str(df[col].dtype),
        "rows_with_text": len(bad),
        "examples": ", ".join(bad.astype(str).unique()[:4]),
    })
pd.DataFrame(rows)


# %% CELL 11 - Impossible and extreme numeric values
def numeric_view(df, col):
    s = pd.to_numeric(df[col], errors="coerce")
    return s

limits = [
    # table, df, column, rule description, mask function
    ("fact_sales",     sales,   "units_sold",          "units_sold <= 0",            lambda s: s <= 0),
    ("fact_batches",   batches, "batch_size_units",    "batch_size_units <= 0",      lambda s: s <= 0),
    ("fact_batches",   batches, "batch_size_units",    "batch_size_units > 1,000,000", lambda s: s > 1_000_000),
    ("fact_batches",   batches, "equipment_age_years", "equipment_age_years < 0",    lambda s: s < 0),
    ("fact_batches",   batches, "equipment_age_years", "equipment_age_years > 60",   lambda s: s > 60),
    ("fact_batches",   batches, "raw_material_purity_pct", "purity > 100 or < 80",   lambda s: (s > 100) | (s < 80)),
    ("dim_plants",     plants,  "year_built",          "year_built < 1900 or > 2026", lambda s: (s < 1900) | (s > 2026)),
    ("fact_adverse_events", adverse_events, "patient_age", "patient_age < 0 or > 110", lambda s: (s < 0) | (s > 110)),
    ("fact_shipments", shipments, "transit_days",      "transit_days < 0 or > 60",   lambda s: (s < 0) | (s > 60)),
]

rows = []
for tname, df, col, rule, mask in limits:
    s = numeric_view(df, col)
    hit = s[mask(s).fillna(False)]
    rows.append({
        "table": tname, "rule": rule, "rows_flagged": len(hit),
        "examples": ", ".join(hit.astype(str).unique()[:5]),
    })
pd.DataFrame(rows)


# %% CELL 12 - Inconsistent spelling, case and spacing in category columns
category_checks = [
    ("dim_customers",       customers,      "country"),
    ("dim_customers",       customers,      "customer_type"),
    ("dim_customers",       customers,      "credit_rating"),
    ("dim_plants",          plants,         "automation_level"),
    ("dim_products",        products,       "therapeutic_area"),
    ("dim_products",        products,       "formulation"),
    ("fact_batches",        batches,        "release_status"),
    ("fact_sales",          sales,          "sales_channel"),
    ("fact_shipments",      shipments,      "transport_mode"),
    ("fact_adverse_events", adverse_events, "severity"),
    ("fact_adverse_events", adverse_events, "patient_sex"),
]

rows = []
for tname, df, col in category_checks:
    s = df[col].dropna().astype(str)
    raw_unique = s.nunique()
    tidy_unique = s.str.strip().str.lower().nunique()
    rows.append({
        "table": tname, "column": col,
        "distinct_as_stored": raw_unique,
        "distinct_after_trim_and_lowercase": tidy_unique,
        "extra_variants": raw_unique - tidy_unique,
    })
pd.DataFrame(rows)


# %% CELL 13 - Yes / No columns written in many ways (Yes, Y, TRUE, 1 ...)
yes_no_checks = [
    ("dim_plants",          plants,         "gmp_certified"),
    ("dim_products",        products,       "requires_cold_chain"),
    ("fact_batches",        batches,        "night_shift"),
    ("fact_shipments",      shipments,      "cold_chain_required"),
    ("fact_shipments",      shipments,      "temperature_excursion"),
    ("fact_shipments",      shipments,      "product_written_off"),
    ("fact_adverse_events", adverse_events, "hepatic_impairment"),
    ("fact_adverse_events", adverse_events, "hospitalised"),
]

for tname, df, col in yes_no_checks:
    print(f"{tname}.{col}")
    print("   ", df[col].value_counts(dropna=False).to_dict())


# %% CELL 14 - Look closer at the country column (the messiest one)
print(customers["country"].value_counts(dropna=False).to_string())


# %% CELL 15 - Mixed date formats
date_checks = [
    ("dim_customers",       customers,      "onboarding_date"),
    ("fact_batches",        batches,        "manufacture_date"),
    ("fact_sales",          sales,          "sale_date"),
    ("fact_shipments",      shipments,      "ship_date"),
    ("fact_shipments",      shipments,      "delivery_date"),
    ("fact_adverse_events", adverse_events, "report_date"),
]

def date_pattern(x):
    """Turn a date string into a pattern, e.g. 2024-03-12 -> 9999-99-99"""
    if pd.isna(x):
        return "missing"
    p = re.sub(r"\d", "9", str(x).strip())
    p = re.sub(r"[A-Za-z]+", "Month", p)
    return p

for tname, df, col in date_checks:
    print(f"{tname}.{col}")
    print(df[col].map(date_pattern).value_counts().to_string())
    print()


# %% CELL 16 - Dates that make no sense (delivery before shipping)
ship_d = pd.to_datetime(shipments["ship_date"], errors="coerce", format="mixed")
deliv_d = pd.to_datetime(shipments["delivery_date"], errors="coerce", format="mixed")

print("ship_date not readable     :", ship_d.isna().sum() - shipments["ship_date"].isna().sum())
print("delivery_date not readable :", deliv_d.isna().sum() - shipments["delivery_date"].isna().sum())
print("delivery before shipping   :", (deliv_d < ship_d).sum())


# %% CELL 17 - Orphan keys (child IDs that do not exist in the parent table)
fk_checks = [
    ("fact_batches.product_id",         batches["product_id"],         products["product_id"]),
    ("fact_batches.plant_id",           batches["plant_id"],           plants["plant_id"]),
    ("fact_sales.customer_id",          sales["customer_id"],          customers["customer_id"]),
    ("fact_sales.product_id",           sales["product_id"],           products["product_id"]),
    ("fact_shipments.sale_id",          shipments["sale_id"],          sales["sale_id"]),
    ("fact_adverse_events.batch_id",    adverse_events["batch_id"],    batches["batch_id"]),
    ("fact_adverse_events.product_id",  adverse_events["product_id"],  products["product_id"]),
]

rows = []
for label, child, parent in fk_checks:
    missing_key = child.isna().sum()
    orphan = (~child.dropna().isin(parent)).sum()
    rows.append({
        "relationship": label,
        "rows": len(child),
        "key_is_blank": missing_key,
        "key_not_in_parent": orphan,
    })
pd.DataFrame(rows)


# %% CELL 18 - Key formatting: IDs written in lower case or with extra spaces
key_cols = [
    ("fact_batches",        batches,        "product_id"),
    ("fact_batches",        batches,        "plant_id"),
    ("fact_sales",          sales,          "customer_id"),
    ("fact_sales",          sales,          "product_id"),
    ("fact_shipments",      shipments,      "sale_id"),
    ("fact_adverse_events", adverse_events, "batch_id"),
    ("fact_adverse_events", adverse_events, "product_id"),
]
rows = []
for tname, df, col in key_cols:
    s = df[col].dropna().astype(str)
    rows.append({"table": tname, "key": col,
                 "lower_case_ids": int((s != s.str.upper()).sum()),
                 "ids_with_extra_spaces": int((s != s.str.strip()).sum()),
                 "example": ", ".join(s[s != s.str.upper()].unique()[:3])})
pd.DataFrame(rows)


# %% CELL 19 - Data quality scorecard (the table for your README)
tables = {
    "dim_customers": (customers, "customer_id"),
    "dim_plants": (plants, "plant_id"),
    "dim_products": (products, "product_id"),
    "fact_batches": (batches, "batch_id"),
    "fact_sales": (sales, "sale_id"),
    "fact_shipments": (shipments, "shipment_id"),
    "fact_adverse_events": (adverse_events, "ae_id"),
}

rows = []
for name, (df, id_col) in tables.items():
    rows.append({
        "table": name,
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_ids": int(df[id_col].duplicated().sum()),
    })

scorecard = pd.DataFrame(rows)
scorecard
```

---

## 5. Data Cleaning and Error Checking

Cleaning is a set of small, reusable helpers applied table by table. Four of them do most of the work: `tidy_key`, `to_number`, `to_yes_no` and `parse_dates`.

**How impossible values were handled** (batches as the example): flag them as missing, then fill with the median of the same product or plant, so one bad value does not distort a group.

| Issue | Fix |
|---|---|
| Duplicate rows | Dropped on the ID column (177 rows) |
| Keys with different case or spaces | `tidy_key` on every ID column |
| Numbers stored as text | `to_number` |
| Yes/No variants | `to_yes_no` (1 or 0) |
| Country spellings | Mapped to 10 standard names |
| Mixed date formats | `parse_dates`, unreadable values left empty |
| Delivery before shipping | Dates repaired (16 rebuilt from transit time), the rest left missing |
| Impossible values | Set to missing, filled with the product or plant median |
| Blank rejection reason | Labelled `No Rejection` (a passed batch, not missing data) |

**Error checking:** the cleaned tables go through 16 automatic checks, and every one prints `OK`, for example no duplicate IDs, purity between 80 and 100, ambient temperature between -60 and 60, patient age between 0 and 110, yes/no columns only 0 or 1, and no delivery before shipping.

### Full code

```python
# %% CELL 1 - Imports and the seven raw tables
import os
import re
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

DATA_DIR = "pharma_raw"
CLEAN_DIR = "pharma_clean"
os.makedirs(CLEAN_DIR, exist_ok=True)

customers      = pd.read_csv(f"{DATA_DIR}/dim_customers.csv")
plants         = pd.read_csv(f"{DATA_DIR}/dim_plants.csv")
products       = pd.read_csv(f"{DATA_DIR}/dim_products.csv")
batches        = pd.read_csv(f"{DATA_DIR}/fact_batches.csv")
sales          = pd.read_csv(f"{DATA_DIR}/fact_sales.csv")
shipments      = pd.read_csv(f"{DATA_DIR}/fact_shipments.csv")
adverse_events = pd.read_csv(f"{DATA_DIR}/fact_adverse_events.csv")


# %% CELL 2 - Cleaning helper functions (run once, used by every table)
def tidy_text(s):
    """Trim spaces and collapse double spaces. Blanks stay missing."""
    s = s.astype("object")
    return s.where(s.isna(), s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True))


def tidy_key(s):
    """ID columns: trim spaces and upper-case, so 'prd-126' matches 'PRD-126'."""
    s = s.astype("object")
    return s.where(s.isna(), s.astype(str).str.strip().str.upper())


def to_number(s):
    """'$1,250.50', 'USD 9.63', '99%', '5d', '3 days', '17 004.44' -> float"""
    s = s.astype("object")
    txt = s.where(s.isna(), s.astype(str).str.lower())
    txt = txt.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(txt, errors="coerce")


def to_yes_no(s):
    """Yes / Y / TRUE / 1 -> 1 ; No / N / FALSE / 0 -> 0 ; anything else -> missing"""
    key = s.astype("object")
    key = key.where(key.isna(), key.astype(str).str.strip().str.lower())
    mapping = {"yes": 1, "y": 1, "true": 1, "1": 1, "1.0": 1,
               "no": 0, "n": 0, "false": 0, "0": 0, "0.0": 0}
    return key.map(mapping).astype("Int64")


def parse_dates(s):
    """
    Mixed date formats -> one proper datetime column.
    The audit showed:  dashes = MM-DD-YYYY,  slashes = DD/MM/YYYY.
    'pending' and '??' become missing (NaT).
    """
    s = s.astype("object")
    txt = s.where(s.isna(), s.astype(str).str.strip())
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    rules = [
        (r"^\d{4}-\d{2}-\d{2}$",           "%Y-%m-%d"),
        (r"^\d{4}/\d{2}/\d{2}$",           "%Y/%m/%d"),
        (r"^\d{2}-\d{2}-\d{4}$",           "%m-%d-%Y"),
        (r"^\d{2}/\d{2}/\d{4}$",           "%d/%m/%Y"),
        (r"^\d{1,2} [A-Za-z]+ \d{4}$",     "%d %B %Y"),
        (r"^[A-Za-z]+ \d{1,2}, \d{4}$",    "%B %d, %Y"),
    ]
    for pattern, fmt in rules:
        mask = txt.str.match(pattern, na=False)
        if mask.any():
            out.loc[mask] = pd.to_datetime(txt[mask], format=fmt, errors="coerce")
    return out


def fill_by_group(df, col, group_col):
    """Fill missing values with the median of the same group, then the overall median."""
    grp_median = df.groupby(group_col)[col].transform("median")
    return df[col].fillna(grp_median).fillna(df[col].median())


# %% CELL 3 - Clean dim_customers
customers_clean = customers.copy()

# key columns first: trim and upper-case (the audit found keys such as 'prd-126')
for col in ["customer_id"]:
    customers_clean[col] = tidy_key(customers_clean[col])

# 3.1 remove repeated customers
customers_clean = customers_clean.drop_duplicates(subset="customer_id", keep="first")

# 3.2 text columns
for col in ["customer_name", "country", "customer_type", "credit_rating"]:
    customers_clean[col] = tidy_text(customers_clean[col])

# 3.3 country: 42 spellings -> 10 countries
country_map = {
    "singapore": "Singapore", "sg": "Singapore",
    "switzerland": "Switzerland", "ch": "Switzerland", "suisse": "Switzerland",
    "united states": "United States", "us": "United States", "usa": "United States",
    "u.s.a": "United States",
    "nigeria": "Nigeria", "ng": "Nigeria",
    "south africa": "South Africa", "rsa": "South Africa", "s. africa": "South Africa",
    "germany": "Germany", "deutschland": "Germany",
    "brazil": "Brazil", "brasil": "Brazil",
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "u.k.": "United Kingdom",
    "great britain": "United Kingdom",
    "uae": "UAE", "u.a.e": "UAE", "united arab emirates": "UAE",
    "india": "India", "in": "India",
}
customers_clean["country"] = customers_clean["country"].str.lower().map(country_map)

# 3.4 customer type and credit rating
customers_clean["customer_type"] = customers_clean["customer_type"].str.title()
customers_clean["credit_rating"] = customers_clean["credit_rating"].str.upper().fillna("Unknown")

# 3.5 date
customers_clean["onboarding_date"] = parse_dates(customers_clean["onboarding_date"])

customers_clean = customers_clean.reset_index(drop=True)
print("dim_customers:", customers.shape, "->", customers_clean.shape)
customers_clean.head(10)


# %% CELL 4 - Clean dim_plants
plants_clean = plants.copy()

# key columns first: trim and upper-case (the audit found keys such as 'prd-126')
for col in ["plant_id"]:
    plants_clean[col] = tidy_key(plants_clean[col])

plants_clean = plants_clean.drop_duplicates(subset="plant_id", keep="first")

for col in ["plant_name", "country", "automation_level"]:
    plants_clean[col] = tidy_text(plants_clean[col])
plants_clean["automation_level"] = plants_clean["automation_level"].str.title()

plants_clean["gmp_certified"] = to_yes_no(plants_clean["gmp_certified"])

# a plant cannot be built before 1900 or after 2026
bad_year = (plants_clean["year_built"] < 1900) | (plants_clean["year_built"] > 2026)
plants_clean.loc[bad_year, "year_built"] = np.nan
plants_clean["year_built"] = plants_clean["year_built"].fillna(plants_clean["year_built"].median()).round().astype(int)

plants_clean = plants_clean.reset_index(drop=True)
print("dim_plants:", plants.shape, "->", plants_clean.shape)
plants_clean


# %% CELL 5 - Clean dim_products
products_clean = products.copy()

# key columns first: trim and upper-case (the audit found keys such as 'prd-126')
for col in ["product_id"]:
    products_clean[col] = tidy_key(products_clean[col])

products_clean = products_clean.drop_duplicates(subset="product_id", keep="first")

for col in ["product_name", "therapeutic_area", "formulation"]:
    products_clean[col] = tidy_text(products_clean[col])
products_clean["therapeutic_area"] = products_clean["therapeutic_area"].str.title().replace({"Anti-Infective": "Anti-infective"})
products_clean["formulation"] = products_clean["formulation"].str.title()

products_clean["requires_cold_chain"] = to_yes_no(products_clean["requires_cold_chain"])

# money and dose stored as text -> numbers
products_clean["dosage_mg"]      = to_number(products_clean["dosage_mg"])
products_clean["unit_cost_usd"]  = to_number(products_clean["unit_cost_usd"])
products_clean["list_price_usd"] = to_number(products_clean["list_price_usd"])

# negative dose is impossible -> missing, then filled
products_clean.loc[products_clean["dosage_mg"] <= 0, "dosage_mg"] = np.nan
products_clean["dosage_mg"] = fill_by_group(products_clean, "dosage_mg", "formulation")
products_clean["shelf_life_months"] = fill_by_group(products_clean, "shelf_life_months", "therapeutic_area")

products_clean = products_clean.reset_index(drop=True)
print("dim_products:", products.shape, "->", products_clean.shape)
products_clean.head(10)


# %% CELL 6 - Clean fact_batches
batches_clean = batches.copy()

# key columns first: trim and upper-case (the audit found keys such as 'prd-126')
for col in ["batch_id", "product_id", "plant_id"]:
    batches_clean[col] = tidy_key(batches_clean[col])

# 6.1 duplicates
batches_clean = batches_clean.drop_duplicates(subset="batch_id", keep="first")

# 6.2 text columns
for col in ["release_status", "rejection_reason", "plant_id"]:
    batches_clean[col] = tidy_text(batches_clean[col])

# 6.3 dates
batches_clean["manufacture_date"] = parse_dates(batches_clean["manufacture_date"])

# 6.4 release status: Released / Rejected only ("Reject" -> "Rejected")
status = batches_clean["release_status"].str.lower()
batches_clean["release_status"] = status.map({"released": "Released", "rejected": "Rejected", "reject": "Rejected"})

# 6.5 rejection reason: blank means the batch passed (labelled 'No Rejection' because
#     the word 'None' is read back as missing by pandas when the CSV is reloaded)
batches_clean["rejection_reason"] = batches_clean["rejection_reason"].str.title().fillna("No Rejection")

# 6.6 missing plant -> its own label (kept for the relationship phase)
batches_clean["plant_id"] = batches_clean["plant_id"].fillna("UNKNOWN")

# 6.7 numbers stored as text
batches_clean["raw_material_purity_pct"] = to_number(batches_clean["raw_material_purity_pct"])

# 6.8 impossible values -> missing -> filled with a sensible median
batches_clean.loc[(batches_clean["batch_size_units"] <= 0) | (batches_clean["batch_size_units"] > 1_000_000), "batch_size_units"] = np.nan
batches_clean.loc[(batches_clean["raw_material_purity_pct"] < 80) | (batches_clean["raw_material_purity_pct"] > 100), "raw_material_purity_pct"] = np.nan
batches_clean.loc[(batches_clean["equipment_age_years"] < 0) | (batches_clean["equipment_age_years"] > 60), "equipment_age_years"] = np.nan

batches_clean["batch_size_units"]         = fill_by_group(batches_clean, "batch_size_units", "product_id").round().astype(int)
batches_clean["raw_material_purity_pct"]  = fill_by_group(batches_clean, "raw_material_purity_pct", "product_id").round(2)
batches_clean["equipment_age_years"]      = fill_by_group(batches_clean, "equipment_age_years", "plant_id").round().astype(int)
batches_clean["max_temp_deviation_c"]     = fill_by_group(batches_clean, "max_temp_deviation_c", "product_id").round(2)

# 6.9 yes/no
batches_clean["night_shift"] = to_yes_no(batches_clean["night_shift"])

batches_clean = batches_clean.reset_index(drop=True)
print("fact_batches:", batches.shape, "->", batches_clean.shape)
batches_clean.head(10)


# %% CELL 7 - Clean fact_sales
sales_clean = sales.copy()

# key columns first: trim and upper-case (the audit found keys such as 'prd-126')
for col in ["sale_id", "customer_id", "product_id"]:
    sales_clean[col] = tidy_key(sales_clean[col])

sales_clean = sales_clean.drop_duplicates(subset="sale_id", keep="first")

sales_clean["sales_channel"] = tidy_text(sales_clean["sales_channel"]).str.lower().replace({"onlinepharmacy": "online pharmacy"}).str.title()
sales_clean["sale_date"] = parse_dates(sales_clean["sale_date"])

# unknown customer kept as its own label (decided in the relationship phase)
sales_clean["customer_id"] = sales_clean["customer_id"].fillna("UNKNOWN")

# money stored as text
sales_clean["unit_price_usd"]    = to_number(sales_clean["unit_price_usd"])
sales_clean["total_amount_usd"]  = to_number(sales_clean["total_amount_usd"])

# discount arrives as 0.15, 15.0 and "15%" -> always a percentage (15)
disc = to_number(sales_clean["discount_pct"])
sales_clean["discount_pct"] = np.where(disc <= 1, disc * 100, disc).round(1)
sales_clean.loc[(sales_clean["discount_pct"] < 0) | (sales_clean["discount_pct"] > 100), "discount_pct"] = np.nan

# a sale cannot have zero or negative units
sales_clean.loc[sales_clean["units_sold"] <= 0, "units_sold"] = np.nan
sales_clean["units_sold"] = fill_by_group(sales_clean, "units_sold", "product_id").round().astype(int)

sales_clean = sales_clean.reset_index(drop=True)
print("fact_sales:", sales.shape, "->", sales_clean.shape)
sales_clean.head(10)


# %% CELL 8 - Clean fact_shipments
shipments_clean = shipments.copy()

# key columns first: trim and upper-case (the audit found keys such as 'prd-126')
for col in ["shipment_id", "sale_id"]:
    shipments_clean[col] = tidy_key(shipments_clean[col])

shipments_clean = shipments_clean.drop_duplicates(subset="shipment_id", keep="first")

shipments_clean["transport_mode"] = tidy_text(shipments_clean["transport_mode"]).str.title()
shipments_clean["packaging_type"] = (tidy_text(shipments_clean["packaging_type"])
                                     .str.replace("_", " ").str.title())

shipments_clean["ship_date"]     = parse_dates(shipments_clean["ship_date"])
shipments_clean["delivery_date"] = parse_dates(shipments_clean["delivery_date"])

for col in ["cold_chain_required", "temperature_excursion", "product_written_off"]:
    shipments_clean[col] = to_yes_no(shipments_clean[col])

# transit days: text -> number, rebuilt from the two dates when blank
shipments_clean["transit_days"] = to_number(shipments_clean["transit_days"])
date_gap = (shipments_clean["delivery_date"] - shipments_clean["ship_date"]).dt.days
shipments_clean["transit_days"] = shipments_clean["transit_days"].fillna(date_gap.where(date_gap >= 0))
shipments_clean["transit_days"] = fill_by_group(shipments_clean, "transit_days", "transport_mode").round().astype(int)

# delivery before shipping is impossible -> rebuild delivery date from ship date + transit days
wrong_order = shipments_clean["delivery_date"] < shipments_clean["ship_date"]
shipments_clean.loc[wrong_order, "delivery_date"] = (
    shipments_clean.loc[wrong_order, "ship_date"] + pd.to_timedelta(shipments_clean.loc[wrong_order, "transit_days"], unit="D"))
print("delivery dates rebuilt:", int(wrong_order.sum()))

# -99 and 999 are placeholder codes, not real temperatures
temp = shipments_clean["avg_ambient_temp_c"]
shipments_clean.loc[(temp < -60) | (temp > 60), "avg_ambient_temp_c"] = np.nan
shipments_clean["avg_ambient_temp_c"] = fill_by_group(shipments_clean, "avg_ambient_temp_c", "transport_mode").round(1)

shipments_clean = shipments_clean.reset_index(drop=True)
print("fact_shipments:", shipments.shape, "->", shipments_clean.shape)
shipments_clean.head(10)


# %% CELL 9 - Clean fact_adverse_events
adverse_clean = adverse_events.copy()

# key columns first: trim and upper-case (the audit found keys such as 'prd-126')
for col in ["ae_id", "batch_id", "product_id"]:
    adverse_clean[col] = tidy_key(adverse_clean[col])

adverse_clean = adverse_clean.drop_duplicates(subset="ae_id", keep="first")

for col in ["patient_sex", "reporter_type", "severity", "outcome"]:
    adverse_clean[col] = tidy_text(adverse_clean[col])

adverse_clean["report_date"] = parse_dates(adverse_clean["report_date"])

adverse_clean["patient_sex"]   = adverse_clean["patient_sex"].str.lower().map(
    {"female": "Female", "f": "Female", "male": "Male", "m": "Male"})
adverse_clean["reporter_type"] = adverse_clean["reporter_type"].str.title()
adverse_clean["severity"]      = (adverse_clean["severity"].str.lower()
                                  .str.replace(" ", "-").str.capitalize())
adverse_clean["outcome"]       = adverse_clean["outcome"].str.lower().str.capitalize()

for col in ["hepatic_impairment", "hospitalised"]:
    adverse_clean[col] = to_yes_no(adverse_clean[col])

# patient age: "60 yrs" -> 60 ; -5 and 999 are impossible -> filled with the median
adverse_clean["patient_age"] = to_number(adverse_clean["patient_age"])
adverse_clean.loc[(adverse_clean["patient_age"] < 0) | (adverse_clean["patient_age"] > 110), "patient_age"] = np.nan
adverse_clean["patient_age"] = adverse_clean["patient_age"].fillna(adverse_clean["patient_age"].median()).round().astype(int)

# blank product_id: recover it from the batch the event belongs to
batch_to_product = batches_clean.drop_duplicates("batch_id").set_index("batch_id")["product_id"]
recovered = adverse_clean["batch_id"].map(batch_to_product)
adverse_clean["product_id"] = adverse_clean["product_id"].fillna(recovered).fillna("UNKNOWN")

# days_to_onset stays missing when unknown (filling it would invent medical facts)

adverse_clean = adverse_clean.reset_index(drop=True)
print("fact_adverse_events:", adverse_events.shape, "->", adverse_clean.shape)
adverse_clean.head(10)


# %% CELL 10 - Before and after comparison (the table for your README)
pairs = [
    ("dim_customers",       customers,      customers_clean),
    ("dim_plants",          plants,         plants_clean),
    ("dim_products",        products,       products_clean),
    ("fact_batches",        batches,        batches_clean),
    ("fact_sales",          sales,          sales_clean),
    ("fact_shipments",      shipments,      shipments_clean),
    ("fact_adverse_events", adverse_events, adverse_clean),
]

rows = []
for name, raw, clean in pairs:
    rows.append({
        "table": name,
        "rows_before": len(raw),
        "rows_after": len(clean),
        "duplicates_removed": len(raw) - len(clean),
        "missing_before": int(raw.isna().sum().sum()),
        "missing_after": int(clean.isna().sum().sum()),
    })
cleaning_summary = pd.DataFrame(rows)
# Note: missing_after can be higher than missing_before because dates written as
# 'pending' or '??' are now proper missing dates (NaT), and cells that were hiding
# placeholder codes (-99, 999, -1) are now honestly missing or filled.
cleaning_summary


# %% CELL 11 - Verification checks (every line should print OK)
def check(label, condition):
    print(("OK    " if condition else "FAIL  ") + label)

check("no duplicate IDs in any table",
      all(df[c].is_unique for df, c in [(customers_clean, "customer_id"), (plants_clean, "plant_id"),
          (products_clean, "product_id"), (batches_clean, "batch_id"), (sales_clean, "sale_id"),
          (shipments_clean, "shipment_id"), (adverse_clean, "ae_id")]))
check("all key columns are upper case",
      all((df[c].dropna() == df[c].dropna().str.upper()).all() for df, c in [
          (sales_clean, "product_id"), (sales_clean, "customer_id"), (batches_clean, "product_id"),
          (batches_clean, "plant_id"), (shipments_clean, "sale_id"), (adverse_clean, "batch_id")]))
check("country has 10 clean values", customers_clean["country"].nunique() == 10 and customers_clean["country"].notna().all())
check("release_status is only Released / Rejected", set(batches_clean["release_status"].dropna()) == {"Released", "Rejected"})
check("severity has 4 clean values", adverse_clean["severity"].nunique() == 4)
check("sales_channel has 4 clean values", sales_clean["sales_channel"].nunique() == 4)
check("units_sold all positive", (sales_clean["units_sold"] > 0).all())
check("discount_pct between 0 and 100", sales_clean["discount_pct"].dropna().between(0, 100).all())
check("purity between 80 and 100", batches_clean["raw_material_purity_pct"].between(80, 100).all())
check("batch_size between 1 and 1,000,000", batches_clean["batch_size_units"].between(1, 1_000_000).all())
check("equipment age between 0 and 60", batches_clean["equipment_age_years"].between(0, 60).all())
check("ambient temperature between -60 and 60", shipments_clean["avg_ambient_temp_c"].between(-60, 60).all())
check("patient_age between 0 and 110", adverse_clean["patient_age"].between(0, 110).all())
check("yes/no columns only 0 or 1", set(batches_clean["night_shift"].dropna()) <= {0, 1}
                                   and set(shipments_clean["temperature_excursion"].dropna()) <= {0, 1})
check("all dates are real datetimes", all(str(df[c].dtype).startswith("datetime64") for df, c in [
      (customers_clean, "onboarding_date"), (batches_clean, "manufacture_date"), (sales_clean, "sale_date"),
      (shipments_clean, "ship_date"), (shipments_clean, "delivery_date"), (adverse_clean, "report_date")]))
check("no delivery before shipping (after cleaning)",
      ((shipments_clean["delivery_date"] >= shipments_clean["ship_date"]) | shipments_clean["delivery_date"].isna() | shipments_clean["ship_date"].isna()).all())


# %% CELL 12 - Save the seven clean tables
clean_tables = {
    "dim_customers": customers_clean,
    "dim_plants": plants_clean,
    "dim_products": products_clean,
    "fact_batches": batches_clean,
    "fact_sales": sales_clean,
    "fact_shipments": shipments_clean,
    "fact_adverse_events": adverse_clean,
}
for name, df in clean_tables.items():
    df.to_csv(f"{CLEAN_DIR}/{name}.csv", index=False)
    print(f"saved {CLEAN_DIR}/{name}.csv   {df.shape}")
```

---

## 6. Feature Engineering

The cleaning phase leaves 7 tables; feature engineering adds **96 new columns** across them, each one answering a business question.

| Table | Columns before | Columns after | Key new features |
|---|---|---|---|
| `fact_batches` | 13 | 34 | `is_rejected` (target), `purity_gap`, `old_equipment_flag` (over 25 years), `high_temp_dev_flag`, `junior_operator_flag`, `night_old_equipment`, `log_batch_size`, plant and product context |
| `fact_sales` | 9 | 35 | `log_units_sold` (target), month and season, `months_since_start`, `price_to_list`, `price_to_cost`, `total_to_expected`, `discount_band` |
| `fact_shipments` | 11 | 29 | `is_excursion` (target), `heat_exposure` (transit days x temperature), `unprotected_cold_chain`, `long_transit_flag`, `hot_climate_flag`, `is_air`, `is_sea` |
| `fact_adverse_events` | 13 | 28 | `is_serious` (target), `age_group`, `polypharmacy_elderly`, `fast_onset_flag`, `onset_unknown`, `batch_to_report_days` |
| Dimensions | 21 | 37 | Customer tenure and credit score, plant age, product dose and shelf-life bands |

**Leakage guard.** Each objective has a list of columns that can never be inputs:

| Objective | Kept out of the model |
|---|---|
| 1 Batch rejection | `release_status`, `rejection_reason`, `process_deviations` |
| 2 Demand | `units_sold`, `total_amount_usd`, `expected_total_usd` |
| 3 Cold chain | `temperature_excursion`, `product_written_off` |
| 4 Adverse events | `severity`, `outcome`, `hospitalised` |
| 5 Pricing | `anomaly_answer_key` |

A quick sanity check (not a model) confirmed the new features carry signal: batches on old equipment are rejected **26.9%** of the time against 10.2% for newer equipment.

### Full code

```python
# %% CELL 1 - Imports and the seven clean tables
import os
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

CLEAN_DIR = "pharma_clean"
FEAT_DIR = "pharma_features"
os.makedirs(FEAT_DIR, exist_ok=True)

customers = pd.read_csv(f"{CLEAN_DIR}/dim_customers.csv", parse_dates=["onboarding_date"])
plants    = pd.read_csv(f"{CLEAN_DIR}/dim_plants.csv")
products  = pd.read_csv(f"{CLEAN_DIR}/dim_products.csv")
batches   = pd.read_csv(f"{CLEAN_DIR}/fact_batches.csv", parse_dates=["manufacture_date"])
sales     = pd.read_csv(f"{CLEAN_DIR}/fact_sales.csv", parse_dates=["sale_date"])
shipments = pd.read_csv(f"{CLEAN_DIR}/fact_shipments.csv", parse_dates=["ship_date", "delivery_date"])
adverse   = pd.read_csv(f"{CLEAN_DIR}/fact_adverse_events.csv", parse_dates=["report_date"])

# one fixed "today" for the whole project, so ages and tenures never change between runs
REF_DATE = sales["sale_date"].max()
print("Reference date:", REF_DATE.date())


# %% CELL 2 - Feature helper functions (run once)
def date_parts(df, col, prefix):
    """Year, month, quarter, day of week and weekend flag from a date column."""
    d = df[col]
    df[f"{prefix}_year"]    = d.dt.year
    df[f"{prefix}_month"]   = d.dt.month
    df[f"{prefix}_quarter"] = d.dt.quarter
    df[f"{prefix}_dow"]     = d.dt.dayofweek
    df[f"{prefix}_is_weekend"] = (d.dt.dayofweek >= 5).astype("Int64").where(d.notna())
    return df


def month_cycle(df, month_col, prefix):
    """Month as a circle: December and January end up next to each other."""
    df[f"{prefix}_month_sin"] = np.sin(2 * np.pi * df[month_col] / 12).round(4)
    df[f"{prefix}_month_cos"] = np.cos(2 * np.pi * df[month_col] / 12).round(4)
    return df


def season_from_month(m):
    """Meteorological season (Northern hemisphere)."""
    return m.map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"})


# %% CELL 3 - dim_customers features
customers_f = customers.copy()

# credit rating as an ordered score (A+ best = 8, Unknown = 0)
rating_score = {"A+": 8, "A": 7, "B+": 6, "B": 5, "C+": 4, "C": 3, "D+": 2, "D": 1, "Unknown": 0}
customers_f["credit_score"] = customers_f["credit_rating"].map(rating_score)
customers_f["credit_known"] = (customers_f["credit_rating"] != "Unknown").astype(int)

# how long the customer has been with Veridian
customers_f["tenure_years"] = ((REF_DATE - customers_f["onboarding_date"]).dt.days / 365.25).round(2)
customers_f["tenure_years"] = customers_f["tenure_years"].fillna(customers_f["tenure_years"].median())
customers_f["tenure_band"] = pd.cut(customers_f["tenure_years"], [-1, 1, 3, 5, 100],
                                    labels=["Under 1y", "1-3y", "3-5y", "Over 5y"]).astype(str)

# large buyers by nature
customers_f["is_bulk_buyer_type"] = customers_f["customer_type"].isin(["Wholesale Distributor", "Government Tender"]).astype(int)

customers_f.head(10)


# %% CELL 4 - dim_plants features
plants_f = plants.copy()

plants_f["plant_age_years"] = REF_DATE.year - plants_f["year_built"]
plants_f["automation_score"] = plants_f["automation_level"].map({"Low": 1, "Medium": 2, "High": 3})
plants_f["is_high_automation"] = (plants_f["automation_level"] == "High").astype(int)
plants_f["is_modern_plant"] = (plants_f["year_built"] >= 2005).astype(int)

plants_f


# %% CELL 5 - dim_products features
products_f = products.copy()

products_f["margin_usd"]       = (products_f["list_price_usd"] - products_f["unit_cost_usd"]).round(2)
products_f["margin_pct"]       = (products_f["margin_usd"] / products_f["list_price_usd"] * 100).round(1)
products_f["price_to_cost"]    = (products_f["list_price_usd"] / products_f["unit_cost_usd"]).round(2)
products_f["is_injectable"]    = (products_f["formulation"] == "Injection").astype(int)
products_f["dose_band"]        = pd.cut(products_f["dosage_mg"], [0, 25, 100, 250, 10000],
                                        labels=["Low", "Medium", "High", "Very high"]).astype(str)
products_f["shelf_life_band"]  = pd.cut(products_f["shelf_life_months"], [0, 18, 30, 100],
                                        labels=["Short", "Medium", "Long"]).astype(str)
products_f["is_short_shelf_life"] = (products_f["shelf_life_months"] <= 18).astype(int)

products_f.head(10)


# %% CELL 6 - Lookups used by the fact tables (small dictionaries, run once)
# product_id -> attribute
prod_form    = products_f.set_index("product_id")["formulation"]
prod_area    = products_f.set_index("product_id")["therapeutic_area"]
prod_dose    = products_f.set_index("product_id")["dosage_mg"]
prod_list    = products_f.set_index("product_id")["list_price_usd"]
prod_cost    = products_f.set_index("product_id")["unit_cost_usd"]
prod_cold    = products_f.set_index("product_id")["requires_cold_chain"]
prod_shelf   = products_f.set_index("product_id")["shelf_life_months"]

# plant_id -> attribute
plant_gmp    = plants_f.set_index("plant_id")["gmp_certified"]
plant_auto   = plants_f.set_index("plant_id")["automation_score"]
plant_age    = plants_f.set_index("plant_id")["plant_age_years"]

# customer_id -> attribute
cust_country = customers_f.set_index("customer_id")["country"]
cust_type    = customers_f.set_index("customer_id")["customer_type"]
cust_credit  = customers_f.set_index("customer_id")["credit_score"]
cust_tenure  = customers_f.set_index("customer_id")["tenure_years"]

# sale_id -> customer / product (used by shipments)
sale_cust    = sales.set_index("sale_id")["customer_id"]
sale_prod    = sales.set_index("sale_id")["product_id"]

# batch_id -> manufacture date (used by adverse events)
batch_date   = batches.set_index("batch_id")["manufacture_date"]


# %% CELL 7 - fact_batches features  (Objective 1: batch rejection)
batches_f = batches.copy()

# TARGET: 1 = rejected
batches_f["is_rejected"] = (batches_f["release_status"] == "Rejected").astype(int)

# calendar
batches_f = date_parts(batches_f, "manufacture_date", "mfg")
batches_f["mfg_season"] = season_from_month(batches_f["mfg_month"])

# process quality
batches_f["purity_gap"]          = (100 - batches_f["raw_material_purity_pct"]).round(2)
batches_f["low_purity_flag"]     = (batches_f["raw_material_purity_pct"] < 98).astype(int)
batches_f["high_temp_dev_flag"]  = (batches_f["max_temp_deviation_c"] > 1.5).astype(int)
batches_f["old_equipment_flag"]  = (batches_f["equipment_age_years"] > 25).astype(int)
batches_f["equipment_age_band"]  = pd.cut(batches_f["equipment_age_years"], [-1, 10, 20, 30, 100],
                                          labels=["0-10y", "11-20y", "21-30y", "Over 30y"]).astype(str)
batches_f["junior_operator_flag"] = (batches_f["operator_experience_years"] < 3).astype(int)
batches_f["log_batch_size"]      = np.log1p(batches_f["batch_size_units"]).round(3)

# product and plant context (lookups)
batches_f["formulation"]         = batches_f["product_id"].map(prod_form)
batches_f["therapeutic_area"]    = batches_f["product_id"].map(prod_area)
batches_f["is_injectable"]       = (batches_f["formulation"] == "Injection").astype(int)
batches_f["plant_gmp_certified"] = batches_f["plant_id"].map(plant_gmp)
batches_f["plant_automation"]    = batches_f["plant_id"].map(plant_auto)
batches_f["plant_age_years"]     = batches_f["plant_id"].map(plant_age)

# combined risk: night shift AND old equipment
batches_f["night_old_equipment"] = (batches_f["night_shift"] * batches_f["old_equipment_flag"]).astype("Int64")

print("Rejection rate:", round(batches_f["is_rejected"].mean() * 100, 1), "%")
batches_f.head(10)


# %% CELL 8 - fact_sales features  (Objective 2: demand, Objective 5: pricing anomalies)
sales_f = sales.copy()

# calendar and trend
sales_f = date_parts(sales_f, "sale_date", "sale")
sales_f = month_cycle(sales_f, "sale_month", "sale")
sales_f["sale_season"]       = season_from_month(sales_f["sale_month"])
first_month = sales_f["sale_date"].min().to_period("M")
sales_f["months_since_start"] = (sales_f["sale_date"].dt.to_period("M") - first_month).apply(lambda x: x.n if pd.notna(x) else np.nan)

# OBJECTIVE 2 TARGET: demand on a log scale (units are heavily skewed)
sales_f["log_units_sold"] = np.log1p(sales_f["units_sold"]).round(4)

# product and customer context (lookups)
sales_f["therapeutic_area"] = sales_f["product_id"].map(prod_area)
sales_f["formulation"]      = sales_f["product_id"].map(prod_form)
sales_f["customer_type"]    = sales_f["customer_id"].map(cust_type).fillna("Unknown")
sales_f["customer_country"] = sales_f["customer_id"].map(cust_country).fillna("Unknown")
sales_f["customer_credit_score"] = sales_f["customer_id"].map(cust_credit)
sales_f["customer_tenure_years"] = sales_f["customer_id"].map(cust_tenure)
sales_f["is_unknown_customer"]   = (sales_f["customer_id"] == "UNKNOWN").astype(int)
sales_f["discount_band"] = pd.cut(sales_f["discount_pct"], [-1, 0, 5, 10, 100],
                                  labels=["None", "Up to 5%", "6-10%", "Over 10%"]).astype(str)

# OBJECTIVE 5 FEATURES: how does the invoice compare with what it should be?
sales_f["list_price_usd"]    = sales_f["product_id"].map(prod_list)
sales_f["unit_cost_usd"]     = sales_f["product_id"].map(prod_cost)
sales_f["price_to_list"]     = (sales_f["unit_price_usd"] / sales_f["list_price_usd"]).round(4)
sales_f["price_to_cost"]     = (sales_f["unit_price_usd"] / sales_f["unit_cost_usd"]).round(4)
sales_f["expected_total_usd"] = (sales_f["units_sold"] * sales_f["unit_price_usd"]).round(2)
sales_f["total_to_expected"]  = (sales_f["total_amount_usd"] / sales_f["expected_total_usd"]).round(4)
sales_f["sold_below_cost"]    = (sales_f["unit_price_usd"] < sales_f["unit_cost_usd"]).astype(int)
sales_f["log_total_amount"]   = np.log1p(sales_f["total_amount_usd"]).round(4)

sales_f.head(10)


# %% CELL 9 - fact_shipments features  (Objective 3: cold-chain failure)
shipments_f = shipments.copy()

# TARGET: 1 = temperature excursion happened
shipments_f["is_excursion"] = shipments_f["temperature_excursion"]

# calendar
shipments_f["ship_month"]  = shipments_f["ship_date"].dt.month
shipments_f["ship_year"]   = shipments_f["ship_date"].dt.year
shipments_f["ship_season"] = season_from_month(shipments_f["ship_month"])
shipments_f["ship_is_weekend"] = (shipments_f["ship_date"].dt.dayofweek >= 5).astype("Int64").where(shipments_f["ship_date"].notna())

# journey risk
shipments_f["long_transit_flag"]  = (shipments_f["transit_days"] > 7).astype(int)
shipments_f["hot_climate_flag"]   = (shipments_f["avg_ambient_temp_c"] > 25).astype(int)
shipments_f["heat_exposure"]      = (shipments_f["avg_ambient_temp_c"].clip(lower=0) * shipments_f["transit_days"]).round(1)
shipments_f["is_air"]             = (shipments_f["transport_mode"] == "Air").astype(int)
shipments_f["is_sea"]             = (shipments_f["transport_mode"] == "Sea").astype(int)
shipments_f["protective_packaging"] = shipments_f["packaging_type"].isin(["Insulated", "Active Cooling"]).astype(int)

# cold-chain product shipped in standard packaging = gap in protection
shipments_f["unprotected_cold_chain"] = ((shipments_f["cold_chain_required"] == 1) &
                                         (shipments_f["packaging_type"] == "Standard")).astype("Int64")

# context through the sale (shipment -> sale -> customer / product)
shipments_f["customer_id"]          = shipments_f["sale_id"].map(sale_cust)
shipments_f["product_id"]           = shipments_f["sale_id"].map(sale_prod)
shipments_f["destination_country"]  = shipments_f["customer_id"].map(cust_country).fillna("Unknown")
shipments_f["customer_type"]        = shipments_f["customer_id"].map(cust_type).fillna("Unknown")
shipments_f["product_requires_cold_chain"] = shipments_f["product_id"].map(prod_cold)
shipments_f["formulation"]          = shipments_f["product_id"].map(prod_form)

print("Excursion rate:", round(shipments_f["is_excursion"].mean() * 100, 1), "%")
shipments_f.head(10)


# %% CELL 10 - fact_adverse_events features  (Objective 4: serious adverse events)
adverse_f = adverse.copy()

# TARGET: 1 = Severe or Life-threatening
adverse_f["is_serious"] = adverse_f["severity"].isin(["Severe", "Life-threatening"]).astype(int)

# calendar
adverse_f["report_year"]  = adverse_f["report_date"].dt.year
adverse_f["report_month"] = adverse_f["report_date"].dt.month

# patient
adverse_f["age_group"] = pd.cut(adverse_f["patient_age"], [-1, 17, 39, 64, 120],
                                labels=["Under 18", "18-39", "40-64", "65 and over"]).astype(str)
adverse_f["is_elderly"]          = (adverse_f["patient_age"] >= 65).astype(int)
adverse_f["many_medications"]    = (adverse_f["concomitant_medications"] >= 4).astype(int)
adverse_f["polypharmacy_elderly"] = adverse_f["is_elderly"] * adverse_f["many_medications"]

# onset speed (missing stays missing, with a flag so the model can use "unknown")
adverse_f["onset_unknown"]   = adverse_f["days_to_onset"].isna().astype(int)
adverse_f["fast_onset_flag"] = (adverse_f["days_to_onset"] <= 3).astype(int)
adverse_f["onset_band"]      = pd.cut(adverse_f["days_to_onset"], [-1, 3, 14, 60, 1000],
                                      labels=["0-3 days", "4-14 days", "15-60 days", "Over 60 days"]).astype(str)

# product context (lookups)
adverse_f["dosage_mg"]        = adverse_f["product_id"].map(prod_dose)
adverse_f["therapeutic_area"] = adverse_f["product_id"].map(prod_area)
adverse_f["formulation"]      = adverse_f["product_id"].map(prod_form)
adverse_f["high_dose_flag"]   = (adverse_f["dosage_mg"] >= 250).astype("Int64").where(adverse_f["dosage_mg"].notna())

# how old the batch was when the event was reported
adverse_f["batch_to_report_days"] = (adverse_f["report_date"] - adverse_f["batch_id"].map(batch_date)).dt.days
adverse_f.loc[adverse_f["batch_to_report_days"] < 0, "batch_to_report_days"] = np.nan

print("Serious rate:", round(adverse_f["is_serious"].mean() * 100, 1), "%")
adverse_f.head(10)


# %% CELL 11 - Leakage guard: columns that must NOT be used as model inputs
LEAKY = {
    "Objective 1 (batch rejection)": ["release_status", "rejection_reason", "is_rejected", "process_deviations"],
    "Objective 2 (demand)":          ["units_sold", "log_units_sold", "total_amount_usd", "expected_total_usd",
                                      "total_to_expected", "log_total_amount"],
    "Objective 3 (cold chain)":      ["temperature_excursion", "is_excursion", "product_written_off"],
    "Objective 4 (adverse events)":  ["severity", "is_serious", "outcome", "hospitalised"],
    "Objective 5 (anomalies)":       ["anomaly_answer_key (never a feature)"],
}
for objective, cols in LEAKY.items():
    print(f"{objective}\n    keep out of X: {', '.join(cols)}\n")


# %% CELL 12 - New features summary (the table for your README)
feature_sets = [
    ("dim_customers",       customers,   customers_f),
    ("dim_plants",          plants,      plants_f),
    ("dim_products",        products,    products_f),
    ("fact_batches",        batches,     batches_f),
    ("fact_sales",          sales,       sales_f),
    ("fact_shipments",      shipments,   shipments_f),
    ("fact_adverse_events", adverse,     adverse_f),
]

rows = []
for name, before, after in feature_sets:
    new_cols = [c for c in after.columns if c not in before.columns]
    rows.append({"table": name, "columns_before": before.shape[1],
                 "columns_after": after.shape[1], "new_features": len(new_cols)})
feature_summary = pd.DataFrame(rows)
feature_summary


# %% CELL 13 - Quick sanity check: do the new features carry signal? (not a model)
def rate_by(df, feature, target):
    return df.groupby(feature)[target].mean().mul(100).round(1)

print("Objective 1 - rejection rate % by old equipment flag\n", rate_by(batches_f, "old_equipment_flag", "is_rejected"), "\n")
print("Objective 1 - rejection rate % by night shift\n", rate_by(batches_f, "night_shift", "is_rejected"), "\n")
print("Objective 3 - excursion rate % by packaging\n", rate_by(shipments_f, "packaging_type", "is_excursion"), "\n")
print("Objective 4 - serious rate % by age group\n", rate_by(adverse_f, "age_group", "is_serious"), "\n")
print("Objective 2 - average units sold by customer type\n", sales_f.groupby("customer_type")["units_sold"].mean().round(0), "\n")
print("Objective 5 - price_to_list summary\n", sales_f["price_to_list"].describe().round(3))


# %% CELL 14 - Save the seven feature tables
feature_tables = {
    "dim_customers": customers_f, "dim_plants": plants_f, "dim_products": products_f,
    "fact_batches": batches_f, "fact_sales": sales_f, "fact_shipments": shipments_f,
    "fact_adverse_events": adverse_f,
}
for name, df in feature_tables.items():
    df.to_csv(f"{FEAT_DIR}/{name}.csv", index=False)
    print(f"saved {FEAT_DIR}/{name}.csv   {df.shape}")
```

---

## 7. Create Relationship Table

The cleaned tables are linked as a **star schema**: four fact tables around three dimensions and a date table.

| Fact table | Links to |
|---|---|
| `fact_sales` | `dim_customers`, `dim_products`, `dim_date` |
| `fact_batches` | `dim_products`, `dim_plants`, `dim_date` |
| `fact_shipments` | `fact_sales` (and through it customers and products), `dim_date` |
| `fact_adverse_events` | `fact_batches`, `dim_products`, `dim_date` |

Some rows pointed at records that do not exist (orphans): 25 batches with an unknown product, 30 with an unknown plant, 50 sales with an unknown customer, 20 adverse events with an unknown batch and 30 shipments with an unknown sale. Deleting them would lose real sales and real safety reports, so each dimension gets an `UNKNOWN` member and the orphans point to it, with a flag to keep the information.

Ten checks confirm every key is unique, no orphan keys remain and every date key exists in `dim_date`. The phase ends with five **modelling tables**, one per objective:

| Modelling table | Rows x columns | Objective |
|---|---|---|
| `model_batches` | 4,000 x 45 | 1 |
| `model_demand` | 6,000 x 47 | 2 |
| `model_shipments` | 3,500 x 38 | 3 |
| `model_adverse` | 1,800 x 37 | 4 |
| `model_pricing` | 6,000 x 40 | 5 |

### Full code

```python
# %% CELL 1 - Imports and the seven feature tables
import os
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

FEAT_DIR = "pharma_features"
STAR_DIR = "pharma_star"
MODEL_DIR = "pharma_model"
os.makedirs(STAR_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

customers = pd.read_csv(f"{FEAT_DIR}/dim_customers.csv", parse_dates=["onboarding_date"])
plants    = pd.read_csv(f"{FEAT_DIR}/dim_plants.csv")
products  = pd.read_csv(f"{FEAT_DIR}/dim_products.csv")
batches   = pd.read_csv(f"{FEAT_DIR}/fact_batches.csv", parse_dates=["manufacture_date"])
sales     = pd.read_csv(f"{FEAT_DIR}/fact_sales.csv", parse_dates=["sale_date"])
shipments = pd.read_csv(f"{FEAT_DIR}/fact_shipments.csv", parse_dates=["ship_date", "delivery_date"])
adverse   = pd.read_csv(f"{FEAT_DIR}/fact_adverse_events.csv", parse_dates=["report_date"])


# %% CELL 2 - The relationships we expect (the map of the schema)
relationships = [
    # child table . key           -> parent table . key         type
    ("fact_batches.product_id",         "dim_products.product_id",     "dimension"),
    ("fact_batches.plant_id",           "dim_plants.plant_id",         "dimension"),
    ("fact_sales.customer_id",          "dim_customers.customer_id",   "dimension"),
    ("fact_sales.product_id",           "dim_products.product_id",     "dimension"),
    ("fact_shipments.sale_id",          "fact_sales.sale_id",          "fact to fact"),
    ("fact_adverse_events.batch_id",    "fact_batches.batch_id",       "fact to fact"),
    ("fact_adverse_events.product_id",  "dim_products.product_id",     "dimension"),
]
pd.DataFrame(relationships, columns=["child_key", "parent_key", "kind"])


# %% CELL 3 - Orphan check BEFORE fixing (child keys with no parent)
tables = {
    "dim_customers": customers, "dim_plants": plants, "dim_products": products,
    "fact_batches": batches, "fact_sales": sales,
    "fact_shipments": shipments, "fact_adverse_events": adverse,
}

def orphan_report(tables):
    rows = []
    for child_key, parent_key, kind in relationships:
        ct, ck = child_key.split(".")
        pt, pk = parent_key.split(".")
        child = tables[ct][ck]
        orphans = ~child.isin(tables[pt][pk])
        rows.append({"relationship": f"{child_key} -> {parent_key}", "kind": kind,
                     "child_rows": len(child), "orphan_rows": int(orphans.sum()),
                     "orphan_%": round(orphans.mean() * 100, 2),
                     "example_orphans": ", ".join(child[orphans].astype(str).unique()[:3])})
    return pd.DataFrame(rows)

orphans_before = orphan_report(tables)
orphans_before


# %% CELL 4 - Add an "UNKNOWN" member to each dimension
# Standard star-schema practice: a row that points at no real customer, plant or
# product is still a real sale or batch. We keep it, and let it join to UNKNOWN.
unknown_customer = {"customer_id": "UNKNOWN", "customer_name": "Unknown customer", "country": "Unknown",
                    "customer_type": "Unknown", "credit_rating": "Unknown", "credit_score": 0, "credit_known": 0,
                    "tenure_years": customers["tenure_years"].median(), "tenure_band": "Unknown",
                    "is_bulk_buyer_type": 0}
unknown_plant    = {"plant_id": "UNKNOWN", "plant_name": "Unknown plant", "country": "Unknown",
                    "automation_level": "Unknown"}
unknown_product  = {"product_id": "UNKNOWN", "product_name": "Unknown product", "therapeutic_area": "Unknown",
                    "formulation": "Unknown", "dose_band": "Unknown", "shelf_life_band": "Unknown"}

def add_unknown(df, member):
    """Add one UNKNOWN row (columns not listed stay blank)."""
    if (df.iloc[:, 0] == "UNKNOWN").any():
        return df
    return pd.concat([df, pd.DataFrame([member])], ignore_index=True)

dim_customers = add_unknown(customers, unknown_customer)
dim_plants    = add_unknown(plants, unknown_plant)
dim_products  = add_unknown(products, unknown_product)

print("dimension rows:", len(dim_customers), len(dim_plants), len(dim_products))


# %% CELL 5 - Repair the orphan keys (point them at UNKNOWN and keep a flag)
fact_batches = batches.copy()
fact_batches["product_known"] = fact_batches["product_id"].isin(products["product_id"]).astype(int)
fact_batches["plant_known"]   = fact_batches["plant_id"].isin(plants["plant_id"]).astype(int)
fact_batches.loc[fact_batches["product_known"] == 0, "product_id"] = "UNKNOWN"
fact_batches.loc[fact_batches["plant_known"] == 0,   "plant_id"]   = "UNKNOWN"

fact_sales = sales.copy()
fact_sales["customer_known"] = fact_sales["customer_id"].isin(customers["customer_id"]).astype(int)
fact_sales["product_known"]  = fact_sales["product_id"].isin(products["product_id"]).astype(int)
fact_sales.loc[fact_sales["customer_known"] == 0, "customer_id"] = "UNKNOWN"
fact_sales.loc[fact_sales["product_known"] == 0,  "product_id"]  = "UNKNOWN"

fact_adverse = adverse.copy()
fact_adverse["product_known"] = fact_adverse["product_id"].isin(products["product_id"]).astype(int)
fact_adverse["batch_known"]   = fact_adverse["batch_id"].isin(batches["batch_id"]).astype(int)
fact_adverse.loc[fact_adverse["product_known"] == 0, "product_id"] = "UNKNOWN"

# fact-to-fact links are kept as they are and flagged (there is no UNKNOWN sale or batch row)
fact_shipments = shipments.copy()
fact_shipments["sale_known"] = fact_shipments["sale_id"].isin(sales["sale_id"]).astype(int)

print("batches   : product orphans", (fact_batches["product_known"] == 0).sum(), "| plant orphans", (fact_batches["plant_known"] == 0).sum())
print("sales     : customer orphans", (fact_sales["customer_known"] == 0).sum(), "| product orphans", (fact_sales["product_known"] == 0).sum())
print("adverse   : product orphans", (fact_adverse["product_known"] == 0).sum(), "| batch orphans", (fact_adverse["batch_known"] == 0).sum())
print("shipments : sale orphans", (fact_shipments["sale_known"] == 0).sum())


# %% CELL 6 - Build dim_date (one row per calendar day) and add date keys to the facts
all_dates = pd.concat([
    fact_batches["manufacture_date"], fact_sales["sale_date"],
    fact_shipments["ship_date"], fact_shipments["delivery_date"], fact_adverse["report_date"]]).dropna()

days = pd.date_range(all_dates.min().normalize(), all_dates.max().normalize(), freq="D")
dim_date = pd.DataFrame({"date": days})
dim_date["date_key"]   = dim_date["date"].dt.strftime("%Y%m%d").astype(int)
dim_date["year"]       = dim_date["date"].dt.year
dim_date["quarter"]    = dim_date["date"].dt.quarter
dim_date["month"]      = dim_date["date"].dt.month
dim_date["month_name"] = dim_date["date"].dt.month_name()
dim_date["week"]       = dim_date["date"].dt.isocalendar().week.astype(int)
dim_date["day_of_week"] = dim_date["date"].dt.day_name()
dim_date["is_weekend"] = (dim_date["date"].dt.dayofweek >= 5).astype(int)
dim_date = dim_date[["date_key", "date", "year", "quarter", "month", "month_name", "week", "day_of_week", "is_weekend"]]

def date_key(s):
    """Date -> 20250312 style key (blank dates stay blank)."""
    return s.dt.strftime("%Y%m%d").astype("float").astype("Int64")

fact_batches["manufacture_date_key"] = date_key(fact_batches["manufacture_date"])
fact_sales["sale_date_key"]          = date_key(fact_sales["sale_date"])
fact_shipments["ship_date_key"]      = date_key(fact_shipments["ship_date"])
fact_shipments["delivery_date_key"]  = date_key(fact_shipments["delivery_date"])
fact_adverse["report_date_key"]      = date_key(fact_adverse["report_date"])

print("dim_date:", dim_date.shape, "from", dim_date["date"].min().date(), "to", dim_date["date"].max().date())
dim_date.head(5)


# %% CELL 7 - Orphan check AFTER fixing (dimension links must now be 0)
star = {
    "dim_customers": dim_customers, "dim_plants": dim_plants, "dim_products": dim_products,
    "fact_batches": fact_batches, "fact_sales": fact_sales,
    "fact_shipments": fact_shipments, "fact_adverse_events": fact_adverse,
}
orphans_after = orphan_report(star)
orphans_after


# %% CELL 8 - Primary key and row-count verification (every line should print OK)
def check(label, condition):
    print(("OK    " if condition else "FAIL  ") + label)

for name, key in [("dim_customers", "customer_id"), ("dim_plants", "plant_id"), ("dim_products", "product_id"),
                  ("fact_batches", "batch_id"), ("fact_sales", "sale_id"),
                  ("fact_shipments", "shipment_id"), ("fact_adverse_events", "ae_id")]:
    check(f"{name}.{key} is unique", star[name][key].is_unique)
check("dim_date.date_key is unique", dim_date["date_key"].is_unique)
check("no orphan dimension keys remain",
      (orphans_after.loc[orphans_after["kind"] == "dimension", "orphan_rows"] == 0).all())
check("every non-blank date key exists in dim_date",
      all(star[t][c].dropna().isin(dim_date["date_key"]).all() for t, c in
          [("fact_batches", "manufacture_date_key"), ("fact_sales", "sale_date_key"),
           ("fact_shipments", "ship_date_key"), ("fact_shipments", "delivery_date_key"),
           ("fact_adverse_events", "report_date_key")]))


# %% CELL 9 - Safe join helper (checks that a join never adds or loses rows)
def safe_merge(left, right, key, keep_cols, prefix):
    """
    Many-to-one left join of a dimension onto a fact table.
    - brings only the columns in keep_cols
    - renames them with a prefix, so no column is ever overwritten or suffixed with _x / _y
    - stops with an error if the row count changes
    """
    cols = [key] + keep_cols
    right_small = right[cols].rename(columns={c: f"{prefix}{c}" for c in keep_cols})
    merged = left.merge(right_small, on=key, how="left", validate="many_to_one")
    assert len(merged) == len(left), f"row count changed when joining on {key}"
    return merged


# %% CELL 10 - Objective 1 table: batch rejection (batches + products + plants)
model_batches = safe_merge(fact_batches, dim_products, "product_id",
                           ["product_name", "dosage_mg", "shelf_life_months", "requires_cold_chain", "margin_pct"], "prod_")
model_batches = safe_merge(model_batches, dim_plants, "plant_id",
                           ["plant_name", "country", "year_built"], "plant_")

print("model_batches:", model_batches.shape, "| rejection rate:", round(model_batches["is_rejected"].mean() * 100, 1), "%")
model_batches.head(5)


# %% CELL 11 - Objective 2 table: demand forecast (sales + customers + products)
model_demand = safe_merge(fact_sales, dim_customers, "customer_id",
                          ["customer_name", "credit_rating", "tenure_band", "is_bulk_buyer_type"], "cust_")
model_demand = safe_merge(model_demand, dim_products, "product_id",
                          ["product_name", "dosage_mg", "shelf_life_months", "requires_cold_chain", "margin_pct"], "prod_")

print("model_demand:", model_demand.shape)
model_demand.head(5)


# %% CELL 12 - Objective 3 table: cold-chain risk (shipments + sales + customers + products)
sale_cols = fact_sales[["sale_id", "sale_date", "units_sold", "sales_channel"]].rename(
    columns={"sale_date": "sale_date_of_order", "units_sold": "order_units", "sales_channel": "order_channel"})
model_shipments = fact_shipments.merge(sale_cols, on="sale_id", how="left", validate="many_to_one")
assert len(model_shipments) == len(fact_shipments)

model_shipments = safe_merge(model_shipments, dim_products.assign(product_id=dim_products["product_id"]),
                             "product_id", ["product_name", "shelf_life_months", "dosage_mg"], "prod_")

print("model_shipments:", model_shipments.shape, "| excursion rate:", round(model_shipments["is_excursion"].mean() * 100, 1), "%")
model_shipments.head(5)


# %% CELL 13 - Objective 4 table: serious adverse events (events + products + batches)
batch_cols = fact_batches[["batch_id", "plant_id", "release_status", "raw_material_purity_pct"]].rename(
    columns={"plant_id": "batch_plant_id", "release_status": "batch_release_status",
             "raw_material_purity_pct": "batch_purity_pct"})
model_adverse = fact_adverse.merge(batch_cols, on="batch_id", how="left", validate="many_to_one")
assert len(model_adverse) == len(fact_adverse)

model_adverse = safe_merge(model_adverse, dim_products, "product_id",
                           ["product_name", "shelf_life_months", "requires_cold_chain"], "prod_")

print("model_adverse:", model_adverse.shape, "| serious rate:", round(model_adverse["is_serious"].mean() * 100, 1), "%")
model_adverse.head(5)


# %% CELL 14 - Objective 5 table: pricing anomalies (sales + products)
model_pricing = safe_merge(fact_sales, dim_products, "product_id", ["product_name", "margin_pct"], "prod_")
model_pricing = model_pricing[model_pricing["product_known"] == 1].copy()   # price_to_list needs a real list price

print("model_pricing:", model_pricing.shape, "(sales whose product has no list price are left out)")
model_pricing.head(5)


# %% CELL 15 - Schema summary (the table for your README)
summary_rows = []
for name, df in star.items():
    summary_rows.append({"layer": "dimension" if name.startswith("dim_") else "fact", "table": name,
                         "rows": len(df), "columns": df.shape[1]})
summary_rows.append({"layer": "dimension", "table": "dim_date", "rows": len(dim_date), "columns": dim_date.shape[1]})
for name, df in [("model_batches", model_batches), ("model_demand", model_demand), ("model_shipments", model_shipments),
                 ("model_adverse", model_adverse), ("model_pricing", model_pricing)]:
    summary_rows.append({"layer": "model", "table": name, "rows": len(df), "columns": df.shape[1]})
schema_summary = pd.DataFrame(summary_rows)
schema_summary


# %% CELL 16 - Save the star schema and the five modelling tables
for name, df in {**star, "dim_date": dim_date}.items():
    df.to_csv(f"{STAR_DIR}/{name}.csv", index=False)
    print(f"saved {STAR_DIR}/{name}.csv   {df.shape}")

for name, df in {"model_batches": model_batches, "model_demand": model_demand,
                 "model_shipments": model_shipments, "model_adverse": model_adverse,
                 "model_pricing": model_pricing}.items():
    df.to_csv(f"{MODEL_DIR}/{name}.csv", index=False)
    print(f"saved {MODEL_DIR}/{name}.csv   {df.shape}")
```

---

## 8. Exploratory Data Analysis

Twelve charts, each answering one question before any model is built.

| Objective | Question | What the data says |
|---|---|---|
| 1 | Which batches get rejected? | Overall rate 15.2%. Old equipment 26.9% vs 10.2%; night shift 19.0% vs 13.5%; non-GMP plants 32.3% vs 13.3%; Injection products 21.2% |
| 1 | Which plants? | Two plants sit at 32.6% and 32.3%, more than double the average. The best plant is at 3.5% |
| 2 | How is demand spread? | Heavily skewed (skewness 2.05), mean 477 units vs median 305. On a log scale it becomes balanced (-0.11), so the model learns on logs |
| 2 | Who buys the volume? | Government tenders average 1,151 units per sale, wholesalers 848, hospitals 399, retail pharmacies 174 |
| 3 | When do shipments overheat? | Sea freight 52.4% vs air 7.7%; unprotected cold chain 36.0% vs 13.7%; long transits 36.6% vs 10.6% |
| 4 | Who is most at risk? | Patients 65 and over 36.3% vs 1.3% for under 18; liver impairment 34.0% vs 18.7%; five or more other medicines 38% vs 9% for none |
| 5 | What is a normal price? | 97.8% of invoices sit between 60% and 130% of list price. 63 invoices are priced below cost |

### Charts

<p align="center">
  <img width="1572" height="575" alt="eda_01_rejection_overview" src="https://github.com/user-attachments/assets/3f865129-432d-4765-85ea-b3af54a306da" />
</p>
<p align="center"><em>Q1.1 How often are batches rejected, and is it improving?</em></p>

<p align="center">
  <img width="1925" height="1176" alt="eda_02_rejection_by_plant_shift" src="https://github.com/user-attachments/assets/5cb52173-6a00-41bf-86b7-ce7dd60998a3" />
</p>
<p align="center"><em>Q1.2 Rejection rate by plant, product form, shift and GMP status</em></p>

<p align="center">
<img width="2226" height="575" alt="eda_03_rejection_drivers" src="https://github.com/user-attachments/assets/692746f1-e6b2-4d5f-98de-625e8675f7f6" />
</p>
<p align="center"><em>Q1.3 Do old equipment, low purity and temperature swings push rejections up?</em></p>

<p align="center">
  <img width="1776" height="575" alt="eda_04_demand_distribution" src="https://github.com/user-attachments/assets/edcb5467-1801-475e-8706-126eb132e0b3" />
</p>
<p align="center"><em>Q2.1 Demand is skewed, so the model learns on a log scale</em></p>

<p align="center">
  <img width="1925" height="650" alt="eda_05_demand_by_group" src="https://github.com/user-attachments/assets/ba9d04ac-5412-4d6f-ba61-e7cac2a55156" />
</p>
<p align="center"><em>Q2.2 Which products and customers drive volume?</em></p>

<p align="center">
  <img width="1926" height="650" alt="eda_06_demand_trend_discount" src="https://github.com/user-attachments/assets/b95e9901-7d67-4e56-9e5f-5e8c66265db5" />
</p>
<p align="center"><em>Q2.3 Is there a trend or season, and does discounting lift volume?</em></p>

<p align="center">
  <img width="1916" height="1176" alt="eda_07_excursions_overview" src="https://github.com/user-attachments/assets/0a198807-c0cc-4a8c-ae5b-a1a7eb9b17f0" />
</p>
<p align="center"><em>Q3.1 Cold-chain excursion rate by packaging, transport mode, season and destination</em></p>

<p align="center">
  <img width="2228" height="575" alt="eda_08_excursion_drivers" src="https://github.com/user-attachments/assets/e0cb231b-919b-4cc2-bb0e-45ae43c8da97" />
</p>
<p align="center"><em>Q3.2 Do longer, hotter journeys fail more often?</em></p>

<p align="center">
  <img width="1926" height="1176" alt="eda_09_adverse_risk_groups" src="https://github.com/user-attachments/assets/6474c530-2d1c-4edd-ab5c-1c33df82f118" />
</p>
<p align="center"><em>Q4.1 Serious adverse event rate by patient risk group</em></p>

<p align="center">
  <img width="2226" height="576" alt="eda_10_serious_by_drug" src="https://github.com/user-attachments/assets/e73992e9-d059-4dda-9153-494bdb27ff8c" />
</p>
<p align="center"><em>Q4.2 Do onset speed, dose and product type matter?</em></p>

<p align="center">
  <img width="1926" height="650" alt="eda_11_pricing_overview" src="https://github.com/user-attachments/assets/9fd54592-6841-4fe8-9fee-1f4f2699b7a9" />
</p>
<p align="center"><em>Q5.1 What does a normal price look like compared with the list price?</em></p>

<p align="center">
  <img width="2526" height="726" alt="eda_12_correlations" src="https://github.com/user-attachments/assets/19f0c988-31c8-4421-94c7-116af210d09b" />
</p>
<p align="center"><em>Which numeric features move with each outcome (outcome columns excluded)</em></p>

### Full code

```python
# %% CELL 1 - Imports and the five modelling tables
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"main": "#1f5f8b", "alert": "#c0392b", "soft": "#9db4c0"}

MODEL_DIR = "pharma_model"
CHART_DIR = "eda_charts"
os.makedirs(CHART_DIR, exist_ok=True)

batches   = pd.read_csv(f"{MODEL_DIR}/model_batches.csv",   parse_dates=["manufacture_date"])
demand    = pd.read_csv(f"{MODEL_DIR}/model_demand.csv",    parse_dates=["sale_date"])
shipments = pd.read_csv(f"{MODEL_DIR}/model_shipments.csv", parse_dates=["ship_date", "delivery_date"])
adverse   = pd.read_csv(f"{MODEL_DIR}/model_adverse.csv",   parse_dates=["report_date"])
pricing   = pd.read_csv(f"{MODEL_DIR}/model_pricing.csv",   parse_dates=["sale_date"])

for name, df in [("batches", batches), ("demand", demand), ("shipments", shipments),
                 ("adverse", adverse), ("pricing", pricing)]:
    print(f"{name:<10} {df.shape}")


# %% CELL 2 - Chart helpers (run once)
def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def rate_bar(ax, df, col, target, title, order=None, min_rows=30):
    """Bar chart of the event rate (%) for each group, with a line at the overall rate."""
    g = (df.dropna(subset=[col]).groupby(col)[target].agg(["mean", "size"]))
    g = g[g["size"] >= min_rows]
    g["rate"] = g["mean"] * 100
    g = g.loc[order] if order is not None else g.sort_values("rate", ascending=False)
    labels = [str(int(i)) if isinstance(i, (float, np.floating)) and float(i).is_integer() else str(i) for i in g.index]
    ax.bar(labels, g["rate"], color=PALETTE["main"])
    ax.axhline(df[target].mean() * 100, color=PALETTE["alert"], ls="--", lw=1.2, label="overall rate")
    for i, v in enumerate(g["rate"]):
        ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)


# %% CELL 3 - First look at every table: size, types, missing values
for name, df in [("batches", batches), ("demand", demand), ("shipments", shipments),
                 ("adverse", adverse), ("pricing", pricing)]:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False).head(5)
    print(f"{name.upper()}  rows={len(df)}  columns={df.shape[1]}")
    print("   top missing:", miss.to_dict() if len(miss) else "none", "\n")


# %% CELL 4 - Summary statistics of the key numeric columns
batches[["batch_size_units", "raw_material_purity_pct", "max_temp_deviation_c",
         "equipment_age_years", "operator_experience_years"]].describe().round(2)


# ------------------------------------------------------------
# OBJECTIVE 1 - BATCH QUALITY
# Decision this supports: where do we spend QA money to stop rejections?
# ------------------------------------------------------------
# %% CELL 5 - Q1.1  How often are batches rejected, and is it getting better or worse?
rejection_rate = batches["is_rejected"].mean() * 100
monthly = (batches.dropna(subset=["manufacture_date"])
           .set_index("manufacture_date")["is_rejected"].resample("QS").mean() * 100)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].pie([100 - rejection_rate, rejection_rate], labels=["Released", "Rejected"], autopct="%1.1f%%",
          colors=[PALETTE["soft"], PALETTE["alert"]], startangle=90)
ax[0].set_title("Share of batches rejected")
ax[1].plot(monthly.index, monthly.values, marker="o", color=PALETTE["main"])
ax[1].set_title("Rejection rate by quarter (%)")
ax[1].set_ylim(0, max(monthly.max() * 1.3, 5))
save(fig, "01_rejection_overview")
print(f"Overall rejection rate: {rejection_rate:.1f}%")


# %% CELL 6 - Q1.2  Which plants, products and shifts reject the most?
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
rate_bar(ax[0, 0], batches, "plant_id", "is_rejected", "Rejection rate by plant")
rate_bar(ax[0, 1], batches, "formulation", "is_rejected", "Rejection rate by formulation")
rate_bar(ax[1, 0], batches, "night_shift", "is_rejected", "Night shift (1) vs day shift (0)", order=[0, 1])
rate_bar(ax[1, 1], batches, "plant_gmp_certified", "is_rejected", "GMP-certified plant (1) vs not (0)", order=[0, 1], min_rows=5)
save(fig, "02_rejection_by_group")


# %% CELL 7 - Q1.3  Do old equipment, low purity and temperature swings push rejections up?
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
rate_bar(ax[0], batches, "equipment_age_band", "is_rejected", "By equipment age",
         order=["0-10y", "11-20y", "21-30y", "Over 30y"])
sns.boxplot(data=batches, x="release_status", y="raw_material_purity_pct", ax=ax[1],
            palette=[PALETTE["soft"], PALETTE["alert"]], hue="release_status", legend=False)
ax[1].set_title("Raw material purity (%) by outcome")
sns.boxplot(data=batches, x="release_status", y="max_temp_deviation_c", ax=ax[2],
            palette=[PALETTE["soft"], PALETTE["alert"]], hue="release_status", legend=False)
ax[2].set_title("Max temperature deviation (C) by outcome")
save(fig, "03_rejection_drivers")


# ------------------------------------------------------------
# OBJECTIVE 2 - DEMAND
# Decision this supports: how much of which product do we plan to make?
# ------------------------------------------------------------
# %% CELL 8 - Q2.1  How is demand distributed, and why do we model it on a log scale?
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(demand["units_sold"], bins=60, color=PALETTE["main"], ax=ax[0])
ax[0].set_title("Units per sale (raw): heavily right-skewed")
sns.histplot(demand["log_units_sold"], bins=60, color=PALETTE["main"], ax=ax[1])
ax[1].set_title("Units per sale (log scale): much more balanced")
save(fig, "04_demand_distribution")
print("Skewness raw:", round(demand["units_sold"].skew(), 2), "| log:", round(demand["log_units_sold"].skew(), 2))


# %% CELL 9 - Q2.2  Which products and which customers drive volume?
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
by_area = demand.groupby("therapeutic_area")["units_sold"].sum().sort_values()
ax[0].barh(by_area.index, by_area.values, color=PALETTE["main"])
ax[0].set_title("Total units sold by therapeutic area")
by_type = demand.groupby("customer_type")["units_sold"].mean().sort_values()
ax[1].barh(by_type.index, by_type.values, color=PALETTE["main"])
ax[1].set_title("Average units per sale by customer type")
save(fig, "05_demand_by_group")


# %% CELL 10 - Q2.3  Is there a trend or season, and does discounting lift volume?
monthly_units = (demand.dropna(subset=["sale_date"]).set_index("sale_date")["units_sold"].resample("MS").sum())
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(monthly_units.index, monthly_units.values, color=PALETTE["main"])
ax[0].set_title("Total units sold per month")
disc = demand.groupby("discount_band")["units_sold"].mean().reindex(["None", "Up to 5%", "6-10%", "Over 10%"])
ax[1].bar(disc.index, disc.values, color=PALETTE["main"])
ax[1].set_title("Average units per sale by discount band")
save(fig, "06_demand_trend_discount")


# ------------------------------------------------------------
# OBJECTIVE 3 - COLD CHAIN
# Decision this supports: which shipments need better packaging or routing?
# ------------------------------------------------------------
# %% CELL 11 - Q3.1  How often do shipments suffer a temperature excursion, and where?
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
rate_bar(ax[0, 0], shipments, "packaging_type", "is_excursion", "By packaging")
rate_bar(ax[0, 1], shipments, "transport_mode", "is_excursion", "By transport mode")
rate_bar(ax[1, 0], shipments, "ship_season", "is_excursion", "By season of shipment")
rate_bar(ax[1, 1], shipments, "destination_country", "is_excursion", "By destination country")
save(fig, "07_excursion_by_group")


# %% CELL 12 - Q3.2  Do longer, hotter journeys fail more often?
shipments["transit_band"] = pd.cut(shipments["transit_days"], [0, 2, 4, 7, 14, 100],
                                   labels=["1-2d", "3-4d", "5-7d", "8-14d", "Over 14d"])
shipments["heat_band"] = pd.qcut(shipments["avg_ambient_temp_c"], 5, duplicates="drop")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
rate_bar(ax[0], shipments, "transit_band", "is_excursion", "By transit time",
         order=["1-2d", "3-4d", "5-7d", "8-14d", "Over 14d"], min_rows=10)
rate_bar(ax[1], shipments, "heat_band", "is_excursion", "By average ambient temperature",
         order=shipments["heat_band"].cat.categories)
rate_bar(ax[2], shipments, "unprotected_cold_chain", "is_excursion",
         "Cold-chain product in standard packaging (1) vs not (0)", order=[0, 1], min_rows=10)
save(fig, "08_excursion_drivers")


# ------------------------------------------------------------
# OBJECTIVE 4 - DRUG SAFETY
# Decision this supports: which adverse-event reports does the safety team review first?
# ------------------------------------------------------------
# %% CELL 13 - Q4.1  Who is most at risk of a serious event?
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
rate_bar(ax[0, 0], adverse, "age_group", "is_serious", "By age group",
         order=["Under 18", "18-39", "40-64", "65 and over"])
rate_bar(ax[0, 1], adverse, "concomitant_medications", "is_serious", "By number of other medicines")
rate_bar(ax[1, 0], adverse, "hepatic_impairment", "is_serious", "Liver impairment (1) vs none (0)", order=[0, 1])
rate_bar(ax[1, 1], adverse, "reporter_type", "is_serious", "By who reported it")
save(fig, "09_serious_by_patient")


# %% CELL 14 - Q4.2  Do onset speed, dose and product type matter?
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
rate_bar(ax[0], adverse, "onset_band", "is_serious", "By time to onset",
         order=["0-3 days", "4-14 days", "15-60 days", "Over 60 days"], min_rows=10)
rate_bar(ax[1], adverse, "high_dose_flag", "is_serious", "High dose (1) vs lower dose (0)", order=[0, 1])
rate_bar(ax[2], adverse, "therapeutic_area", "is_serious", "By therapeutic area")
save(fig, "10_serious_by_drug")


# ------------------------------------------------------------
# OBJECTIVE 5 - PRICING ANOMALIES
# Decision this supports: which invoices does finance audit?
# ------------------------------------------------------------
# %% CELL 15 - Q5.1  What does a normal price look like compared with the list price?
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(pricing["price_to_list"].clip(upper=7), bins=80, color=PALETTE["main"], ax=ax[0])
ax[0].axvline(1, color=PALETTE["alert"], ls="--", label="list price")
ax[0].set_title("Invoice price / list price")
ax[0].legend()
sns.scatterplot(data=pricing, x="expected_total_usd", y="total_amount_usd", s=8, alpha=0.4,
                color=PALETTE["main"], ax=ax[1])
ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_title("Invoice total vs units x unit price")
save(fig, "11_pricing_overview")

normal = pricing["price_to_list"].between(0.6, 1.3).mean() * 100
print(f"{normal:.1f}% of invoices sit between 60% and 130% of list price")
print("Invoices below cost:", int(pricing["sold_below_cost"].sum()))


# ------------------------------------------------------------
# WHICH NUMBERS MOVE TOGETHER?
# ------------------------------------------------------------
# %% CELL 16 - Q.  Which numeric features relate to each outcome?  (correlation, outcome columns excluded from X)
corr_sets = {
    "Batch rejection": (batches, "is_rejected",
        ["raw_material_purity_pct", "max_temp_deviation_c", "equipment_age_years", "night_shift",
         "operator_experience_years", "batch_size_units", "plant_automation", "plant_age_years"]),
    "Cold-chain excursion": (shipments, "is_excursion",
        ["transit_days", "avg_ambient_temp_c", "heat_exposure", "protective_packaging", "is_air", "cold_chain_required"]),
    "Serious adverse event": (adverse, "is_serious",
        ["patient_age", "concomitant_medications", "hepatic_impairment", "days_to_onset", "dosage_mg"]),
}
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, (title, (df, target, cols)) in zip(ax, corr_sets.items()):
    c = df[cols + [target]].astype(float).corr()[target].drop(target).sort_values()
    a.barh(c.index, c.values, color=[PALETTE["alert"] if v > 0 else PALETTE["main"] for v in c.values])
    a.axvline(0, color="black", lw=0.8)
    a.set_title(f"Correlation with: {title}", fontsize=11)
save(fig, "12_correlations")


# %% CELL 17 - EDA findings table (numbers to quote in the README)
def rate(df, col, target, value):
    s = df[df[col] == value][target]
    return round(s.mean() * 100, 1)

findings = pd.DataFrame([
    ["Objective 1", "Overall batch rejection rate (%)",              round(batches["is_rejected"].mean() * 100, 1)],
    ["Objective 1", "Rejection % when equipment is over 25 years",   rate(batches, "old_equipment_flag", "is_rejected", 1)],
    ["Objective 1", "Rejection % on night shift",                    rate(batches, "night_shift", "is_rejected", 1)],
    ["Objective 1", "Rejection % on day shift",                      rate(batches, "night_shift", "is_rejected", 0)],
    ["Objective 2", "Mean units per sale",                           round(demand["units_sold"].mean(), 0)],
    ["Objective 2", "Median units per sale",                         round(demand["units_sold"].median(), 0)],
    ["Objective 3", "Overall excursion rate (%)",                    round(shipments["is_excursion"].mean() * 100, 1)],
    ["Objective 3", "Excursion % in standard packaging",             rate(shipments, "packaging_type", "is_excursion", "Standard")],
    ["Objective 3", "Excursion % in active cooling",                 rate(shipments, "packaging_type", "is_excursion", "Active Cooling")],
    ["Objective 4", "Overall serious-event rate (%)",                round(adverse["is_serious"].mean() * 100, 1)],
    ["Objective 4", "Serious % for patients 65 and over",            rate(adverse, "age_group", "is_serious", "65 and over")],
    ["Objective 4", "Serious % for patients under 18",               rate(adverse, "age_group", "is_serious", "Under 18")],
    ["Objective 5", "Invoices priced below 60% or above 130% of list (%)",
        round((~pricing["price_to_list"].between(0.6, 1.3)).mean() * 100, 2)],
], columns=["objective", "finding", "value"])
findings
```

---

## 9. Analytical Base Table

One table per objective with only what a model is allowed to see: an ID and date (kept aside as labels), the input features and the target. Everything that leaks the answer is dropped here.

| Objective | Rows | Input columns (numeric + text) | Target |
|---|---|---|---|
| 1 Batch rejection | 4,000 | 14 (12 + 2) | `is_rejected` |
| 2 Demand forecast | 5,618 (sales with a valid date) | 16 (11 + 5) | `log_units_sold` |
| 3 Cold chain | 3,500 | 17 (11 + 6) | `is_excursion` |
| 4 Adverse events | 1,800 | 13 (9 + 4) | `is_serious` |
| 5 Pricing anomalies | 6,000 | 7 (all numeric ratios) | none |

Objective 5 is unsupervised, so its table holds seven pricing ratios only: price to list price, price to cost, total to expected total, invoice size, discount, quantity and unit price.

### Full code

#### Objectives 1 to 4 (one script; the `OBJECTIVE` setting picks the objective)

```python
# ---------- 1. Imports, settings and the objective we model first ----------
import os, json, shutil, warnings
from datetime import date
import joblib, sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, classification_report, roc_curve,
                             mean_absolute_error, mean_squared_error, r2_score)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None); pd.set_option("display.width", 200)
pd.reset_option("display.float_format")       # undo any one-decimal display setting left by an earlier notebook cell
sns.set_theme(style="whitegrid", context="notebook")
SEED = 42
MODEL_DIR = "pharma_model"

# ---- choose one: "batch_rejection", "demand_forecast", "cold_chain", "adverse_events" ----
OBJECTIVE = "batch_rejection"

CONFIG = {
    "batch_rejection": dict(
        title="Batch rejection", file="model_batches.csv", id_col="batch_id", date_col="manufacture_date",
        target="is_rejected", task="classification", split="random",
        numeric=["raw_material_purity_pct", "max_temp_deviation_c", "equipment_age_years", "night_shift",
                 "operator_experience_years", "log_batch_size", "plant_automation", "plant_age_years",
                 "plant_gmp_certified", "mfg_month", "prod_dosage_mg", "prod_shelf_life_months"],
        categorical=["formulation", "therapeutic_area"],
        leakage=["release_status", "rejection_reason", "process_deviations", "is_rejected"],
        business_problem="Rejected batches waste raw materials and plant capacity and delay supply to customers, and QA cannot inspect every batch in depth.",
        decision="Which batches should get extra inspection or testing before release.",
        ml_objective="Predict whether a batch will be rejected at quality control, using process, plant and product information known when the batch is made.",
        action="Send batches scored above the threshold to extra inspection before release."),
    "demand_forecast": dict(
        title="Demand forecast", file="model_demand.csv", id_col="sale_id", date_col="sale_date",
        target="log_units_sold", task="regression", split="time",
        numeric=["discount_pct", "sale_month_sin", "sale_month_cos", "months_since_start", "customer_credit_score",
                 "customer_tenure_years", "prod_dosage_mg", "prod_shelf_life_months", "prod_margin_pct",
                 "unit_price_usd", "is_unknown_customer"],
        categorical=["therapeutic_area", "customer_type", "sales_channel", "formulation", "customer_country"],
        leakage=["units_sold", "log_units_sold", "total_amount_usd", "expected_total_usd", "total_to_expected", "log_total_amount"],
        business_problem="Production and stock must be planned before orders arrive. Too little stock loses sales; too much ties up cash and risks expiry.",
        decision="How many units of each product to make and hold for each customer type and period.",
        ml_objective="Predict the units in a sale (on a log scale) from product, customer, price, discount and calendar information, testing on later dates than training.",
        action="Use predicted units, by product and customer type, as the starting point for the production plan."),
    "cold_chain": dict(
        title="Cold-chain excursion", file="model_shipments.csv", id_col="shipment_id", date_col="ship_date",
        target="is_excursion", task="classification", split="random",
        numeric=["transit_days", "avg_ambient_temp_c", "heat_exposure", "cold_chain_required", "protective_packaging",
                 "is_air", "is_sea", "ship_month", "unprotected_cold_chain", "product_requires_cold_chain",
                 "prod_shelf_life_months"],
        categorical=["packaging_type", "transport_mode", "ship_season", "destination_country", "customer_type", "formulation"],
        leakage=["temperature_excursion", "is_excursion", "product_written_off"],
        business_problem="Temperature excursions during shipping spoil medicines. More than half of excursions end in a write-off.",
        decision="Which shipments need better packaging, a faster route or a closer watch before they leave.",
        ml_objective="Predict whether a shipment will suffer a temperature excursion, using journey, packaging, route and product information known at dispatch.",
        action="Upgrade packaging or change the route for shipments scored above the threshold."),
    "adverse_events": dict(
        title="Serious adverse events", file="model_adverse.csv", id_col="ae_id", date_col="report_date",
        target="is_serious", task="classification", split="random",
        numeric=["patient_age", "concomitant_medications", "hepatic_impairment", "days_to_onset", "dosage_mg",
                 "batch_to_report_days", "report_month", "prod_shelf_life_months", "onset_unknown"],
        categorical=["therapeutic_area", "formulation", "reporter_type", "patient_sex"],
        leakage=["severity", "is_serious", "outcome", "hospitalised"],
        business_problem="The safety team receives more adverse-event reports than it can review at once, and serious cases must not wait.",
        decision="Which reports the pharmacovigilance team reviews first.",
        ml_objective="Predict whether a report describes a serious event (severe or life-threatening), using patient, drug and report details available when it arrives.",
        action="Review reports scored above the threshold first."),
}
cfg = CONFIG[OBJECTIVE]
OUT_DIR = f"models/{OBJECTIVE}"
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)
os.makedirs("abt", exist_ok=True)
print(f"Objective: {cfg['title']}   |   target: {cfg['target']}   |   task: {cfg['task']}")


# ---------- 2. Build the analytical base table (ABT) ----------
raw = pd.read_csv(f"{MODEL_DIR}/{cfg['file']}", parse_dates=[cfg["date_col"]])
features = cfg["numeric"] + cfg["categorical"]

# guard: no leaking column may be used as a feature
assert not set(features) & set(cfg["leakage"]), "a leaking column is in the feature list"

abt = raw[[cfg["id_col"], cfg["date_col"]] + features + [cfg["target"]]].copy()
abt = abt.drop_duplicates(subset=cfg["id_col"]).dropna(subset=[cfg["target"]])
if cfg["split"] == "time":                       # a forecast needs a date to split on
    before = len(abt)
    abt = abt.dropna(subset=[cfg["date_col"]]).sort_values(cfg["date_col"])
    print(f"Rows without a usable date left out of the forecast: {before - len(abt)}")

# ---------- 3. Drop the columns the model must not learn from ----------
# The ID and the date only label a row. They carry no information about the outcome,
# so they are moved out of the table into case_info (used later to trace each prediction).
case_info = abt[[cfg["id_col"], cfg["date_col"]]].reset_index(drop=True)
abt = abt.drop(columns=[cfg["id_col"], cfg["date_col"]]).reset_index(drop=True)
abt.to_csv(f"abt/{OBJECTIVE}_abt.csv", index=False)
case_info.to_csv(f"abt/{OBJECTIVE}_case_info.csv", index=False)

print(f"Dropped from the table: {cfg['id_col']}, {cfg['date_col']}  (kept aside in case_info)")
print(f"ABT: {abt.shape[0]} rows, {len(features)} features + target '{cfg['target']}'")
print(f"Left out on purpose (leakage): {', '.join(cfg['leakage'])}")
if cfg["task"] == "classification":
    print(f"Positive rate: {abt[cfg['target']].mean()*100:.1f}%")
missing = abt[features].isna().sum(); missing = missing[missing > 0]
print("Columns with missing values:", missing.to_dict() if len(missing) else "none")
abt.head(10)
```

#### Objective 5: Pricing anomalies (unsupervised)

```python
import os, warnings
import joblib, sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None); pd.set_option("display.width", 200)
pd.reset_option("display.float_format")
sns.set_theme(style="whitegrid", context="notebook")
SEED = 42
MODEL_DIR = "pharma_model"
OBJECTIVE = "pricing_anomalies"
OUT_DIR = f"models/{OBJECTIVE}"
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True); os.makedirs("abt", exist_ok=True)

# ---------- 1. Load the pricing table ----------
raw = pd.read_csv(f"{MODEL_DIR}/model_pricing.csv")
print(f"Invoices loaded: {len(raw)}")

# ---------- 2. The features: how unusual is the way this invoice was priced? ----------
raw["log_units"] = np.log1p(raw["units_sold"])            # quantity on a log scale, so one huge order does not dominate
raw["log_price"] = np.log1p(raw["unit_price_usd"])
features = ["price_to_list",        # price charged divided by list price (1.0 = list price)
            "price_to_cost",        # price divided by unit cost (below 1.0 = sold at a loss)
            "total_to_expected",    # invoice total divided by units x price (1.0 = the arithmetic adds up)
            "log_total_amount",     # size of the invoice
            "discount_pct",         # discount given
            "log_units",            # quantity
            "log_price"]            # unit price

abt = raw[["sale_id", "sale_date"] + features].drop_duplicates("sale_id")
before = len(abt)
abt = abt.dropna(subset=features).reset_index(drop=True)
print(f"Invoices with a blank in a feature, left out: {before - len(abt)}")

# ---------- 3. Keep the ID and date aside; they only label a row ----------
case_info = abt[["sale_id", "sale_date"]].copy()
abt = abt.drop(columns=["sale_id", "sale_date"])
abt.to_csv(f"abt/{OBJECTIVE}_abt.csv", index=False)
case_info.to_csv(f"abt/{OBJECTIVE}_case_info.csv", index=False)

print(f"\nABT: {abt.shape[0]} invoices, {abt.shape[1]} features, NO target column (unsupervised)")
print("Features:", ", ".join(features))
print("\nA quick look at what 'normal' looks like (most invoices sit near 1.0 on the ratio columns):")
abt.describe().T[["min", "25%", "50%", "75%", "max"]].round(2)
```

---

## 10. Encode Categorical Columns

Models need numbers. Text columns become 0/1 columns, one per category. Blank categories become their own `Unknown` label instead of being mistaken for the first category, and `drop_first=True` removes the one column that would repeat information.

| Objective | Columns before | Columns after encoding |
|---|---|---|
| 1 | 14 | 22 |
| 2 | 16 | 36 |
| 3 | 17 | 37 |
| 4 | 13 | 21 |
| 5 | 7 | 7 (nothing to encode) |

### Full code

#### Objectives 1 to 4

```python
# ---------- 1. Look at the text columns before encoding ----------
print("Text columns to encode:", cfg["categorical"], "\n")
for col in cfg["categorical"]:
    print(f"{col}: {abt[col].nunique()} categories, {abt[col].isna().sum()} missing")
    print("   ", abt[col].value_counts(dropna=False).to_dict())

# ---------- 2. Encode: text -> 0/1 columns ----------
# A blank category gets its own label, so it is not mistaken for the first category.
X_text = abt[features].copy()
X_text[cfg["categorical"]] = X_text[cfg["categorical"]].fillna("Unknown")

# drop_first=True removes one category per column (it is implied when all the others are 0),
# which avoids repeating the same information twice.
X_all = pd.get_dummies(X_text, columns=cfg["categorical"], drop_first=True, dtype=int)
feature_columns = list(X_all.columns)

# ---------- 3. Check the result ----------
new_columns = [c for c in feature_columns if c not in cfg["numeric"]]
print(f"\nColumns before encoding: {len(features)}   after encoding: {len(feature_columns)}")
print(f"{len(new_columns)} new 0/1 columns:", new_columns)
print("Text columns left over (should be none):", [c for c in X_all.columns if X_all[c].dtype == 'object'])
print("Rows unchanged:", len(X_all) == len(abt))

X_all.head(10)
```

#### Objective 5

```python
text_cols = abt.select_dtypes(exclude="number").columns.tolist()
print(f"Columns in the table: {abt.shape[1]}   Text (categorical) columns: {len(text_cols)} {text_cols}")

if text_cols:
    X_all = pd.get_dummies(abt, columns=text_cols, drop_first=True, dtype=int)
    print(f"Encoded {len(text_cols)} text columns into {X_all.shape[1] - (abt.shape[1] - len(text_cols))} number columns.")
else:
    X_all = abt.copy()
    print("Every column is already a number, so there is nothing to encode.")
    print("Pricing is judged on ratios (price / list price, price / cost, total / expected total), discount, quantity and price level.")

feature_columns = X_all.columns.tolist()
print(f"\nColumns the models will use ({len(feature_columns)}): {', '.join(feature_columns)}")
print("All numeric:", bool(X_all.dtypes.map(lambda t: np.issubdtype(t, np.number)).all()))
X_all.head(10)
```

---

## 11. Separate X and y

Inputs (`X`) and the answer (`y`) are separated before anything is split or scaled, so the target can never leak into the inputs.

For Objective 5 there is no `y`. The planted-anomaly answer key is held aside and opened once, at evaluation, to score the four models. No model ever trains on it.

### Full code

#### Objectives 1 to 4

```python
# ---------- 1. Separate ----------
y_all = abt[cfg["target"]]            # the target: what we want to predict
                                      # X_all (from Step 2) is every input column, already encoded

# ---------- 2. Check the separation ----------
print(f"X_all (inputs):  {X_all.shape[0]} rows x {X_all.shape[1]} columns")
print(f"y_all (target):  {y_all.shape[0]} rows, named '{y_all.name}'")
print("\nTarget is NOT inside the inputs:", cfg["target"] not in X_all.columns)
print("No leaking column is inside the inputs:", not any(c in X_all.columns for c in cfg["leakage"]))
print("No ID or date column is inside the inputs:", not any(c in X_all.columns for c in [cfg["id_col"], cfg["date_col"]]))
print("X and y have the same rows, in the same order:", bool((X_all.index == y_all.index).all()))

if cfg["task"] == "classification":
    print(f"\nTarget values: {y_all.value_counts().to_dict()}  ({y_all.mean()*100:.1f}% are 1)")
else:
    print(f"\nTarget (log of units): average {y_all.mean():.2f}, from {y_all.min():.2f} to {y_all.max():.2f}")

# ---------- 3. Look at both ----------
display_pair = pd.concat([X_all.head(5), y_all.head(5).rename(f"y: {cfg['target']}")], axis=1)
display_pair
```

#### Objective 5

```python
key = pd.read_csv("pharma_raw/anomaly_answer_key.csv").drop_duplicates("sale_id").set_index("sale_id")["is_anomaly"]
y_all = case_info["sale_id"].map(key).fillna(0).astype(int)
y_all.index = X_all.index
y_all.name = "is_anomaly"

print(f"X (inputs): {X_all.shape[0]} invoices x {X_all.shape[1]} columns")
print(f"y (answer key, held aside): {len(y_all)} values, {int(y_all.sum())} real anomalies ({y_all.mean()*100:.1f}%)")
assert "is_anomaly" not in X_all.columns and len(X_all) == len(y_all), "the answer key must not be inside X"
print("The answer key is not inside X: True")
print("The models will never see y. It is only used to score them.")
pd.DataFrame({"invoices": [len(X_all), int(y_all.sum())]}, index=["total", "real anomalies"])
```

---

## 12. Train Test Split

| Objective | Method | Train | Test | Real cases in test |
|---|---|---|---|---|
| 1 | Random, stratified 75/25 | 3,000 | 1,000 | 152 rejections |
| 2 | **By time**: earlier 80% to train, latest 20% to test (Jan to Sep 2025) | 4,494 | 1,124 | n/a |
| 3 | Random, stratified 75/25 | 2,625 | 875 | 138 excursions |
| 4 | Random, stratified 75/25 | 1,350 | 450 | 90 serious events |
| 5 | Random 75/25 | 4,500 | 1,500 | 40 anomalies |

Stratified keeps the problem rate the same in train and test. The forecast splits by time because a real forecast always predicts the future from the past.

### Full code

#### Objectives 1 to 4

```python
# ---------- 1. Split ----------
if cfg["split"] == "random":
    # 75% train, 25% test. 'stratify' keeps the same share of rejected batches in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=SEED,
        stratify=y_all if cfg["task"] == "classification" else None)
    print("Split type: random, stratified 75% / 25%")
else:
    # a forecast must predict the future: train on the past, test on the most recent 20%
    cut = int(len(X_all) * 0.8)
    X_train, X_test = X_all.iloc[:cut], X_all.iloc[cut:]
    y_train, y_test = y_all.iloc[:cut], y_all.iloc[cut:]
    print("Split type: by time, earlier 80% to train, latest 20% to test")
    print(f"Train: up to {case_info[cfg['date_col']].iloc[cut-1].date()}   Test: from {case_info[cfg['date_col']].iloc[cut].date()}")

# ---------- 2. Check the split ----------
print(f"\nX_train: {X_train.shape}   X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}      y_test: {y_test.shape}")
print("No case appears in both sets:", len(set(X_train.index) & set(X_test.index)) == 0)
print("Case IDs kept aside, traceable by row number:", case_info.loc[X_test.index[:3], cfg["id_col"]].tolist(), "...")
print("Same columns in both sets:", list(X_train.columns) == list(X_test.columns))

if cfg["task"] == "classification":
    summary = pd.DataFrame({
        "rows": [len(y_train), len(y_test)],
        "positives": [int(y_train.sum()), int(y_test.sum())],
        "positive_rate_%": [round(y_train.mean() * 100, 1), round(y_test.mean() * 100, 1)],
    }, index=["train", "test"])
else:
    summary = pd.DataFrame({
        "rows": [len(y_train), len(y_test)],
        "average_units": [round(np.expm1(y_train).mean(), 0), round(np.expm1(y_test).mean(), 0)],
    }, index=["train", "test"])
summary
```

#### Objective 5

```python
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, random_state=SEED)
# y_train and y_test are the answer key for those invoices. They are NOT used for training.

print("Split type: random 75% / 25% (no labels, so no stratification)")
print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}      y_test: {y_test.shape}   (answer key, held aside; real anomalies {int(y_train.sum())} train, {int(y_test.sum())} test)")
print("No invoice appears in both sets:", len(set(X_train.index) & set(X_test.index)) == 0)
print("Same columns in both sets:", list(X_train.columns) == list(X_test.columns))
print("Invoice IDs kept aside, traceable by row number:", case_info.loc[X_test.index[:3], "sale_id"].tolist(), "...")
pd.DataFrame({"invoices": [len(X_train), len(X_test)]}, index=["train", "test"])
```

---

## 13. Feature Scaling of Numerical Columns

The scaler is fitted on the **training rows only**, then applied to the test rows, so nothing from the test set leaks in. Blanks are filled with the training median first. Yes/No columns stay as 0/1 and only genuine numbers are scaled.

The fill values, scaled columns and scaler are saved with the model in section 19, so new data is prepared in exactly the same way in the app.

### Full code

#### Objectives 1 to 4

```python
# ---------- 1. Fill the few blank cells (using TRAIN medians only) ----------
blank_before = int(X_train.isna().sum().sum() + X_test.isna().sum().sum())
fill_values = X_train.median().to_dict()          # saved later, so new data is filled the same way
X_train = X_train.fillna(fill_values)
X_test = X_test.fillna(fill_values)
print(f"Blank cells filled: {blank_before}   Blank cells left: {int(X_train.isna().sum().sum() + X_test.isna().sum().sum())}")

# ---------- 2. Choose the columns to scale ----------
# Real numbers are scaled. Columns that are only 0/1 (flags and the encoded columns) stay as they are.
scale_cols = [c for c in cfg["numeric"] if X_train[c].nunique() > 2]
print(f"\nScaling {len(scale_cols)} numeric columns: {scale_cols}")

# ---------- 3. Fit the scaler on TRAIN only, then apply to train and test ----------
scaler = StandardScaler().fit(X_train[scale_cols])

X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[scale_cols] = scaler.transform(X_train[scale_cols])
X_test_s[scale_cols] = scaler.transform(X_test[scale_cols])

# ---------- 4. Check the result ----------
check = pd.DataFrame({
    "train_mean_before": X_train[scale_cols].mean().round(2),
    "train_mean_after": X_train_s[scale_cols].mean().round(2),
    "train_std_before": X_train[scale_cols].std().round(2),
    "train_std_after": X_train_s[scale_cols].std().round(2),
    "test_mean_after": X_test_s[scale_cols].mean().round(2),
})
print("\nAfter scaling the train columns should have mean about 0 and spread (std) about 1.")
print("Test columns will be close but not exact, because the scaler never saw them.\n")
print(check.to_string())
print(f"\nShapes unchanged: train {X_train_s.shape}, test {X_test_s.shape}")

X_train_s.head(10)
```

#### Objective 5

```python
fill_values = X_train.median().to_dict()                      # saved later, so new invoices are filled the same way
X_train = X_train.fillna(fill_values); X_test = X_test.fillna(fill_values)

scale_cols = list(X_train.columns)                            # every feature here is a real number
scaler = StandardScaler().fit(X_train[scale_cols])
X_train_s = pd.DataFrame(scaler.transform(X_train[scale_cols]), columns=scale_cols, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test[scale_cols]), columns=scale_cols, index=X_test.index)

check = pd.DataFrame({
    "train_mean_before": X_train.mean().round(2), "train_mean_after": X_train_s.mean().round(2),
    "train_std_before": X_train.std().round(2), "train_std_after": X_train_s.std().round(2),
    "test_mean_after": X_test_s.mean().round(2)})
print("After scaling the train columns should have mean about 0 and spread (std) about 1.")
print("Test columns will be close but not exact, because the scaler never saw them.\n")
print(check.to_string())
print(f"\nShapes: train {X_train_s.shape}, test {X_test_s.shape}")
X_train_s.head(10)
```

---

## 14. Train Model

Four models per objective, tuned with randomised search and cross-validation on the training rows only, plus a "no skill" baseline to show what guessing scores.

| Objective | Models tried | Cross-validation |
|---|---|---|
| 1, 3, 4 | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting | Stratified 5-fold, scored on ROC AUC |
| 2 | Ridge Regression, Decision Tree, Random Forest, Gradient Boosting | Time-ordered 5-fold, scored on error |
| 5 | Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope | Compared on the training invoices |

**How the winner is picked:** by cross-validated score on the training data. If several models sit within one standard deviation of the best, the simplest wins. The test set is not used to choose, only to confirm.

| Cross-validated ROC AUC | Logistic Regression | Gradient Boosting | Random Forest | Decision Tree | Baseline |
|---|---|---|---|---|---|
| 1 Batch rejection | **0.794** | 0.779 | 0.776 | 0.728 | 0.500 |
| 3 Cold chain | **0.796** | 0.793 | 0.795 | 0.771 | 0.500 |
| 4 Adverse events | **0.883** | 0.875 | 0.861 | 0.837 | 0.500 |

For the forecast, Ridge had the lowest cross-validated error (0.299 on the log scale, against 0.306 for Gradient Boosting). For anomalies, Elliptic Envelope scored a training PR AUC of 0.952 against 0.917 for Isolation Forest.

### Full code

#### Objective 1: Batch rejection

```python
# BUSINESS OBJECTIVE 1: REDUCE BATCH REJECTIONS
# TRAIN MODEL
import time
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

SEED = globals().get("SEED", 42)
CV_FOLDS = 5
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}   # ranking quality; PR AUC suits the rare rejected class
REFIT_ON = "roc_auc"

# --- 1. Data check ---
# (In the notebook a self-preparing block sits here. If another objective's data is loaded, it silently re-runs
#  sections 9 to 13 for this objective first. It is left out of this README to keep the training logic readable.)
if set(np.unique(y_train)) != {0, 1} or X_train_s.isna().any().any():
    raise ValueError("y_train must be 0/1 and X_train_s must have no blanks; re-run the preparation phases.")

print(f"Training data: {X_train_s.shape[0]:,} batches x {X_train_s.shape[1]} features | "
      f"rejected: {int(y_train.sum()):,} ({y_train.mean():.1%})")

# --- 2. Models and their search spaces ---
# class_weight="balanced" keeps the rare rejected class from being ignored.
candidates = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        {"C": loguniform(1e-3, 1e2)}, 10),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"max_depth": randint(3, 9), "min_samples_leaf": randint(10, 80)}, 20),
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        {"max_depth": [6, 10, 14, None], "min_samples_leaf": randint(3, 30), "max_features": ["sqrt", 0.4, 0.6]}, 10),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {"learning_rate": loguniform(0.02, 0.2), "max_depth": randint(2, 6), "max_iter": randint(100, 400),
         "min_samples_leaf": randint(20, 80), "l2_regularization": loguniform(1e-3, 10)}, 15),
}
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# --- 3. Baseline: what a model with no skill scores (always answers with the overall rejection rate) ---
baseline = DummyClassifier(strategy="prior").fit(X_train_s, y_train)
base_cv = cross_validate(DummyClassifier(strategy="prior"), X_train_s, y_train, cv=cv, scoring=SCORING)
trained_models = {"Baseline (no skill)": baseline}
rows = [{"model": "Baseline (no skill)", "cv_roc_auc": base_cv["test_roc_auc"].mean(), "cv_roc_auc_std": base_cv["test_roc_auc"].std(),
         "cv_pr_auc": base_cv["test_pr_auc"].mean(), "fit_seconds": 0.0, "best_settings": "-"}]

# --- 4. Tune every model on the training data, then keep the best version refitted on all of it ---
for name, (estimator, space, n_iter) in candidates.items():
    start = time.perf_counter()
    search = RandomizedSearchCV(estimator, space, n_iter=n_iter, scoring=SCORING, refit=REFIT_ON,
                                cv=cv, random_state=SEED, n_jobs=-1)
    search.fit(X_train_s, y_train)
    best = search.best_index_
    trained_models[name] = search.best_estimator_
    rows.append({"model": name,
                 "cv_roc_auc": search.cv_results_["mean_test_roc_auc"][best],
                 "cv_roc_auc_std": search.cv_results_["std_test_roc_auc"][best],
                 "cv_pr_auc": search.cv_results_["mean_test_pr_auc"][best],
                 "fit_seconds": time.perf_counter() - start,
                 "best_settings": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()}})

tuning_results = pd.DataFrame(rows).set_index("model")
print(f"\nTrained {len(trained_models)} models (1 baseline + {len(candidates)} tuned). The test set has not been used.\n")
tuning_results.style.format({"cv_roc_auc": "{:.3f}", "cv_roc_auc_std": "{:.3f}", "cv_pr_auc": "{:.3f}", "fit_seconds": "{:.1f}"})
```

#### Objective 2: Demand forecast

```python
# BUSINESS OBJECTIVE 2: FORECAST DEMAND
# TRAIN MODEL
import time
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, cross_validate
from sklearn.tree import DecisionTreeRegressor

SEED = globals().get("SEED", 42)
CV_SPLITS = 5
SCORING = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}   # error in log units; R-squared for context
REFIT_ON = "rmse"

# --- 1. Data check ---
# (In the notebook a self-preparing block sits here. If another objective's data is loaded, it silently re-runs
#  sections 9 to 13 for this objective first. It is left out of this README to keep the training logic readable.)
if X_train_s.isna().any().any():
    raise ValueError("X_train_s has blanks; the scaling phase should have filled them.")

print(f"Training data: {X_train_s.shape[0]:,} sales x {X_train_s.shape[1]} features | "
      f"average units per sale: {np.expm1(y_train).mean():,.0f}")

# --- 2. Models and their search spaces ---
candidates = {
    "Linear Regression (Ridge)": (
        Ridge(random_state=SEED), {"alpha": loguniform(1e-3, 1e3)}, 12),
    "Decision Tree": (
        DecisionTreeRegressor(random_state=SEED),
        {"max_depth": randint(3, 12), "min_samples_leaf": randint(10, 60)}, 20),
    "Random Forest": (
        RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        {"max_depth": [8, 12, None], "min_samples_leaf": randint(2, 20), "max_features": [1.0, 0.6, 0.4]}, 10),
    "Gradient Boosting": (
        HistGradientBoostingRegressor(random_state=SEED),
        {"learning_rate": loguniform(0.02, 0.2), "max_depth": randint(2, 6), "max_iter": randint(100, 400),
         "min_samples_leaf": randint(20, 80), "l2_regularization": loguniform(1e-3, 10)}, 15),
}
cv = TimeSeriesSplit(n_splits=CV_SPLITS)      # forward in time: never trains on the future

# --- 3. Baseline: what a model with no skill scores (always predicts the training average) ---
base_cv = cross_validate(DummyRegressor(strategy="mean"), X_train_s, y_train, cv=cv, scoring=SCORING)
trained_models = {"Baseline (no skill)": DummyRegressor(strategy="mean").fit(X_train_s, y_train)}
rows = [{"model": "Baseline (no skill)", "cv_rmse": -base_cv["test_rmse"].mean(), "cv_rmse_std": base_cv["test_rmse"].std(),
         "cv_r2": base_cv["test_r2"].mean(), "fit_seconds": 0.0, "best_settings": "-"}]

# --- 4. Tune every model on the training data, then keep the best version refitted on all of it ---
for name, (estimator, space, n_iter) in candidates.items():
    start = time.perf_counter()
    search = RandomizedSearchCV(estimator, space, n_iter=n_iter, scoring=SCORING, refit=REFIT_ON,
                                cv=cv, random_state=SEED, n_jobs=-1)
    search.fit(X_train_s, y_train)
    best = search.best_index_
    trained_models[name] = search.best_estimator_
    rows.append({"model": name,
                 "cv_rmse": -search.cv_results_["mean_test_rmse"][best],
                 "cv_rmse_std": search.cv_results_["std_test_rmse"][best],
                 "cv_r2": search.cv_results_["mean_test_r2"][best],
                 "fit_seconds": time.perf_counter() - start,
                 "best_settings": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()}})

tuning_results = pd.DataFrame(rows).set_index("model")
print(f"\nTrained {len(trained_models)} models (1 baseline + {len(candidates)} tuned). The test set has not been used.\n")
tuning_results.style.format({"cv_rmse": "{:.3f}", "cv_rmse_std": "{:.3f}", "cv_r2": "{:.3f}", "fit_seconds": "{:.1f}"})
```

#### Objective 3: Cold-chain excursions

```python
# BUSINESS OBJECTIVE 3: PREVENT COLD-CHAIN EXCURSIONS
# TRAIN MODEL
import time
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

SEED = globals().get("SEED", 42)
CV_FOLDS = 5
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}   # ranking quality; PR AUC suits the rare excursion class
REFIT_ON = "roc_auc"

# --- 1. Data check ---
# (In the notebook a self-preparing block sits here. If another objective's data is loaded, it silently re-runs
#  sections 9 to 13 for this objective first. It is left out of this README to keep the training logic readable.)
if set(np.unique(y_train)) != {0, 1} or X_train_s.isna().any().any():
    raise ValueError("y_train must be 0/1 and X_train_s must have no blanks; re-run the preparation phases.")

print(f"Training data: {X_train_s.shape[0]:,} shipments x {X_train_s.shape[1]} features | "
      f"with an excursion: {int(y_train.sum()):,} ({y_train.mean():.1%})")

# --- 2. Models and their search spaces ---
# class_weight="balanced" keeps the rare excursion class from being ignored.
candidates = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        {"C": loguniform(1e-3, 1e2)}, 10),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"max_depth": randint(3, 9), "min_samples_leaf": randint(10, 80)}, 20),
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        {"max_depth": [6, 10, 14, None], "min_samples_leaf": randint(3, 30), "max_features": ["sqrt", 0.4, 0.6]}, 10),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {"learning_rate": loguniform(0.02, 0.2), "max_depth": randint(2, 6), "max_iter": randint(100, 400),
         "min_samples_leaf": randint(20, 80), "l2_regularization": loguniform(1e-3, 10)}, 15),
}
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# --- 3. Baseline: what a model with no skill scores (always answers with the overall excursion rate) ---
baseline = DummyClassifier(strategy="prior").fit(X_train_s, y_train)
base_cv = cross_validate(DummyClassifier(strategy="prior"), X_train_s, y_train, cv=cv, scoring=SCORING)
trained_models = {"Baseline (no skill)": baseline}
rows = [{"model": "Baseline (no skill)", "cv_roc_auc": base_cv["test_roc_auc"].mean(), "cv_roc_auc_std": base_cv["test_roc_auc"].std(),
         "cv_pr_auc": base_cv["test_pr_auc"].mean(), "fit_seconds": 0.0, "best_settings": "-"}]

# --- 4. Tune every model on the training data, then keep the best version refitted on all of it ---
for name, (estimator, space, n_iter) in candidates.items():
    start = time.perf_counter()
    search = RandomizedSearchCV(estimator, space, n_iter=n_iter, scoring=SCORING, refit=REFIT_ON,
                                cv=cv, random_state=SEED, n_jobs=-1)
    search.fit(X_train_s, y_train)
    best = search.best_index_
    trained_models[name] = search.best_estimator_
    rows.append({"model": name,
                 "cv_roc_auc": search.cv_results_["mean_test_roc_auc"][best],
                 "cv_roc_auc_std": search.cv_results_["std_test_roc_auc"][best],
                 "cv_pr_auc": search.cv_results_["mean_test_pr_auc"][best],
                 "fit_seconds": time.perf_counter() - start,
                 "best_settings": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()}})

tuning_results = pd.DataFrame(rows).set_index("model")
print(f"\nTrained {len(trained_models)} models (1 baseline + {len(candidates)} tuned). The test set has not been used.\n")
tuning_results.style.format({"cv_roc_auc": "{:.3f}", "cv_roc_auc_std": "{:.3f}", "cv_pr_auc": "{:.3f}", "fit_seconds": "{:.1f}"})
```

#### Objective 4: Serious adverse events

```python
# BUSINESS OBJECTIVE 4: REDUCE SERIOUS ADVERSE EVENTS
# TRAIN MODEL
import time
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

SEED = globals().get("SEED", 42)
CV_FOLDS = 5
SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}   # ranking quality; PR AUC suits the rare serious class
REFIT_ON = "roc_auc"

# --- 1. Data check ---
# (In the notebook a self-preparing block sits here. If another objective's data is loaded, it silently re-runs
#  sections 9 to 13 for this objective first. It is left out of this README to keep the training logic readable.)
if set(np.unique(y_train)) != {0, 1} or X_train_s.isna().any().any():
    raise ValueError("y_train must be 0/1 and X_train_s must have no blanks; re-run the preparation phases.")

print(f"Training data: {X_train_s.shape[0]:,} reports x {X_train_s.shape[1]} features | "
      f"serious events: {int(y_train.sum()):,} ({y_train.mean():.1%})")

# --- 2. Models and their search spaces ---
# class_weight="balanced" keeps the rare serious class from being ignored.
candidates = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED),
        {"C": loguniform(1e-3, 1e2)}, 10),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"max_depth": randint(3, 9), "min_samples_leaf": randint(10, 80)}, 20),
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        {"max_depth": [6, 10, 14, None], "min_samples_leaf": randint(3, 30), "max_features": ["sqrt", 0.4, 0.6]}, 10),
    "Gradient Boosting": (
        HistGradientBoostingClassifier(random_state=SEED),
        {"learning_rate": loguniform(0.02, 0.2), "max_depth": randint(2, 6), "max_iter": randint(100, 400),
         "min_samples_leaf": randint(20, 80), "l2_regularization": loguniform(1e-3, 10)}, 15),
}
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

# --- 3. Baseline: what a model with no skill scores (always answers with the overall serious-event rate) ---
baseline = DummyClassifier(strategy="prior").fit(X_train_s, y_train)
base_cv = cross_validate(DummyClassifier(strategy="prior"), X_train_s, y_train, cv=cv, scoring=SCORING)
trained_models = {"Baseline (no skill)": baseline}
rows = [{"model": "Baseline (no skill)", "cv_roc_auc": base_cv["test_roc_auc"].mean(), "cv_roc_auc_std": base_cv["test_roc_auc"].std(),
         "cv_pr_auc": base_cv["test_pr_auc"].mean(), "fit_seconds": 0.0, "best_settings": "-"}]

# --- 4. Tune every model on the training data, then keep the best version refitted on all of it ---
for name, (estimator, space, n_iter) in candidates.items():
    start = time.perf_counter()
    search = RandomizedSearchCV(estimator, space, n_iter=n_iter, scoring=SCORING, refit=REFIT_ON,
                                cv=cv, random_state=SEED, n_jobs=-1)
    search.fit(X_train_s, y_train)
    best = search.best_index_
    trained_models[name] = search.best_estimator_
    rows.append({"model": name,
                 "cv_roc_auc": search.cv_results_["mean_test_roc_auc"][best],
                 "cv_roc_auc_std": search.cv_results_["std_test_roc_auc"][best],
                 "cv_pr_auc": search.cv_results_["mean_test_pr_auc"][best],
                 "fit_seconds": time.perf_counter() - start,
                 "best_settings": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in search.best_params_.items()}})

tuning_results = pd.DataFrame(rows).set_index("model")
print(f"\nTrained {len(trained_models)} models (1 baseline + {len(candidates)} tuned). The test set has not been used.\n")
tuning_results.style.format({"cv_roc_auc": "{:.3f}", "cv_roc_auc_std": "{:.3f}", "cv_pr_auc": "{:.3f}", "fit_seconds": "{:.1f}"})
```

#### Objective 5: Pricing anomalies

```python
# BUSINESS OBJECTIVE 5: FIND UNUSUAL PRICES AND INVOICES
# TRAIN MODEL
import time
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

SEED = globals().get("SEED", 42)
CONTAMINATION = 0.02      # finance can review about 2% of invoices; raise it to review more, lower it to review fewer

# --- 1. Data check ---
# (In the notebook a self-preparing block sits here. If another objective's data is loaded, it silently re-runs
#  sections 9 to 13 for this objective first. It is left out of this README to keep the training logic readable.)
if X_train_s.isna().any().any():
    raise ValueError("X_train_s has blanks; the scaling phase should have filled them.")
print(f"Training data: {len(X_train_s):,} invoices x {X_train_s.shape[1]} features | no labels are used for training")

# --- 2. Detectors ---
detectors = {
    # builds random trees; an unusual invoice is cut off from the rest in very few splits
    "Isolation Forest": IsolationForest(n_estimators=400, contamination=CONTAMINATION, random_state=SEED, n_jobs=-1),
    # compares each invoice with its closest neighbours; an invoice far from every group is unusual
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=35, contamination=CONTAMINATION, novelty=True, n_jobs=-1),
    # draws a smooth boundary around the normal invoices; anything outside it is unusual
    "One-Class SVM": OneClassSVM(kernel="rbf", gamma="scale", nu=CONTAMINATION),
    # assumes normal invoices form one bell-shaped cloud; an invoice far out in the tails is unusual
    "Elliptic Envelope": EllipticEnvelope(contamination=CONTAMINATION, support_fraction=0.9, random_state=SEED),
}

# --- 3. Train every detector on the training invoices ---
trained_models, rows = {}, []
for name, detector in detectors.items():
    start = time.perf_counter()
    detector.fit(X_train_s)
    elapsed = time.perf_counter() - start
    score = -detector.score_samples(X_train_s)                # higher = more unusual
    flagged = (detector.predict(X_train_s) == -1)
    trained_models[name] = detector
    rows.append({"model": name, "invoices_flagged_in_training": int(flagged.sum()), "share_flagged": flagged.mean(),
                 "score_median": np.median(score), "score_p99": np.percentile(score, 99), "fit_seconds": elapsed})

training_summary = pd.DataFrame(rows).set_index("model")
print(f"\nTrained {len(trained_models)} detectors; each was told to expect about {CONTAMINATION:.0%} unusual invoices. "
      "The test invoices and the answer key have not been used.\n")
training_summary.style.format({"share_flagged": "{:.1%}", "score_median": "{:.3f}", "score_p99": "{:.3f}", "fit_seconds": "{:.1f}"})
```

---

## 15. Predict X_test

Every trained model scores the test rows it has never seen. Classification models give a **risk probability** instead of a hard yes/no, so the business can choose how many cases to act on.

A sanity check follows: the average predicted risk of batches that really were rejected should be clearly higher than for batches that were fine. For the forecast, predictions are converted back from logs to units. For anomalies, each detector gives a score, and a vote counts how many detectors flag the same invoice (7 invoices were flagged by all four, 27 by three).

### Full code

#### Objective 1: Batch rejection

```python
# BUSINESS OBJECTIVE 1: REDUCE BATCH REJECTIONS
# PREDICT X_TEST
import numpy as np
import pandas as pd

# --- 1. Pre-flight checks ---
for name in ("trained_models", "X_test_s", "y_test", "case_info"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 1 train-model cell first (it prepares the data itself).")
if "Baseline (no skill)" not in trained_models or not hasattr(trained_models["Logistic Regression"], "predict_proba"):
    raise ValueError("trained_models does not hold the Objective 1 classifiers. Re-run the train-model cell.")
if len(X_test_s) != len(y_test):
    raise ValueError(f"X_test_s has {len(X_test_s)} rows but y_test has {len(y_test)}: they belong to different objectives.")

# --- 2. Every model predicts the probability that a batch is rejected ---
test_proba = {name: model_.predict_proba(X_test_s)[:, 1] for name, model_ in trained_models.items()}

predictions = pd.DataFrame(
    {"batch_id": case_info.loc[X_test_s.index, "batch_id"].to_numpy(), "actual_rejected": y_test.to_numpy()},
    index=X_test_s.index)
for name, proba in test_proba.items():
    predictions[name] = proba.round(3)

# --- 3. Sanity check: rejected batches should receive a clearly higher risk than good batches ---
is_rejected = y_test.to_numpy() == 1
risk_check = pd.DataFrame({
    "avg risk, rejected batches": {n: p[is_rejected].mean() for n, p in test_proba.items()},
    "avg risk, good batches": {n: p[~is_rejected].mean() for n, p in test_proba.items()},
})
risk_check["gap"] = risk_check.iloc[:, 0] - risk_check.iloc[:, 1]

print(f"{len(predictions):,} test batches scored by {len(test_proba)} models | "
      f"actually rejected: {int(is_rejected.sum())} ({is_rejected.mean():.1%})\n")
print("A useful model gives rejected batches a clearly higher average risk (a bigger gap is better):")
print(risk_check.round(3).to_string())
print("\nFirst 10 test batches (risk = probability of rejection):")
predictions.head(10)
```

#### Objective 2: Demand forecast

```python
# BUSINESS OBJECTIVE 2: FORECAST DEMAND
# PREDICT X_TEST
import numpy as np
import pandas as pd

for name in ("trained_models", "X_test_s", "y_test", "case_info"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 2 train-model cell first (it prepares the data itself).")
if "Baseline (no skill)" not in trained_models or "Linear Regression (Ridge)" not in trained_models:
    raise ValueError("trained_models does not hold the Objective 2 regressors. Re-run the train-model cell.")
if len(X_test_s) != len(y_test):
    raise ValueError("X_test_s and y_test have different lengths: they belong to different objectives.")

# --- 1. Every model forecasts the test sales ---
actual_units = np.expm1(y_test.to_numpy())
test_pred_units = {name: np.expm1(m.predict(X_test_s)) for name, m in trained_models.items()}

predictions = pd.DataFrame({
    "sale_id": case_info.loc[X_test_s.index, "sale_id"].to_numpy(),
    "sale_date": pd.to_datetime(case_info.loc[X_test_s.index, "sale_date"]).dt.date.to_numpy(),
    "actual_units": actual_units.round(0)}, index=X_test_s.index)
for name, units in test_pred_units.items():
    predictions[name] = units.round(0)

# --- 2. Sanity check against the real units ---
check = pd.DataFrame({
    "average forecast": {n: u.mean() for n, u in test_pred_units.items()},
    "smallest forecast": {n: u.min() for n, u in test_pred_units.items()},
    "largest forecast": {n: u.max() for n, u in test_pred_units.items()},
    "correlation with actual": {n: np.corrcoef(u, actual_units)[0, 1] if u.std() > 0 else np.nan for n, u in test_pred_units.items()}})
check.loc["ACTUAL"] = [actual_units.mean(), actual_units.min(), actual_units.max(), 1.0]

print(f"{len(predictions):,} test sales forecast by {len(test_pred_units)} models | "
      f"period {predictions['sale_date'].min()} to {predictions['sale_date'].max()} (later than all training data)\n")
print("A useful model has an average near the actual and a correlation near 1 (the baseline forecasts one number, so it has none):")
print(check.round(2).to_string())
print("\nFirst 10 test sales (units):")
predictions.head(10)
```

#### Objective 3: Cold-chain excursions

```python
# BUSINESS OBJECTIVE 3: PREVENT COLD-CHAIN EXCURSIONS
# PREDICT X_TEST
import numpy as np
import pandas as pd

# --- 1. Pre-flight checks ---
for name in ("trained_models", "X_test_s", "y_test", "case_info"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 3 train-model cell first (it prepares the data itself).")
if "Baseline (no skill)" not in trained_models or not hasattr(trained_models["Logistic Regression"], "predict_proba"):
    raise ValueError("trained_models does not hold the Objective 3 classifiers. Re-run the train-model cell.")
if len(X_test_s) != len(y_test):
    raise ValueError(f"X_test_s has {len(X_test_s)} rows but y_test has {len(y_test)}: they belong to different objectives.")

# --- 2. Every model predicts the probability that a shipment is excursion ---
test_proba = {name: model_.predict_proba(X_test_s)[:, 1] for name, model_ in trained_models.items()}

predictions = pd.DataFrame(
    {"shipment_id": case_info.loc[X_test_s.index, "shipment_id"].to_numpy(), "actual_excursion": y_test.to_numpy()},
    index=X_test_s.index)
for name, proba in test_proba.items():
    predictions[name] = proba.round(3)

# --- 3. Sanity check: shipments with an excursion should receive a clearly higher risk than shipments without an excursion ---
is_excursion = y_test.to_numpy() == 1
risk_check = pd.DataFrame({
    "avg risk, shipments with an excursion": {n: p[is_excursion].mean() for n, p in test_proba.items()},
    "avg risk, shipments without an excursion": {n: p[~is_excursion].mean() for n, p in test_proba.items()},
})
risk_check["gap"] = risk_check.iloc[:, 0] - risk_check.iloc[:, 1]

print(f"{len(predictions):,} test shipments scored by {len(test_proba)} models | "
      f"with an excursion: {int(is_excursion.sum())} ({is_excursion.mean():.1%})\n")
print("A useful model gives shipments with an excursion a clearly higher average risk (a bigger gap is better):")
print(risk_check.round(3).to_string())
print("\nFirst 10 test shipments (risk = probability of excursion):")
predictions.head(10)
```

#### Objective 4: Serious adverse events

```python
# BUSINESS OBJECTIVE 4: REDUCE SERIOUS ADVERSE EVENTS
# PREDICT X_TEST
import numpy as np
import pandas as pd

# --- 1. Pre-flight checks ---
for name in ("trained_models", "X_test_s", "y_test", "case_info"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 4 train-model cell first (it prepares the data itself).")
if "Baseline (no skill)" not in trained_models or not hasattr(trained_models["Logistic Regression"], "predict_proba"):
    raise ValueError("trained_models does not hold the Objective 4 classifiers. Re-run the train-model cell.")
if len(X_test_s) != len(y_test):
    raise ValueError(f"X_test_s has {len(X_test_s)} rows but y_test has {len(y_test)}: they belong to different objectives.")

# --- 2. Every model predicts the probability that a report is serious ---
test_proba = {name: model_.predict_proba(X_test_s)[:, 1] for name, model_ in trained_models.items()}

predictions = pd.DataFrame(
    {"ae_id": case_info.loc[X_test_s.index, "ae_id"].to_numpy(), "actual_serious": y_test.to_numpy()},
    index=X_test_s.index)
for name, proba in test_proba.items():
    predictions[name] = proba.round(3)

# --- 3. Sanity check: serious reports should receive a clearly higher risk than non-serious reports ---
is_serious = y_test.to_numpy() == 1
risk_check = pd.DataFrame({
    "avg risk, serious reports": {n: p[is_serious].mean() for n, p in test_proba.items()},
    "avg risk, non-serious reports": {n: p[~is_serious].mean() for n, p in test_proba.items()},
})
risk_check["gap"] = risk_check.iloc[:, 0] - risk_check.iloc[:, 1]

print(f"{len(predictions):,} test reports scored by {len(test_proba)} models | "
      f"actually serious: {int(is_serious.sum())} ({is_serious.mean():.1%})\n")
print("A useful model gives serious reports a clearly higher average risk (a bigger gap is better):")
print(risk_check.round(3).to_string())
print("\nFirst 10 test reports (risk = probability of serious event):")
predictions.head(10)
```

#### Objective 5: Pricing anomalies

```python
# BUSINESS OBJECTIVE 5: FIND UNUSUAL PRICES AND INVOICES
# PREDICT X_TEST
import numpy as np
import pandas as pd

for name in ("trained_models", "X_test_s", "X_test", "case_info"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 5 train-model cell first (it prepares the data itself).")
if "Isolation Forest" not in trained_models or "OBJECTIVE" not in globals() or OBJECTIVE != "pricing_anomalies":
    raise ValueError("The notebook holds a different objective. Run the Objective 5 train-model cell first.")

# --- 1. Every detector scores and flags the test invoices ---
test_score, test_flag = {}, {}
for name, m in trained_models.items():
    test_score[name] = -m.score_samples(X_test_s)          # flip the sign so that higher = more unusual
    test_flag[name] = (m.predict(X_test_s) == -1).astype(int)

predictions = pd.DataFrame({"sale_id": case_info.loc[X_test.index, "sale_id"].values}, index=X_test.index)
for name in trained_models:
    predictions[f"{name} score"] = test_score[name].round(3)
for name in trained_models:
    predictions[f"{name} flag"] = test_flag[name]
flag_cols = [f"{n} flag" for n in trained_models]
predictions["models that flag it"] = predictions[flag_cols].sum(axis=1)

print(f"{len(predictions)} test invoices scored by {len(trained_models)} models.\n")

# --- 2. Do the flags make sense? (no answer key needed) ----------
check = pd.DataFrame({
    "invoices flagged": {n: int(f.sum()) for n, f in test_flag.items()},
    "share flagged %": {n: f.mean() * 100 for n, f in test_flag.items()},
    "average price / list, flagged": {n: X_test.loc[test_flag[n] == 1, "price_to_list"].mean() for n in trained_models},
    "average price / list, not flagged": {n: X_test.loc[test_flag[n] == 0, "price_to_list"].mean() for n in trained_models},
}).round(2)
print("Each model was told to expect about 2% unusual invoices. Flagged invoices should look different from the rest:")
print(check.to_string())

agree = predictions["models that flag it"].value_counts().reindex(range(len(trained_models) + 1), fill_value=0)
print("\nHow many models flag the same invoice (more agreement = more trustworthy flag):")
print("  " + "   ".join(f"{k} model(s): {v}" for k, v in agree.items()))

# --- 3. The 10 invoices the models agree are most unusual ----------
look = X_test.join(predictions[["sale_id", "models that flag it"]])
look["average score"] = predictions[[f"{n} score" for n in trained_models]].mean(axis=1)
print("\nThe 10 most unusual test invoices (average score across the four models):")
look.sort_values("average score", ascending=False).head(10)[
    ["sale_id", "price_to_list", "price_to_cost", "total_to_expected", "discount_pct", "models that flag it", "average score"]].round(2)
```

---

## 16. Compare X and y

Which inputs drive each outcome? Permutation importance is calculated for each of the four models, and a feature counts as "agreed" when at least 3 of the 4 models rank it in their top 5.

| Objective | Features the models agree on |
|---|---|
| 1 Batch rejection | Raw material purity, maximum temperature deviation, equipment age, plant age, Injection products |
| 2 Demand | Therapeutic area (Oncology, Vaccines), sales channel (Distributor, Tender), Retail Pharmacy customers |
| 3 Cold chain | Transit days, ambient temperature, unprotected cold chain, heat exposure |
| 4 Adverse events | Dosage, patient age, number of other medicines, liver impairment, Oncology |
| 5 Pricing | Price against list price and against cost drive the anomaly scores |

These drivers match what the EDA found, which is a useful check that the models learned real patterns and not noise.

### Charts

<p align="center">
  <img width="3126" height="876" alt="obj1_features" src="https://github.com/user-attachments/assets/db18d096-0567-4453-8ce0-8537dd961c1a" />
</p>
<p align="center"><em>Objective 1: Batch rejection</em></p>

<p align="center">
  <img width="3202" height="950" alt="obj2_features" src="https://github.com/user-attachments/assets/5e3c979f-4be4-4b2f-9d51-e2d7cf458b7d" />
</p>
<p align="center"><em>Objective 2: Demand forecast</em></p>

<p align="center">
  <img width="3126" height="876" alt="obj3_features" src="https://github.com/user-attachments/assets/0cf6a966-f4c5-4221-b4b5-b25cca0c246f" />
</p>
<p align="center"><em>Objective 3: Cold-chain excursions</em></p>

<p align="center">
  <img width="3126" height="876" alt="obj4_features" src="https://github.com/user-attachments/assets/cd9f7cb5-eef9-4b0c-b044-3af6fd69eaa4" />
</p>
<p align="center"><em>Objective 4: Serious adverse events</em></p>

<p align="center">
  <img width="3126" height="875" alt="obj5_features" src="https://github.com/user-attachments/assets/1e47293c-d52c-4700-abdb-6fee999c57a8" />
</p>
<p align="center"><em>Objective 5: Pricing anomalies</em></p>

### Full code

#### Objective 1: Batch rejection

```python
# BUSINESS OBJECTIVE 1: REDUCE BATCH REJECTIONS
# COMPARE X AND y
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

SEED = globals().get("SEED", 42)
OUT_DIR = globals().get("OUT_DIR", "models/batch_rejection")
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)
N_REPEATS, TOP_N, MIN_AGREEMENT = 8, 10, 3

for name in ("trained_models", "X_train", "X_train_s", "y_train", "X_test_s", "y_test"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 1 train-model cell first (it prepares the data itself).")
real_models = [n for n in trained_models if not n.startswith("Baseline")]

# --- A. Correlation of each feature with rejection ---
correlations = X_train_s.corrwith(y_train.astype(float))
top_corr = correlations.reindex(correlations.abs().nlargest(12).index).sort_values()

# --- B. What each model relies on (permutation importance, scored by ROC AUC) ---
importance_by_model = pd.DataFrame({
    name: permutation_importance(trained_models[name], X_test_s, y_test, scoring="roc_auc",
                                 n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1).importances_mean
    for name in real_models}, index=X_test_s.columns)
importance_by_model["average"] = importance_by_model.mean(axis=1)
top_features = importance_by_model.nlargest(TOP_N, "average")

in_top5 = importance_by_model[real_models].rank(ascending=False) <= 5
agreed_features = in_top5.sum(axis=1)[lambda s: s >= MIN_AGREEMENT].sort_values(ascending=False).index.tolist()

# --- C. Rejection rate in the lowest vs highest quarter of the main numeric drivers ---
drivers = [c for c in top_features.index if c in X_train.columns and X_train[c].nunique() > 2][:5]
def quarter_rates(col):
    by_q = y_train.groupby(pd.qcut(X_train[col], 4, duplicates="drop"), observed=True).mean() * 100
    return by_q.iloc[0], by_q.iloc[-1]
rates = pd.DataFrame({c: quarter_rates(c) for c in drivers}, index=["lowest quarter (%)", "highest quarter (%)"]).T

# --- Chart ---
fig, ax = plt.subplots(1, 3, figsize=(21, 6))
ax[0].barh(top_corr.index, top_corr.values, color=np.where(top_corr.values > 0, "#c0392b", "#1f5f8b"))
ax[0].axvline(0, color="black", lw=0.8)
ax[0].set_title("A. Correlation with rejection\n(red = more rejection, blue = less)", fontsize=11)
sns.heatmap(top_features[real_models].iloc[::-1], annot=True, fmt=".3f", cmap="Blues", cbar=False, ax=ax[1])
ax[1].set_title("B. What each model relies on\n(drop in ROC AUC when the column is shuffled)", fontsize=11)
ax[1].tick_params(axis="x", rotation=20)
x = np.arange(len(rates))
ax[2].bar(x - 0.2, rates.iloc[:, 0], 0.4, color="#9db4c0", label="lowest quarter of the feature")
ax[2].bar(x + 0.2, rates.iloc[:, 1], 0.4, color="#c0392b", label="highest quarter of the feature")
ax[2].axhline(y_train.mean() * 100, color="black", ls="--", lw=1)
ax[2].set_xticks(x); ax[2].set_xticklabels(rates.index, rotation=25, ha="right")
ax[2].set_ylabel("rejection rate (%)"); ax[2].legend(fontsize=9)
ax[2].set_title("C. Rejection rate: lowest vs highest quarter\n(dashed line = overall rate)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/charts/features_vs_target.png", dpi=150, bbox_inches="tight"); plt.show()

# --- Findings ---
print(f"Features that at least {MIN_AGREEMENT} of {len(real_models)} models rank in their top 5: "
      f"{', '.join(agreed_features) if agreed_features else 'none'}")
print("Because the models agree on these, they are the safest factors to act on.\n")
summary = rates.round(1).astype(str) + "%"
summary.insert(0, "correlation with rejection", correlations[rates.index].round(3))
print(f"Main numeric drivers in plain numbers (overall rejection rate {y_train.mean():.1%}):")
summary
```

#### Objective 2: Demand forecast

```python
# BUSINESS OBJECTIVE 2: FORECAST DEMAND
# COMPARE X AND y
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

SEED = globals().get("SEED", 42)
OUT_DIR = globals().get("OUT_DIR", "models/demand_forecast")
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)
N_REPEATS, TOP_N, MIN_AGREEMENT = 8, 10, 3
GROUP_COLUMNS = ["therapeutic_area", "customer_type", "sales_channel"]

for name in ("trained_models", "X_train", "X_train_s", "y_train", "X_test_s", "y_test", "X_text"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 2 train-model cell first (it prepares the data itself).")
real_models = [n for n in trained_models if not n.startswith("Baseline")]
units_train = np.expm1(y_train)

# --- A. Correlation with demand ---
correlations = X_train_s.corrwith(y_train.astype(float))
top_corr = correlations.reindex(correlations.abs().nlargest(12).index).sort_values()

# --- B. What each model relies on (permutation importance, scored by R-squared) ---
importance_by_model = pd.DataFrame({
    name: permutation_importance(trained_models[name], X_test_s, y_test, scoring="r2",
                                 n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1).importances_mean
    for name in real_models}, index=X_test_s.columns)
importance_by_model["average"] = importance_by_model.mean(axis=1)
top_features = importance_by_model.nlargest(TOP_N, "average")
in_top5 = importance_by_model[real_models].rank(ascending=False) <= 5
agreed_features = in_top5.sum(axis=1)[lambda s: s >= MIN_AGREEMENT].sort_values(ascending=False).index.tolist()

# --- C. Average units by group ---
group_averages = pd.concat(
    [units_train.groupby(X_text.loc[X_train.index, c]).mean().rename(lambda v, c=c: f"{c}: {v}") for c in GROUP_COLUMNS]
).sort_values()
discount_quarters = units_train.groupby(pd.qcut(X_train["discount_pct"], 4, duplicates="drop"), observed=True).mean()

# --- Chart ---
fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
ax[0].barh(top_corr.index, top_corr.values, color=np.where(top_corr.values > 0, "#c0392b", "#1f5f8b"))
ax[0].axvline(0, color="black", lw=0.8)
ax[0].set_title("A. Correlation with units sold\n(red = more units, blue = fewer)", fontsize=11)
sns.heatmap(top_features[real_models].iloc[::-1], annot=True, fmt=".3f", cmap="Blues", cbar=False, ax=ax[1])
ax[1].set_title("B. What each model relies on\n(drop in R-squared when the column is shuffled)", fontsize=11)
ax[1].tick_params(axis="x", rotation=20)
ax[2].barh(group_averages.index, group_averages.values, color="#2e7d6b")
ax[2].axvline(units_train.mean(), color="black", ls="--", lw=1)
ax[2].set_xlabel("average units per sale")
ax[2].set_title("C. Average units by product area, customer type and channel\n(dashed line = overall average)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/charts/features_vs_target.png", dpi=150, bbox_inches="tight"); plt.show()

# --- Findings ---
print(f"Features that at least {MIN_AGREEMENT} of {len(real_models)} models rank in their top 5: "
      f"{', '.join(agreed_features) if agreed_features else 'none'}")
print("Because the models agree on these, they are the safest factors to plan around.\n")
print(f"Discount: average units are {discount_quarters.iloc[0]:.0f} in the lowest quarter of discount and "
      f"{discount_quarters.iloc[-1]:.0f} in the highest (correlation {correlations['discount_pct']:.2f}).\n")
print(f"Average units per sale by group (training data, overall average {units_train.mean():.0f}):")
table = group_averages.sort_values(ascending=False).round(0).astype(int).to_frame("average units")
table["vs overall"] = ((table["average units"] / units_train.mean() - 1) * 100).round(0).astype(int).astype(str) + "%"
table
```

#### Objective 3: Cold-chain excursions

```python
# BUSINESS OBJECTIVE 3: PREVENT COLD-CHAIN EXCURSIONS
# COMPARE X AND y
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

SEED = globals().get("SEED", 42)
OUT_DIR = globals().get("OUT_DIR", "models/cold_chain")
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)
N_REPEATS, TOP_N, MIN_AGREEMENT = 8, 10, 3

for name in ("trained_models", "X_train", "X_train_s", "y_train", "X_test_s", "y_test"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 3 train-model cell first (it prepares the data itself).")
real_models = [n for n in trained_models if not n.startswith("Baseline")]

# --- A. Correlation of each feature with excursion ---
correlations = X_train_s.corrwith(y_train.astype(float))
top_corr = correlations.reindex(correlations.abs().nlargest(12).index).sort_values()

# --- B. What each model relies on (permutation importance, scored by ROC AUC) ---
importance_by_model = pd.DataFrame({
    name: permutation_importance(trained_models[name], X_test_s, y_test, scoring="roc_auc",
                                 n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1).importances_mean
    for name in real_models}, index=X_test_s.columns)
importance_by_model["average"] = importance_by_model.mean(axis=1)
top_features = importance_by_model.nlargest(TOP_N, "average")

in_top5 = importance_by_model[real_models].rank(ascending=False) <= 5
agreed_features = in_top5.sum(axis=1)[lambda s: s >= MIN_AGREEMENT].sort_values(ascending=False).index.tolist()

# --- C. Excursion rate in the lowest vs highest quarter of the main numeric drivers ---
drivers = [c for c in top_features.index if c in X_train.columns and X_train[c].nunique() > 2][:5]
def quarter_rates(col):
    by_q = y_train.groupby(pd.qcut(X_train[col], 4, duplicates="drop"), observed=True).mean() * 100
    return by_q.iloc[0], by_q.iloc[-1]
rates = pd.DataFrame({c: quarter_rates(c) for c in drivers}, index=["lowest quarter (%)", "highest quarter (%)"]).T

# --- Chart ---
fig, ax = plt.subplots(1, 3, figsize=(21, 6))
ax[0].barh(top_corr.index, top_corr.values, color=np.where(top_corr.values > 0, "#c0392b", "#1f5f8b"))
ax[0].axvline(0, color="black", lw=0.8)
ax[0].set_title("A. Correlation with excursion\n(red = more excursion, blue = less)", fontsize=11)
sns.heatmap(top_features[real_models].iloc[::-1], annot=True, fmt=".3f", cmap="Blues", cbar=False, ax=ax[1])
ax[1].set_title("B. What each model relies on\n(drop in ROC AUC when the column is shuffled)", fontsize=11)
ax[1].tick_params(axis="x", rotation=20)
x = np.arange(len(rates))
ax[2].bar(x - 0.2, rates.iloc[:, 0], 0.4, color="#9db4c0", label="lowest quarter of the feature")
ax[2].bar(x + 0.2, rates.iloc[:, 1], 0.4, color="#c0392b", label="highest quarter of the feature")
ax[2].axhline(y_train.mean() * 100, color="black", ls="--", lw=1)
ax[2].set_xticks(x); ax[2].set_xticklabels(rates.index, rotation=25, ha="right")
ax[2].set_ylabel("excursion rate (%)"); ax[2].legend(fontsize=9)
ax[2].set_title("C. Excursion rate: lowest vs highest quarter\n(dashed line = overall rate)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/charts/features_vs_target.png", dpi=150, bbox_inches="tight"); plt.show()

# --- Findings ---
print(f"Features that at least {MIN_AGREEMENT} of {len(real_models)} models rank in their top 5: "
      f"{', '.join(agreed_features) if agreed_features else 'none'}")
print("Because the models agree on these, they are the safest factors to act on.\n")
summary = rates.round(1).astype(str) + "%"
summary.insert(0, "correlation with excursion", correlations[rates.index].round(3))
print(f"Main numeric drivers in plain numbers (overall excursion rate {y_train.mean():.1%}):")
summary
```

#### Objective 4: Serious adverse events

```python
# BUSINESS OBJECTIVE 4: REDUCE SERIOUS ADVERSE EVENTS
# COMPARE X AND y
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

SEED = globals().get("SEED", 42)
OUT_DIR = globals().get("OUT_DIR", "models/adverse_events")
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)
N_REPEATS, TOP_N, MIN_AGREEMENT = 8, 10, 3

for name in ("trained_models", "X_train", "X_train_s", "y_train", "X_test_s", "y_test"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 4 train-model cell first (it prepares the data itself).")
real_models = [n for n in trained_models if not n.startswith("Baseline")]

# --- A. Correlation of each feature with serious event ---
correlations = X_train_s.corrwith(y_train.astype(float))
top_corr = correlations.reindex(correlations.abs().nlargest(12).index).sort_values()

# --- B. What each model relies on (permutation importance, scored by ROC AUC) ---
importance_by_model = pd.DataFrame({
    name: permutation_importance(trained_models[name], X_test_s, y_test, scoring="roc_auc",
                                 n_repeats=N_REPEATS, random_state=SEED, n_jobs=-1).importances_mean
    for name in real_models}, index=X_test_s.columns)
importance_by_model["average"] = importance_by_model.mean(axis=1)
top_features = importance_by_model.nlargest(TOP_N, "average")

in_top5 = importance_by_model[real_models].rank(ascending=False) <= 5
agreed_features = in_top5.sum(axis=1)[lambda s: s >= MIN_AGREEMENT].sort_values(ascending=False).index.tolist()

# --- C. Serious-event rate in the lowest vs highest quarter of the main numeric drivers ---
drivers = [c for c in top_features.index if c in X_train.columns and X_train[c].nunique() > 2][:5]
def quarter_rates(col):
    by_q = y_train.groupby(pd.qcut(X_train[col], 4, duplicates="drop"), observed=True).mean() * 100
    return by_q.iloc[0], by_q.iloc[-1]
rates = pd.DataFrame({c: quarter_rates(c) for c in drivers}, index=["lowest quarter (%)", "highest quarter (%)"]).T

# --- Chart ---
fig, ax = plt.subplots(1, 3, figsize=(21, 6))
ax[0].barh(top_corr.index, top_corr.values, color=np.where(top_corr.values > 0, "#c0392b", "#1f5f8b"))
ax[0].axvline(0, color="black", lw=0.8)
ax[0].set_title("A. Correlation with serious event\n(red = more serious event, blue = less)", fontsize=11)
sns.heatmap(top_features[real_models].iloc[::-1], annot=True, fmt=".3f", cmap="Blues", cbar=False, ax=ax[1])
ax[1].set_title("B. What each model relies on\n(drop in ROC AUC when the column is shuffled)", fontsize=11)
ax[1].tick_params(axis="x", rotation=20)
x = np.arange(len(rates))
ax[2].bar(x - 0.2, rates.iloc[:, 0], 0.4, color="#9db4c0", label="lowest quarter of the feature")
ax[2].bar(x + 0.2, rates.iloc[:, 1], 0.4, color="#c0392b", label="highest quarter of the feature")
ax[2].axhline(y_train.mean() * 100, color="black", ls="--", lw=1)
ax[2].set_xticks(x); ax[2].set_xticklabels(rates.index, rotation=25, ha="right")
ax[2].set_ylabel("serious-event rate (%)"); ax[2].legend(fontsize=9)
ax[2].set_title("C. Serious-event rate: lowest vs highest quarter\n(dashed line = overall rate)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/charts/features_vs_target.png", dpi=150, bbox_inches="tight"); plt.show()

# --- Findings ---
print(f"Features that at least {MIN_AGREEMENT} of {len(real_models)} models rank in their top 5: "
      f"{', '.join(agreed_features) if agreed_features else 'none'}")
print("Because the models agree on these, they are the safest factors to act on.\n")
summary = rates.round(1).astype(str) + "%"
summary.insert(0, "correlation with serious event", correlations[rates.index].round(3))
print(f"Main numeric drivers in plain numbers (overall serious-event rate {y_train.mean():.1%}):")
summary
```

#### Objective 5: Pricing anomalies

```python
# BUSINESS OBJECTIVE 5: FIND UNUSUAL PRICES AND INVOICES
# COMPARE X AND y
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = globals().get("OUT_DIR", "models/pricing_anomalies")
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)
for name in ("trained_models", "X_test", "X_test_s", "predictions"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 5 train and predict cells first.")
models = list(trained_models)
score_cols = [f"{n} score" for n in models]
avg_score = predictions[score_cols].mean(axis=1)
features = list(X_test_s.columns)

# ---------- A. How far from normal is each feature in flagged invoices? (in standard deviations) ----------
# 0 = typical invoice, +3 = three standard deviations above typical, -3 = three below
shift = pd.DataFrame({n: X_test_s[predictions[f"{n} flag"] == 1].mean() for n in models})
shift["agreed by 3+ models"] = X_test_s[predictions["models that flag it"] >= 3].mean()

# ---------- B. Which features does the anomaly score follow? (distance from normal vs score) ----------
distance = X_test_s.sub(X_test_s.median()).abs()
corr_score = distance.corrwith(avg_score).sort_values()

# ---------- C. The picture ----------
fig, ax = plt.subplots(1, 3, figsize=(21, 6))
sns.heatmap(shift.round(1), annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax[0], cbar=False)
ax[0].set_title("A. How flagged invoices differ from normal\n(standard deviations from the typical invoice)", fontsize=11)
ax[0].tick_params(axis="x", rotation=25)

ax[1].barh(corr_score.index, corr_score.values, color="#c0392b")
ax[1].set_title("B. Which feature drives the anomaly score\n(correlation of distance-from-normal with score)", fontsize=11)

votes = predictions["models that flag it"]
ax[2].scatter(X_test["price_to_list"].clip(upper=7), X_test["price_to_cost"].clip(upper=26), s=8, alpha=0.3, color="#9db4c0", label="not flagged")
for k, col in [(1, "#f5b041"), (2, "#e67e22"), (3, "#c0392b"), (4, "#7b241c")]:
    m = votes == k
    ax[2].scatter(X_test.loc[m, "price_to_list"].clip(upper=7), X_test.loc[m, "price_to_cost"].clip(upper=26), s=45, color=col, label=f"flagged by {k} model(s)")
ax[2].set_xlabel("price / list price"); ax[2].set_ylabel("price / unit cost"); ax[2].legend(fontsize=8)
ax[2].set_title("C. Where the flagged invoices sit", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/charts/features_vs_scores.png", dpi=150, bbox_inches="tight"); plt.show()

# ---------- D. In words ----------
top_drivers = corr_score.sort_values(ascending=False).head(3).index.tolist()
print(f"Features that move the anomaly score most: {', '.join(top_drivers)}")
for n in models:
    biggest = shift[n].abs().sort_values(ascending=False).head(2).index.tolist()
    direction = ", ".join(f"{c} ({shift.loc[c, n]:+.1f} sd)" for c in biggest)
    print(f"  {n}: flagged invoices differ most on {direction}")
over = (X_test["price_to_list"][votes >= 3] > 1.5).sum(); under = (X_test["price_to_list"][votes >= 3] < 0.5).sum()
print(f"\nInvoices flagged by 3 or more models: {int((votes >= 3).sum())}  ->  {over} priced above 1.5x list, {under} below 0.5x list")

rows = []
for c in ["price_to_list", "price_to_cost", "total_to_expected", "discount_pct"]:
    rows.append({"feature": c, "typical invoice": X_test[c].median().round(2),
                 "flagged by 3+ models (average)": X_test.loc[votes >= 3, c].mean().round(2),
                 "not flagged (average)": X_test.loc[votes == 0, c].mean().round(2)})
print("\nIn plain numbers (test invoices):")
pd.DataFrame(rows).set_index("feature")
```

---

## 17. Evaluate Model

All figures below are on the unseen test rows.

**Classification models (ROC AUC: 0.5 is guessing, 1.0 is perfect)**

| Objective | Logistic Regression | Gradient Boosting | Random Forest | Decision Tree |
|---|---|---|---|---|
| 1 Batch rejection | **0.817** | 0.810 | 0.798 | 0.748 |
| 3 Cold chain | **0.817** | 0.804 | 0.806 | 0.773 |
| 4 Adverse events | **0.910** | 0.914 | 0.898 | 0.871 |

The more complex models are never meaningfully better, so the simplest model wins. In Objective 4, Gradient Boosting is 0.004 higher, which is within noise for 90 serious cases.

**The cut-off is a business choice.** Each model gives a risk percentage, so I set two cut-offs using only training data:

| Objective | Cut-off | Cases flagged | Real cases caught | Precision | Recall |
|---|---|---|---|---|---|
| 1 Batch rejection | Balanced (0.68) | 176 | 80 of 152 | 45% | 53% |
| 1 Batch rejection | Cost-based (0.39) | 471 | 131 of 152 | 28% | 86% |
| 3 Cold chain | Balanced (0.60) | 165 | 73 of 138 | 44% | 53% |
| 3 Cold chain | Cost-based (0.26) | 622 | 134 of 138 | 22% | 97% |
| 4 Adverse events | Balanced (0.65) | 106 | 66 of 90 | 62% | 73% |
| 4 Adverse events | Cost-based (0.25) | 244 | 89 of 90 | 36% | 99% |

**Demand forecast (Ridge):** average error **108 units per sale (21%)** against 337 for guessing the average, with 56% of forecasts within 20% of the actual. Total bias is just -5.8 units.

**Anomaly detection (Elliptic Envelope):** 36 of 40 anomalies caught, 4 false alarms (90% precision and 90% recall). A fixed rule (price below 0.5x or above 1.5x list) catches all 40 with no false alarms, which beats every model on this data.

### Charts

<p align="center">
  <img width="2976" height="815" alt="obj1_evaluation" src="https://github.com/user-attachments/assets/a90eb23a-5b84-4b42-84dd-7a54d9d4a9ff" />
</p>
<p align="center"><em>Objective 1: Batch rejection, evaluation on unseen data</em></p>

<p align="center">
  <img width="2976" height="845" alt="obj2_evaluation" src="https://github.com/user-attachments/assets/f12b8083-1ff1-4918-bc29-171a4193fedd" />
</p>
<p align="center"><em>Objective 2: Demand forecast, evaluation on unseen data</em></p>

<p align="center">
  <img width="2976" height="815" alt="obj3_evaluation" src="https://github.com/user-attachments/assets/f1c51e28-82e1-4511-8940-349ebafba33d" />
</p>
<p align="center"><em>Objective 3: Cold-chain excursions, evaluation on unseen data</em></p>

<p align="center">
  <img width="2976" height="815" alt="obj4_evaluation" src="https://github.com/user-attachments/assets/94c8b2a7-5a79-44ad-94d7-eb56c9fcd710" />
</p>
<p align="center"><em>Objective 4: Serious adverse events, evaluation on unseen data</em></p>

<p align="center">
  <img width="2975" height="815" alt="obj5_evaluation" src="https://github.com/user-attachments/assets/d85067cb-1f2b-4ed9-899b-d229d3b1b06c" />
</p>
<p align="center"><em>Objective 5: Pricing anomalies, evaluation on unseen data</em></p>

### Full code

#### Objective 1: Batch rejection

```python
# BUSINESS OBJECTIVE 1: REDUCE BATCH REJECTIONS
# EVALUATE MODEL
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
try:
    from IPython.display import display
except ImportError:
    display = print

SEED = globals().get("SEED", 42)
OUT_DIR = globals().get("OUT_DIR", "models/batch_rejection")
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)

# Example business costs: replace with real figures and run the cell again.
COST_PER_INSPECTION = 500          # one extra inspection
LOSS_PER_MISSED_REJECTION = 5000   # one rejected batch that slips through (inspection finds it and avoids this loss)
SIMPLICITY_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]   # simplest first

for name in ("trained_models", "tuning_results", "test_proba", "X_train_s", "y_train", "y_test", "X_test_s", "importance_by_model"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 1 train-model cell, then predict and compare, first.")
real_models = [n for n in SIMPLICITY_ORDER if n in trained_models]
y_true = y_test.to_numpy()
n_test, n_pos = len(y_true), int(y_true.sum())

# --- 1. CHOOSE the model on training evidence ---
cv_auc = tuning_results.loc[real_models, "cv_roc_auc"]
spread = tuning_results.loc[cv_auc.idxmax(), "cv_roc_auc_std"]
contenders = [n for n in real_models if cv_auc[n] >= cv_auc.max() - spread]
best_name = contenders[0]                       # simplest model within one standard deviation of the best

# --- 2. SET the cut-offs on out-of-fold training predictions ---
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
grid = np.arange(0.05, 0.96, 0.01)
def best_cutoff(oof, score):
    return float(grid[np.argmax([score(oof >= t) for t in grid])])
def net_benefit(inspected, caught):
    return caught * LOSS_PER_MISSED_REJECTION - inspected * COST_PER_INSPECTION
oof = {n: cross_val_predict(clone(trained_models[n]), X_train_s, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
       for n in real_models}
balanced_cutoff = {n: best_cutoff(oof[n], lambda f: f1_score(y_train, f, zero_division=0)) for n in real_models}
y_tr = y_train.to_numpy()
money_cutoff = best_cutoff(oof[best_name], lambda f: net_benefit(f.sum(), y_tr[f].sum()))
threshold = balanced_cutoff[best_name]

# --- 3. CONFIRM on the test set ---
def test_row(name, proba, cutoff):
    flag = (proba >= cutoff).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, flag).ravel())
    top20 = np.argsort(-proba)[: n_test // 5]
    return {"roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
            "found_in_riskiest_20%": y_true[top20].sum() / n_pos * 100, "flagged": tp + fp, "caught": tp, "missed": fn,
            "false_alarms": fp, "precision_%": precision_score(y_true, flag, zero_division=0) * 100,
            "recall_%": recall_score(y_true, flag) * 100, "f1": f1_score(y_true, flag), "tn": tn, "fp": fp, "fn": fn, "tp": tp}
test_table = pd.DataFrame({n: test_row(n, test_proba[n], balanced_cutoff[n]) for n in real_models}).T
count_cols = ["flagged", "caught", "missed", "false_alarms", "tn", "fp", "fn", "tp"]
test_table[count_cols] = test_table[count_cols].astype(int)

model = trained_models[best_name]
metrics = {k: round(float(v), 4) for k, v in test_table.loc[best_name].items()}   # numbers only; counts are converted below
metrics.update(true_negatives=metrics.pop("tn"), false_positives=metrics.pop("fp"),
               false_negatives=metrics.pop("fn"), true_positives=metrics.pop("tp"),
               baseline_roc_auc=round(float(roc_auc_score(y_true, test_proba["Baseline (no skill)"])), 4))
importance = importance_by_model[best_name].sort_values(ascending=False)
best_on_test = test_table["roc_auc"].idxmax()

# --- 4. PRICE the decision ---
p_best = test_proba[best_name]
def strategy(label, flag):
    return {"strategy": label, "batches inspected": int(flag.sum()), "rejections caught": int(y_true[flag].sum()),
            "net benefit": int(net_benefit(flag.sum(), y_true[flag].sum()))}
rng = np.random.default_rng(SEED)
random_flag = np.zeros(n_test, bool); random_flag[rng.choice(n_test, int((p_best >= threshold).sum()), replace=False)] = True
money = pd.DataFrame([
    strategy("Inspect nothing", np.zeros(n_test, bool)),
    strategy("Inspect every batch", np.ones(n_test, bool)),
    strategy(f"{best_name}, balanced cut-off ({threshold:.2f})", p_best >= threshold),
    strategy(f"{best_name}, money-best cut-off ({money_cutoff:.2f})", p_best >= money_cutoff),
    strategy("Inspect the same number at random", random_flag)]).set_index("strategy")

# --- Charts ---
fig, ax = plt.subplots(1, 3, figsize=(20, 5.6))
cm = np.array([[metrics["true_negatives"], metrics["false_positives"]], [metrics["false_negatives"], metrics["true_positives"]]]).astype(int)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0],
            xticklabels=["predicted good", "predicted rejected"], yticklabels=["actually good", "actually rejected"])
ax[0].set_title(f"Confusion matrix: {best_name}\n(cut-off {threshold:.2f}, test set)")
for n in real_models:
    fpr, tpr, _ = roc_curve(y_true, test_proba[n])
    ax[1].plot(fpr, tpr, lw=3 if n == best_name else 1.3, label=f"{n} ({test_table.loc[n, 'roc_auc']:.2f})")
ax[1].plot([0, 1], [0, 1], "k--", lw=0.8, label="random guessing (0.50)")
ax[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves (test set)"); ax[1].legend(fontsize=8)
shares = np.linspace(0.02, 1.0, 50)
for n in real_models:
    order = np.argsort(-test_proba[n])
    ax[2].plot(shares * 100, [y_true[order[: max(int(n_test * s), 1)]].sum() / n_pos * 100 for s in shares],
               lw=3 if n == best_name else 1.3, label=n)
ax[2].plot([0, 100], [0, 100], "k--", lw=0.8, label="random picking")
ax[2].set(xlabel="share of batches inspected, riskiest first (%)", ylabel="share of all rejections found (%)",
          title="How quickly each model finds the rejections"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/charts/evaluation.png", dpi=150, bbox_inches="tight"); plt.show()

# --- Report ---
print("1. CHOICE (training evidence)")
print(tuning_results.loc[real_models, ["cv_roc_auc", "cv_roc_auc_std", "cv_pr_auc"]].round(3).to_string())
print(f"   Within one standard deviation of the best: {', '.join(contenders)}  ->  chosen: {best_name}\n")
print("2. TEST SET, ALL MODELS (cut-off per model set on training data)")
print(test_table.drop(columns=["tn", "fp", "fn", "tp"]).round(3).to_string())
if best_on_test == best_name:
    print(f"   The test set CONFIRMS the choice: {best_name} also has the best test ROC AUC.\n")
else:
    gap = test_table.loc[best_on_test, "roc_auc"] - test_table.loc[best_name, "roc_auc"]
    print(f"   {best_on_test} scores {gap:.3f} ROC AUC higher on the test set; with {n_pos} rejections that is within noise, so the choice stands.\n")
print(f"3. {best_name.upper()} AT THE BALANCED CUT-OFF ({threshold:.2f})")
print(f"   Flags {metrics['flagged']:.0f} of {n_test:,} batches, catches {metrics['caught']:.0f} of {n_pos} rejections "
      f"(recall {metrics['recall_%']:.0f}%) with {metrics['false_alarms']:.0f} false alarms (precision {metrics['precision_%']:.0f}%).\n")
print(f"4. IS IT WORTH USING? Costs: {COST_PER_INSPECTION} per inspection, {LOSS_PER_MISSED_REJECTION} per missed rejection.")
print(f"   An inspection pays for itself above {COST_PER_INSPECTION / LOSS_PER_MISSED_REJECTION:.0%} precision; the model's is {metrics['precision_%']:.0f}%.")
print("   Use the balanced cut-off when inspectors are scarce, the money-best cut-off to maximise savings.\n")
display(money)
```

#### Objective 2: Demand forecast

```python
# BUSINESS OBJECTIVE 2: FORECAST DEMAND
# EVALUATE MODEL
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from IPython.display import display
except ImportError:
    display = print

OUT_DIR = globals().get("OUT_DIR", "models/demand_forecast")
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)

# Example stock costs: replace with real figures and run the cell again.
COST_PER_UNIT_OVERSTOCK = 2.0    # holding cost, expiry risk and tied-up cash for each unit made but not sold
COST_PER_UNIT_SHORTAGE = 6.0     # lost margin and goodwill for each unit demanded but not available
SIMPLICITY_ORDER = ["Linear Regression (Ridge)", "Decision Tree", "Random Forest", "Gradient Boosting"]   # simplest first

for name in ("trained_models", "tuning_results", "test_pred_units", "y_test", "X_test_s", "X_text", "case_info", "importance_by_model"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 2 train-model cell, then predict and compare, first.")
real_models = [n for n in SIMPLICITY_ORDER if n in trained_models]
actual = np.expm1(y_test.to_numpy())

# --- 1. CHOOSE the model on training evidence (lower error is better) ---
cv_rmse = tuning_results.loc[real_models, "cv_rmse"]
spread = tuning_results.loc[cv_rmse.idxmin(), "cv_rmse_std"]
contenders = [n for n in real_models if cv_rmse[n] <= cv_rmse.min() + spread]
best_name = contenders[0]                        # simplest model within one standard deviation of the best

# --- 2. CONFIRM on the test set ---
def test_row(units):
    err = units - actual
    return {"r2 (log scale)": r2_score(y_test, np.log1p(units)),
            "avg error (units)": mean_absolute_error(actual, units),
            "typical error (units)": np.sqrt(mean_squared_error(actual, units)),
            "error % of units (WAPE)": np.abs(err).sum() / actual.sum() * 100,
            "within 20% of actual (%)": (np.abs(err) / actual <= 0.20).mean() * 100,
            "bias (units, + = over-forecast)": err.mean()}
test_table = pd.DataFrame({n: test_row(u) for n, u in test_pred_units.items()}).T
best_on_test = test_table.loc[real_models, "avg error (units)"].idxmin()

model = trained_models[best_name]
p_best = test_pred_units[best_name]
threshold = None                                  # a forecast has no yes/no cut-off (kept so the save step uses the same names)
metrics = {k: round(float(v), 4) for k, v in test_table.loc[best_name].items()}
metrics["baseline_avg_error_units"] = round(float(test_table.loc["Baseline (no skill)", "avg error (units)"]), 4)
importance = importance_by_model[best_name].sort_values(ascending=False)

# planning view: production plans work with totals by product area and month
plan = pd.DataFrame({"area": X_text.loc[X_test_s.index, "therapeutic_area"].to_numpy(),
                     "month": pd.to_datetime(case_info.loc[X_test_s.index, "sale_date"]).dt.to_period("M").astype(str).to_numpy(),
                     "actual": actual, "forecast": p_best}).groupby(["area", "month"], as_index=False).sum()
plan_wape = np.abs(plan["forecast"] - plan["actual"]).sum() / plan["actual"].sum() * 100

# --- 3. PRICE the decision ---
def stock_cost(forecast):
    over, short = np.clip(forecast - actual, 0, None).sum(), np.clip(actual - forecast, 0, None).sum()
    return {"units over-made": round(over), "units short": round(short),
            "total stock cost": round(over * COST_PER_UNIT_OVERSTOCK + short * COST_PER_UNIT_SHORTAGE)}
y_train_units = np.expm1(y_train)
money = pd.DataFrame({
    f"Forecast with {best_name}": stock_cost(p_best),
    "Plan every sale at the typical (median) size": stock_cost(np.full(len(actual), y_train_units.median())),
    "Plan every sale at the average size": stock_cost(np.full(len(actual), y_train_units.mean()))}).T
saving = money.iloc[1:]["total stock cost"].min() - money.iloc[0]["total stock cost"]

# --- Charts ---
fig, ax = plt.subplots(1, 3, figsize=(20, 5.8))
ax[0].scatter(actual, p_best, s=8, alpha=0.4, color="#1f5f8b")
lim = [min(actual.min(), p_best.min()), max(actual.max(), p_best.max())]
ax[0].plot(lim, lim, "k--", lw=1, label="perfect forecast")
ax[0].set(xscale="log", yscale="log", xlabel="actual units (log scale)", ylabel="forecast units (log scale)",
          title=f"Forecast vs actual: {best_name}\n(R-squared {metrics['r2 (log scale)']:.2f}, test set)"); ax[0].legend(fontsize=9)
pct = np.clip((p_best - actual) / actual * 100, -100, 150)
ax[1].hist(pct, bins=50, color="#2e7d6b"); ax[1].axvline(0, color="black", lw=1)
ax[1].axvspan(-20, 20, color="gold", alpha=0.25, label=f"within 20% ({metrics['within 20% of actual (%)']:.0f}% of sales)")
ax[1].set(xlabel="forecast error (%)   negative = under-forecast, positive = over-forecast", ylabel="number of sales",
          title="How far off is each forecast?"); ax[1].legend(fontsize=9)
err_plot = test_table["avg error (units)"].sort_values(ascending=False)
ax[2].barh(err_plot.index, err_plot.values, color=np.where(err_plot.index == best_name, "#c0392b", "#9db4c0"))
ax[2].set(xlabel="average error per sale (units), lower is better", title="All models on the test set\n(chosen model in red)")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/charts/evaluation.png", dpi=150, bbox_inches="tight"); plt.show()

# --- Report ---
print("1. CHOICE (training evidence, forward-in-time cross-validation; lower error is better)")
print(tuning_results.loc[real_models, ["cv_rmse", "cv_rmse_std", "cv_r2"]].round(3).to_string())
print(f"   Within one standard deviation of the best: {', '.join(contenders)}  ->  chosen: {best_name}\n")
print("2. TEST SET (latest 20% of sales), ALL MODELS")
print(test_table.round(2).to_string())
if best_on_test == best_name:
    print(f"   The test set CONFIRMS the choice: {best_name} also has the lowest average error.\n")
else:
    gap = test_table.loc[best_name, "avg error (units)"] - test_table.loc[best_on_test, "avg error (units)"]
    print(f"   {best_on_test} has an average error {gap:.1f} units lower on the test set. That is within normal test-sample noise, so the choice stands.\n")
print(f"3. {best_name.upper()}")
print(f"   Average error {metrics['avg error (units)']:.0f} units per sale ({metrics['error % of units (WAPE)']:.0f}% of units sold) against "
      f"{metrics['baseline_avg_error_units']:.0f} for the no-skill baseline; {metrics['within 20% of actual (%)']:.0f}% of sales forecast within 20%.")
print(f"   Bias {metrics['bias (units, + = over-forecast)']:+.0f} units per sale. Planning view: summed by product area and month, the error falls to {plan_wape:.0f}% of units.\n")
print(f"4. IS IT WORTH USING? Costs: {COST_PER_UNIT_OVERSTOCK} per unit over-made, {COST_PER_UNIT_SHORTAGE} per unit short.")
print(f"   Saving against the better of the two simple plans over the test period: {saving:,.0f}.\n")
display(money)
```

#### Objective 3: Cold-chain excursions

```python
# BUSINESS OBJECTIVE 3: PREVENT COLD-CHAIN EXCURSIONS
# EVALUATE MODEL
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
try:
    from IPython.display import display
except ImportError:
    display = print

SEED = globals().get("SEED", 42)
OUT_DIR = globals().get("OUT_DIR", "models/cold_chain")
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)

# Example business costs: replace with real figures and run the cell again.
COST_PER_PROTECTION = 300          # upgrading the packaging or route of one shipment
LOSS_PER_MISSED_EXCURSION = 4000   # average loss when an unprotected shipment suffers an excursion (protection avoids it)
SIMPLICITY_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]   # simplest first

for name in ("trained_models", "tuning_results", "test_proba", "X_train_s", "y_train", "y_test", "X_test_s", "importance_by_model"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 3 train-model cell, then predict and compare, first.")
real_models = [n for n in SIMPLICITY_ORDER if n in trained_models]
y_true = y_test.to_numpy()
n_test, n_pos = len(y_true), int(y_true.sum())

# --- 1. CHOOSE the model on training evidence ---
cv_auc = tuning_results.loc[real_models, "cv_roc_auc"]
spread = tuning_results.loc[cv_auc.idxmax(), "cv_roc_auc_std"]
contenders = [n for n in real_models if cv_auc[n] >= cv_auc.max() - spread]
best_name = contenders[0]                       # simplest model within one standard deviation of the best

# --- 2. SET the cut-offs on out-of-fold training predictions ---
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
grid = np.arange(0.05, 0.96, 0.01)
def best_cutoff(oof, score):
    return float(grid[np.argmax([score(oof >= t) for t in grid])])
def net_benefit(protected, caught):
    return caught * LOSS_PER_MISSED_EXCURSION - protected * COST_PER_PROTECTION
oof = {n: cross_val_predict(clone(trained_models[n]), X_train_s, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
       for n in real_models}
balanced_cutoff = {n: best_cutoff(oof[n], lambda f: f1_score(y_train, f, zero_division=0)) for n in real_models}
y_tr = y_train.to_numpy()
money_cutoff = best_cutoff(oof[best_name], lambda f: net_benefit(f.sum(), y_tr[f].sum()))
threshold = balanced_cutoff[best_name]

# --- 3. CONFIRM on the test set ---
def test_row(name, proba, cutoff):
    flag = (proba >= cutoff).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, flag).ravel())
    top20 = np.argsort(-proba)[: n_test // 5]
    return {"roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
            "found_in_riskiest_20%": y_true[top20].sum() / n_pos * 100, "flagged": tp + fp, "caught": tp, "missed": fn,
            "false_alarms": fp, "precision_%": precision_score(y_true, flag, zero_division=0) * 100,
            "recall_%": recall_score(y_true, flag) * 100, "f1": f1_score(y_true, flag), "tn": tn, "fp": fp, "fn": fn, "tp": tp}
test_table = pd.DataFrame({n: test_row(n, test_proba[n], balanced_cutoff[n]) for n in real_models}).T
count_cols = ["flagged", "caught", "missed", "false_alarms", "tn", "fp", "fn", "tp"]
test_table[count_cols] = test_table[count_cols].astype(int)

model = trained_models[best_name]
metrics = {k: round(float(v), 4) for k, v in test_table.loc[best_name].items()}   # numbers only; counts are converted below
metrics.update(true_negatives=metrics.pop("tn"), false_positives=metrics.pop("fp"),
               false_negatives=metrics.pop("fn"), true_positives=metrics.pop("tp"),
               baseline_roc_auc=round(float(roc_auc_score(y_true, test_proba["Baseline (no skill)"])), 4))
importance = importance_by_model[best_name].sort_values(ascending=False)
best_on_test = test_table["roc_auc"].idxmax()

# --- 4. PRICE the decision ---
p_best = test_proba[best_name]
def strategy(label, flag):
    return {"strategy": label, "shipments protected": int(flag.sum()), "excursions caught": int(y_true[flag].sum()),
            "net benefit": int(net_benefit(flag.sum(), y_true[flag].sum()))}
rng = np.random.default_rng(SEED)
random_flag = np.zeros(n_test, bool); random_flag[rng.choice(n_test, int((p_best >= threshold).sum()), replace=False)] = True
money = pd.DataFrame([
    strategy("Protect nothing", np.zeros(n_test, bool)),
    strategy("Protect every shipment", np.ones(n_test, bool)),
    strategy(f"{best_name}, balanced cut-off ({threshold:.2f})", p_best >= threshold),
    strategy(f"{best_name}, money-best cut-off ({money_cutoff:.2f})", p_best >= money_cutoff),
    strategy("Protect the same number at random", random_flag)]).set_index("strategy")

# --- Charts ---
fig, ax = plt.subplots(1, 3, figsize=(20, 5.6))
cm = np.array([[metrics["true_negatives"], metrics["false_positives"]], [metrics["false_negatives"], metrics["true_positives"]]]).astype(int)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0],
            xticklabels=["predicted good", "predicted excursion"], yticklabels=["actually good", "actually excursion"])
ax[0].set_title(f"Confusion matrix: {best_name}\n(cut-off {threshold:.2f}, test set)")
for n in real_models:
    fpr, tpr, _ = roc_curve(y_true, test_proba[n])
    ax[1].plot(fpr, tpr, lw=3 if n == best_name else 1.3, label=f"{n} ({test_table.loc[n, 'roc_auc']:.2f})")
ax[1].plot([0, 1], [0, 1], "k--", lw=0.8, label="random guessing (0.50)")
ax[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves (test set)"); ax[1].legend(fontsize=8)
shares = np.linspace(0.02, 1.0, 50)
for n in real_models:
    order = np.argsort(-test_proba[n])
    ax[2].plot(shares * 100, [y_true[order[: max(int(n_test * s), 1)]].sum() / n_pos * 100 for s in shares],
               lw=3 if n == best_name else 1.3, label=n)
ax[2].plot([0, 100], [0, 100], "k--", lw=0.8, label="random picking")
ax[2].set(xlabel="share of shipments protected, riskiest first (%)", ylabel="share of all excursions found (%)",
          title="How quickly each model finds the excursions"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/charts/evaluation.png", dpi=150, bbox_inches="tight"); plt.show()

# --- Report ---
print("1. CHOICE (training evidence)")
print(tuning_results.loc[real_models, ["cv_roc_auc", "cv_roc_auc_std", "cv_pr_auc"]].round(3).to_string())
print(f"   Within one standard deviation of the best: {', '.join(contenders)}  ->  chosen: {best_name}\n")
print("2. TEST SET, ALL MODELS (cut-off per model set on training data)")
print(test_table.drop(columns=["tn", "fp", "fn", "tp"]).round(3).to_string())
if best_on_test == best_name:
    print(f"   The test set CONFIRMS the choice: {best_name} also has the best test ROC AUC.\n")
else:
    gap = test_table.loc[best_on_test, "roc_auc"] - test_table.loc[best_name, "roc_auc"]
    print(f"   {best_on_test} scores {gap:.3f} ROC AUC higher on the test set; with {n_pos} excursions that is within noise, so the choice stands.\n")
print(f"3. {best_name.upper()} AT THE BALANCED CUT-OFF ({threshold:.2f})")
print(f"   Flags {metrics['flagged']:.0f} of {n_test:,} shipments, catches {metrics['caught']:.0f} of {n_pos} excursions "
      f"(recall {metrics['recall_%']:.0f}%) with {metrics['false_alarms']:.0f} false alarms (precision {metrics['precision_%']:.0f}%).\n")
print(f"4. IS IT WORTH USING? Costs: {COST_PER_PROTECTION} per protection, {LOSS_PER_MISSED_EXCURSION} per missed excursion.")
print(f"   A protection pays for itself above {COST_PER_PROTECTION / LOSS_PER_MISSED_EXCURSION:.0%} precision; the model's is {metrics['precision_%']:.0f}%.")
print("   Use the balanced cut-off when protection capacity is limited, the money-best cut-off to maximise savings.\n")
display(money)
```

#### Objective 4: Serious adverse events

```python
# BUSINESS OBJECTIVE 4: REDUCE SERIOUS ADVERSE EVENTS
# EVALUATE MODEL
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
try:
    from IPython.display import display
except ImportError:
    display = print

SEED = globals().get("SEED", 42)
OUT_DIR = globals().get("OUT_DIR", "models/adverse_events")
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)

# Example business costs: replace with real figures and run the cell again.
COST_PER_REVIEW = 150              # staff time to review one report urgently, ahead of the queue
LOSS_PER_LATE_SERIOUS_CASE = 3000  # cost when a serious case waits in the queue (regulatory deadline, patient risk, legal exposure)
SIMPLICITY_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]   # simplest first

for name in ("trained_models", "tuning_results", "test_proba", "X_train_s", "y_train", "y_test", "X_test_s", "importance_by_model"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 4 train-model cell, then predict and compare, first.")
real_models = [n for n in SIMPLICITY_ORDER if n in trained_models]
y_true = y_test.to_numpy()
n_test, n_pos = len(y_true), int(y_true.sum())

# --- 1. CHOOSE the model on training evidence ---
cv_auc = tuning_results.loc[real_models, "cv_roc_auc"]
spread = tuning_results.loc[cv_auc.idxmax(), "cv_roc_auc_std"]
contenders = [n for n in real_models if cv_auc[n] >= cv_auc.max() - spread]
best_name = contenders[0]                       # simplest model within one standard deviation of the best

# --- 2. SET the cut-offs on out-of-fold training predictions ---
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
grid = np.arange(0.05, 0.96, 0.01)
def best_cutoff(oof, score):
    return float(grid[np.argmax([score(oof >= t) for t in grid])])
def net_benefit(reviewed, caught):
    return caught * LOSS_PER_LATE_SERIOUS_CASE - reviewed * COST_PER_REVIEW
oof = {n: cross_val_predict(clone(trained_models[n]), X_train_s, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
       for n in real_models}
balanced_cutoff = {n: best_cutoff(oof[n], lambda f: f1_score(y_train, f, zero_division=0)) for n in real_models}
y_tr = y_train.to_numpy()
money_cutoff = best_cutoff(oof[best_name], lambda f: net_benefit(f.sum(), y_tr[f].sum()))
threshold = balanced_cutoff[best_name]

# --- 3. CONFIRM on the test set ---
def test_row(name, proba, cutoff):
    flag = (proba >= cutoff).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y_true, flag).ravel())
    top20 = np.argsort(-proba)[: n_test // 5]
    return {"roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
            "found_in_riskiest_20%": y_true[top20].sum() / n_pos * 100, "flagged": tp + fp, "caught": tp, "missed": fn,
            "false_alarms": fp, "precision_%": precision_score(y_true, flag, zero_division=0) * 100,
            "recall_%": recall_score(y_true, flag) * 100, "f1": f1_score(y_true, flag), "tn": tn, "fp": fp, "fn": fn, "tp": tp}
test_table = pd.DataFrame({n: test_row(n, test_proba[n], balanced_cutoff[n]) for n in real_models}).T
count_cols = ["flagged", "caught", "missed", "false_alarms", "tn", "fp", "fn", "tp"]
test_table[count_cols] = test_table[count_cols].astype(int)

model = trained_models[best_name]
metrics = {k: round(float(v), 4) for k, v in test_table.loc[best_name].items()}   # numbers only; counts are converted below
metrics.update(true_negatives=metrics.pop("tn"), false_positives=metrics.pop("fp"),
               false_negatives=metrics.pop("fn"), true_positives=metrics.pop("tp"),
               baseline_roc_auc=round(float(roc_auc_score(y_true, test_proba["Baseline (no skill)"])), 4))
importance = importance_by_model[best_name].sort_values(ascending=False)
best_on_test = test_table["roc_auc"].idxmax()

# --- 4. PRICE the decision ---
p_best = test_proba[best_name]
def strategy(label, flag):
    return {"strategy": label, "reports reviewed urgently": int(flag.sum()), "serious cases caught": int(y_true[flag].sum()),
            "net benefit": int(net_benefit(flag.sum(), y_true[flag].sum()))}
rng = np.random.default_rng(SEED)
random_flag = np.zeros(n_test, bool); random_flag[rng.choice(n_test, int((p_best >= threshold).sum()), replace=False)] = True
money = pd.DataFrame([
    strategy("Review none urgently", np.zeros(n_test, bool)),
    strategy("Review every report urgently", np.ones(n_test, bool)),
    strategy(f"{best_name}, balanced cut-off ({threshold:.2f})", p_best >= threshold),
    strategy(f"{best_name}, money-best cut-off ({money_cutoff:.2f})", p_best >= money_cutoff),
    strategy("Review the same number at random", random_flag)]).set_index("strategy")

# --- Charts ---
fig, ax = plt.subplots(1, 3, figsize=(20, 5.6))
cm = np.array([[metrics["true_negatives"], metrics["false_positives"]], [metrics["false_negatives"], metrics["true_positives"]]]).astype(int)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0],
            xticklabels=["predicted good", "predicted serious"], yticklabels=["actually good", "actually serious"])
ax[0].set_title(f"Confusion matrix: {best_name}\n(cut-off {threshold:.2f}, test set)")
for n in real_models:
    fpr, tpr, _ = roc_curve(y_true, test_proba[n])
    ax[1].plot(fpr, tpr, lw=3 if n == best_name else 1.3, label=f"{n} ({test_table.loc[n, 'roc_auc']:.2f})")
ax[1].plot([0, 1], [0, 1], "k--", lw=0.8, label="random guessing (0.50)")
ax[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves (test set)"); ax[1].legend(fontsize=8)
shares = np.linspace(0.02, 1.0, 50)
for n in real_models:
    order = np.argsort(-test_proba[n])
    ax[2].plot(shares * 100, [y_true[order[: max(int(n_test * s), 1)]].sum() / n_pos * 100 for s in shares],
               lw=3 if n == best_name else 1.3, label=n)
ax[2].plot([0, 100], [0, 100], "k--", lw=0.8, label="random picking")
ax[2].set(xlabel="share of reports reviewed urgently, riskiest first (%)", ylabel="share of all serious cases found (%)",
          title="How quickly each model finds the serious cases"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/charts/evaluation.png", dpi=150, bbox_inches="tight"); plt.show()

# --- Report ---
print("1. CHOICE (training evidence)")
print(tuning_results.loc[real_models, ["cv_roc_auc", "cv_roc_auc_std", "cv_pr_auc"]].round(3).to_string())
print(f"   Within one standard deviation of the best: {', '.join(contenders)}  ->  chosen: {best_name}\n")
print("2. TEST SET, ALL MODELS (cut-off per model set on training data)")
print(test_table.drop(columns=["tn", "fp", "fn", "tp"]).round(3).to_string())
if best_on_test == best_name:
    print(f"   The test set CONFIRMS the choice: {best_name} also has the best test ROC AUC.\n")
else:
    gap = test_table.loc[best_on_test, "roc_auc"] - test_table.loc[best_name, "roc_auc"]
    print(f"   {best_on_test} scores {gap:.3f} ROC AUC higher on the test set; with {n_pos} serious cases that is within noise, so the choice stands.\n")
print(f"3. {best_name.upper()} AT THE BALANCED CUT-OFF ({threshold:.2f})")
print(f"   Flags {metrics['flagged']:.0f} of {n_test:,} reports, catches {metrics['caught']:.0f} of {n_pos} serious cases "
      f"(recall {metrics['recall_%']:.0f}%) with {metrics['false_alarms']:.0f} false alarms (precision {metrics['precision_%']:.0f}%).\n")
print(f"4. IS IT WORTH USING? Costs: {COST_PER_REVIEW} per urgent review, {LOSS_PER_LATE_SERIOUS_CASE} per serious case left waiting.")
print(f"   An urgent review pays for itself above {COST_PER_REVIEW / LOSS_PER_LATE_SERIOUS_CASE:.0%} precision; the model's is {metrics['precision_%']:.0f}%.")
print("   Use the balanced cut-off when reviewers are scarce, the money-best cut-off to maximise savings.\n")
display(money)
```

#### Objective 5: Pricing anomalies

```python
# BUSINESS OBJECTIVE 5: FIND UNUSUAL PRICES AND INVOICES
# EVALUATE MODEL
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
try:
    from IPython.display import display
except ImportError:
    display = print

# ---------- REVIEW COSTS (example values: replace with your real figures) ----------
COST_PER_REVIEW = 50          # finance time to check one flagged invoice
RECOVERY_RATE = 0.5           # share of the price gap that is recovered or corrected when a real anomaly is found

OUT_DIR = globals().get("OUT_DIR", "models/pricing_anomalies")
os.makedirs(f"{OUT_DIR}/charts", exist_ok=True)
for name in ("trained_models", "X_train_s", "X_test_s", "X_test", "y_train", "y_test", "predictions", "case_info", "raw"):
    if name not in globals():
        raise NameError(f"{name} is missing. Run the Objective 5 train, predict and compare cells first.")
if "OBJECTIVE" not in globals() or OBJECTIVE != "pricing_anomalies" or "models that flag it" not in predictions.columns:
    raise ValueError("The notebook holds a different objective. Run the Objective 5 train cell, then predict, first.")
models = list(trained_models)
simplicity_order = ["Elliptic Envelope", "Isolation Forest", "One-Class SVM", "Local Outlier Factor"]   # simplest first

# ---------- 0. The answer key (separated from X during data preparation, never used for training) ----------
truth_train, truth_test = y_train.to_numpy().astype(int), y_test.to_numpy().astype(int)
print(f"Real anomalies in the answer key: train {truth_train.sum()} of {len(truth_train)}, test {truth_test.sum()} of {len(truth_test)}\n")

# ---------- 1. CHOOSE the winner on the TRAIN invoices ----------
train_rows = {}
for n in models:
    s = -trained_models[n].score_samples(X_train_s)
    train_rows[n] = {"roc_auc": roc_auc_score(truth_train, s), "pr_auc": average_precision_score(truth_train, s)}
train_table = pd.DataFrame(train_rows).T
best_pr = train_table["pr_auc"].max()
contenders = [n for n in simplicity_order if train_table.loc[n, "pr_auc"] >= best_pr - 0.05]
best_name = contenders[0]
print("1. EVIDENCE FROM THE TRAIN INVOICES (used to choose the model)\n")
print(train_table.round(3).to_string())
print(f"\nModels within 0.05 PR AUC of the best: {', '.join(contenders)}")
print(f"Chosen: {best_name} (best score, or the simplest among near-ties)")

# ---------- 2. CONFIRM on the TEST invoices ----------
n_test, n_pos = len(truth_test), int(truth_test.sum())
def row(name, score, flag):
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(truth_test, flag).ravel())
    top = np.argsort(-score)[: int(n_test * 0.05)]
    return {"model": name, "roc_auc": roc_auc_score(truth_test, score), "pr_auc": average_precision_score(truth_test, score),
            "found_in_top_5%": truth_test[top].sum() / n_pos * 100, "flagged": tp + fp, "caught": tp, "missed": fn,
            "false_alarms": fp, "precision_%": precision_score(truth_test, flag) * 100, "recall_%": recall_score(truth_test, flag) * 100,
            "f1": f1_score(truth_test, flag)}
rows = [row(n, predictions[f"{n} score"].values, predictions[f"{n} flag"].values) for n in models]
vote_flag = (predictions["models that flag it"] >= 2).astype(int).values
vote_score = predictions[[f"{n} score" for n in models]].rank().mean(axis=1).values
rows.append(row("Vote: 2 of 4 models agree", vote_score, vote_flag))
rule_flag = ((X_test["price_to_list"] < 0.5) | (X_test["price_to_list"] > 1.5)).astype(int).values
rows.append(row("Fixed rule (price below 0.5x or above 1.5x list)", np.abs(np.log(X_test["price_to_list"].clip(lower=0.01).values)), rule_flag))
test_table = pd.DataFrame(rows).set_index("model")
print("\n2. TEST-INVOICE RESULTS (invoices the models never trained on)\n")
print(test_table.round(2).to_string())
print(f"\nBaseline for comparison: random flagging has ROC AUC 0.50 and finds about 5% of anomalies in the top 5%.")

best_on_test = test_table.loc[models, "pr_auc"].idxmax()
if best_on_test == best_name:
    print(f"The test invoices CONFIRM the choice: {best_name} also has the best test PR AUC.")
else:
    gap = test_table.loc[best_on_test, "pr_auc"] - test_table.loc[best_name, "pr_auc"]
    print(f"On the test invoices {best_on_test} scores {gap:.3f} PR AUC higher than {best_name}. With only {n_pos} real anomalies in the test set "
          "that gap is within noise, so the choice made on the train invoices stands.")

# ---------- 3. The chosen model in detail ----------
model = trained_models[best_name]
s_best = predictions[f"{best_name} score"].values
f_best = predictions[f"{best_name} flag"].values
tn, fp, fn, tp = (int(v) for v in confusion_matrix(truth_test, f_best).ravel())
threshold = None            # the model's own boundary decides the flag; kept so the save step has the same names
metrics = {k: float(v) for k, v in test_table.loc[best_name].items()}
metrics.update({"true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp,
                "rule_recall_pct": float(test_table.loc["Fixed rule (price below 0.5x or above 1.5x list)", "recall_%"]),
                "rule_precision_pct": float(test_table.loc["Fixed rule (price below 0.5x or above 1.5x list)", "precision_%"])})
metrics = {k: round(v, 4) for k, v in metrics.items()}
distance = X_test_s.sub(X_test_s.median()).abs()
importance = distance.corrwith(pd.Series(s_best, index=X_test_s.index)).sort_values(ascending=False)   # what drives the score

# ---------- 4. Pictures ----------
fig, ax = plt.subplots(1, 3, figsize=(20, 5.6))
for n in models:
    fpr, tpr, _ = roc_curve(truth_test, predictions[f"{n} score"].values)
    ax[0].plot(fpr, tpr, lw=3 if n == best_name else 1.3, label=f"{n} ({test_table.loc[n, 'roc_auc']:.2f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="random (0.50)")
ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate"); ax[0].legend(fontsize=8)
ax[0].set_title("ROC curves on the test invoices")

ax[1].hist(s_best[truth_test == 0], bins=60, alpha=0.7, color="#9db4c0", label="normal invoice")
ax[1].hist(s_best[truth_test == 1], bins=30, alpha=0.85, color="#c0392b", label="real anomaly")
ax[1].set_yscale("log"); ax[1].set_xlabel("anomaly score (higher = more unusual)"); ax[1].legend(fontsize=9)
ax[1].set_title(f"{best_name}: do real anomalies score higher?")

names = list(test_table.index)
short = [n.split(" (")[0].replace("Vote: 2 of 4 models agree", "Vote 2 of 4") for n in names]
x = np.arange(len(names))
ax[2].bar(x - 0.2, test_table["recall_%"], 0.4, label="share of real anomalies found (recall)", color="#2e7d6b")
ax[2].bar(x + 0.2, test_table["precision_%"], 0.4, label="share of flags that are real (precision)", color="#e67e22")
ax[2].set_xticks(x); ax[2].set_xticklabels(short, rotation=30, ha="right", fontsize=8); ax[2].set_ylim(0, 105); ax[2].legend(fontsize=8)
ax[2].set_title("Recall and precision: models vs the fixed rule")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/charts/evaluation.png", dpi=150, bbox_inches="tight"); plt.show()

# ---------- 5. Is it worth using? Money recovered against review cost ----------
gap = (raw.set_index("sale_id").loc[case_info.loc[X_test_s.index, "sale_id"], ["total_amount_usd", "units_sold", "list_price_usd"]]
       .eval("abs(total_amount_usd - units_sold * list_price_usd)").values)
def net_value(flag): return (gap[(flag == 1) & (truth_test == 1)].sum() * RECOVERY_RATE) - flag.sum() * COST_PER_REVIEW
money = pd.DataFrame({
    "invoices reviewed": [int(f_best.sum()), int(rule_flag.sum()), int(vote_flag.sum()), n_test],
    "real anomalies caught": [int(((f_best == 1) & (truth_test == 1)).sum()), int(((rule_flag == 1) & (truth_test == 1)).sum()),
                               int(((vote_flag == 1) & (truth_test == 1)).sum()), n_pos],
    "net value": [round(net_value(f_best)), round(net_value(rule_flag)), round(net_value(vote_flag)), round(net_value(np.ones(n_test, dtype=int)))],
}, index=[f"{best_name}", "Fixed price rule", "2 of 4 models agree", "Review every invoice"])

print(f"\n3. THE CHOSEN MODEL: {best_name}")
print(f"   Flags {tp + fp} of {n_test} invoices, catches {tp} of {n_pos} real anomalies (recall {metrics['recall_%']:.0f}%), "
      f"{fp} false alarms (precision {metrics['precision_%']:.0f}%), {fn} missed.")
print(f"   The fixed price rule finds {metrics['rule_recall_pct']:.0f}% of real anomalies with {metrics['rule_precision_pct']:.0f}% precision.")
print(f"\n4. IS IT WORTH USING? Example: {COST_PER_REVIEW} per review, {RECOVERY_RATE:.0%} of the price gap recovered on each real anomaly caught.")
print("   Replace the two figures at the top with real ones and run the cell again.\n")
display(money)

test_table.round(2)
```

---

## 18. Business Analysis

A model score is only useful if it changes a decision. Each risk is split into **High, Medium and Low tiers**, and each policy is priced. The costs below are examples; change them in the notebook and the numbers update.

| # | Policy | Work created | Problems caught | Net benefit (example costs) | vs best simple policy |
|---|---|---|---|---|---|
| 1 | Inspect High + Medium batches | 47% of batches | 86% of rejections | 419,500 | +159,500 |
| 2 | Plan from the forecast, by area and month | n/a | Error falls to 10% | Stock cost 499,180 vs 1,721,109 | -71% cost |
| 3 | Protect High + Medium shipments | 71% of shipments | 97% of excursions | 349,400 | +59,900 |
| 4 | Review High + Medium reports first | 54% of reports | 99% of serious cases | 230,400 | +27,900 |
| 5 | Fixed price rule, model as second check | 3% of invoices | 100% of anomalies | 1,183,836 | +73,000 |

**Where to look first**
- **Batches:** Delta Works 6 (33%) and Gamma Works 7 (28%) reject far more than the 15% average. Injection products reject 21%.
- **Shipments:** sea freight (47% excursions) and Nigeria-bound shipments (31%).
- **Safety reports:** patients 65 and over (37% serious) and Anti-infective products (28%).
- **Demand:** Vaccines is the hardest to plan (17% error by area and month) and tends to be over-forecast.

The risk tiers hold up well: High-tier batches are rejected 45% of the time (3.0 times the average), against 4% in the Low tier. High-tier adverse event reports are serious 62% of the time, and only 1 of 206 Low-tier reports was serious.

### Charts

<p align="center">
  <img width="2384" height="860" alt="business_all_objectives_summary" src="https://github.com/user-attachments/assets/a1351007-bd68-44fe-a114-deb728c0379a" />
</p>
<p align="center"><em>All five objectives: work created against problems caught, and gain over the best simple alternative</em></p>

<p align="center">
  <img width="2382" height="1551" alt="business_objective1_batch_rejection" src="https://github.com/user-attachments/assets/b0159ad6-c20c-4053-aa37-e7b3772408e3" />
</p>
<p align="center"><em>Objective 1: Batch rejection, business analysis</em></p>

<p align="center">
  <img width="2379" height="1551" alt="business_objective2_demand_forecast" src="https://github.com/user-attachments/assets/4467c9c9-4597-430f-9783-38fec11a2154" />
</p>
<p align="center"><em>Objective 2: Demand forecast, business analysis</em></p>

<p align="center">
  <img width="2382" height="1551" alt="business_objective3_cold_chain" src="https://github.com/user-attachments/assets/46a46697-21b3-4b7b-ad65-1dbcd2237111" />
</p>
<p align="center"><em>Objective 3: Cold-chain excursions, business analysis</em></p>

<p align="center">
  <img width="2383" height="1551" alt="business_objective4_adverse_events" src="https://github.com/user-attachments/assets/ee08ca3d-4958-4481-bd55-59f54b3c3751" />
</p>
<p align="center"><em>Objective 4: Serious adverse events, business analysis</em></p>

<p align="center">
  <img width="2384" height="1551" alt="business_objective5_pricing_anomalies" src="https://github.com/user-attachments/assets/47cb2f09-7070-4e23-a380-4b4f38aacf4c" />
</p>
<p align="center"><em>Objective 5: Pricing anomalies, business analysis</em></p>

### Full code (one cell for all five objectives)

```python
# BUSINESS ANALYSIS FOR ALL FIVE BUSINESS OBJECTIVES (ONE CELL)
# Purpose : turn the five winning models into business decisions, with charts, in a single cell.
#   Objective 1  Batch rejection        -> which batches get extra inspection
#   Objective 2  Demand forecast        -> how much to make, and what a wrong plan costs
#   Objective 3  Cold-chain excursions  -> which shipments get better protection
#   Objective 4  Serious adverse events -> which reports the safety team reviews first
#   Objective 5  Pricing anomalies      -> which invoices finance checks
import base64, contextlib, io, os, time, warnings, zlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
try:
    from IPython.display import display
    from IPython.utils.capture import capture_output
except ImportError:
    display, capture_output = print, None
warnings.filterwarnings("ignore")

# ---------- EXAMPLE COSTS (replace with real figures) ----------
OBJ1_COST_PER_INSPECTION, OBJ1_LOSS_PER_MISSED_REJECTION = 500, 5000      # inspect one batch / a rejection that slips through
OBJ2_COST_PER_UNIT_OVERSTOCK, OBJ2_COST_PER_UNIT_SHORTAGE = 2.0, 6.0      # per unit made but unsold / demanded but unavailable
OBJ3_COST_PER_PROTECTION, OBJ3_LOSS_PER_EXCURSION = 300, 4000             # protect one shipment / an excursion on an unprotected one
OBJ4_COST_PER_REVIEW, OBJ4_LOSS_PER_LATE_SERIOUS_CASE = 150, 3000         # urgent review of one report / a serious case left waiting
OBJ5_COST_PER_REVIEW, OBJ5_RECOVERY_RATE = 50, 0.5                        # check one invoice / share of the price gap recovered
REBUILD = False                       # True = ignore the cache and retrain everything

OUT = "models/business_analysis"; CACHE = "models/_business_cache"
for d in (f"{OUT}/charts", CACHE):
    os.makedirs(d, exist_ok=True)

# ---------- chart style (one palette for the whole cell) ----------
BLUE, ORANGE, AQUA, AMBER, RED, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#d03b3b", "#9a9892", "#0b0b0b"
TIER_COLOR = {"High": RED, "Medium": AMBER, "Low": BLUE}
DIVERGE = LinearSegmentedColormap.from_list("gain", [RED, "#f0efec", BLUE])

# 1. THE FIVE EVALUATED MODELS (packed; re-run silently only when the cache is missing)

_PIPELINES = {...}   # packed copy of every objective's train, predict, compare and evaluate cells (sections 14 to 17).
                    # It is what lets this cell run on its own; it is left out here to keep the README readable.

def _run_pipeline(n):
    """Run objective n's train, predict, compare and evaluate cells in a private namespace."""
    ns = {"__name__": "__main__"}
    code = zlib.decompress(base64.b64decode(_PIPELINES[n])).decode()
    shown, plt.show = plt.show, (lambda *a, **k: None)
    try:
        if capture_output is not None:
            with capture_output():
                exec(code, ns)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, ns)
    finally:
        plt.show = shown
        plt.close("all")
    return ns

def _context(ns, key, cols):
    """Descriptive columns for the test rows, looked up from the source table by the row key."""
    ids = ns["case_info"].loc[ns["X_test_s"].index, key].to_numpy()
    ctx = ns["raw"].drop_duplicates(key).set_index(key).reindex(ids)[cols].reset_index(drop=True)
    return ids, ctx

def _extract(n, ns):
    """Keep only what the business analysis needs from an evaluated objective (compact and cacheable)."""
    best = ns["best_name"]
    if n == 1:
        ids, ctx = _context(ns, "batch_id", ["plant_plant_name", "plant_country", "formulation", "therapeutic_area", "prod_product_name"])
        cases = pd.DataFrame({"batch_id": ids, "risk": ns["test_proba"][best], "rejected": ns["y_test"].to_numpy()}).join(ctx)
        meta = {"threshold": float(ns["threshold"]), "money_cutoff": float(ns["money_cutoff"])}
    elif n == 3:
        ids, ctx = _context(ns, "shipment_id", ["destination_country", "transport_mode", "packaging_type", "ship_season", "customer_type", "formulation"])
        cases = pd.DataFrame({"shipment_id": ids, "risk": ns["test_proba"][best], "excursion": ns["y_test"].to_numpy()}).join(ctx)
        meta = {"threshold": float(ns["threshold"]), "money_cutoff": float(ns["money_cutoff"])}
    elif n == 4:
        ids, ctx = _context(ns, "ae_id", ["therapeutic_area", "formulation", "age_group", "reporter_type", "onset_band"])
        cases = pd.DataFrame({"ae_id": ids, "risk": ns["test_proba"][best], "serious": ns["y_test"].to_numpy()}).join(ctx)
        meta = {"threshold": float(ns["threshold"]), "money_cutoff": float(ns["money_cutoff"])}
    elif n == 2:
        idx = ns["X_test_s"].index
        cases = pd.DataFrame({"sale_id": ns["case_info"].loc[idx, "sale_id"].to_numpy(),
                              "month": pd.to_datetime(ns["case_info"].loc[idx, "sale_date"]).dt.to_period("M").astype(str).to_numpy(),
                              "area": ns["X_text"].loc[idx, "therapeutic_area"].to_numpy(), "channel": ns["X_text"].loc[idx, "sales_channel"].to_numpy(),
                              "customer_type": ns["X_text"].loc[idx, "customer_type"].to_numpy(),
                              "actual": np.asarray(ns["actual"], float), "forecast": np.asarray(ns["p_best"], float)})
        units = np.expm1(ns["y_train"])
        meta = {"median_units": float(units.median()), "mean_units": float(units.mean())}
    else:
        ids, ctx = _context(ns, "sale_id", ["customer_type", "sales_channel", "therapeutic_area", "formulation", "price_to_list"])
        cases = pd.DataFrame({"sale_id": ids, "model_flag": np.asarray(ns["f_best"]).astype(int), "rule_flag": np.asarray(ns["rule_flag"]).astype(int),
                              "vote_flag": np.asarray(ns["vote_flag"]).astype(int), "anomaly": np.asarray(ns["truth_test"]).astype(int),
                              "gap_usd": np.asarray(ns["gap"], float)}).join(ctx)
        meta = {}
    meta["best_name"] = best
    return cases, meta

INPUTS, _t0 = {}, time.time()
for n in (1, 2, 3, 4, 5):
    path = f"{CACHE}/objective_{n}.joblib"
    if os.path.exists(path) and not REBUILD:
        INPUTS[n] = joblib.load(path)
    else:
        print(f"Objective {n}: training and evaluating (first run only) ...")
        INPUTS[n] = _extract(n, _run_pipeline(n))
        joblib.dump(INPUTS[n], path)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c9c8c2", "axes.grid": True, "grid.color": "#ececea", "axes.axisbelow": True, "figure.facecolor": "white"})  # set AFTER the model cells run (they change the plot style)
print(f"Evaluated models loaded for all 5 objectives ({time.time() - _t0:.0f}s).\n")


# 2. SMALL SHARED HELPERS (tables and chart panels; all wording is set per objective below)

def _rates(df, col, target, min_n=30):
    g = df.groupby(col)[target].agg(n="size", rate="mean")
    return g[g["n"] >= min_n].sort_values("rate", ascending=False)

def _policy(label, flag, actual, cost, loss, unit):
    flag = np.asarray(flag, bool); caught = int(actual[flag].sum()); worked = int(flag.sum())
    return {"policy": label, f"{unit} worked": worked, "problems caught": caught, "problems missed": int(actual.sum()) - caught,
            "action cost": worked * cost, "loss avoided": caught * loss, "net benefit": caught * loss - worked * cost}

def _tiers(df, target, threshold, money_cutoff, unit):
    hi, lo = max(threshold, money_cutoff), min(threshold, money_cutoff)
    df["tier"] = np.select([df["risk"] >= hi, df["risk"] >= lo], ["High", "Medium"], default="Low")
    t = df.groupby("tier").agg(n=(target, "size"), problems=(target, "sum")).reindex(["High", "Medium", "Low"])
    avg = df[target].mean()
    t["share of all %"] = t["n"] / len(df) * 100
    t["problem rate %"] = t["problems"] / t["n"] * 100
    t["share of problems %"] = t["problems"] / df[target].sum() * 100
    t["lift vs average"] = t["problem rate %"] / (avg * 100)
    return t.rename(columns={"n": unit})

def _sensitivity(actual, flag, costs, losses):
    """Gain of a fixed policy over the best simple policy (do nothing or act on everything) at other cost levels."""
    k, n, caught, worked = actual.sum(), len(actual), actual[flag].sum(), flag.sum()
    return pd.DataFrame({c: [(caught * l - worked * c) - max(0, k * l - n * c) for l in losses] for c in costs}, index=losses)

def _heat(ax, grid, title, xlabel, ylabel, fmt="{:,.0f}"):
    sns.heatmap(grid, annot=grid.map(lambda v: fmt.format(v)), fmt="", cmap=DIVERGE, center=0, cbar=False, ax=ax, linewidths=2, linecolor="white")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(False)

def _four_panel(path, headline, tiers, unit, plans, spots, spots_title, sens, sens_title, sens_x, sens_y):
    fig, ax = plt.subplots(2, 2, figsize=(16, 10.5)); fig.suptitle(headline, fontsize=14, fontweight="bold", x=0.01, ha="left")
    order = ["High", "Medium", "Low"]; avg = (tiers["problems"].sum() / tiers[unit].sum()) * 100
    ax[0, 0].bar(order, tiers["problem rate %"], color=[TIER_COLOR[t] for t in order], width=0.6)
    ax[0, 0].axhline(avg, color=INK, ls="--", lw=1, label=f"average {avg:.0f}%"); ax[0, 0].legend(frameon=False, loc="upper right")
    for i, t in enumerate(order):
        ax[0, 0].text(i, tiers.loc[t, "problem rate %"] + 1.2, f"{tiers.loc[t, 'problem rate %']:.{1 if tiers.loc[t, 'problem rate %'] < 1 else 0}f}%\n({int(tiers.loc[t, unit])} {unit})", ha="center", fontsize=9, bbox=_label_box, zorder=5)
    ax[0, 0].set(ylabel="problem rate (%)", title="A. Problem rate by risk tier", ylim=(0, max(tiers["problem rate %"].max() * 1.3, 10)))
    p = plans["net benefit"]; best = p.idxmax()
    ax[0, 1].barh(p.index[::-1], p.values[::-1], color=[BLUE if i == best else "#b9c7d6" for i in p.index[::-1]], height=0.6)
    for y, v in enumerate(p.values[::-1]):
        ax[0, 1].text(v + p.abs().max() * 0.01, y, f"{v:,.0f}", va="center", fontsize=9)
    ax[0, 1].set(xlabel="net benefit on the test rows (example costs)", title="B. Which policy pays best? (highlighted = best)"); ax[0, 1].xaxis.set_major_formatter(_thousands); ax[0, 1].xaxis.set_major_locator(MaxNLocator(5))
    top = spots.head(8).sort_values("rate")
    ax[1, 0].barh(top.index.astype(str), top["rate"] * 100, color=ORANGE, height=0.6); ax[1, 0].axvline(avg, color=INK, ls="--", lw=1)
    for y, (v, n_) in enumerate(zip(top["rate"] * 100, top["n"])):
        ax[1, 0].text(v + 0.4, y, f"{v:.0f}%  (n={int(n_)})", va="center", fontsize=8.5)
    ax[1, 0].set(xlabel="problem rate (%); dashed = average", title=spots_title); ax[1, 0].set_xlim(0, top["rate"].max() * 100 * 1.25)
    _heat(ax[1, 1], sens, sens_title, sens_x, sens_y)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.show()

SUMMARY = []
_thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")
_label_box = dict(fc="white", ec="none", pad=1.2)


# 3. BUSINESS OBJECTIVE 1: REDUCE BATCH REJECTIONS  ->  which batches to inspect

print("=" * 100 + "\nBUSINESS OBJECTIVE 1: REDUCE BATCH REJECTIONS\n" + "=" * 100)
c1, m1 = INPUTS[1]
n1, k1, y1 = len(c1), int(c1["rejected"].sum()), c1["rejected"].to_numpy()
tiers1 = _tiers(c1, "rejected", m1["threshold"], m1["money_cutoff"], "batches")
plans1 = pd.DataFrame([_policy("Inspect nothing", np.zeros(n1), y1, OBJ1_COST_PER_INSPECTION, OBJ1_LOSS_PER_MISSED_REJECTION, "batches"),
                       _policy("Inspect every batch", np.ones(n1), y1, OBJ1_COST_PER_INSPECTION, OBJ1_LOSS_PER_MISSED_REJECTION, "batches"),
                       _policy("Inspect High tier only", c1["tier"] == "High", y1, OBJ1_COST_PER_INSPECTION, OBJ1_LOSS_PER_MISSED_REJECTION, "batches"),
                       _policy("Inspect High + Medium tiers", c1["tier"] != "Low", y1, OBJ1_COST_PER_INSPECTION, OBJ1_LOSS_PER_MISSED_REJECTION, "batches")]).set_index("policy")
rec1 = plans1["net benefit"].idxmax(); r1 = plans1.loc[rec1]; simple1 = plans1.loc[["Inspect nothing", "Inspect every batch"], "net benefit"]
plants1, forms1, areas1 = _rates(c1, "plant_plant_name", "rejected", 40), _rates(c1, "formulation", "rejected", 40), _rates(c1, "therapeutic_area", "rejected", 40)
sens1 = _sensitivity(y1, (c1["tier"] != "Low").to_numpy(), [100, 250, 500, 1000, 1500], [2000, 3500, 5000, 7500, 10000])
sens1.index.name, sens1.columns.name = "loss per missed rejection", "cost per inspection"
_four_panel(f"{OUT}/charts/objective1_batch_rejection.png", "Objective 1: Reduce batch rejections (inspect the risky batches)", tiers1, "batches", plans1, plants1,
            "C. Plants with the highest rejection rate", sens1, "D. Gain of 'High + Medium' over the best simple policy\n(blue = still worth doing)", "cost per inspection", "loss per missed rejection")
print(f"- Problem size: {k1} of {n1:,} test batches ({k1 / n1:.0%}) were rejected; with none caught, that is {k1 * OBJ1_LOSS_PER_MISSED_REJECTION:,.0f} of losses (example cost).")
print(f"- The model ({m1['best_name']}) separates risk well: High-tier batches are rejected {tiers1.loc['High', 'problem rate %']:.0f}% of the time ({tiers1.loc['High', 'lift vs average']:.1f}x average); Low-tier only {tiers1.loc['Low', 'problem rate %']:.0f}%.")
print(f"- Recommended: '{rec1}': {r1['batches worked'] / n1:.0%} of batches, catching {r1['problems caught'] / k1:.0%} of rejections; net benefit {r1['net benefit']:,.0f}, {r1['net benefit'] - simple1.max():,.0f} better than the best simple policy.")
print(f"- Hot spots: {plants1.index[0]} has the highest plant rejection rate ({plants1['rate'].iloc[0]:.0%}); {forms1.index[0]} products ({forms1['rate'].iloc[0]:.0%}) and {areas1.index[0]} ({areas1['rate'].iloc[0]:.0%}) are the riskiest lines.")
print(f"- The policy beats the best simple policy in {int((sens1 > 0).to_numpy().sum())} of {sens1.size} cost scenarios; it loses when inspections are very cheap or very dear against the loss.\n")
display(tiers1.round(1)); display(plans1)
SUMMARY.append({"objective": "1 Batch rejection", "best model": m1["best_name"], "recommended action": rec1, "workload %": r1["batches worked"] / n1 * 100,
                "problems caught %": r1["problems caught"] / k1 * 100, "net benefit": r1["net benefit"], "gain over best simple": r1["net benefit"] - simple1.max(),
                "hot spot": f"{plants1.index[0]} ({plants1['rate'].iloc[0]:.0%} rejected)"})


# 4. BUSINESS OBJECTIVE 2: DEMAND FORECAST  ->  how much to make, and what a wrong plan costs

print("\n" + "=" * 100 + "\nBUSINESS OBJECTIVE 2: FORECAST DEMAND\n" + "=" * 100)
c2, m2 = INPUTS[2]
a2, f2 = c2["actual"].to_numpy(), c2["forecast"].to_numpy()
wape2 = np.abs(f2 - a2).sum() / a2.sum() * 100
def _stock_cost(forecast):
    over, short = np.clip(forecast - a2, 0, None).sum(), np.clip(a2 - forecast, 0, None).sum()
    return {"units over-made": round(over), "units short": round(short), "stock cost": round(over * OBJ2_COST_PER_UNIT_OVERSTOCK + short * OBJ2_COST_PER_UNIT_SHORTAGE)}
plans2 = pd.DataFrame({f"Forecast ({m2['best_name']})": _stock_cost(f2), "Flat plan at the typical sale size": _stock_cost(np.full(len(a2), m2["median_units"])),
                       "Flat plan at the average sale size": _stock_cost(np.full(len(a2), m2["mean_units"]))}).T
best_flat2 = plans2.iloc[1:]["stock cost"].min(); saving2 = best_flat2 - plans2.iloc[0]["stock cost"]
by_area2 = c2.groupby("area").apply(lambda g: pd.Series({"sale-level error %": np.abs(g.forecast - g.actual).sum() / g.actual.sum() * 100,
                                                          "bias %": (g.forecast - g.actual).sum() / g.actual.sum() * 100, "units sold": g.actual.sum()}))
plan_am2 = c2.groupby(["area", "month"], as_index=False)[["actual", "forecast"]].sum()
by_area2["area-month error %"] = plan_am2.groupby("area").apply(lambda g: np.abs(g.forecast - g.actual).sum() / g.actual.sum() * 100)
plan_wape2 = np.abs(plan_am2.forecast - plan_am2.actual).sum() / plan_am2.actual.sum() * 100
monthly2 = c2.groupby("month")[["actual", "forecast"]].sum()
def _saving_grid(over_costs, short_costs):
    g = pd.DataFrame(index=short_costs, columns=over_costs, dtype=float)
    for s in short_costs:
        for o in over_costs:
            cost = lambda f: np.clip(f - a2, 0, None).sum() * o + np.clip(a2 - f, 0, None).sum() * s
            g.loc[s, o] = min(cost(np.full(len(a2), m2["median_units"])), cost(np.full(len(a2), m2["mean_units"]))) - cost(f2)
    return g
sens2 = _saving_grid([1, 1.5, 2, 3, 4], [3, 4.5, 6, 9, 12]); sens2.index.name, sens2.columns.name = "cost per unit short", "cost per unit over-made"
fig, ax = plt.subplots(2, 2, figsize=(16, 10.5)); fig.suptitle("Objective 2: Forecast demand (plan production from the forecast)", fontsize=14, fontweight="bold", x=0.01, ha="left")
ax[0, 0].plot(monthly2.index, monthly2["actual"], color=INK, lw=2, marker="o", ms=5, label="actual units sold")
ax[0, 0].plot(monthly2.index, monthly2["forecast"], color=BLUE, lw=2, marker="o", ms=5, label=f"forecast ({m2['best_name']})")
ax[0, 0].set(title="A. Monthly demand: forecast vs actual (test period)", ylabel="units"); ax[0, 0].legend(frameon=False); ax[0, 0].tick_params(axis="x", rotation=45); ax[0, 0].yaxis.set_major_formatter(_thousands)
ba = by_area2.sort_values("sale-level error %"); yy = np.arange(len(ba))
ax[0, 1].barh(yy + 0.2, ba["sale-level error %"], height=0.38, color=ORANGE, label="per sale"); ax[0, 1].barh(yy - 0.2, ba["area-month error %"], height=0.38, color=BLUE, label="per product area and month")
for y, (a_, b_) in enumerate(zip(ba["sale-level error %"], ba["area-month error %"])):
    ax[0, 1].text(a_ + 0.3, y + 0.2, f"{a_:.0f}%", va="center", fontsize=8.5); ax[0, 1].text(b_ + 0.3, y - 0.2, f"{b_:.0f}%", va="center", fontsize=8.5)
ax[0, 1].set_yticks(yy); ax[0, 1].set_yticklabels(ba.index); ax[0, 1].set(xlabel="forecast error (% of units)", title="B. Error falls when plans use totals by area and month"); ax[0, 1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
sc = plans2["stock cost"]; ax[1, 0].bar(range(3), sc.values, color=[BLUE, "#b9c7d6", "#b9c7d6"], width=0.6)
ax[1, 0].set_xticks(range(3)); ax[1, 0].set_xticklabels(["Forecast", "Flat plan:\ntypical size", "Flat plan:\naverage size"])
for i, v in enumerate(sc.values):
    ax[1, 0].text(i, v + sc.max() * 0.015, f"{v:,.0f}", ha="center", fontsize=9)
ax[1, 0].set(ylabel="stock cost (example costs)", title="C. What a wrong plan costs"); ax[1, 0].set_ylim(0, sc.max() * 1.15); ax[1, 0].yaxis.set_major_formatter(_thousands)
_heat(ax[1, 1], sens2, "D. Saving of the forecast over the best flat plan\n(blue = forecast cheaper)", "cost per unit over-made", "cost per unit short")
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"{OUT}/charts/objective2_demand_forecast.png", dpi=150, bbox_inches="tight"); plt.show()
worst2, bestarea2 = by_area2["bias %"].abs().idxmax(), by_area2["area-month error %"].idxmin()
print(f"- Accuracy: the forecast ({m2['best_name']}) is off by {wape2:.0f}% of units per sale, but only {plan_wape2:.0f}% when summed by product area and month, which is how production is actually planned.")
print(f"- Money (example costs): stock cost {plans2.iloc[0]['stock cost']:,.0f} against {best_flat2:,.0f} for the best flat plan, a saving of {saving2:,.0f} ({saving2 / best_flat2:.0%}).")
print(f"- Over-made {plans2.iloc[0]['units over-made']:,.0f} units and short {plans2.iloc[0]['units short']:,.0f} units on {a2.sum():,.0f} sold; shortages cost more per unit, so plan with a safety margin where stock-outs hurt most.")
print(f"- By product area: easiest {bestarea2} ({by_area2.loc[bestarea2, 'area-month error %']:.0f}% error by area-month); largest bias {worst2} ({by_area2.loc[worst2, 'bias %']:+.1f}%; negative = under-forecast).")
print(f"- The forecast beats the best flat plan in {int((sens2 > 0).to_numpy().sum())} of {sens2.size} cost scenarios tested.\n")
display(by_area2.round(1)); display(plans2)
SUMMARY.append({"objective": "2 Demand forecast", "best model": m2["best_name"], "recommended action": "Plan production from the forecast, by product area and month",
                "workload %": np.nan, "problems caught %": np.nan, "net benefit": saving2, "gain over best simple": saving2,
                "hot spot": f"{worst2} (bias {by_area2.loc[worst2, 'bias %']:+.1f}%)"})


# 5. BUSINESS OBJECTIVE 3: PREVENT COLD-CHAIN EXCURSIONS  ->  which shipments to protect

print("\n" + "=" * 100 + "\nBUSINESS OBJECTIVE 3: PREVENT COLD-CHAIN EXCURSIONS\n" + "=" * 100)
c3, m3 = INPUTS[3]
n3, k3, y3 = len(c3), int(c3["excursion"].sum()), c3["excursion"].to_numpy()
tiers3 = _tiers(c3, "excursion", m3["threshold"], m3["money_cutoff"], "shipments")
plans3 = pd.DataFrame([_policy("Protect nothing", np.zeros(n3), y3, OBJ3_COST_PER_PROTECTION, OBJ3_LOSS_PER_EXCURSION, "shipments"),
                       _policy("Protect every shipment", np.ones(n3), y3, OBJ3_COST_PER_PROTECTION, OBJ3_LOSS_PER_EXCURSION, "shipments"),
                       _policy("Protect High tier only", c3["tier"] == "High", y3, OBJ3_COST_PER_PROTECTION, OBJ3_LOSS_PER_EXCURSION, "shipments"),
                       _policy("Protect High + Medium tiers", c3["tier"] != "Low", y3, OBJ3_COST_PER_PROTECTION, OBJ3_LOSS_PER_EXCURSION, "shipments")]).set_index("policy")
rec3 = plans3["net benefit"].idxmax(); r3 = plans3.loc[rec3]; simple3 = plans3.loc[["Protect nothing", "Protect every shipment"], "net benefit"]
dest3, mode3, pack3 = _rates(c3, "destination_country", "excursion", 40), _rates(c3, "transport_mode", "excursion", 40), _rates(c3, "packaging_type", "excursion", 40)
sens3 = _sensitivity(y3, (c3["tier"] != "Low").to_numpy(), [100, 200, 300, 600, 1000], [1500, 2500, 4000, 6000, 9000])
sens3.index.name, sens3.columns.name = "loss per excursion", "cost per protection"
_four_panel(f"{OUT}/charts/objective3_cold_chain.png", "Objective 3: Prevent cold-chain excursions (protect the risky shipments)", tiers3, "shipments", plans3, dest3,
            "C. Destinations with the highest excursion rate", sens3, "D. Gain of 'High + Medium' over the best simple policy\n(blue = still worth doing)", "cost per protection", "loss per excursion")
print(f"- Problem size: {k3} of {n3:,} test shipments ({k3 / n3:.0%}) had a temperature excursion; with none prevented, that is {k3 * OBJ3_LOSS_PER_EXCURSION:,.0f} of losses (example cost).")
print(f"- The model ({m3['best_name']}) separates risk well: High-tier shipments suffer an excursion {tiers3.loc['High', 'problem rate %']:.0f}% of the time ({tiers3.loc['High', 'lift vs average']:.1f}x average); Low-tier only {tiers3.loc['Low', 'problem rate %']:.0f}%.")
print(f"- Recommended: '{rec3}': {r3['shipments worked'] / n3:.0%} of shipments, preventing up to {r3['problems caught'] / k3:.0%} of excursions; net benefit {r3['net benefit']:,.0f}, {r3['net benefit'] - simple3.max():,.0f} better than the best simple policy.")
print(f"- Hot spots: {dest3.index[0]} is the riskiest destination ({dest3['rate'].iloc[0]:.0%}); {mode3.index[0]} transport ({mode3['rate'].iloc[0]:.0%}) and {pack3.index[0]} packaging ({pack3['rate'].iloc[0]:.0%}) have the highest excursion rates.")
print(f"- The policy beats the best simple policy in {int((sens3 > 0).to_numpy().sum())} of {sens3.size} cost scenarios; it loses when protection is very cheap or very dear against the loss.\n")
display(tiers3.round(1)); display(plans3)
SUMMARY.append({"objective": "3 Cold-chain excursions", "best model": m3["best_name"], "recommended action": rec3, "workload %": r3["shipments worked"] / n3 * 100,
                "problems caught %": r3["problems caught"] / k3 * 100, "net benefit": r3["net benefit"], "gain over best simple": r3["net benefit"] - simple3.max(),
                "hot spot": f"{dest3.index[0]} ({dest3['rate'].iloc[0]:.0%} excursions)"})


# 6. BUSINESS OBJECTIVE 4: REDUCE SERIOUS ADVERSE EVENTS  ->  which reports the safety team reviews first

print("\n" + "=" * 100 + "\nBUSINESS OBJECTIVE 4: REDUCE SERIOUS ADVERSE EVENTS\n" + "=" * 100)
c4, m4 = INPUTS[4]
n4, k4, y4 = len(c4), int(c4["serious"].sum()), c4["serious"].to_numpy()
tiers4 = _tiers(c4, "serious", m4["threshold"], m4["money_cutoff"], "reports")
plans4 = pd.DataFrame([_policy("Review none urgently", np.zeros(n4), y4, OBJ4_COST_PER_REVIEW, OBJ4_LOSS_PER_LATE_SERIOUS_CASE, "reports"),
                       _policy("Review every report urgently", np.ones(n4), y4, OBJ4_COST_PER_REVIEW, OBJ4_LOSS_PER_LATE_SERIOUS_CASE, "reports"),
                       _policy("Review High tier first", c4["tier"] == "High", y4, OBJ4_COST_PER_REVIEW, OBJ4_LOSS_PER_LATE_SERIOUS_CASE, "reports"),
                       _policy("Review High + Medium tiers first", c4["tier"] != "Low", y4, OBJ4_COST_PER_REVIEW, OBJ4_LOSS_PER_LATE_SERIOUS_CASE, "reports")]).set_index("policy")
rec4 = plans4["net benefit"].idxmax(); r4 = plans4.loc[rec4]; simple4 = plans4.loc[["Review none urgently", "Review every report urgently"], "net benefit"]
area4, age4, rep4 = _rates(c4, "therapeutic_area", "serious", 30), _rates(c4, "age_group", "serious", 30), _rates(c4, "reporter_type", "serious", 30)
sens4 = _sensitivity(y4, (c4["tier"] != "Low").to_numpy(), [50, 100, 150, 300, 500], [1000, 2000, 3000, 5000, 8000])
sens4.index.name, sens4.columns.name = "loss per late serious case", "cost per urgent review"
_four_panel(f"{OUT}/charts/objective4_adverse_events.png", "Objective 4: Reduce serious adverse events (review the risky reports first)", tiers4, "reports", plans4, area4,
            "C. Therapeutic areas with the highest serious-event rate", sens4, "D. Gain of 'High + Medium' over the best simple policy\n(blue = still worth doing)", "cost per urgent review", "loss per late serious case")
print(f"- Problem size: {k4} of {n4:,} test reports ({k4 / n4:.0%}) describe a serious event; if every one waits in the queue, that is {k4 * OBJ4_LOSS_PER_LATE_SERIOUS_CASE:,.0f} of losses (example cost).")
print(f"- The model ({m4['best_name']}) separates risk well: High-tier reports are serious {tiers4.loc['High', 'problem rate %']:.0f}% of the time ({tiers4.loc['High', 'lift vs average']:.1f}x average); Low-tier only {tiers4.loc['Low', 'problem rate %']:.0f}%.")
print(f"- Recommended: '{rec4}': {r4['reports worked'] / n4:.0%} of reports, reaching {r4['problems caught'] / k4:.0%} of serious cases early; net benefit {r4['net benefit']:,.0f}, {r4['net benefit'] - simple4.max():,.0f} better than the best simple policy.")
print(f"- Hot spots: {area4.index[0]} has the highest serious-event rate ({area4['rate'].iloc[0]:.0%}); age group {age4.index[0]} ({age4['rate'].iloc[0]:.0%}) and {rep4.index[0]} reporters ({rep4['rate'].iloc[0]:.0%}) stand out.")
print(f"- For patient safety the lower cut-off is the one to use: a missed serious case costs far more than an extra review. The policy beats the best simple policy in {int((sens4 > 0).to_numpy().sum())} of {sens4.size} cost scenarios.\n")
display(tiers4.round(1)); display(plans4)
SUMMARY.append({"objective": "4 Serious adverse events", "best model": m4["best_name"], "recommended action": rec4, "workload %": r4["reports worked"] / n4 * 100,
                "problems caught %": r4["problems caught"] / k4 * 100, "net benefit": r4["net benefit"], "gain over best simple": r4["net benefit"] - simple4.max(),
                "hot spot": f"{area4.index[0]} ({area4['rate'].iloc[0]:.0%} serious)"})


# 7. BUSINESS OBJECTIVE 5: FIND PRICING ANOMALIES  ->  which invoices finance should check

print("\n" + "=" * 100 + "\nBUSINESS OBJECTIVE 5: FIND UNUSUAL PRICES AND INVOICES\n" + "=" * 100)
c5, m5 = INPUTS[5]
n5, k5, y5 = len(c5), int(c5["anomaly"].sum()), c5["anomaly"].to_numpy()
gap5 = c5["gap_usd"].to_numpy()
def _review_policy(label, flag):
    flag = np.asarray(flag, bool); caught = int((flag & (y5 == 1)).sum()); recovered = gap5[flag & (y5 == 1)].sum() * OBJ5_RECOVERY_RATE
    return {"policy": label, "invoices checked": int(flag.sum()), "anomalies caught": caught, "anomalies missed": k5 - caught,
            "false alarms": int(flag.sum()) - caught, "money recovered": round(recovered), "review cost": int(flag.sum()) * OBJ5_COST_PER_REVIEW,
            "net value": round(recovered - flag.sum() * OBJ5_COST_PER_REVIEW)}
model_flag5, rule_flag5 = c5["model_flag"].to_numpy(bool), c5["rule_flag"].to_numpy(bool)
plans5 = pd.DataFrame([_review_policy(f"Model only ({m5['best_name']})", model_flag5), _review_policy("Fixed price rule only", rule_flag5),
                       _review_policy("2 of 4 models agree", c5["vote_flag"].to_numpy(bool)), _review_policy("Rule + model (either flags)", model_flag5 | rule_flag5),
                       _review_policy("Check every invoice", np.ones(n5))]).set_index("policy")
rec5 = plans5["net value"].idxmax(); r5 = plans5.loc[rec5]
spots5 = pd.concat([_rates(c5.assign(**{"": c5[col]}), "", "anomaly", 40).rename(index=lambda v, c=col: f"{c.replace('_', ' ')}: {v}") for col in ("customer_type", "sales_channel", "therapeutic_area")]).sort_values("rate", ascending=False)
def _grid5(costs, rates):
    g = pd.DataFrame(index=rates, columns=costs, dtype=float)
    for rr in rates:
        for cc in costs:
            f = model_flag5 | rule_flag5
            g.loc[rr, cc] = (gap5[f & (y5 == 1)].sum() * rr - f.sum() * cc) - max(0, gap5[y5 == 1].sum() * rr - n5 * cc)
    return g
sens5 = _grid5([10, 25, 50, 100, 200], [0.1, 0.25, 0.5, 0.75, 1.0]); sens5.index.name, sens5.columns.name = "share of price gap recovered", "cost per invoice check"
fig, ax = plt.subplots(2, 2, figsize=(16, 10.5)); fig.suptitle("Objective 5: Find unusual prices and invoices (what finance should check)", fontsize=14, fontweight="bold", x=0.01, ha="left")
nv = plans5["net value"]; ax[0, 0].barh(nv.index[::-1], nv.values[::-1], color=[BLUE if i == rec5 else "#b9c7d6" for i in nv.index[::-1]], height=0.6)
for y, v in enumerate(nv.values[::-1]):
    ax[0, 0].text(v + nv.abs().max() * 0.01, y, f"{v:,.0f}", va="center", fontsize=9)
ax[0, 0].set(xlabel="net value on the test invoices (example costs)", title="A. Which check policy pays best? (highlighted = best)"); ax[0, 0].xaxis.set_major_formatter(_thousands); ax[0, 0].set_xlim(0, nv.max() * 1.18)
yy = np.arange(len(plans5))[::-1]
ax[0, 1].barh(yy + 0.2, plans5["anomalies caught"] / k5 * 100, height=0.38, color=BLUE, label="real anomalies caught (%)")
ax[0, 1].barh(yy - 0.2, plans5["invoices checked"] / n5 * 100, height=0.38, color=ORANGE, label="invoices checked (% of all)")
ax[0, 1].set_yticks(yy); ax[0, 1].set_yticklabels(plans5.index, fontsize=9); ax[0, 1].set(xlabel="%", title="B. What each policy catches against the work it creates"); ax[0, 1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
top = spots5.head(8).sort_values("rate"); avg5 = k5 / n5 * 100
ax[1, 0].barh(top.index, top["rate"] * 100, color=ORANGE, height=0.6); ax[1, 0].axvline(avg5, color=INK, ls="--", lw=1)
for y, (v, n_) in enumerate(zip(top["rate"] * 100, top["n"])):
    ax[1, 0].text(v + 0.05, y, f"{v:.1f}%  (n={int(n_)})", va="center", fontsize=8.5)
ax[1, 0].set(xlabel="anomaly rate (%); dashed = average", title="C. Where the anomalies concentrate"); ax[1, 0].set_xlim(0, max(top["rate"].max() * 100 * 1.3, avg5 * 1.6))
_heat(ax[1, 1], sens5, "D. Gain of 'rule + model' over the best simple policy\n(blue = still worth doing)", "cost per invoice check", "share of price gap recovered")
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(f"{OUT}/charts/objective5_pricing_anomalies.png", dpi=150, bbox_inches="tight"); plt.show()
print(f"- Problem size: {k5} of {n5:,} test invoices ({k5 / n5:.1%}) are mispriced; their price gaps total {gap5[y5 == 1].sum():,.0f}.")
print(f"- Recommended: '{rec5}': checks {r5['invoices checked']} invoices ({r5['invoices checked'] / n5:.1%}), catches {r5['anomalies caught']} of {k5} anomalies with {r5['false alarms']} false alarms; net value {r5['net value']:,.0f} (example costs).")
print(f"- The fixed price rule (below 0.5x or above 1.5x list) is the dependable check here; the model ({m5['best_name']}) adds a second opinion for patterns the rule does not cover. Check every invoice would net {plans5.loc['Check every invoice', 'net value']:,.0f}.")
print(f"- Concentration: highest anomaly rate is {spots5.index[0]} ({spots5['rate'].iloc[0]:.1%}, n={int(spots5['n'].iloc[0])}) against {avg5 / 100:.1%} overall; sample sizes are small, so treat this as a lead, not a finding.")
eith = plans5.loc["Rule + model (either flags)"]
print(f"- 'Rule + model (either flags)' checks {eith['invoices checked']} invoices for {eith['net value']:,.0f}: {int(eith['invoices checked'] - r5['invoices checked'])} more checks than the rule alone, in exchange for the model's second opinion. It beats the best simple policy in {int((sens5 > 0).to_numpy().sum())} of {sens5.size} cost scenarios.\n")
display(plans5)
print("Largest flagged price gaps for finance to check first:")
display(c5.loc[model_flag5 | rule_flag5].sort_values("gap_usd", ascending=False).head(10)[["sale_id", "customer_type", "sales_channel", "price_to_list", "gap_usd", "anomaly"]].round(2))
SUMMARY.append({"objective": "5 Pricing anomalies", "best model": m5["best_name"], "recommended action": rec5, "workload %": r5["invoices checked"] / n5 * 100,
                "problems caught %": r5["anomalies caught"] / k5 * 100, "net benefit": r5["net value"], "gain over best simple": r5["net value"] - max(0, plans5.loc["Check every invoice", "net value"]),
                "hot spot": f"{spots5.index[0]} ({spots5['rate'].iloc[0]:.1%} anomalies)"})


# 8. ALL FIVE TOGETHER

print("\n" + "=" * 100 + "\nALL FIVE OBJECTIVES: THE DECISION IN ONE TABLE\n" + "=" * 100)
business_summary = pd.DataFrame(SUMMARY).set_index("objective")
business_summary.round(1).to_csv(f"{OUT}/business_summary.csv")
display(business_summary.style.format({"workload %": "{:.0f}", "problems caught %": "{:.0f}", "net benefit": "{:,.0f}", "gain over best simple": "{:,.0f}"}, na_rep="n/a"))
fig, ax = plt.subplots(1, 2, figsize=(16, 5.8)); fig.suptitle("All five objectives: effort against result", fontsize=14, fontweight="bold", x=0.01, ha="left")
cls = business_summary.dropna(subset=["workload %"]); xx = np.arange(len(cls))
ax[0].bar(xx - 0.2, cls["workload %"], width=0.38, color=ORANGE, label="share of cases worked (%)"); ax[0].bar(xx + 0.2, cls["problems caught %"], width=0.38, color=BLUE, label="share of problems caught (%)")
for i, (w, c_) in enumerate(zip(cls["workload %"], cls["problems caught %"])):
    ax[0].text(i - 0.2, w + 1.5, f"{w:.0f}%", ha="center", fontsize=9); ax[0].text(i + 0.2, c_ + 1.5, f"{c_:.0f}%", ha="center", fontsize=9)
ax[0].set_xticks(xx); ax[0].set_xticklabels([s.split(" ", 1)[1] for s in cls.index], fontsize=9); ax[0].set(ylim=(0, 115), title="A. Work created vs problems caught (recommended policies)"); ax[0].legend(frameon=False, loc="upper left")
gain = business_summary["gain over best simple"]; ax[1].barh([s for s in gain.index][::-1], gain.values[::-1], color=BLUE, height=0.6)
for y, v in enumerate(gain.values[::-1]):
    ax[1].text(v + gain.max() * 0.01, y, f"{v:,.0f}", va="center", fontsize=9)
ax[1].set(xlabel="example currency units on the test rows", title="B. Gain over the best simple alternative (example costs)"); ax[1].set_xlim(0, gain.max() * 1.15); ax[1].xaxis.set_major_formatter(_thousands)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f"{OUT}/charts/all_objectives_summary.png", dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved: {OUT}/business_summary.csv and five objective charts + the summary chart in {OUT}/charts/")
print("All money figures use the EXAMPLE costs at the top of this cell; replace them with real figures and run it again.")
```

---

## 19. Save Model, Scaler and Features

Each winning model is saved as **three `.joblib` files**, enough to score new data without the notebook:

| File | What it holds |
|---|---|
| `model.joblib` | The trained model |
| `scaler.joblib` | The scaler fitted on the training data |
| `features.joblib` | Exact input columns and their order, fill values for blanks, allowed category values, cut-offs and risk tiers, test results and the scikit-learn version |

| Folder | Model |
|---|---|
| `models/batch_rejection/` | Logistic Regression |
| `models/demand_forecast/` | Linear Regression (Ridge) |
| `models/cold_chain/` | Logistic Regression |
| `models/adverse_events/` | Logistic Regression |
| `models/pricing_anomalies/` | Elliptic Envelope |

**Proof that it works:** each objective is reloaded from its three files alone and has to reproduce the notebook's own test predictions. If any one fails, the cell stops with an error, so a broken file cannot be saved without anyone noticing. All five passed.

### Full code (one cell for all five objectives)

```python
# SAVE MODEL, SCALER AND FEATURES FOR ALL FIVE OBJECTIVES (ONE CELL)
# Purpose : save the winning model of each business objective so it can be deployed without this notebook.
#   Objective 1  Batch rejection        -> Logistic Regression
#   Objective 2  Demand forecast        -> Linear Regression (Ridge)
#   Objective 3  Cold-chain excursions  -> Logistic Regression
#   Objective 4  Serious adverse events -> Logistic Regression
#   Objective 5  Pricing anomalies      -> Elliptic Envelope
# Output  : three .joblib files per objective, in models/<objective>/ :
#             model.joblib     the trained model
#             scaler.joblib    the scaler fitted on the training data
#             features.joblib  the exact input columns and order, fill values for blanks, allowed category
#                              values, cut-offs and risk tiers, the test results and the library version
#           and models/saved_models_summary.csv listing every file.

import base64, contextlib, io, os, time, warnings, zlib
from datetime import date
import joblib, sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from IPython.display import display
    from IPython.utils.capture import capture_output
except ImportError:
    display, capture_output = print, None
warnings.filterwarnings("ignore")

_PIPELINES = {...}   # packed copy of every objective's train, predict, compare and evaluate cells (sections 14 to 17).
                    # It is what lets this cell run on its own; it is left out here to keep the README readable.

def _run_pipeline(n):
    """Run objective n's train, predict, compare and evaluate cells in a private namespace."""
    ns = {"__name__": "__main__"}
    code = zlib.decompress(base64.b64decode(_PIPELINES[n])).decode()
    shown, plt.show = plt.show, (lambda *a, **k: None)
    try:
        if capture_output is not None:
            with capture_output():
                exec(code, ns)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, ns)
    finally:
        plt.show = shown
        plt.close("all")
    return ns

def _floats(d):
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float, np.integer, np.floating))}

def _save_three(ns, features_info):
    out = ns["OUT_DIR"]; os.makedirs(out, exist_ok=True)
    joblib.dump(ns["model"], f"{out}/model.joblib")
    joblib.dump(ns["scaler"], f"{out}/scaler.joblib")
    joblib.dump(features_info, f"{out}/features.joblib")
    return out

def _scale(X, info, scaler):
    """Shared preparation of new rows: same columns and order as training, blanks filled, scaled columns scaled."""
    X = X.reindex(columns=info["feature_columns"], fill_value=0).fillna(info["fill_values"])
    X[info["scaled_columns"]] = scaler.transform(X[info["scaled_columns"]])
    return X

def _encode(raw_rows, info):
    """Raw input columns -> encoded table. Blank or unseen text becomes "Unknown", as in training."""
    X = raw_rows[info["numeric_inputs"] + info["categorical_inputs"]].copy()
    for c in info["categorical_inputs"]:
        X[c] = X[c].fillna("Unknown").where(X[c].fillna("Unknown").isin(info["categories_seen"][c]), "Unknown")
    return pd.get_dummies(X, columns=info["categorical_inputs"], dtype=int)

# -------- the scoring functions: each uses ONLY the three saved files --------
def predict_batch_rejection(raw_rows, folder="models/batch_rejection"):
    info, mdl, scl = joblib.load(f"{folder}/features.joblib"), joblib.load(f"{folder}/model.joblib"), joblib.load(f"{folder}/scaler.joblib")
    p = mdl.predict_proba(_scale(_encode(raw_rows, info), info, scl))[:, 1]
    tier = np.select([p >= info["risk_tiers"]["high_from"], p >= info["risk_tiers"]["medium_from"]], ["High", "Medium"], default="Low")
    return pd.DataFrame({"rejection_risk": p.round(4), "inspection_tier": tier}, index=raw_rows.index)

def predict_demand(raw_rows, folder="models/demand_forecast"):
    info, mdl, scl = joblib.load(f"{folder}/features.joblib"), joblib.load(f"{folder}/model.joblib"), joblib.load(f"{folder}/scaler.joblib")
    return pd.DataFrame({"forecast_units": np.expm1(mdl.predict(_scale(_encode(raw_rows, info), info, scl))).round(1)}, index=raw_rows.index)

def predict_cold_chain(raw_rows, folder="models/cold_chain"):
    info, mdl, scl = joblib.load(f"{folder}/features.joblib"), joblib.load(f"{folder}/model.joblib"), joblib.load(f"{folder}/scaler.joblib")
    p = mdl.predict_proba(_scale(_encode(raw_rows, info), info, scl))[:, 1]
    tier = np.select([p >= info["risk_tiers"]["high_from"], p >= info["risk_tiers"]["medium_from"]], ["High", "Medium"], default="Low")
    return pd.DataFrame({"excursion_risk": p.round(4), "protection_tier": tier}, index=raw_rows.index)

def predict_adverse_events(raw_rows, folder="models/adverse_events"):
    info, mdl, scl = joblib.load(f"{folder}/features.joblib"), joblib.load(f"{folder}/model.joblib"), joblib.load(f"{folder}/scaler.joblib")
    p = mdl.predict_proba(_scale(_encode(raw_rows, info), info, scl))[:, 1]
    tier = np.select([p >= info["risk_tiers"]["high_from"], p >= info["risk_tiers"]["medium_from"]], ["High", "Medium"], default="Low")
    return pd.DataFrame({"serious_event_risk": p.round(4), "review_tier": tier}, index=raw_rows.index)

def predict_pricing_anomalies(raw_rows, folder="models/pricing_anomalies"):
    info, mdl, scl = joblib.load(f"{folder}/features.joblib"), joblib.load(f"{folder}/model.joblib"), joblib.load(f"{folder}/scaler.joblib")
    X = _scale(raw_rows[info["feature_columns"]].copy(), info, scl)
    return pd.DataFrame({"anomaly_score": (-mdl.score_samples(X)).round(4), "flagged": (mdl.predict(X) == -1).astype(int)}, index=raw_rows.index)

def _classification_info(ns, tier_names):
    cfg, t, m = ns["cfg"], float(ns["threshold"]), float(ns["money_cutoff"])
    return {"objective": ns["OBJECTIVE"], "title": cfg["title"], "task": "classification", "target": cfg["target"], "model_name": ns["best_name"],
            "threshold": t, "money_cutoff": m, "risk_tiers": {"high_from": max(t, m), "medium_from": min(t, m)}, "tier_names": tier_names,
            "numeric_inputs": cfg["numeric"], "categorical_inputs": cfg["categorical"],
            "categories_seen": {c: sorted(ns["X_text"][c].unique().tolist()) for c in cfg["categorical"]},
            "feature_columns": ns["feature_columns"], "scaled_columns": ns["scale_cols"],
            "fill_values": {k: float(v) for k, v in ns["fill_values"].items()}, "left_out_for_leakage": cfg["leakage"],
            "metrics": _floats(ns["metrics"]), "trained_on": str(date.today()), "sklearn_version": sklearn.__version__}

RESULTS, FILES = [], []
def _record(label, folder, ok, detail):
    RESULTS.append({"objective": label, "folder": folder, "reload reproduces notebook": "YES" if ok else "NO - investigate", "check": detail})
    for f in ("model.joblib", "scaler.joblib", "features.joblib"):
        FILES.append({"objective": label, "file": f"{folder}/{f}", "size KB": round(os.path.getsize(f"{folder}/{f}") / 1024, 1)})

t0 = time.time()

# OBJECTIVE 1: BATCH REJECTION (Logistic Regression)

print("Objective 1: training, then saving ...")
ns = _run_pipeline(1)
folder = _save_three(ns, _classification_info(ns, ["Inspect first", "Inspect if capacity allows", "Release as normal"]))
reloaded = predict_batch_rejection(ns["abt"].loc[ns["X_test"].index], folder)
ok = bool(np.allclose(reloaded["rejection_risk"], ns["test_proba"][ns["best_name"]], atol=1e-3))
_record("1 Batch rejection", folder, ok, f"{len(reloaded)} test batches: risks match the notebook's within 0.001")
sample1 = reloaded.head(5)


# OBJECTIVE 2: DEMAND FORECAST (Linear Regression, Ridge)

print("Objective 2: training, then saving ...")
ns = _run_pipeline(2)
info = {
    "objective": ns["OBJECTIVE"], "title": ns["cfg"]["title"], "task": "regression", "target": ns["cfg"]["target"], "target_note": "log of units sold; predictions are converted back with expm1",
    "model_name": ns["best_name"], "numeric_inputs": ns["cfg"]["numeric"], "categorical_inputs": ns["cfg"]["categorical"],
    "categories_seen": {c: sorted(ns["X_text"][c].unique().tolist()) for c in ns["cfg"]["categorical"]},
    "feature_columns": ns["feature_columns"], "scaled_columns": ns["scale_cols"], "fill_values": {k: float(v) for k, v in ns["fill_values"].items()},
    "left_out_for_leakage": ns["cfg"]["leakage"], "metrics": _floats(ns["metrics"]), "trained_on": str(date.today()), "sklearn_version": sklearn.__version__}
folder = _save_three(ns, info)
reloaded = predict_demand(ns["abt"].loc[ns["X_test"].index], folder)
ok = bool(np.allclose(reloaded["forecast_units"], ns["p_best"], atol=1.0))
_record("2 Demand forecast", folder, ok, f"{len(reloaded)} test sales: forecasts match the notebook's within 1 unit")
sample2 = reloaded.head(5)


# OBJECTIVE 3: COLD-CHAIN EXCURSIONS (Logistic Regression)

print("Objective 3: training, then saving ...")
ns = _run_pipeline(3)
folder = _save_three(ns, _classification_info(ns, ["Protect first", "Protect if budget allows", "Ship as normal"]))
reloaded = predict_cold_chain(ns["abt"].loc[ns["X_test"].index], folder)
ok = bool(np.allclose(reloaded["excursion_risk"], ns["test_proba"][ns["best_name"]], atol=1e-3))
_record("3 Cold-chain excursions", folder, ok, f"{len(reloaded)} test shipments: risks match the notebook's within 0.001")
sample3 = reloaded.head(5)


# OBJECTIVE 4: SERIOUS ADVERSE EVENTS (Logistic Regression)

print("Objective 4: training, then saving ...")
ns = _run_pipeline(4)
folder = _save_three(ns, _classification_info(ns, ["Review first", "Review next", "Standard queue"]))
reloaded = predict_adverse_events(ns["abt"].loc[ns["X_test"].index], folder)
ok = bool(np.allclose(reloaded["serious_event_risk"], ns["test_proba"][ns["best_name"]], atol=1e-3))
_record("4 Serious adverse events", folder, ok, f"{len(reloaded)} test reports: risks match the notebook's within 0.001")
sample4 = reloaded.head(5)


# OBJECTIVE 5: PRICING ANOMALIES (Elliptic Envelope; unsupervised, so no target and no cut-off)

print("Objective 5: training, then saving ...")
ns = _run_pipeline(5)
info = {"objective": ns["OBJECTIVE"], "title": "Find unusual prices and invoices", "task": "anomaly_detection", "target": None,
        "model_name": ns["best_name"], "decision_rule": "flag when model.predict(X) == -1; a higher anomaly_score (= -score_samples) is more unusual",
        "companion_rule": "price_to_list below 0.5 or above 1.5 (found every planted anomaly in testing); use it alongside the model",
        "numeric_inputs": list(ns["feature_columns"]), "categorical_inputs": [], "categories_seen": {}, "feature_columns": ns["feature_columns"],
        "scaled_columns": ns["scale_cols"], "fill_values": {k: float(v) for k, v in ns["fill_values"].items()}, "left_out_for_leakage": ["anomaly_answer_key"],
        "metrics": _floats(ns["metrics"]), "trained_on": str(date.today()), "sklearn_version": sklearn.__version__}
folder = _save_three(ns, info)
reloaded = predict_pricing_anomalies(ns["X_test"], folder)
ok = bool(np.allclose(reloaded["anomaly_score"], ns["s_best"], atol=1e-3) and (reloaded["flagged"].to_numpy() == np.asarray(ns["f_best"])).all())
_record("5 Pricing anomalies", folder, ok, f"{len(reloaded)} test invoices: scores and flags match the notebook's")
sample5 = reloaded.head(5)


# SUMMARY

saved_models_summary = pd.DataFrame(FILES)
saved_models_summary.to_csv("models/saved_models_summary.csv", index=False)
print(f"\nFinished in {time.time() - t0:.0f}s. Deployment-readiness check (each objective reloaded from its three files alone):\n")
display(pd.DataFrame(RESULTS).set_index("objective"))
print("\nFiles saved:"); display(saved_models_summary.set_index("objective"))
if not all(r["reload reproduces notebook"] == "YES" for r in RESULTS):
    raise AssertionError("At least one reloaded model did not reproduce the notebook; do not deploy it. See the table above.")
print("\nAll 15 files reload and reproduce the notebook. Score new rows with predict_batch_rejection, predict_demand, predict_cold_chain,")
print("predict_adverse_events and predict_pricing_anomalies (each takes the raw input columns and uses only the three saved files).")
print("\nExample, first 5 test batches scored from the saved files alone:"); display(sample1)
```

---

## 20. Download Model, Scaler and Features

One cell packages all 15 files into a deployment bundle:

```
deployment/veridian_pharma_models.zip
    batch_rejection/   model.joblib  scaler.joblib  features.joblib
    demand_forecast/   ...
    cold_chain/        ...
    adverse_events/    ...
    pricing_anomalies/ ...
    manifest.json       file sizes, SHA-256 fingerprints, library versions, test results
    README_DEPLOY.txt   how to load the files and score new rows
```

The zip is reopened and every file is fingerprinted again before the cell reports success, so the deployment team can confirm nothing was damaged in transfer.

### Full code

```python
# DOWNLOAD MODEL, SCALER AND FEATURES FOR DEPLOYMENT (ONE CELL)
#
# Purpose : package the 15 saved files (3 per objective) into one deployment bundle.
# Inputs  : the files written by the save cell, in models/<objective>/ (model, scaler, features .joblib)
# Output  : deployment/veridian_pharma_models.zip   the bundle to download and hand to the deployment team
#           deployment/veridian_pharma_models/      the same files unzipped, ready to use
#           Inside the bundle: manifest.json (file sizes and SHA-256 fingerprints, library versions, results)
#                              and README_DEPLOY.txt (how to load and score with the files)
import hashlib, json, os, shutil, zipfile
from datetime import datetime
import joblib, sklearn
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print

OBJECTIVES = {   # folder name -> (label, model name expected, how to score)
    "batch_rejection":   ("1 Batch rejection",        "predict_batch_rejection"),
    "demand_forecast":   ("2 Demand forecast",        "predict_demand"),
    "cold_chain":        ("3 Cold-chain excursions",  "predict_cold_chain"),
    "adverse_events":    ("4 Serious adverse events", "predict_adverse_events"),
    "pricing_anomalies": ("5 Pricing anomalies",      "predict_pricing_anomalies"),
}
FILES = ("model.joblib", "scaler.joblib", "features.joblib")
BUNDLE, ZIP_PATH = "deployment/veridian_pharma_models", "deployment/veridian_pharma_models.zip"

# ---------- 1. Every file must exist (they come from the save cell) ----------
missing = [f"models/{o}/{f}" for o in OBJECTIVES for f in FILES if not os.path.exists(f"models/{o}/{f}")]
if missing:
    raise FileNotFoundError(f"{len(missing)} saved file(s) not found, for example {missing[0]}. Run the save model, scaler and features cell first.")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()

# ---------- 2. Copy into a clean bundle folder and fingerprint ----------
shutil.rmtree(BUNDLE, ignore_errors=True)
manifest = {"created": datetime.now().isoformat(timespec="seconds"), "sklearn_version_here": sklearn.__version__, "joblib_version_here": joblib.__version__, "objectives": {}}
rows = []
for folder, (label, scorer) in OBJECTIVES.items():
    os.makedirs(f"{BUNDLE}/{folder}", exist_ok=True)
    info = joblib.load(f"models/{folder}/features.joblib")
    entry = {"label": label, "model_name": info["model_name"], "task": info["task"], "scoring_function": scorer, "trained_on": info["trained_on"],
             "sklearn_version_trained": info["sklearn_version"], "n_input_columns": len(info["feature_columns"]), "metrics": info["metrics"], "files": {}}
    for f in FILES:
        shutil.copy2(f"models/{folder}/{f}", f"{BUNDLE}/{folder}/{f}")
        entry["files"][f] = {"size_bytes": os.path.getsize(f"{BUNDLE}/{folder}/{f}"), "sha256": sha256(f"{BUNDLE}/{folder}/{f}")}
        rows.append({"objective": label, "file": f"{folder}/{f}", "size KB": round(entry["files"][f]["size_bytes"] / 1024, 1), "sha256 (first 12)": entry["files"][f]["sha256"][:12]})
    manifest["objectives"][folder] = entry
    if info["sklearn_version"] != sklearn.__version__:
        print(f"WARNING: {label} was trained with scikit-learn {info['sklearn_version']} but this environment has {sklearn.__version__}. Deploy with the training version.")
json.dump(manifest, open(f"{BUNDLE}/manifest.json", "w"), indent=2)

readme = f"""VERIDIAN PHARMA GLOBAL: DEPLOYMENT BUNDLE
Created {manifest['created']} | trained with scikit-learn {sklearn.__version__} | joblib {joblib.__version__}

Contents: one folder per business objective, each with three files:
  model.joblib     the trained model
  scaler.joblib    the scaler fitted on the training data
  features.joblib  input columns and order, fill values for blanks, allowed category values,
                   cut-offs and risk tiers, test results and library versions
manifest.json lists every file with its size and SHA-256 fingerprint; compare them after transfer.

REQUIREMENTS: Python 3, pandas, numpy, scikit-learn=={sklearn.__version__}, joblib. Use the same scikit-learn
version that trained the models; joblib files are not guaranteed to load across versions.
SECURITY: joblib files can run code when loaded. Load only files you received from a trusted source.

OBJECTIVES AND MODELS
""" + "".join(f"  {v['label']:<28} {v['model_name']:<26} scoring function: {v['scoring_function']}\n" for v in manifest["objectives"].values()) + """
HOW TO SCORE NEW ROWS (example, batch rejection; the other objectives work the same way)
    import joblib, numpy as np, pandas as pd
    folder = "batch_rejection"
    info, model, scaler = (joblib.load(f"{folder}/features.joblib"), joblib.load(f"{folder}/model.joblib"), joblib.load(f"{folder}/scaler.joblib"))
    X = new_rows[info["numeric_inputs"] + info["categorical_inputs"]].copy()          # new_rows = your raw input columns
    for c in info["categorical_inputs"]:                                              # blank or unseen text -> "Unknown"
        X[c] = X[c].fillna("Unknown").where(X[c].fillna("Unknown").isin(info["categories_seen"][c]), "Unknown")
    X = pd.get_dummies(X, columns=info["categorical_inputs"], dtype=int)
    X = X.reindex(columns=info["feature_columns"], fill_value=0).fillna(info["fill_values"])
    X[info["scaled_columns"]] = scaler.transform(X[info["scaled_columns"]])
    risk = model.predict_proba(X)[:, 1]                                               # probability of the problem
    tier = np.select([risk >= info["risk_tiers"]["high_from"], risk >= info["risk_tiers"]["medium_from"]], ["High", "Medium"], default="Low")

NOTES BY OBJECTIVE
  Demand forecast: the model predicts the LOG of units sold; convert with numpy.expm1(model.predict(X)).
  Pricing anomalies: no target and no probability. Score with  -model.score_samples(X)  (higher = more unusual);
                     flag where model.predict(X) == -1. Also check price_to_list below 0.5 or above 1.5, which found
                     every planted anomaly in testing. Its inputs are the 7 numeric pricing features in features.joblib.
  Cut-offs and risk tiers are in features.joblib (threshold, money_cutoff, risk_tiers). Recompute them if the
  business costs of acting or missing change, and retrain regularly: models age as the business changes.
"""
open(f"{BUNDLE}/README_DEPLOY.txt", "w", encoding="utf-8").write(readme)

# ---------- 3. Zip ----------
os.makedirs("deployment", exist_ok=True)
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, names in os.walk(BUNDLE):
        for name in sorted(names):
            full = os.path.join(root, name)
            zf.write(full, os.path.relpath(full, BUNDLE))

# ---------- 4. Check the zip: reopen it and re-fingerprint every file ----------
with zipfile.ZipFile(ZIP_PATH) as zf:
    assert zf.testzip() is None, "The zip is corrupt."
    bad = []
    for folder, entry in manifest["objectives"].items():
        for f, meta in entry["files"].items():
            h = hashlib.sha256(zf.read(f"{folder}/{f}")).hexdigest()
            if h != meta["sha256"]:
                bad.append(f"{folder}/{f}")
    n_in_zip = len([n for n in zf.namelist() if n.endswith(".joblib")])
if bad or n_in_zip != 15:
    raise AssertionError(f"Zip check failed: {bad or n_in_zip}")

print(f"Bundle ready: {n_in_zip} .joblib files + manifest.json + README_DEPLOY.txt, all fingerprints verified.")
print(f"  Zip file : {os.path.abspath(ZIP_PATH)}   ({os.path.getsize(ZIP_PATH) / 1024:.0f} KB)")
print(f"  Unzipped : {os.path.abspath(BUNDLE)}\n")
display(pd.DataFrame(rows).set_index("objective"))

# ---------- 5. Download ----------
try:                                                   # Google Colab: sends the zip to the browser's downloads
    from google.colab import files
    files.download(ZIP_PATH)
    print("\nDownload started in your browser.")
except ImportError:
    print("\nRunning locally (VS Code / Jupyter): the zip is already on your computer at the path above. Copy it to the deployment machine.")
```

### The Streamlit app

The saved files power the app, [`app.py`](app.py). It has a Home page and one page for each model: **Batch Quality, Demand Planning, Cold-Chain Protection, Drug Safety and Pricing Control.**

- **Single case:** fill in a short form and press Predict. The app returns the risk percentage, a High, Medium or Low tier, a recommended action and the main factors behind the result.
- **Many cases:** download a CSV template, upload your filled file and download the scored results.
- **Pricing Control** combines the model with the fixed price rule, and says why an invoice was flagged.
- The forms ask plain questions. The app works out the model's derived inputs itself, such as heat exposure and months since start.

Live app: [Open the Streamlit app](YOUR-STREAMLIT-LINK)

<!-- Add a screenshot: <p align="center"><img src="images/app_screenshot.png" width="100%"></p> -->

### Full code: app.py

```python
# PHARMACEUTICAL BUSINESS OPERATIONS PREDICTION SYSTEM (Veridian Pharma Global)
# A Streamlit app that uses the five saved models (model, scaler and features .joblib files).
#
#   Page            Business problem                 Model
#   Batch Quality         Will this batch be rejected?     Logistic Regression
#   Demand Planning       How many units will we sell?     Linear Regression (Ridge)
#   Cold-Chain Protection Will this shipment get too hot?  Logistic Regression
#   Drug Safety           Is this report a serious event?  Logistic Regression
#   Pricing Control       Is this invoice priced oddly?    Elliptic Envelope

from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="Pharmaceutical Business Operations Prediction System", page_icon="💊", layout="wide")


# PART 1. FIND AND LOAD THE SAVED FILES

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


# PART 2. SCORING HELPERS

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


# PART 3. THE FIVE OBJECTIVES
# Each objective has: a list of questions (fields) and a function that turns the answers into the model's inputs.
# Field helpers: num = a number, pick = a drop-down, yesno = Yes/No (stored as 1/0), when = a date.

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


# PART 4. DRAWING THE PAGES

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
st.sidebar.title("Pharmaceutical Business Operations Prediction System")
choice = st.sidebar.radio("Go to", ["Home"] + list(PAGES))
if choice == "Home":
    home_page()
else:
    objective_page(choice)
```

---

## 21. Key Findings and Insights

**Objective 1: batch rejection**
- The model ranks risk well: High-tier batches are rejected 45% of the time, Low-tier only 4%.
- Purity, temperature deviation, equipment age and plant age matter most.
- Two plants account for most of the problem, at about double the average rate.

**Objective 2: demand forecast**
- Ridge is off by 21% on a single sale but only 10% when sales are added up by product area and month, which is how production is planned.
- Oncology is the easiest to plan (4% error), Vaccines the hardest (17%).
- Planning from the forecast cuts stock cost by 71% against the best flat plan, in every cost scenario tested.

**Objective 3: cold chain**
- Transit time, outside temperature and cold-chain products shipped without protective packaging drive excursions.
- Low-tier shipments had an excursion only 2% of the time. Only 4 of 138 excursions fell there.
- Sea freight (47%) and Nigeria-bound shipments (31%) stand out.

**Objective 4: serious adverse events**
- Dosage, patient age, other medicines and liver impairment drive the model.
- High-tier reports are serious 62% of the time. Only 1 of 206 Low-tier reports was.
- Patients aged 65 and over (37%) and Anti-infective products (28%) have the highest rates.

**Objective 5: pricing anomalies**
- 40 of 1,500 test invoices (2.7%) were mispriced, with price gaps adding up to about 2.37 million.
- Elliptic Envelope caught 36 of 40 with 4 false alarms; Isolation Forest caught 33 with 3.
- A simple rule caught all 40 with no false alarms, which beats every model here.

**Across all five**
- **The simplest model won almost every time.** Logistic Regression won three objectives, Ridge a fourth. Random Forest and Gradient Boosting finished within about 0.01 AUC.
- **The cut-off matters more than the algorithm.** In Objective 1, the balanced cut-off catches 80 of 152 rejections and the cost-based one catches 131.
- **Leakage was the biggest risk.** Columns that reveal the answer were excluded, so results reflect what is known in advance.
- **Models rank risk; they do not predict it exactly.** At the balanced cut-offs, fewer than half of flagged cases in Objectives 1 and 3 were real problems. Use them to decide where to look first.

---

## 22. Recommendations

1. **Use the model to set inspection order, not to replace inspection.** Put batches, shipments and safety reports into High, Medium and Low tiers. Work High and Medium first and release Low as normal.
2. **Use the lower, cost-based cut-off where a miss is expensive.** For serious adverse events and cold-chain excursions, an extra review or protection costs far less than a missed case. Replace the example costs with real ones and recompute the cut-offs.
3. **Target the hot spots.** Investigate Delta Works 6 and Gamma Works 7 for rejections, and sea freight and Nigeria-bound shipments for excursions. Add protective packaging to cold-chain products now shipped unprotected.
4. **Plan production from the forecast, by product area and month.** Hold extra safety stock where stock-outs cost most, and review the Vaccines forecast first.
5. **Run the fixed price rule on every invoice, with the model as a second check.** Send flagged invoices to finance, largest price gap first.
6. **Prioritise safety review** for patients 65 and over, high doses and patients on several medicines.
7. **Deploy the saved files in a controlled way.** Use the same scikit-learn version that trained them, keep the manifest fingerprints and score new rows with the matching function.
8. **Monitor and retrain.** Track the problem rate in each tier every month, retrain when it drifts and keep a person in the loop for every flagged case.
9. **Improve the data.** Capture real costs for inspection, protection, review and shortage, and record the outcome of every flagged case.

---

## 23. Impact

This section shows what the models are worth if they are used the way section 22 recommends. Every figure comes from the unseen test rows. Money figures use **example costs** (set out below), so treat them as a way to compare policies, not as a forecast of real savings.

### The size of each problem today

| # | Business problem | What the data shows |
|---|---|---|
| 1 | Batch rejection | 605 batches (15.2%) were rejected. Two plants, PLT-16 and PLT-17, carry 44% of the rejected production cost |
| 2 | Demand planning | 21 of 40 products bring in 80% of revenue. Month 3 runs 15% above the average month |
| 3 | Cold chain | 15.8% of shipments had an excursion and 55% of those were written off, about $8.2M of sales value |
| 4 | Drug safety | 20% of reports are serious, rising to 36% for patients aged 65 and over |
| 5 | Pricing | 131 invoices (2.2%) are priced below 50% or above 150% of list: a $1.89M shortfall and $6.2M of over-billing risk |

### What the models change

| # | Policy | Effort | Result on unseen data | Net benefit (example costs) | Gain over the best simple policy |
|---|---|---|---|---|---|
| 1 | Inspect High and Medium risk batches | 47% of batches | Catches **86%** of rejections | 419,500 | +159,500 |
| 2 | Plan production from the forecast, by product area and month | n/a | Forecast error falls to **10%**. Stock cost 499,180 against 1,721,109 | **-71%** stock cost | in every cost scenario tested |
| 3 | Protect High and Medium risk shipments | 71% of shipments | Prevents up to **97%** of excursions | 349,400 | +59,900 |
| 4 | Review High and Medium risk reports first | 54% of reports | Reaches **99%** of serious cases | 230,400 | +27,900 |
| 5 | Fixed price rule on every invoice, model as a second check | 3% of invoices | Flags **100%** of the 40 mispriced invoices | 1,183,836 | +73,000 |

### In plain terms

- **Batches:** inspecting under half the batches finds about 9 in 10 rejections. Low-tier batches are rejected only 4% of the time, so they can be released as normal. At the cost-based cut-off the model catches 131 of 152 rejections, against 80 at the balanced cut-off.
- **Demand:** a single sale is forecast to within 21%, but once sales are added up by product area and month (how production is actually planned) the error is only 10%. That is where the 71% fall in stock cost comes from. Vaccines is the hardest to plan (17% error) and Oncology the easiest (4%).
- **Cold chain:** Low-tier shipments had an excursion only 2% of the time, and only 4 of 138 excursions fell there. Protecting the rest is far cheaper than writing off the 55% of excursions that are lost.
- **Safety:** High-tier reports are serious 62% of the time. Only 1 of 206 Low-tier reports was serious, so urgent review effort can go where the risk is.
- **Pricing:** a transparent price rule caught all 40 mispriced invoices with no false alarms and beat every model. The Elliptic Envelope model caught 36 of 40 with 4 false alarms, so it stays as a second check for odd patterns the rule misses.

### What it cost to get there

The net benefit figures above are built on these example costs, which you can change in section 18 and re-run:

| # | Example cost used |
|---|---|
| 1 | 500 to inspect a batch, 5,000 for a rejection that slips through |
| 2 | 2 per unit made but unsold, 6 per unit demanded but unavailable |
| 3 | 300 to protect a shipment, 4,000 for an excursion on an unprotected one |
| 4 | 150 for an urgent review, 3,000 for a serious case left waiting |
| 5 | 50 to check an invoice, 50% of the price gap recovered |

### Where to start

The biggest and easiest gains are the two plants behind 44% of rejected cost, sea freight and Nigeria-bound shipments, patients aged 65 and over, and the monthly invoice exception review. If those two plants matched the others, about 176 fewer batches and 25% less rejected cost would follow.

### What to keep in mind

- The dollar figures rest on example costs. Replace them with real inspection, protection, review and shortage costs before quoting a saving.
- Each result comes from one held-out sample, so measure it against current practice in a pilot before scaling.
- At the balanced cut-offs, fewer than half of the flagged batches and shipments are real problems. The models decide **where to look first**, and a person still makes the call.
- The data is fictional, so the figures show how the method works and not how a real plant performs.

---

## 24. Conclusion

I built five models for five business problems, compared at least four algorithms for each, chose the winners on training evidence and confirmed them on unseen data. Logistic Regression solved batch rejection, cold-chain excursions and serious adverse events. Ridge regression handled the demand forecast, and Elliptic Envelope handled pricing anomalies. All five are saved with their scalers and feature definitions, reload and reproduce the notebook's results, and run in a Streamlit app that scores one case or a whole file.

Used as a ranking tool, the models point inspection, protection and review effort at a minority of cases that hold most of the problems. In the test data, that means inspecting under half the batches to catch 86% of rejections, reviewing just over half of the safety reports to reach 99% of serious cases, and cutting forecast stock cost by about 70%. For pricing, a transparent rule beat every model, a reminder to test a simple baseline before choosing a complex one.

**Limits to keep in mind**
- The money figures use example costs, not real ones.
- Results come from a single held-out sample per objective.
- The data is fictional, so the findings demonstrate the method rather than the performance of a real plant.

**Next steps:** replace the example costs with real figures, pilot the tiered policies next to current practice, measure the difference, and retrain on fresh data as the business changes.

---

**Author:** Adewale Oshinukan | Data Scientist | [GitHub](YOUR-GITHUB-LINK) | [Live App](YOUR-STREAMLIT-LINK)
