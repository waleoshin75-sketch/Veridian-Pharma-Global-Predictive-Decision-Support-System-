# Veridian Pharma Global: Predictive Decision-Support System

**Pharmaceutical Business Operations Analytics**

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
23. [Conclusion](#23-conclusion)

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

The audit changes nothing. It counts every problem first, one reusable function per table:

```python
def audit_table(df, name, id_col):
    miss = df.isna().sum()                                   # missing values
    print(f"fully duplicated rows : {df.duplicated().sum()}")
    print(f"repeated {id_col:<12}: {df[id_col].duplicated().sum()}")
    print(df.dtypes.astype(str).to_string())                 # numbers stored as text show up here
```

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

---

## 5. Data Cleaning and Error Checking

Cleaning is a set of small, reusable helpers applied table by table. Four of them do most of the work:

```python
def tidy_key(s):        # 'prd-126 ' -> 'PRD-126', so keys match across tables
    s = s.astype("object")
    return s.where(s.isna(), s.astype(str).str.strip().str.upper())

def to_number(s):       # '$1,250.50', '99%', '3 days' -> float
    txt = s.where(s.isna(), s.astype(str).str.lower()).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(txt, errors="coerce")

def to_yes_no(s):       # Yes / Y / TRUE / 1 -> 1 ; No / N / FALSE / 0 -> 0
    key = s.astype("object")
    key = key.where(key.isna(), key.astype(str).str.strip().str.lower())
    return key.map({"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}).astype("Int64")

def parse_dates(s):     # six formats -> one datetime column; '??' becomes missing
    ...
```

**How impossible values were handled** (batches as the example): flag them as missing, then fill with the median of the same product or plant, so one bad value does not distort a group.

```python
batches_clean.loc[(batches_clean["raw_material_purity_pct"] < 80) |
                  (batches_clean["raw_material_purity_pct"] > 100), "raw_material_purity_pct"] = np.nan
batches_clean["raw_material_purity_pct"] = fill_by_group(batches_clean, "raw_material_purity_pct", "product_id")
```

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

```python
batches_f["is_rejected"]         = (batches_f["release_status"] == "Rejected").astype(int)
batches_f["low_purity_flag"]     = (batches_f["raw_material_purity_pct"] < 98).astype(int)
batches_f["old_equipment_flag"]  = (batches_f["equipment_age_years"] > 25).astype(int)
batches_f["night_old_equipment"] = batches_f["night_shift"] * batches_f["old_equipment_flag"]
```

**Leakage guard.** Each objective has a list of columns that can never be inputs:

| Objective | Kept out of the model |
|---|---|
| 1 Batch rejection | `release_status`, `rejection_reason`, `process_deviations` |
| 2 Demand | `units_sold`, `total_amount_usd`, `expected_total_usd` |
| 3 Cold chain | `temperature_excursion`, `product_written_off` |
| 4 Adverse events | `severity`, `outcome`, `hospitalised` |
| 5 Pricing | `anomaly_answer_key` |

A quick sanity check (not a model) confirmed the new features carry signal: batches on old equipment are rejected **26.9%** of the time against 10.2% for newer equipment.

---

## 7. Create Relationship Table

The cleaned tables are linked as a **star schema**: four fact tables around three dimensions and a date table.

| Fact table | Links to |
|---|---|
| `fact_sales` | `dim_customers`, `dim_products`, `dim_date` |
| `fact_batches` | `dim_products`, `dim_plants`, `dim_date` |
| `fact_shipments` | `fact_sales` (and through it customers and products), `dim_date` |
| `fact_adverse_events` | `fact_batches`, `dim_products`, `dim_date` |

Some rows pointed at records that do not exist (orphans): 25 batches with an unknown product, 30 with an unknown plant, 50 sales with an unknown customer, 20 adverse events with an unknown batch and 30 shipments with an unknown sale. Deleting them would lose real sales and real safety reports, so each dimension gets an `UNKNOWN` member and the orphans point to it, with a flag to keep the information:

```python
unknown_plant = {"plant_id": "UNKNOWN", "plant_name": "Unknown plant", "country": "Unknown", "automation_level": "Unknown"}
dim_plants = pd.concat([plants, pd.DataFrame([unknown_plant])], ignore_index=True)
```

Ten checks confirm every key is unique, no orphan keys remain and every date key exists in `dim_date`. The phase ends with five **modelling tables**, one per objective:

| Modelling table | Rows x columns | Objective |
|---|---|---|
| `model_batches` | 4,000 x 45 | 1 |
| `model_demand` | 6,000 x 47 | 2 |
| `model_shipments` | 3,500 x 38 | 3 |
| `model_adverse` | 1,800 x 37 | 4 |
| `model_pricing` | 6,000 x 40 | 5 |

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

<p align="center">
  <img src="images/eda_02_rejection_by_plant_shift.png" alt="Batch rejection rate by plant, formulation, shift and GMP status" width="100%">
</p>
<p align="center"><em>Rejection rate by plant, product form, shift and GMP status</em></p>

<p align="center">
  <img src="images/eda_07_excursions_overview.png" alt="Excursion rate by packaging, transport mode, season and destination" width="100%">
</p>
<p align="center"><em>Cold-chain excursion rate by packaging, transport mode, season and destination</em></p>

<p align="center">
  <img src="images/eda_09_adverse_risk_groups.png" alt="Serious event rate by age group and risk factors" width="100%">
</p>
<p align="center"><em>Serious adverse event rate by patient risk group</em></p>

<p align="center">
  <img src="images/eda_12_correlations.png" alt="Correlation of numeric features with each outcome" width="100%">
</p>
<p align="center"><em>Which numeric features move with each outcome (outcome columns excluded)</em></p>

The other eight charts are in the [`images`](images) folder.

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

```python
CONFIG = {"batch_rejection": dict(file="model_batches.csv", id_col="batch_id", target="is_rejected",
                                  numeric=[...], categorical=["formulation", "therapeutic_area"],
                                  leakage=["release_status", "rejection_reason", "process_deviations"], ...)}
abt = raw[[cfg["id_col"], cfg["date_col"]] + features + [cfg["target"]]].copy()
case_info = abt[[cfg["id_col"], cfg["date_col"]]]          # labels, never model inputs
```

Objective 5 is unsupervised, so its table holds seven pricing ratios only: price to list price, price to cost, total to expected total, invoice size, discount, quantity and unit price.

---

## 10. Encode Categorical Columns

Models need numbers. Text columns become 0/1 columns, one per category. Blank categories become their own `Unknown` label instead of being mistaken for the first category, and `drop_first=True` removes the one column that would repeat information.

```python
X_text[cfg["categorical"]] = X_text[cfg["categorical"]].fillna("Unknown")
X_all = pd.get_dummies(X_text, columns=cfg["categorical"], drop_first=True, dtype=int)
feature_columns = list(X_all.columns)
```

| Objective | Columns before | Columns after encoding |
|---|---|---|
| 1 | 14 | 22 |
| 2 | 16 | 36 |
| 3 | 17 | 37 |
| 4 | 13 | 21 |
| 5 | 7 | 7 (nothing to encode) |

---

## 11. Separate X and y

Inputs (`X`) and the answer (`y`) are separated before anything is split or scaled, so the target can never leak into the inputs.

```python
y_all = abt[cfg["target"]]            # 1 = rejected / excursion / serious event ; log of units for the forecast
X_all = X_all.copy()                  # inputs only
```

For Objective 5 there is no `y`. The planted-anomaly answer key is held aside and opened once, at evaluation, to score the four models. No model ever trains on it.

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

```python
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.25, stratify=y_all, random_state=42)
```

---

## 13. Feature Scaling of Numerical Columns

The scaler is fitted on the **training rows only**, then applied to the test rows, so nothing from the test set leaks in. Blanks are filled with the training median first. Yes/No columns stay as 0/1 and only genuine numbers are scaled.

```python
fill_values = X_train.median().to_dict()
scale_cols  = [c for c in cfg["numeric"] if X_train[c].nunique() > 2]
scaler      = StandardScaler().fit(X_train[scale_cols])
X_train_s[scale_cols] = scaler.transform(X_train[scale_cols])
X_test_s[scale_cols]  = scaler.transform(X_test[scale_cols])
```

The fill values, scaled columns and scaler are saved with the model in section 19, so new data is prepared in exactly the same way in the app.

---

## 14. Train Model

Four models per objective, tuned with randomised search and cross-validation on the training rows only, plus a "no skill" baseline to show what guessing scores.

| Objective | Models tried | Cross-validation |
|---|---|---|
| 1, 3, 4 | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting | Stratified 5-fold, scored on ROC AUC |
| 2 | Ridge Regression, Decision Tree, Random Forest, Gradient Boosting | Time-ordered 5-fold, scored on error |
| 5 | Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope | Compared on the training invoices |

```python
search = RandomizedSearchCV(estimator, search_space, n_iter=10, scoring=SCORING, refit="roc_auc",
                            cv=StratifiedKFold(5, shuffle=True, random_state=42), n_jobs=-1)
```

**How the winner is picked:** by cross-validated score on the training data. If several models sit within one standard deviation of the best, the simplest wins. The test set is not used to choose, only to confirm.

| Cross-validated ROC AUC | Logistic Regression | Gradient Boosting | Random Forest | Decision Tree | Baseline |
|---|---|---|---|---|---|
| 1 Batch rejection | **0.794** | 0.779 | 0.776 | 0.728 | 0.500 |
| 3 Cold chain | **0.796** | 0.793 | 0.795 | 0.771 | 0.500 |
| 4 Adverse events | **0.883** | 0.875 | 0.861 | 0.837 | 0.500 |

For the forecast, Ridge had the lowest cross-validated error (0.299 on the log scale, against 0.306 for Gradient Boosting). For anomalies, Elliptic Envelope scored a training PR AUC of 0.952 against 0.917 for Isolation Forest.

---

## 15. Predict X_test

Every trained model scores the test rows it has never seen. Classification models give a **risk probability** instead of a hard yes/no, so the business can choose how many cases to act on.

```python
test_proba = {name: model.predict_proba(X_test_s)[:, 1] for name, model in trained_models.items()}
predictions = pd.DataFrame({"batch_id": ids, "actual_rejected": y_test.to_numpy(), **test_proba})
```

A sanity check follows: the average predicted risk of batches that really were rejected should be clearly higher than for batches that were fine. For the forecast, predictions are converted back from logs to units. For anomalies, each detector gives a score, and a vote counts how many detectors flag the same invoice (7 invoices were flagged by all four, 27 by three).

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

<p align="center">
  <img src="images/obj1_features.png" alt="Batch rejection: feature correlation, permutation importance and risk by driver" width="100%">
</p>
<p align="center"><em>Objective 1: what drives batch rejection</em></p>

<p align="center">
  <img src="images/obj3_features.png" alt="Cold chain: feature correlation, permutation importance and risk by driver" width="100%">
</p>
<p align="center"><em>Objective 3: what drives cold-chain excursions</em></p>

These drivers match what the EDA found, which is a useful check that the models learned real patterns and not noise.

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

<p align="center">
  <img src="images/obj1_evaluation.png" alt="Objective 1 evaluation: confusion matrix, ROC curves and gains" width="100%">
</p>
<p align="center"><em>Objective 1: confusion matrix, ROC curves and how quickly each model finds the rejections</em></p>

<p align="center">
  <img src="images/obj4_evaluation.png" alt="Objective 4 evaluation: confusion matrix, ROC curves and gains" width="100%">
</p>
<p align="center"><em>Objective 4: serious adverse events</em></p>

<p align="center">
  <img src="images/obj5_evaluation.png" alt="Objective 5 evaluation: anomaly models against the fixed price rule" width="100%">
</p>
<p align="center"><em>Objective 5: four detectors and a fixed price rule</em></p>

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

<p align="center">
  <img src="images/business_all_objectives_summary.png" alt="Effort against result for all five objectives" width="100%">
</p>
<p align="center"><em>Work created against problems caught, and gain over the best simple alternative</em></p>

<p align="center">
  <img src="images/business_objective1_batch_rejection.png" alt="Business analysis for objective 1" width="100%">
</p>
<p align="center"><em>Objective 1 business analysis: risk tiers, policies, plant hot spots and cost sensitivity</em></p>

The analysis for objectives 2 to 5 follows the same layout and is in the [`images`](images) folder.

---

## 19. Save Model, Scaler and Features

Each winning model is saved as **three `.joblib` files**, enough to score new data without the notebook:

| File | What it holds |
|---|---|
| `model.joblib` | The trained model |
| `scaler.joblib` | The scaler fitted on the training data |
| `features.joblib` | Exact input columns and their order, fill values for blanks, allowed category values, cut-offs and risk tiers, test results and the scikit-learn version |

```python
joblib.dump(model,  f"{out}/model.joblib")
joblib.dump(scaler, f"{out}/scaler.joblib")
joblib.dump(features_info, f"{out}/features.joblib")
```

| Folder | Model |
|---|---|
| `models/batch_rejection/` | Logistic Regression |
| `models/demand_forecast/` | Linear Regression (Ridge) |
| `models/cold_chain/` | Logistic Regression |
| `models/adverse_events/` | Logistic Regression |
| `models/pricing_anomalies/` | Elliptic Envelope |

**Proof that it works:** each objective is reloaded from its three files alone and has to reproduce the notebook's own test predictions. If any one fails, the cell stops with an error, so a broken file cannot be saved without anyone noticing. All five passed.

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

### The Streamlit app

The saved files power the app, [`app.py`](app.py). It has a Home page and one page for each model: **Batch Quality, Demand Planning, Cold-Chain Protection, Drug Safety and Pricing Control.**

- **Single case:** fill in a short form and press Predict. The app returns the risk percentage, a High, Medium or Low tier, a recommended action and the main factors behind the result.
- **Many cases:** download a CSV template, upload your filled file and download the scored results.
- **Pricing Control** combines the model with the fixed price rule, and says why an invoice was flagged.
- The forms ask plain questions. The app works out the model's derived inputs itself, such as heat exposure and months since start.

**Run it yourself**

```bash
pip install streamlit pandas numpy scikit-learn joblib
streamlit run app.py
```

Live app: [Open the Streamlit app](YOUR-STREAMLIT-LINK)

<!-- Add a screenshot: <p align="center"><img src="images/app_screenshot.png" width="100%"></p> -->

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

## 23. Conclusion

I built five models for five business problems, compared at least four algorithms for each, chose the winners on training evidence and confirmed them on unseen data. Logistic Regression solved batch rejection, cold-chain excursions and serious adverse events. Ridge regression handled the demand forecast, and Elliptic Envelope handled pricing anomalies. All five are saved with their scalers and feature definitions, reload and reproduce the notebook's results, and run in a Streamlit app that scores one case or a whole file.

Used as a ranking tool, the models point inspection, protection and review effort at a minority of cases that hold most of the problems. In the test data, that means inspecting under half the batches to catch 86% of rejections, reviewing just over half of the safety reports to reach 99% of serious cases, and cutting forecast stock cost by about 70%. For pricing, a transparent rule beat every model, a reminder to test a simple baseline before choosing a complex one.

**Limits to keep in mind**
- The money figures use example costs, not real ones.
- Results come from a single held-out sample per objective.
- The data is fictional, so the findings demonstrate the method rather than the performance of a real plant.

**Next steps:** replace the example costs with real figures, pilot the tiered policies next to current practice, measure the difference, and retrain on fresh data as the business changes.

---

**Author:** Adewale Oshinukan | Data Scientist | [GitHub](YOUR-GITHUB-LINK) | [Live App](YOUR-STREAMLIT-LINK)
