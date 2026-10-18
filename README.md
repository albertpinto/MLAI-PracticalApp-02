# What Drives the Price of a Used Car?

**UC Berkeley ML/AI — Practical Application II**

This project analyzes about 426,000 used-car listings (a Craigslist subset from Kaggle) to find out what makes a used car more or less expensive, and turns the answer into advice for a used-car dealership.

📓 **Notebook:** [`used_car_price_analysis.ipynb`](used_car_price_analysis.ipynb)

---

## The Question

A dealership wants to know what consumers value in a used car so it can decide what to stock and how to price it. This is a **supervised regression** problem: predict `price` from vehicle attributes, then explain which attributes matter and in which direction.

The work follows **CRISP-DM**: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.

---

## How the Data Was Prepared

- **Removed clearly wrong rows with fixed rules** (price outside $500–$150,000, model year before 1990, mileage of 0 or over 500,000). The notebook logs how many rows each rule removes.
- **Kept rows with missing values.** Missing fields are filled *inside the modeling pipeline* (median for numbers, an explicit `"unknown"` category for text), so the dataset is not shrunk by dropping every incomplete row.
- **Dropped** `id`, `VIN` (unique identifiers) and `size` (~72% missing).
- **Engineered** `vehicle_age` (newest year in the data minus model year) and a numeric `cylinders`.
- **Split 80/20 before learning anything** from the data, so imputation values, scaling and encodings come from the training set only and the test score is honest.
- **High-cardinality columns** (`model`, `region`) use cross-fitted target encoding, so location and model name are kept instead of dropped.
- **Price is modeled on a log scale** and converted back to dollars.

---

## Models

| Model | Purpose |
|---|---|
| Baseline (median price) | The bar every model must beat |
| Linear Regression | Simple and interpretable |
| Ridge | Linear with shrinkage; `alpha` tuned with cross-validation |
| Lasso | Linear with feature selection; `alpha` tuned with cross-validation |
| Gradient Boosting | Non-linear; captures interactions; tuned with cross-validation |

Metrics are R², MAE and RMSE on a held-out test set, plus 5-fold cross-validated R² and the train/test gap to check for overfitting.

---

## Results

All scores are on a held-out 20% test set (~74,000 listings).

| Model | Test R² | Test MAE |
|---|---|---|
| Baseline (median price) | -0.05 | $11,211 |
| Linear Regression | 0.710 | $4,559 |
| Ridge | 0.710 | $4,559 |
| Lasso | 0.712 | $4,554 |
| **Gradient Boosting** | **0.833** | **$3,255** |

Cross-validated R² for Gradient Boosting is 0.836 (test 0.833), with a train/test gap of about 0.03, so the result is stable with mild overfitting.

**What drives price (permutation importance, MAE increase when shuffled):**

1. Make and model (+$3,457)
2. Vehicle age (+$3,321)
3. Odometer (+$1,939)
4. Cylinders, manufacturer, condition, fuel, drive, type (about $400–$650 each)
5. Paint color (+$83) and transmission (+$78) barely matter

**Direction of effects:** median price falls from about $36K (0–2 years old) to about $5K (18+ years), and from about $31K (under 20K miles) to about $5.6K (over 200K miles). Salvage and fair condition sell at roughly 40% discounts; excellent, new and like-new at 24–29% premiums; pickups and trucks have the highest medians.

**Recommendations for the dealership** are in the notebook's client report (section 6).

---|---|---|
| Baseline | | |
| Linear Regression | | |
| Ridge | | |
| Lasso | | |
| Gradient Boosting | | |

**Top price drivers (permutation importance):** *to be filled in*

**Recommendations for the dealership:** *to be filled in*

---

## How to Run

```bash
uv sync
# put vehicles.csv in data/ (see data/README.md)
uv run jupyter lab used_car_price_analysis.ipynb
```

Requires Python 3.12+ and the packages in `pyproject.toml`.
