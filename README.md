# Walmart Sales Forcasting

Weekly sales forecasting with XGBoost. The repository holds two things:

- `src/data_optimized.ipynb` - the original notebook, written for the Walmart dataset
- `src/general_pipeline.py` - a Streamlit app that runs the same pipeline on any uploaded dataset

The notebook is the reference. The app repeats its method without hardcoding any column
name, so it can be pointed at a different dataset.

## Starting Guide
### 1. Clone the repo
```bash
git clone <repo-url>
cd <repo-folder>
```
---
### 2. Create Venv and Activate
#### Linux and MacOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```
#### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
```
---
### 3. Install Libraries
```bash
pip install -r requirements.txt
```
---
### 4. Run the App
```bash
streamlit run src/general_pipeline.py
```
Upload `data/weekly.csv` and pick `Weekly_Sales` as the target to try it out.

---
**Dataset Link:** https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast

## Files

| Path | What it is |
| --- | --- |
| `data/train.csv`, `stores.csv`, `features.csv` | Raw Kaggle files |
| `data/weekly.csv` | Store 1 only, all departments collapsed into one row per week. Built from the three raw files with the notebook's aggregation rules. Ready to upload to the app. |
| `src/data_optimized.ipynb` | The notebook the app is based on |
| `src/general_pipeline.py` | The Streamlit app |

## How the App Works

Upload a CSV or Excel file, pick the target column, press Run Pipeline. Everything below
happens inside `run_pipeline()`, which is split into commented steps in that order.

1. **Join the tables.** With a separate test file, train and test are concatenated first.
2. **Drop rows without a target value.**
3. **Feature engineering.** Optional columns the user ticks: Year, Month, Week of Year,
   Quarter of Year, Target Lag 1, Target Lag 52. Calendar features need a date column;
   lag features are past values of the target.
4. **Train / test split.** Either the last N% of the rows, or the uploaded test file.
5. **Feature ranking.** MRMR, RFE and SFS each return an ordered list of columns.
6. **One model per (algorithm, feature count).** For every algorithm, every feature count
   between Min and Max Features is trained and scored.
7. **Baseline.** One more model using every feature, so feature selection has something
   to be compared against.
8. **Best run per algorithm**, ranked by MAE.
9. **Hyperparameter tuning** of the overall winner: Grid Search, Bayesian Optimization,
   or none.

Results are a table (RMSE, MAE, MAPE per algorithm) and charts of the predictions against
the actual values.

## Decisions

### MAE decides which model is best

MAPE was the original ranking metric and it produced meaningless numbers. `Weekly_Sales`
in `data/train.csv` contains 73 exact zeros and 770 rows below 1 in absolute value, and a
percentage error divides by the actual value. scikit-learn guards that division with an
epsilon of about 2.2e-16 rather than skipping the row, so a single zero actual is enough
to blow the result up: in a four row test, 29% became 5.6e+18%.

MAE was chosen instead: it cannot divide by zero and it is in the target's own unit, so
the number means something to read. RMSE and MAPE are still shown in the table, they just
do not decide the ranking. The tuning step uses `neg_mean_absolute_error` so it optimises
the same thing the ranking uses.

### MAPE skips rows whose actual value is zero

MAPE is still reported, but rows with a zero actual are left out of it, and the app says
how many were skipped. Without this the metric is not just wrong, it is unreadable.

### Rows with a missing target are dropped, not filled with zero

The earlier version filled missing targets with 0. That invents data points and creates
exactly the zero denominators that broke MAPE. They are dropped now and the app reports
the count.

### Train and test files are joined before feature engineering

When two files are uploaded, they are concatenated before the lag features are built.
Otherwise `Target_Lag_1` restarts at 0 on the first test row and `Target_Lag_52` is 0 for
the whole test set, which makes the predictions worthless.

### Grid Search takes ranges, and learning rate is spaced logarithmically

Grid Search now takes range sliders like Bayesian Optimization does, plus a
"Values per parameter" input. The two searches are not the same kind of thing, so the
extra input is needed:

- Bayesian has a budget (`n_iter`). Widening the range does not cost anything.
- Grid Search is exhaustive. A range means nothing to it until it is cut into a list of
  values, and the cost is the product of the list lengths.

At 3 values per parameter the cost is 27 combinations, the same as the old hardcoded
lists. At 5 it is 125. The app prints the combination count so the cost is visible before
running.

`n_estimators` and `max_depth` are cut linearly. `learning_rate` is cut logarithmically,
because it is a multiplicative parameter - going from 0.01 to 0.02 changes the model far
more than going from 0.29 to 0.30 does, even though the distance is smaller. Cutting the
same 0.01-0.30 range five ways:

```
linear: 0.010  0.082  0.155  0.227  0.300     equal distances, 4 of 5 above 0.08
log   : 0.010  0.023  0.055  0.128  0.300     equal ratios, each 2.34x the previous
```

Linear spacing spends most of its budget in a region where the model barely changes.

### Charts compare against the actual values

Every prediction chart draws the real test values as a black line. Plotting the algorithms
alone shows nothing about whether they are right.

## Known Limitations

- The best model is selected on the test set, so the test set doubles as a validation set.
  The reported scores are optimistic.
- `GridSearchCV` and `BayesSearchCV` use `cv=3`, which splits time-ordered rows randomly.
  `TimeSeriesSplit` would be the correct choice.
- SFS returns an unordered subset of columns rather than a ranking, so slicing it to K
  features takes the first K in column order, not the K strongest.

All three match the notebook the project is based on and were left as they are.
