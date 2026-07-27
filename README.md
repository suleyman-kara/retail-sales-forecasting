# Walmart Sales Forcasting

Weekly sales forecasting with XGBoost.

- `src/data_optimized.ipynb` - the original notebook, written for the Walmart dataset
- `src/general_pipeline.py` - a Streamlit app that runs the same pipeline on any uploaded dataset

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
| `data/weekly.csv` | Store 1, all departments collapsed into one row per week |
| `src/data_optimized.ipynb` | The notebook the app is based on |
| `src/general_pipeline.py` | The Streamlit app |

## How the App Works

Upload a file, pick the target column, press Run Pipeline. The steps below are the
commented sections of `run_pipeline()`, in order.

1. Join train and test, so lag features do not restart in the test set
2. Drop rows without a target value
3. Feature engineering: Year, Month, Week of Year, Quarter of Year, Target Lag 1, Target Lag 52
4. Train / test split
5. Feature ranking with MRMR, RFE and SFS
6. One model per algorithm and feature count
7. A baseline using every feature
8. Best run per algorithm, ranked by MAE
9. Optional tuning of the winner with Grid Search or Bayesian Optimization

Results are a table of RMSE, MAE and MAPE, plus charts against the actual values.

## Decisions

| Decision | Reason |
| --- | --- |
| MAE ranks the models, not MAPE | The app accepts any dataset and MAPE is undefined when an actual value is 0. |
| MAPE skips rows whose actual value is 0 | scikit-learn divides by an epsilon instead of skipping, which turned a 29% error into 5.6e+18%. |
| Rows with a missing target are dropped | Filling them with 0 invents data and creates the zero denominators that break MAPE. |
| Train and test files are joined before feature engineering | Otherwise `Target_Lag_1` restarts at 0 on the first test row and `Target_Lag_52` is 0 for the whole test set. |
| Grid Search takes ranges plus a "values per parameter" input | A range means nothing to an exhaustive search until it is cut into a list, and the cost is the product of the list lengths. |
| Grid Search cuts `learning_rate` logarithmically | It is multiplicative: 0.01 to 0.02 changes the model more than 0.29 to 0.30 does, so equal distances waste most of the budget. |
| Grid Search and Bayesian use the same scoring | Different searches judged by the same metric can be compared; it also matches the metric the table is ranked by. |
| Charts draw the actual values | Prediction lines alone say nothing about whether they are right. |

The notebook ranks by MAPE instead, which is safe there because the weekly totals never
reach 0. On this dataset both metrics pick the same models.

## Known Limitations

- The best model is selected on the test set, so the reported scores are optimistic.
- `cv=3` splits time-ordered rows randomly; `TimeSeriesSplit` would be correct.
- SFS returns an unordered subset, so slicing it to K takes the first K in column order.

All three match the notebook and were left as they are.
