import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from mrmr import mrmr_regression
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer

st.set_page_config(layout="wide")

# Default conf
default_config = {
    "train_df": None,
    "test_df": None,
    "use_separate_test": False,
    "test_split_pct": 10,
    "feature_list": [],
    "raw_feature_count": 0,
    "selected_target": None,
    "feature_sel_algo_list": ["MRMR", "RFE", "SFS"],
    "approved_feature_sel_algo_list": [],
    "feature_eng_list": ["Year", "Month", "Week of Year", "Quarter of Year", "Target Lag 1", "Target Lag 52"],
    "approved_feature_eng_list": [],
    "feature_lower_limit": 1,
    "feature_upper_limit": 1,
    "xgboost_base": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6, "random_state": 42, "n_jobs": 1},
    "tuning_method": "None",
    "bo": {"n_estimators": (50, 200), "max_depth": (3, 8), "learning_rate": (0.01, 0.30), "subsample": (0.5, 1.0), "n_iter": 10},
    "gs": {"n_estimators": (50, 200), "max_depth": (3, 8), "learning_rate": (0.01, 0.30), "values_per_param": 3},
    "results": None,
    "best_per_algo": None,
    "best_predictions": {},
    "actual_values": None,
    "optimized_comparison": None,
    "dropped_rows": 0,
    "mape_skipped_rows": 0
}

if "config" not in st.session_state:
    st.session_state.config = default_config
else:
    # Ensure all missing default keys exist in existing sessions
    for k, v in default_config.items():
        if k not in st.session_state.config:
            st.session_state.config[k] = v

Config = st.session_state.config

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # MAPE cannot divide by 0
    usable = y_true != 0
    if usable.any():
        mape = float(np.mean(np.abs((y_true[usable] - y_pred[usable]) / y_true[usable])) * 100)
    else:
        mape = np.nan

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE (%)": mape
    }

def format_metric(value):
    return "N/A" if np.isnan(value) else f"{value:.2f}"

def read_uploaded_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def apply_feature_engineering(df, target_col, selected_eng_features):
    df = df.copy()

    # Calendar features
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)

        if "Year" in selected_eng_features:
            df['Year'] = df[date_col].dt.year
        if "Month" in selected_eng_features:
            df['Month'] = df[date_col].dt.month
        if "Week of Year" in selected_eng_features:
            df['Week_of_Year'] = df[date_col].dt.isocalendar().week.astype(int)
        if "Quarter of Year" in selected_eng_features:
            df['Quarter_of_Year'] = df[date_col].dt.quarter

    # Lag features
    if target_col in df.columns:
        if "Target Lag 1" in selected_eng_features:
            df['Target_Lag_1'] = df[target_col].shift(1).fillna(0)
        if "Target Lag 52" in selected_eng_features:
            df['Target_Lag_52'] = df[target_col].shift(52).fillna(0)

    # Replacing NaN values with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df

def optimize_best_model(X_train, y_train, X_test, y_test, active_cols):
    tuning_method = Config["tuning_method"]
    if tuning_method == "None":
        return None

    base_model = xgb.XGBRegressor(random_state=42, n_jobs=1)

    # Only the search object differs
    if tuning_method == "Grid Search":
        gs_params = Config["gs"]
        steps = int(gs_params["values_per_param"])

        # Range to list, np.unique drops the repeats
        n_estimators = np.unique(np.linspace(*gs_params["n_estimators"], steps).astype(int)).tolist()
        max_depth = np.unique(np.linspace(*gs_params["max_depth"], steps).astype(int)).tolist()

        # Log spacing, 0.01->0.02 matters more than 0.29->0.30
        lr_low, lr_high = gs_params["learning_rate"]
        learning_rate = np.logspace(np.log10(lr_low), np.log10(lr_high), steps).tolist()

        search = GridSearchCV(estimator=base_model, cv=3, scoring='neg_mean_absolute_error',
                              param_grid={'n_estimators': n_estimators,
                                          'max_depth': max_depth,
                                          'learning_rate': learning_rate})
    else:
        bo_params = Config["bo"]
        search = BayesSearchCV(estimator=base_model, cv=3, scoring='neg_mean_absolute_error',
                               n_iter=int(bo_params["n_iter"]), random_state=42,
                               search_spaces={'n_estimators': Integer(*bo_params["n_estimators"]),
                                              'max_depth': Integer(*bo_params["max_depth"]),
                                              'learning_rate': Real(*bo_params["learning_rate"], prior='uniform'),
                                              'subsample': Real(*bo_params["subsample"], prior='uniform')})

    # cv=3 splits randomly, TimeSeriesSplit would fit better
    search.fit(X_train[active_cols], y_train)
    predictions = search.best_estimator_.predict(X_test[active_cols])

    return {
        "method": tuning_method,
        "best_params": search.best_params_,
        "predictions": predictions,
        "metrics": calculate_metrics(y_test, predictions)
    }


# Pipeline

def run_pipeline():
    target_col = Config["selected_target"]
    train_df = Config["train_df"]
    test_df = Config["test_df"] if Config["use_separate_test"] else None

    if test_df is not None and target_col not in test_df.columns:
        st.error(f"Test data must contain the target column '{target_col}'!")
        return

    # Join tables for lag features
    if test_df is not None:
        shared_columns = [c for c in train_df.columns if c in test_df.columns]
        data = pd.concat([train_df[shared_columns], test_df[shared_columns]], ignore_index=True)
        data["Is_Test_Row"] = [False] * len(train_df) + [True] * len(test_df)
    else:
        data = train_df.copy()
        data["Is_Test_Row"] = False

    # Drop the target column
    data[target_col] = pd.to_numeric(data[target_col], errors='coerce')
    row_count = len(data)
    data = data.dropna(subset=[target_col]).reset_index(drop=True)
    Config["dropped_rows"] = row_count - len(data)

    # Feature engineering
    data = apply_feature_engineering(data, target_col, Config["approved_feature_eng_list"])

    # Train and test split
    if test_df is None:
        split_point = int(len(data) * (1 - Config["test_split_pct"] / 100.0))
        data.loc[split_point:, "Is_Test_Row"] = True

    train_data = data[~data["Is_Test_Row"]].reset_index(drop=True)
    test_data = data[data["Is_Test_Row"]].reset_index(drop=True)

    date_cols = [c for c in data.columns if 'date' in c.lower() or 'time' in c.lower()]
    drop_cols = [target_col, "Is_Test_Row"] + date_cols

    y_train = train_data[target_col]
    X_train = train_data.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])
    y_test = test_data[target_col]
    X_test = test_data.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number])

    total_features = len(X_train.columns)
    min_features = max(1, min(Config["feature_lower_limit"], total_features))
    max_features = max(min_features, min(Config["feature_upper_limit"], total_features))

    # Base model
    params = Config["xgboost_base"]
    model = xgb.XGBRegressor(n_estimators=int(params["n_estimators"]),
                             learning_rate=float(params["learning_rate"]),
                             max_depth=int(params["max_depth"]),
                             random_state=42, n_jobs=1)

    # Feature ranking
    selected_algos = Config["approved_feature_sel_algo_list"]
    ranked = {}

    if "MRMR" in selected_algos:
        ranked["MRMR"] = mrmr_regression(X=X_train, y=y_train, K=max_features, show_progress=False, n_jobs=1)

    if "RFE" in selected_algos:
        selector = RFE(estimator=model, n_features_to_select=1).fit(X_train, y_train) # Selecting 1 feature gives a full ranking
        ranked["RFE"] = [col for rank, col in sorted(zip(selector.ranking_, X_train.columns))]

    if "SFS" in selected_algos:
        sfs_count = min(max_features, max(1, total_features - 1))
        selector = SequentialFeatureSelector(estimator=model, n_features_to_select=sfs_count, direction='forward', cv=2, n_jobs=1).fit(X_train, y_train)
        ranked["SFS"] = X_train.columns[selector.get_support()].tolist()

    all_results = []
    best_predictions = {}
    best_features = {}
    best_score = {name: float('inf') for name in ranked}

    for k in range(min_features, max_features + 1):
        for name, ranking in ranked.items():
            if k > len(ranking):
                continue

            active_cols = ranking[:k]
            model.fit(X_train[active_cols], y_train)
            predictions = model.predict(X_test[active_cols])

            metrics = calculate_metrics(y_test, predictions)
            all_results.append({"Method": name, "K": k, **metrics})

            # Best model = lowest MAE
            if metrics["MAE"] < best_score[name]:
                best_score[name] = metrics["MAE"]
                best_predictions[name] = predictions
                best_features[name] = active_cols

    # Baseline with every feature
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    best_predictions["All Features"] = predictions
    all_results.append({"Method": "All Features", "K": total_features, **calculate_metrics(y_test, predictions)})

    # Best run per algorithm
    results_df = pd.DataFrame(all_results)
    best_per_algo = results_df.loc[results_df.groupby("Method")["MAE"].idxmin()]
    best_per_algo = best_per_algo.sort_values(by="MAE").reset_index(drop=True)

    # Tune the winner
    winner = best_per_algo.iloc[0]["Method"]
    winner_columns = best_features.get(winner, X_train.columns.tolist())
    optimized_result = optimize_best_model(X_train, y_train, X_test, y_test, winner_columns)

    # Results section
    Config["results"] = results_df.round(2)
    Config["best_per_algo"] = best_per_algo.round(2)
    Config["best_predictions"] = best_predictions
    Config["actual_values"] = y_test.values  # charts compare the predictions to these
    Config["optimized_comparison"] = optimized_result
    Config["mape_skipped_rows"] = int(np.sum(y_test == 0))

# Interface

def load_data():
    # UI elements
    st.header("Load Data")
    data_choice = st.radio("Please make your data choice",
                           options=["Only train data", "Train and test data"],
                           horizontal=True)
    train_file = st.file_uploader("Upload Train Data", type=["csv", "xls", "xlsx"], accept_multiple_files=False)
    if data_choice == "Train and test data":
        test_file = st.file_uploader("Upload Test Data", type=["csv", "xls", "xlsx"], accept_multiple_files=False)

    # Data operations
    Config["use_separate_test"] = data_choice == "Train and test data"

    if train_file:
        Config["train_df"] = read_uploaded_file(train_file)
        Config["feature_list"] = list(Config["train_df"].columns)
        Config["raw_feature_count"] = len(Config["feature_list"])
        Config["selected_target"] = st.selectbox("Select Target Variable", Config["feature_list"])

    if data_choice == "Only train data" and train_file:
        Config["test_split_pct"] = st.slider("Test data split ratio (%)",
                                             min_value=5, max_value=40, value=10, step=5)
    elif data_choice == "Train and test data" and test_file:
        Config["test_df"] = read_uploaded_file(test_file)

def feature_selection():
    fe_list = Config["feature_eng_list"]
    fsa_list = Config["feature_sel_algo_list"]

    st.header("Feature Selection & Engineering")

    Config["approved_feature_eng_list"] = st.multiselect("Feature Engineering", fe_list, default=fe_list)

    total_fc = max(1, Config["raw_feature_count"] + len(Config["approved_feature_eng_list"]) - 1) #Date
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("Estimated feature count: " + str(total_fc))
    with col2:
        Config["feature_lower_limit"] = st.number_input("Min Features", min_value=1, max_value=total_fc, value=1)
    with col3:
        Config["feature_upper_limit"] = st.number_input("Max Features", min_value=1, max_value=total_fc, value=total_fc, key=f"max_features_{total_fc}")

    Config["approved_feature_sel_algo_list"] = st.multiselect("Feature Selection Algorithms", fsa_list, default=fsa_list)

def model_tuning_settings():
    st.header("XGBoost Hyperparameters")

    xgb_params = Config["xgboost_base"]
    col1, col2, col3 = st.columns(3)
    with col1:
        xgb_params["n_estimators"] = st.number_input("n_estimators", value=xgb_params["n_estimators"], step=10)
    with col2:
        xgb_params["learning_rate"] = st.number_input("learning_rate", value=xgb_params["learning_rate"], step=0.01)
    with col3:
        xgb_params["max_depth"] = st.number_input("max_depth", value=xgb_params["max_depth"], step=1)

    Config["tuning_method"] = st.radio("Final Optimization Method", ["None", "Grid Search", "Bayesian Optimization"], horizontal=True)

    if Config["tuning_method"] == "Grid Search":
        gs = Config["gs"]
        col1, col2 = st.columns(2)
        with col1:
            gs["n_estimators"] = st.slider("n_estimators range", 10, 500, gs["n_estimators"])
            gs["max_depth"] = st.slider("max_depth range", 1, 15, gs["max_depth"])
        with col2:
            gs["learning_rate"] = st.slider("learning_rate range", 0.001, 0.5, gs["learning_rate"])
            gs["values_per_param"] = st.number_input("Values per parameter", value=gs["values_per_param"], min_value=2, max_value=6, step=1)
        # Every combination is tried, cost grows fast
        st.info(f"Up to {gs['values_per_param'] ** 3} combinations x 3 folds = {gs['values_per_param'] ** 3 * 3} model fits")

    elif Config["tuning_method"] == "Bayesian Optimization":
        bo = Config["bo"]
        col1, col2 = st.columns(2)
        with col1:
            bo["n_estimators"] = st.slider("n_estimators range", 10, 500, bo["n_estimators"])
            bo["max_depth"] = st.slider("max_depth range", 1, 15, bo["max_depth"])
        with col2:
            bo["learning_rate"] = st.slider("learning_rate range", 0.001, 0.5, bo["learning_rate"])
            bo["subsample"] = st.slider("subsample range", 0.1, 1.0, bo["subsample"])
            bo["n_iter"] = st.number_input("Search Iterations", value=bo["n_iter"], min_value=1, step=5)

def results_view():
    st.header("Results")
    Config["show_table"] = st.checkbox("Show results table", value=True)
    Config["show_individual_charts"] = st.checkbox("Show individual charts per algorithm", value=False)

st.title("Financial Forecasting Pipeline")
col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        load_data()
    with st.container(border=True):
        model_tuning_settings()

with col2:
    with st.container(border=True):
        feature_selection()
    with st.container(border=True):
        results_view()
        if st.button("Run Pipeline", type="primary", use_container_width=True):
            if Config["train_df"] is None:
                st.error("Please upload train data first!")
            else:
                with st.spinner("Pipeline started..."):
                    run_pipeline()
                    st.success("Execution Completed!")


# Results
if Config["results"] is not None:
    st.header("Model Results")

    if Config.get("show_table"):
        st.subheader("Best Performance Per Algorithm")
        st.dataframe(Config["best_per_algo"], use_container_width=True)

        if Config["dropped_rows"] or Config["mape_skipped_rows"]:
            st.caption(f"{Config['dropped_rows']} row(s) dropped (no target value), {Config['mape_skipped_rows']} test row(s) skipped by MAPE (zero actual value).")

        opt_info = Config.get("optimized_comparison")
        if opt_info:
            st.subheader("Final Model Optimization: " + opt_info['method'])
            c1, c2, c3 = st.columns(3)
            c1.metric("Optimized RMSE", format_metric(opt_info["metrics"]["RMSE"]))
            c2.metric("Optimized MAE", format_metric(opt_info["metrics"]["MAE"]))
            c3.metric("Optimized MAPE (%)", format_metric(opt_info["metrics"]["MAPE (%)"]))
            st.json(opt_info["best_params"])

        with st.expander("Show All Iterations"):
            st.dataframe(Config["results"], use_container_width=True)

    preds_dict = Config["best_predictions"]
    opt_info = Config.get("optimized_comparison")
    if opt_info:
        preds_dict[f"Optimized ({opt_info['method']})"] = opt_info["predictions"]

    actual_values = Config["actual_values"]

    col1, col2, col3 = st.columns([1,4,1])

    with col2:
        if Config.get("show_individual_charts"):
            st.subheader("Individual Prediction Charts")
            for m_name, p_vals in preds_dict.items():
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(actual_values, label="Actual", color='black', marker='o')
                ax.plot(p_vals, label=m_name, color='tab:blue', linestyle='--', marker='s')
                ax.set_title(f"Algorithm: {m_name}")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.subheader("Combined Prediction Comparison")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(actual_values, label="Actual", color='black', linewidth=2.5)
            for m_name, p_vals in preds_dict.items():
                ax.plot(p_vals, label=m_name, linestyle='--')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)