
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_wine,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# --------------------------------------------------
# PERSISTENT EXPERIMENT STORAGE
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "neurolab.db"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def init_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                dataset_name TEXT NOT NULL,
                target TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                ranking_metric TEXT NOT NULL,
                ranking_score REAL NOT NULL,
                metrics_json TEXT NOT NULL,
                random_seed INTEGER NOT NULL
            )
        """)


def save_experiment(result):
    leaderboard = result["leaderboard"]
    best_row = leaderboard.iloc[0].to_dict()

    # Store numeric metrics from the winning leaderboard row.
    metrics = {
        key: float(value)
        for key, value in best_row.items()
        if key not in ("Model", "Rank")
        and isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(value)
    }

    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute("""
            INSERT INTO experiments (
                created_at, dataset_name, target, problem_type,
                algorithm, ranking_metric, ranking_score,
                metrics_json, random_seed
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            str(result["dataset_name"]),
            str(result["target"]),
            str(result["problem_type"]),
            str(result["algorithm"]),
            str(result["ranking_metric"]),
            float(result["leaderboard"]["Ranking Score"].iloc[0]),
            json.dumps(metrics),
            int(result["seed"]),
        ))

        return cursor.lastrowid


def get_experiment_history():
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query("""
            SELECT id, created_at, dataset_name, target,
                   problem_type, algorithm, ranking_metric,
                   ranking_score, metrics_json, random_seed
            FROM experiments
            ORDER BY id DESC
        """, connection)


init_database()


st.set_page_config(
    page_title="NeuroLab AI",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeuroLab AI")
st.caption("Machine Learning Experimentation Studio")
st.divider()

# --------------------------------------------------
# SIDEBAR: DATA SOURCE AND EXPERIMENT SETTINGS
# --------------------------------------------------


with st.sidebar:
    st.title("🧠 NeuroLab AI")
    st.caption("ML Experimentation Studio")
    st.divider()

    st.header("Dataset Configuration")

    source = st.radio(
        "Data source",
        ["Built-in dataset", "Upload CSV"],
    )

    uploaded_file = None
    dataset_name = None
    builtin_data = None

    if source == "Built-in dataset":
        datasets = {
            "Breast Cancer": load_breast_cancer,
            "Iris": load_iris,
            "Wine": load_wine,
        }

        dataset_name = st.selectbox(
            "Choose dataset",
            list(datasets.keys()),
        )

        data = datasets[dataset_name](as_frame=True)
        builtin_data = data.frame.copy()
        builtin_data["target"] = data.target

    else:
        uploaded_file = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
        )

    st.divider()
    st.header("Experiment Settings")

    problem_type = st.selectbox(
        "Problem type",
        ["Classification", "Regression"],
    )

    if problem_type == "Classification":
        ranking_metric = st.selectbox(
            "Rank models by",
            ["F1 Score", "Accuracy", "Precision", "Recall"],
        )
    else:
        ranking_metric = st.selectbox(
            "Rank models by",
            ["RMSE", "MAE", "R² Score"],
        )

    test_percent = st.slider(
        "Test set (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42,
    )

    submitted = st.button(
        "🚀 Run AutoML",
        type="primary",
        use_container_width=True,
    )


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = None

if source == "Built-in dataset":
    df = builtin_data

elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("The uploaded CSV has no data rows.")
            st.stop()
        if len(df.columns) < 2:
            st.error("The dataset needs at least two columns.")
            st.stop()
        if len(df) > 200_000 or len(df.columns) > 300:
            st.error("For this MVP, CSV files are limited to 200,000 rows and 300 columns.")
            st.stop()
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        st.stop()

if df is None:
    st.info(
        "Choose a built-in dataset or upload a CSV file "
        "from the sidebar to begin."
    )
    st.stop()

if not df.columns.is_unique:
    st.error("Your dataset contains duplicate column names. Rename them and upload again.")
    st.stop()

# --------------------------------------------------
# DATASET INTELLIGENCE
# --------------------------------------------------

st.subheader("Dataset Intelligence")

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
categorical_columns = df.select_dtypes(
    include=["object", "category", "bool", "string"]
).columns.tolist()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns):,}")
c3.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

tab1, tab2, tab3 = st.tabs(
    ["Data Preview", "Data Quality", "Statistics"]
)

with tab1:
    st.dataframe(df.head(100), use_container_width=True)
    st.caption("Preview limited to the first 100 rows.")

with tab2:
    quality = pd.DataFrame({
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isna().sum(),
        "Missing (%)": (df.isna().mean() * 100).round(2),
        "Unique Values": df.nunique(dropna=True),
    })

    st.dataframe(quality, use_container_width=True)

    if df.isna().sum().sum() == 0:
        st.success("No missing cells detected.")
    else:
        st.warning("Missing values will be imputed during model training.")

with tab3:
    st.dataframe(
        df.describe(include="all").transpose(),
        use_container_width=True,
    )

# --------------------------------------------------
# TARGET AND PROBLEM CONFIGURATION
# --------------------------------------------------

st.divider()
st.subheader("Configure Your ML Experiment")

with st.form("experiment_form"):
    target = st.selectbox(
        "Target column — what should the model predict?",
        options=list(df.columns),
        index=(
            list(df.columns).index("target")
            if "target" in df.columns
            else len(df.columns) - 1
        ),
    )

    problem_type = st.selectbox(
        "Problem type",
        ["Classification", "Regression"],
        help=(
            "Classification predicts categories. "
            "Regression predicts numeric values."
        ),
    )

    
    st.info("AutoML will compare all supported algorithms.")

    if problem_type == "Classification":
        ranking_metric = st.selectbox(
            "Rank models by",
            ["F1 Score", "Accuracy", "Precision", "Recall"],
        )
    else:
        ranking_metric = st.selectbox(
            "Rank models by",
            ["R² Score", "MAE", "RMSE"],
            help="For MAE and RMSE, lower scores are better.",
        )


    test_percent = st.slider(
        "Test set (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=42,
    )

    submitted = st.form_submit_button(
        "Train and Evaluate Model",
        type="primary",
        use_container_width=True,
    )

# --------------------------------------------------
# TRAINING AND EVALUATION
# --------------------------------------------------


if submitted:
    import time

    try:
        X = df.drop(columns=[target]).copy()
        y = df[target].copy()

        valid_target = y.notna()
        X = X.loc[valid_target].reset_index(drop=True)
        y = y.loc[valid_target].reset_index(drop=True)

        if problem_type == "Regression":
            y = pd.to_numeric(y, errors="coerce")
            valid_numeric = y.notna()
            X = X.loc[valid_numeric].reset_index(drop=True)
            y = y.loc[valid_numeric].reset_index(drop=True)

        if len(X) < 10 or y.nunique() < 2:
            st.error("Insufficient usable rows or target classes.")
            st.stop()

        if problem_type == "Classification" and y.nunique() > 100:
            st.error("Too many classes for this AutoML MVP.")
            st.stop()

        # Remove features containing no useful information.
        unusable = [
            col for col in X.columns
            if X[col].isna().all()
            or X[col].nunique(dropna=True) <= 1
        ]
        X = X.drop(columns=unusable)

        numeric_features = X.select_dtypes(
            include=np.number
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()

        if not numeric_features and not categorical_features:
            st.error("No supported feature columns remain.")
            st.stop()

        transformers = []

        if numeric_features:
            transformers.append((
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ))

        if categorical_features:
            transformers.append((
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ))

        preprocessor = ColumnTransformer(transformers)

        if problem_type == "Classification":
            models = {
                "Logistic Regression": LogisticRegression(max_iter=2000),
                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    random_state=int(seed),
                    n_jobs=-1,
                ),
                "Support Vector Machine": SVC(),
            }
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(
                    n_estimators=200,
                    random_state=int(seed),
                    n_jobs=-1,
                ),
                "Support Vector Regressor": SVR(),
            }

        stratify = None

        if problem_type == "Classification":
            counts = y.value_counts()
            test_rows = int(np.ceil(len(y) * test_percent / 100))
            train_rows = len(y) - test_rows

            if (
                counts.min() >= 2
                and test_rows >= y.nunique()
                and train_rows >= y.nunique()
            ):
                stratify = y
            else:
                st.warning(
                    "Some classes are too rare for stratified splitting."
                )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_percent / 100,
            random_state=int(seed),
            stratify=stratify,
        )

        leaderboard = []
        successful_models = {}
        failures = []

        progress = st.progress(0, text="Starting AutoML...")

        for index, (name, estimator) in enumerate(models.items()):
            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", estimator),
            ])

            try:
                started = time.perf_counter()
                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)
                elapsed = time.perf_counter() - started

                if problem_type == "Classification":
                    accuracy = accuracy_score(y_test, predictions)
                    precision = precision_score(
                        y_test, predictions,
                        average="weighted", zero_division=0,
                    )
                    recall = recall_score(
                        y_test, predictions,
                        average="weighted", zero_division=0,
                    )
                    f1 = f1_score(
                        y_test, predictions,
                        average="weighted", zero_division=0,
                    )

                    row = {
                        "Model": name,
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "Training + Prediction (s)": elapsed,
                    }

                    score = {
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                    }[ranking_metric]

                else:
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = float(np.sqrt(
                        mean_squared_error(y_test, predictions)
                    ))
                    r2 = r2_score(y_test, predictions)

                    row = {
                        "Model": name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "R² Score": r2,
                        "Training + Prediction (s)": elapsed,
                    }

                    score = {
                        "MAE": mae,
                        "RMSE": rmse,
                        "R² Score": r2,
                    }[ranking_metric]

                row["Ranking Score"] = score
                leaderboard.append(row)

                successful_models[name] = {
                    "pipeline": pipeline,
                    "predictions": predictions,
                    "score": score,
                }

            except Exception as model_error:
                failures.append(f"{name}: {model_error}")

            progress.progress(
                (index + 1) / len(models),
                text=f"Evaluated {index + 1}/{len(models)} models",
            )

        progress.empty()

        if not successful_models:
            st.error("All models failed. Check your data and target column.")
            for failure in failures:
                st.write(failure)
            st.stop()

        # Higher is better for accuracy, F1, precision, recall and R².
        # Lower is better for MAE and RMSE.
        lower_is_better = ranking_metric in ["MAE", "RMSE"]

        best_name = (
            min if lower_is_better else max
        )(
            successful_models,
            key=lambda name: successful_models[name]["score"],
        )

        best = successful_models[best_name]

        leaderboard_df = pd.DataFrame(leaderboard)
        leaderboard_df = leaderboard_df.sort_values(
            "Ranking Score",
            ascending=lower_is_better,
        ).reset_index(drop=True)

        leaderboard_df.insert(
            0, "Rank", range(1, len(leaderboard_df) + 1)
        )

        st.session_state["custom_experiment"] = {
            "pipeline": best["pipeline"],
            "dataset_name": (
                uploaded_file.name if uploaded_file else dataset_name
            ),
            "target": target,
            "problem_type": problem_type,
            "algorithm": best_name,
            "feature_names": X.columns.tolist(),
            "X_test": X_test,
            "y_test": y_test,
            "predictions": best["predictions"],
            "seed": int(seed),
            "leaderboard": leaderboard_df,
            "ranking_metric": ranking_metric,
            "failures": failures,
        }


        
        # Save this successful run permanently.
        saved_result = st.session_state["custom_experiment"]

        experiment_id = save_experiment(saved_result)
        saved_result["experiment_id"] = experiment_id

        st.success(
            f"Experiment #{experiment_id} saved to SQLite."
        )


    except Exception as exc:
        st.error(f"AutoML failed: {exc}")
    

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

result = st.session_state.get("custom_experiment")


if result:
    st.divider()
    st.subheader("AutoML Leaderboard")

    st.write(
        f"Models ranked by **{result['ranking_metric']}**"
    )

    st.dataframe(
        result["leaderboard"].round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        f"Best model: {result['algorithm']}"
    )

    for failure in result["failures"]:
        st.warning(f"Model skipped: {failure}")

    st.subheader("Winning Model Evaluation")

    
    st.subheader("Model Registry")

    model_filename = (
        f"neurolab_model_{result['experiment_id']}.joblib"
        if result.get("experiment_id")
        else "neurolab_model.joblib"
    )

    model_path = MODEL_DIR / model_filename

    model_bytes = joblib.dumps(result["pipeline"]) if hasattr(joblib, "dumps") else None

    # Serialize the fitted pipeline for download.
    import io

    buffer = io.BytesIO()
    joblib.dump(result["pipeline"], buffer)
    model_bytes = buffer.getvalue()

    st.download_button(
        "Export Winning Model",
        data=model_bytes,
        file_name=model_filename,
        mime="application/octet-stream",
    )



    st.write(
        f"**Dataset:** {result['dataset_name']}  \n"
        f"**Target:** {result['target']}  \n"
        f"**Algorithm:** {result['algorithm']}"
    )

    y_test = result["y_test"]
    predictions = result["predictions"]

    if result["problem_type"] == "Classification":
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions,
            average="weighted", zero_division=0,
        )
        recall = recall_score(
            y_test, predictions,
            average="weighted", zero_division=0,
        )
        f1 = f1_score(
            y_test, predictions,
            average="weighted", zero_division=0,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{accuracy:.2%}")
        m2.metric("Precision", f"{precision:.2%}")
        m3.metric("Recall", f"{recall:.2%}")
        m4.metric("F1 Score", f"{f1:.2%}")

        labels = sorted(pd.unique(y_test).tolist())
        label_names = [str(label) for label in labels]

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        )

        left, right = st.columns(2)

        with left:
            st.plotly_chart(
                px.imshow(
                    cm,
                    x=label_names,
                    y=label_names,
                    text_auto=True,
                    labels={
                        "x": "Predicted",
                        "y": "Actual",
                        "color": "Count",
                    },
                    title="Confusion Matrix",
                    color_continuous_scale="Blues",
                ),
                use_container_width=True,
            )

        with right:
            report = classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(
                pd.DataFrame(report).transpose().round(3),
                use_container_width=True,
            )

    else:
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.4f}")
        m2.metric("RMSE", f"{rmse:.4f}")
        m3.metric("R² Score", f"{r2:.4f}")

        plot_data = pd.DataFrame({
            "Actual": np.asarray(y_test),
            "Predicted": predictions,
        })

        st.plotly_chart(
            px.scatter(
                plot_data,
                x="Actual",
                y="Predicted",
                title="Actual vs Predicted Values",
            ),
            use_container_width=True,
        )

    # Download predictions for independent inspection.
    export = result["X_test"].copy()
    export["Actual"] = np.asarray(y_test)
    export["Predicted"] = predictions

    st.download_button(
        "Download Predictions CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="neurolab_predictions.csv",
        mime="text/csv",
    )

    st.success("Experiment complete. Results are based on held-out test data.")

st.divider()
st.caption("NeuroLab AI | Python · Streamlit · scikit-learn")

# --------------------------------------------------
# EXPLAINABLE AI: PERMUTATION FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()
st.subheader("🔍 Explainable AI")
st.write(
    "Identify which input features most influence "
    "your model's evaluation score."
)

with st.expander("Analyze feature importance", expanded=True):
    if st.button("Calculate Feature Importance"):
        try:
            with st.spinner("Analyzing model features..."):
                scoring = (
                    "f1_weighted"
                    if result["problem_type"] == "Classification"
                    else "r2"
                )

                importance_result = permutation_importance(
                    result["pipeline"],
                    result["X_test"],
                    result["y_test"],
                    scoring=scoring,
                    n_repeats=5,
                    random_state=42,
                    n_jobs=-1,
                )

                importance_df = pd.DataFrame({
                    "Feature": result["X_test"].columns,
                    "Importance": importance_result.importances_mean,
                    "Std": importance_result.importances_std,
                }).sort_values(
                    "Importance", ascending=False
                ).reset_index(drop=True)

                st.session_state["feature_importance"] = {
                    "experiment_id": result.get("experiment_id"),
                    "data": importance_df,
                }

        except Exception as exc:
            st.error(f"Feature importance failed: {exc}")

    importance_state = st.session_state.get("feature_importance")

    if importance_state is not None:
        if importance_state["experiment_id"] != result.get("experiment_id"):
            st.info("Calculate feature importance for this experiment.")
        else:
            importance_df = importance_state["data"]

            st.caption(
                "Positive importance means shuffling a feature "
                "reduced the selected evaluation score. Values near "
                "zero suggest little measured contribution."
            )

            chart = px.bar(
                importance_df.sort_values(
                    "Importance", ascending=True
                ),
                x="Importance",
                y="Feature",
                error_x="Std",
                orientation="h",
                title="Permutation Feature Importance",
                labels={
                    "Importance": "Mean score decrease",
                    "Feature": "Input feature",
                },
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )

            st.dataframe(
                importance_df.round(5),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download Feature Importance CSV",
                data=importance_df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="neurolab_feature_importance.csv",
                mime="text/csv",
            )


# --------------------------------------------------
# EXPERIMENT HISTORY
# --------------------------------------------------

st.divider()
st.subheader("📚 Experiment History")

history = get_experiment_history()

if history.empty:
    st.info("No saved experiments yet. Run AutoML to create your first record.")
else:
    display_history = history.drop(
        columns=["metrics_json"]
    ).copy()

    display_history["created_at"] = pd.to_datetime(
        display_history["created_at"], utc=True
    ).dt.strftime("%Y-%m-%d %H:%M UTC")

    st.dataframe(
        display_history,
        use_container_width=True,
        hide_index=True,
    )

    history_csv = history.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Experiment History",
        data=history_csv,
        file_name="neurolab_experiment_history.csv",
        mime="text/csv",
    )
