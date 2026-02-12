
import duckdb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


CSV_PATH = r"rideshare_kaggle.csv"  


def build_clean_table(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    con.execute(f"""
        CREATE OR REPLACE TABLE rides_raw AS
        SELECT * FROM read_csv_auto('{CSV_PATH}');
    """)

    con.execute("""
        CREATE OR REPLACE TABLE rides_dedup AS
        SELECT *
        FROM rides_raw
        QUALIFY row_number() OVER (PARTITION BY id ORDER BY timestamp DESC) = 1;
    """)

    con.execute("""
        CREATE OR REPLACE TABLE rides_kda AS
        WITH base AS (
          SELECT
            id,
            timestamp,

            CAST(datetime AS TIMESTAMP) AS dt,
            CAST(hour AS INTEGER) AS hour,
            CAST(day AS INTEGER) AS day,
            CAST(month AS INTEGER) AS month,

            source,
            destination,
            (source || ' -> ' || destination) AS route,
            cab_type,
            name AS product_name,
            short_summary,

            TRY_CAST(distance AS DOUBLE) AS distance,
            TRY_CAST(surge_multiplier AS DOUBLE) AS surge_multiplier,

            TRY_CAST(temperature AS DOUBLE) AS temperature,
            TRY_CAST(apparentTemperature AS DOUBLE) AS apparent_temperature,
            TRY_CAST(precipIntensity AS DOUBLE) AS precip_intensity,
            TRY_CAST(precipProbability AS DOUBLE) AS precip_probability,
            TRY_CAST(humidity AS DOUBLE) AS humidity,
            TRY_CAST(windSpeed AS DOUBLE) AS wind_speed,
            TRY_CAST(cloudCover AS DOUBLE) AS cloud_cover,
            TRY_CAST(pressure AS DOUBLE) AS pressure,
            TRY_CAST(uvIndex AS DOUBLE) AS uv_index
          FROM rides_dedup
        )
        SELECT
          *,
          CASE WHEN surge_multiplier > 1 THEN 1 ELSE 0 END AS surge_flag
        FROM base
        WHERE
          cab_type IS NOT NULL
          AND surge_multiplier IS NOT NULL
          AND surge_multiplier BETWEEN 1 AND 10
          AND distance IS NOT NULL
          AND distance BETWEEN 0.1 AND 100;
    """)

    df = con.execute("SELECT * FROM rides_kda").df()
    if df.empty:
        raise RuntimeError("rides_kda is empty after cleaning. Check CSV_PATH / schema / filters.")
    return df


def run_kda_for_subset(df: pd.DataFrame, subset_name: str) -> None:
    target = "surge_flag"

    candidate_cat = ["route", "product_name", "source", "destination", "short_summary"]
    candidate_num = [
        "distance", "hour", "day", "month",
        "temperature", "apparent_temperature", "precip_intensity", "precip_probability",
        "humidity", "wind_speed", "cloud_cover", "pressure", "uv_index"
    ]

    cat_cols = [c for c in candidate_cat if c in df.columns]
    num_cols = [c for c in candidate_num if c in df.columns]

    X = df[cat_cols + num_cols].copy()
    y = df[target].astype(int)

    if y.nunique() < 2:
        print(f"\n=== {subset_name} ===")
        print("Only one class present in surge_flag; cannot fit/evaluate classification meaningfully.")
        print("Surge rate:", round(y.mean(), 4), "| Rows:", len(df))
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
    )

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", clf),
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)

    print(f"\n=== {subset_name} | Model Evaluation (surge_flag) ===")
    print(f"Rows used: {len(df):,}")
    print("Surge rate:", round(df[target].mean(), 4))
    print("Drivers (numeric):", num_cols)
    print("Drivers (categorical):", cat_cols)
    print("ROC AUC:", round(auc, 4))
    print("Accuracy:", round(acc, 4))

    rng = np.random.RandomState(42)
    base_auc = auc
    drivers = cat_cols + num_cols

    rows = []
    n_repeats = 3  

    for d in drivers:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[d] = rng.permutation(X_perm[d].values)

            proba_perm = pipe.predict_proba(X_perm)[:, 1]
            auc_perm = roc_auc_score(y_test, proba_perm)
            drops.append(base_auc - auc_perm)

        rows.append({
            "driver": d,
            "importance_auc_drop": float(np.mean(drops)),
            "importance_std": float(np.std(drops))
        })

    driver_rank = pd.DataFrame(rows).sort_values("importance_auc_drop", ascending=False)

    print(f"\n=== {subset_name} | Driver Ranking (AUC drop) ===")
    print(driver_rank.head(20).to_string(index=False))


def main():
    con = duckdb.connect()
    df = build_clean_table(con)

    cab_types = df["cab_type"].dropna().unique().tolist()
    cab_types_sorted = sorted([str(x) for x in cab_types])

    print("\n=== Stratified KDA Setup ===")
    print("Cab types found:", cab_types_sorted)

    for ct in cab_types_sorted:
        df_sub = df[df["cab_type"].astype(str) == ct].copy()
        run_kda_for_subset(df_sub, subset_name=f"cab_type = {ct}")


if __name__ == "__main__":
    main()
