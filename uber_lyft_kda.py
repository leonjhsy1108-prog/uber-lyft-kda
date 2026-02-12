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


def main():
    con = duckdb.connect()

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

            -- time fields
            CAST(datetime AS TIMESTAMP) AS dt,
            CAST(hour AS INTEGER) AS hour,
            CAST(day AS INTEGER) AS day,
            CAST(month AS INTEGER) AS month,

            -- categorical drivers
            source,
            destination,
            (source || ' -> ' || destination) AS route,
            cab_type,
            name AS product_name,
            short_summary,

            -- numeric drivers / fields (TRY_CAST is safe for numeric or string input)
            TRY_CAST(distance AS DOUBLE) AS distance,
            TRY_CAST(price AS DOUBLE) AS price,
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
          surge_multiplier IS NOT NULL
          AND surge_multiplier BETWEEN 1 AND 10
          AND distance IS NOT NULL
          AND distance BETWEEN 0.1 AND 100;
    """)

    df = con.execute("SELECT * FROM rides_kda").df()
    if df.empty:
        raise RuntimeError("rides_kda is empty after cleaning. Check CSV_PATH / schema / filters.")

    target = "surge_flag"

    candidate_cat = ["route", "cab_type", "product_name", "source", "destination", "short_summary"]
    candidate_num = [
        "distance", "hour", "day", "month",
        "temperature", "apparent_temperature", "precip_intensity", "precip_probability",
        "humidity", "wind_speed", "cloud_cover", "pressure", "uv_index"
    ]

    cat_cols = [c for c in candidate_cat if c in df.columns]
    num_cols = [c for c in candidate_num if c in df.columns]

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Available columns: {list(df.columns)}")

    X = df[cat_cols + num_cols].copy()
    y = df[target].astype(int)

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
        n_jobs=None
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

    print("\n=== Model Evaluation (surge_flag) ===")
    print(f"Rows used: {len(df):,}")
    print("Surge rate in cleaned data:", round(df[target].mean(), 4))
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

    print("\n=== Driver Ranking (Permutation importance on raw drivers; AUC drop) ===")
    print(driver_rank.head(20).to_string(index=False))

    perf_rows = []
    for d in driver_rank["driver"]:
        if d in num_cols:
            v = df[d].astype(float)
            vmin, vmax = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
            scaled = np.clip((v - vmin) / (vmax - vmin + 1e-9), 0, 1) * 100
            perf = float(np.nanmean(scaled))
        else:
            perf = float(df[d].value_counts(normalize=True, dropna=True).iloc[0] * 100)

        perf_rows.append((d, perf))

    perf_df = pd.DataFrame(perf_rows, columns=["driver", "performance_0_100"])
    kda_summary = driver_rank.merge(perf_df, on="driver", how="left")

    print("\n=== KDA Summary (Importance + Performance proxy) ===")
    print(kda_summary.head(25).to_string(index=False))

   
if __name__ == "__main__":
    main()
