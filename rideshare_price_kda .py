import duckdb
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

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
            TRY_CAST(price AS DOUBLE) AS price,

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
          price / distance AS price_per_mile,
          LOG(price / distance) AS log_price_intensity
        FROM base
        WHERE
          price IS NOT NULL
          AND distance IS NOT NULL
          AND distance > 0.1
          AND price > 0
          AND price / distance BETWEEN 0.5 AND 50;
    """)

    df = con.execute("SELECT * FROM rides_kda").df()

    if df.empty:
        raise RuntimeError("rides_kda is empty after cleaning.")

    target = "log_price_intensity"

    candidate_cat = ["route", "cab_type", "product_name", "source", "destination", "short_summary"]
    candidate_num = [
        "distance", "hour", "day", "month",
        "temperature", "apparent_temperature", "precip_intensity",
        "precip_probability", "humidity", "wind_speed",
        "cloud_cover", "pressure", "uv_index"
    ]

    cat_cols = [c for c in candidate_cat if c in df.columns]
    num_cols = [c for c in candidate_num if c in df.columns]

    X = df[cat_cols + num_cols].copy()
    y = df[target].astype(float)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    model = LinearRegression()

    pipe = Pipeline([
        ("prep", preprocess),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n=== Model Evaluation (log price per mile) ===")
    print(f"Rows used: {len(df):,}")
    print("R²:", round(r2, 4))
    print("MAE:", round(mae, 4))
    print("Drivers (numeric):", num_cols)
    print("Drivers (categorical):", cat_cols)

    rng = np.random.RandomState(42)
    base_r2 = r2

    drivers = cat_cols + num_cols
    rows = []
    n_repeats = 3

    for d in drivers:
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[d] = rng.permutation(X_perm[d].values)

            y_perm_pred = pipe.predict(X_perm)
            r2_perm = r2_score(y_test, y_perm_pred)

            drops.append(base_r2 - r2_perm)

        rows.append({
            "driver": d,
            "importance_r2_drop": float(np.mean(drops)),
            "importance_std": float(np.std(drops))
        })

    driver_rank = pd.DataFrame(rows).sort_values(
        "importance_r2_drop", ascending=False
    )

    print("\n=== Driver Ranking (Permutation importance; R² drop) ===")
    print(driver_rank.head(20).to_string(index=False))


    perf_rows = []

    for d in driver_rank["driver"]:
        if d in num_cols:
            v = df[d].astype(float)
            vmin, vmax = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
            scaled = np.clip((v - vmin) / (vmax - vmin + 1e-9), 0, 1) * 100
            perf = float(np.nanmean(scaled))
        else:
            perf = float(df[d].value_counts(normalize=True).iloc[0] * 100)

        perf_rows.append((d, perf))

    perf_df = pd.DataFrame(perf_rows, columns=["driver", "performance_0_100"])
    kda_summary = driver_rank.merge(perf_df, on="driver", how="left")

    print("\n=== KDA Summary (Importance + Performance proxy) ===")
    print(kda_summary.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
