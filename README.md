# Structural vs Behavioral Pricing Analysis (Uber & Lyft)

## Project Overview

This project analyzes ride-share pricing by decomposing it into two
distinct components:

1.  Behavioral pricing (surge occurrence)
2.  Structural pricing (baseline rate intensity)

Rather than modeling raw price directly, the analysis separates dynamic
demand-driven adjustments from structural pricing rules to identify the
true drivers behind each.

The objective is not only predictive performance, but clear,
interpretable driver analysis aligned with business reasoning.

------------------------------------------------------------------------

## Business Questions

-   What drives surge pricing events?
-   What determines baseline price intensity?
-   Are pricing differences platform-driven or tier-driven?

------------------------------------------------------------------------

## Dataset

The dataset contains over 600,000 ride records from Uber and Lyft,
including:

-   Ride characteristics (distance, source, destination, product tier)
-   Platform identifier
-   Weather features
-   Time variables
-   Surge multiplier
-   Price

Note: The dataset exceeds GitHub's 100MB limit and is not included in
this repository.\
It can be downloaded from:

https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma

------------------------------------------------------------------------

## How to Reproduce

1.  Download the dataset from the Kaggle link above.
2.  Extract the CSV file(s).
3.  Place the CSV in the same directory as the Python scripts, or update
    the CSV_PATH variable in the scripts.
4.  Install dependencies:

pip install duckdb pandas numpy scikit-learn

5.  Run the scripts:

python uber_lyft_kda.py python kda_stratified.py

------------------------------------------------------------------------

# Part I -- Behavioral Pricing (Surge Analysis)

## KPI Definition

surge_flag = 1 if surge_multiplier \> 1 else 0

Why binary?

-   Surge events are rare (\~3% of rides).
-   Modeling occurrence provides stronger signal than modeling
    magnitude.
-   Aligns with rare-event classification methodology.

## Model

-   Logistic regression (L2-regularized)
-   Class-weight balancing
-   Evaluation metric: ROC AUC
-   Driver ranking via permutation importance (AUC drop)

## Results

Surge rate: \~3%\
ROC AUC: \~0.88

Key drivers:

1.  cab_type\
2.  product_name\
3.  source\
4.  route

Stratified analysis showed:

-   Lyft exhibits surge events.
-   Uber shows no surge in the cleaned dataset.
-   Within Lyft, surge is primarily driven by ride tier and pickup
    location.

Insight:

Surge pricing differences are platform-driven and behaviorally
asymmetric.

------------------------------------------------------------------------

# Part II -- Structural Pricing (Price Intensity Analysis)

## KPI Engineering

price_per_mile = price / distance\
log_price_intensity = log(price_per_mile)

Why transform?

-   Removes mechanical dominance of distance.
-   Isolates effective rate differences.
-   Reduces skew via log transformation.
-   Better captures pricing structure.

## Model

-   Linear regression
-   Evaluation metrics: R² and MAE
-   Driver ranking via permutation importance (R² drop)

## Results

R² ≈ 0.94

Top drivers:

1.  product_name\
2.  distance\
3.  route\
4.  source\
5.  destination

Platform importance becomes minimal after controlling for tier and
route.

Insight:

Baseline pricing differences are primarily tier-based rather than
platform-based.

------------------------------------------------------------------------

## Combined Insights

By modeling surge and price intensity separately, the analysis reveals:

-   Behavioral pricing (surge) is platform-driven.
-   Structural pricing (rate intensity) is tier- and route-driven.
-   Platform differences largely disappear once structural pricing rules
    are controlled for.
-   Time-of-day and weather have limited impact on baseline rate but
    minor influence on surge.

------------------------------------------------------------------------

## Technical Stack

-   Python
-   DuckDB (SQL-based data cleaning)
-   pandas, numpy
-   scikit-learn
-   Logistic Regression
-   Linear Regression
-   Permutation Importance

------------------------------------------------------------------------

## Skills Demonstrated

-   KPI engineering and metric design
-   Rare-event classification
-   Continuous regression modeling
-   Feature transformation (log scaling)
-   Structural vs behavioral decomposition
-   Large-scale SQL data preparation
-   Business-focused interpretation of model results

------------------------------------------------------------------------

## Final Takeaway

Ride-share pricing contains two fundamentally different mechanisms:

1.  Dynamic, behavior-driven surge adjustments
2.  Structural, rule-based pricing schedules

Understanding both components separately shows that surge differences
are platform-specific, while structural rate differences are primarily
tier-driven.

This project demonstrates the importance of careful KPI definition and
structural reasoning in data analysis.
