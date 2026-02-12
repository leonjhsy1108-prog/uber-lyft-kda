# Key Driver Analysis of Surge Pricing (Uber vs Lyft)

## Project Overview

This project implements a Key Driver Analysis (KDA) on ride-sharing data
to understand:

What drives surge pricing events?

Instead of building a generic predictive model, the goal is to:

-   Identify structurally important drivers
-   Rank them using model-based importance
-   Interpret findings in a business-meaningful way

The analysis follows a structured KDA framework:

1.  Define a meaningful KPI
2.  Clean and prepare data (SQL-based)
3.  Model the outcome
4.  Rank drivers via permutation importance
5.  Interpret importance vs performance

------------------------------------------------------------------------

## Dataset Availability (GitHub Size Limit)

GitHub does not allow files larger than 100 MB in a standard repository.
The original CSV dataset exceeds this limit, so this repository does not
include the raw data. Only the code and documentation are uploaded.

You can download the dataset from Kaggle here:

https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma?resource=download

------------------------------------------------------------------------

## How to Reproduce / Run the Code

### 1) Download the dataset

1.  Go to the Kaggle link above and download the dataset.
2.  Extract the downloaded archive if needed.
3.  Locate the main CSV file(s).

### 2) Place the CSV in your project folder

Put the required CSV file in the same folder as the Python scripts, or
update the CSV_PATH variable in the scripts to point to your local
dataset path.

### 3) Install dependencies

pip install duckdb pandas numpy scikit-learn

### 4) Run the scripts

Run the combined model:
```
python uber_lyft_kda.py
```
Run the stratified model:
```
python kda_stratified.py
```
The scripts will print:
```
Model performance (ROC AUC, etc.) - Key driver rankings (permutation importance)
```
------------------------------------------------------------------------

## Why Surge Instead of Price?

### Why Not Use price as KPI?

Using price as the KPI would mostly rediscover the pricing formula:
```
price approx base_fare + distance times rate + surge
```
If we used price: - Distance would dominate - The model would reflect
mechanical pricing logic - Insights would be trivial

That is not true Key Driver Analysis. It becomes pricing replication.

------------------------------------------------------------------------

### Why Not Use Raw surge_multiplier?

Regression on exact surge multiplier had R2 approx 0.046.

Reasons: - Surge multiplier has low variation - Surge is rare - Many
unobserved supply-demand variables are missing

This produced weak signal.

------------------------------------------------------------------------

## Final KPI Choice: Binary surge_flag

We defined:
```
surge_flag = 1 if surge_multiplier > 1 else 0
```
This improves signal strength and aligns with classical KDA methodology.

------------------------------------------------------------------------

## Why Classification (Logistic Regression)?

Since surge_flag is binary, we use logistic regression.

Logistic regression was chosen because it is interpretable, stable, and
aligned with traditional KDA methodology.

------------------------------------------------------------------------

## Combined Model Results (Uber + Lyft)
```
Surge rate: 3.03
ROC AUC: 0.8796

Top Drivers: 1. cab_type 2. product_name 3. source 4. short_summary 5.
route
```
Interpretation: - Platform dominates surge behavior - Ride tier matters
significantly - Pickup location influences surge probability

------------------------------------------------------------------------

## Stratified Analysis (Platform-Specific KDA)

### Lyft

Surge rate: 6.82
ROC AUC: 0.7194

Top Drivers: - product_name - source - short_summary - route - distance

### Uber

Surge rate: 0.0

Uber had no surge events in the cleaned dataset.

This indicates structural platform differences.

------------------------------------------------------------------------

## Key Insights

1.  Surge pricing is platform-driven.
2.  Lyft exhibits dynamic surge behavior; Uber does not in this dataset.
3.  Within Lyft, ride tier and pickup location dominate surge
    probability.

------------------------------------------------------------------------

## Tech Stack

-   Python
-   DuckDB
-   scikit-learn
-   Logistic Regression
-   Permutation Importance
