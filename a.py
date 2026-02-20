# =====================================================================
# GLOBAL FOOD WASTE ANALYSIS & REDISTRIBUTION POTENTIAL PIPELINE
# FINAL PRODUCTION VERSION
# =====================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

print("\nStarting Food Waste Analysis Pipeline...\n")

# ============================================================
# 1. LOAD DATA (REAL OR SYNTHETIC BACKUP)
# ============================================================
FILE_NAME = "global_food_wastage_dataset.csv"

if os.path.exists(FILE_NAME):
    df = pd.read_csv(FILE_NAME)
    print("Real dataset loaded.")
else:
    print("Dataset not found. Generating synthetic dataset.")
    countries_list = ["USA","India","China","Germany","Brazil","UK","Japan","France","Canada","Australia"]
    temp_data = []
    for c in countries_list:
        base_waste = np.random.uniform(20000,80000)
        for yr in range(2018,2025):
            waste = base_waste + (yr-2018)*np.random.uniform(1000,3000)
            temp_data.append([c,yr,waste,waste*1.5,waste/50])
    df = pd.DataFrame(temp_data, columns=["country","year","total_waste_tons","economic_loss_usd","per_capita_waste_kg"])

df.columns = [str(c).strip().lower().replace(" ","_") for c in df.columns]
df.drop_duplicates(inplace=True)
df.dropna(subset=["country","year"], inplace=True)

print("Data prepared successfully.")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
RECOVERY_RATE = 0.35
MEAL_KG = 0.5
CO2_PER_TON = 2.5

df = df.sort_values(["country","year"])
df["recoverable_tons"] = df["total_waste_tons"] * RECOVERY_RATE
df["meals_saved"] = df["recoverable_tons"] * 1000 / MEAL_KG
df["people_fed_yearly"] = df["meals_saved"] / (365 * 3)
df["co2_saved_tons"] = df["recoverable_tons"] * CO2_PER_TON

df["prev_year_waste"] = df.groupby("country")["total_waste_tons"].shift(1)
df["waste_growth_rate"] = (df["total_waste_tons"] - df["prev_year_waste"]) / df["prev_year_waste"]
df["waste_growth_rate"] = df["waste_growth_rate"].fillna(0)

print("Feature engineering completed.")

# ============================================================
# 3. GLOBAL KPI EXPORT
# ============================================================
kpi_summary = pd.DataFrame({
    "metric":[
        "Total Waste",
        "Recoverable Waste",
        "Meals Possible",
        "People Fed Per Year",
        "CO2 Saved"
    ],
    "value":[
        df["total_waste_tons"].sum(),
        df["recoverable_tons"].sum(),
        df["meals_saved"].sum(),
        df["people_fed_yearly"].sum(),
        df["co2_saved_tons"].sum()
    ]
})

kpi_summary.to_csv("global_kpi_summary.csv", index=False)
print("Global KPI summary saved.")

# ============================================================
# 4. MACHINE LEARNING MODEL
# ============================================================
print("\nTraining forecasting model...")

ml_df = df.groupby(["country","year"]).agg({
    "total_waste_tons":"sum",
    "waste_growth_rate":"mean"
}).reset_index()

countries = ml_df["country"].unique()
ml_encoded = pd.get_dummies(ml_df, columns=["country"], drop_first=True)

X = ml_encoded.drop("total_waste_tons", axis=1)
y = ml_encoded["total_waste_tons"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.07, max_depth=5)
model.fit(X_train, y_train)

print("Model R2 Score:", round(r2_score(y_test, model.predict(X_test)), 3))

# ============================================================
# 5. IMPROVED RECURSIVE FORECAST (WITH DERIVED FEATURES)
# ============================================================
future_years = list(range(2025,2031))
forecast_rows = []
dummy_cols = [c for c in X.columns if c.startswith("country_")]

for country in countries:

    country_history = ml_df[ml_df["country"] == country].sort_values("year")
    last_value = country_history.iloc[-1]["total_waste_tons"]
    avg_growth = country_history["waste_growth_rate"].mean()

    for year in future_years:

        projected_value = last_value * (1 + avg_growth)

        row = {"year": year, "waste_growth_rate": avg_growth}
        for col in dummy_cols:
            row[col] = 1 if col == f"country_{country}" else 0

        row_df = pd.DataFrame([row]).reindex(columns=X.columns, fill_value=0)
        ml_prediction = model.predict(row_df)[0]

        final_prediction = (ml_prediction * 0.6) + (projected_value * 0.4)

        # Derived features for forecast rows
        recoverable = final_prediction * RECOVERY_RATE
        meals = recoverable * 1000 / MEAL_KG
        people = meals / (365 * 3)
        co2 = recoverable * CO2_PER_TON

        forecast_rows.append([
            country, year, final_prediction, None, None,
            recoverable, meals, people, co2,
            last_value, avg_growth, "Forecast"
        ])

        last_value = final_prediction

df_forecast = pd.DataFrame(
    forecast_rows,
    columns=[
        "country","year","total_waste_tons",
        "economic_loss_usd","per_capita_waste_kg",
        "recoverable_tons","meals_saved","people_fed_yearly",
        "co2_saved_tons","prev_year_waste","waste_growth_rate","data_type"
    ]
)

df["data_type"] = "Actual"
combined_df = pd.concat([df, df_forecast], ignore_index=True)

print("Forecasting completed.")

# ============================================================
# 6. COUNTRY CLUSTERING
# ============================================================
cluster_data = df.groupby("country").agg({
    "total_waste_tons":"mean",
    "economic_loss_usd":"mean",
    "per_capita_waste_kg":"mean"
}).reset_index()

scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_data.drop("country", axis=1))

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_data["cluster_id"] = kmeans.fit_predict(scaled)

labels = {
    0:"High Waste Economies",
    1:"Growing Waste Countries",
    2:"Efficient Low Waste"
}
cluster_data["segment"] = cluster_data["cluster_id"].map(labels)

print("Clustering completed.")

# ============================================================
# 7. FINAL EXPORT
# ============================================================
final_df = combined_df.merge(cluster_data[["country","segment"]], on="country", how="left")
final_df.to_csv("processed_waste_data_final.csv", index=False)

print("\nPipeline completed successfully.")
print("Generated files:")
print("processed_waste_data_final.csv")
print("global_kpi_summary.csv")