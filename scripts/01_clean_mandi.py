import numpy as np
import pandas as pd

# 1. Load Raw CSV
df = pd.read_csv("data/raw/agmarknet_rice_2020_2025.csv")

# 2. Standardize column names (strip spaces, lowercase)
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# 3. Clean and convert Date
df["arrival_date"] = pd.to_datetime(
    df["arrival_date"], dayfirst=True, errors="coerce"
)
df = df.dropna(subset=["arrival_date"])

# 4. Standardize String Fields
for col in ["state", "district", "market", "commodity"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.title()

# 5. Clean Numeric Fields & Convert ₹/Quintal to ₹/kg
numeric_cols = ["modal_price", "min_price", "max_price", "arrivals_tonnes"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "")
            .str.extract(r"(\d+\.?\d*)")[0]
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert price to ₹/kg (1 Quintal = 100 kg)
df["modal_price_kg"] = df["modal_price"] / 100.0

# 6. Remove obvious outliers & negative prices
df = df[(df["modal_price_kg"] >= 10) & (df["modal_price_kg"] <= 150)]

# 7. Aggregate duplicate entries per market per day
daily_df = (
    df.groupby(["arrival_date", "state", "district", "market"])
    .agg(
        {
            "modal_price_kg": "mean",
            "arrivals_tonnes": "sum",
        }
    )
    .reset_index()
)

# 8. Sort and Forward-Fill missing non-trading days
daily_df = daily_df.sort_values(by=["market", "arrival_date"])
daily_df.to_parquet("data/processed/cleaned_rice_prices.parquet", index=False)
print("Data cleaned and saved successfully!")