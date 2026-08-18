import os
import json
import pandas as pd
import numpy as np

# Ingestion Modules
from src.ingestion.market_api import AgmarknetClient
from src.ingestion.weather_api import WeatherApiClient
from src.ingestion.procurement_api import ProcurementClient
from src.ingestion.stock_api import BufferStockClient
from src.ingestion.allocation_api import AllocationClient

# Cleaning Modules
from src.cleaning.clean_production import clean_production_dataset
from src.cleaning.clean_market import clean_market_data
from src.cleaning.clean_weather import clean_weather_dataset
from src.cleaning.clean_procurement import clean_procurement_dataset
from src.cleaning.clean_stock import clean_stock_dataset
from src.cleaning.clean_allocation import clean_allocation_dataset
from src.cleaning.clean_msp import clean_msp_dataset

# Feature Engineering Modules
from src.feature_engineering.price_features import add_price_features
from src.feature_engineering.arrival_features import add_arrival_features
from src.feature_engineering.production_features import add_production_features
from src.feature_engineering.weather_features import add_weather_features
from src.feature_engineering.government_features import add_government_features

def run_ingestion_phase():
    """Step 1: Execute all dataset ingestion clients and save raw files."""
    print("=== PHASE 1: INGESTION PHASE ===")
    
    # 1. Market Data Ingestion
    market_client = AgmarknetClient()
    rice_raw = market_client.fetch(commodity="Rice", limit=5000)
    market_client.save_raw(rice_raw, "data/raw/market/rice_market_raw.json")
    
    paddy_raw = market_client.fetch(commodity="Paddy(Dhan)", limit=5000)
    market_client.save_raw(paddy_raw, "data/raw/market/paddy_market_raw.json")
    
    # Extract unique state + district locations from market raw records
    df_rice_temp = pd.DataFrame(rice_raw.get("records", []))
    locations = []
    if not df_rice_temp.empty and "state" in df_rice_temp.columns and "district" in df_rice_temp.columns:
        pairs = df_rice_temp[["state", "district"]].drop_duplicates().to_records(index=False)
        locations = [(p[0], p[1]) for p in pairs]
        
    # 2. Open-Meteo Weather Ingestion
    weather_client = WeatherApiClient()
    raw_weather = weather_client.fetch(locations=locations, start_date="2024-01-01", end_date="2026-07-31")
    weather_client.save_raw(raw_weather, "data/raw/weather/openmeteo_weather_raw.json")
    
    # 3. Procurement Ingestion
    proc_client = ProcurementClient()
    raw_proc = proc_client.fetch()
    proc_client.save_raw(raw_proc, "data/raw/procurement/procurement_raw.json")
    
    # 4. Buffer Stock Ingestion
    stock_client = BufferStockClient()
    raw_stock = stock_client.fetch()
    stock_client.save_raw(raw_stock, "data/raw/stock/stock_raw.json")
    
    # 5. Allocation Ingestion
    alloc_client = AllocationClient()
    raw_alloc = alloc_client.fetch()
    alloc_client.save_raw(raw_alloc, "data/raw/allocation/allocation_raw.json")
    
    print("Ingestion Phase Complete.\n")

def run_cleaning_phase():
    """Step 2: Clean all raw datasets and save individual processed Parquet files."""
    print("=== PHASE 1: CLEANING PHASE ===")
    
    prod_df = clean_production_dataset()
    rice_mkt_df = clean_market_data("data/raw/market/rice_market_raw.json", "data/processed/rice_market.parquet", "Rice")
    paddy_mkt_df = clean_market_data("data/raw/market/paddy_market_raw.json", "data/processed/paddy_market.parquet", "Paddy (Dhan)")
    weather_df = clean_weather_dataset()
    proc_df = clean_procurement_dataset()
    stock_df = clean_stock_dataset()
    alloc_df = clean_allocation_dataset()
    msp_df = clean_msp_dataset()
    
    print("Cleaning Phase Complete.\n")
    return {
        "production": prod_df,
        "rice_market": rice_mkt_df,
        "paddy_market": paddy_mkt_df,
        "weather": weather_df,
        "procurement": proc_df,
        "stock": stock_df,
        "allocation": alloc_df,
        "msp": msp_df
    }

def build_merged_modelling_datasets(dfs: dict):
    """Step 3: Build multi-granularity merged datasets (Market Level & State Level)."""
    print("=== PHASE 1: MERGING PHASE ===")
    
    rice_mkt = dfs["rice_market"].copy()
    weather = dfs["weather"].copy()
    
    if rice_mkt.empty:
        raise ValueError("Rice market dataset is empty!")
        
    # 1. Market Level Dataset (State + District + Market + Date)
    weather_cols = ["date", "state", "district", "latitude", "longitude", "rainfall_mm", "rainfall_deviation_percent",
                    "temperature_mean_c", "temperature_max_c", "temperature_min_c", "soil_moisture", "evapotranspiration"]
    existing_w_cols = [c for c in weather_cols if c in weather.columns]
    
    market_level_df = rice_mkt.merge(
        weather[existing_w_cols],
        on=["date", "state", "district"],
        how="left"
    )
    market_level_path = "data/processed/unified_market_daily_modelling.parquet"
    market_level_df.to_parquet(market_level_path, index=False)
    print(f"Saved Market Level dataset ({len(market_level_df)} rows) to {market_level_path}")
    
    # 2. State Level Dataset (State + Date Aggregation)
    state_daily_df = (
        rice_mkt.groupby(["date", "state"])
        .agg({
            "price_rs_per_qtl": "mean",
            "arrival_mt": "sum"
        })
        .reset_index()
    )
    
    w_agg_dict = {
        "rainfall_mm": "mean",
        "rainfall_deviation_percent": "mean",
        "temperature_mean_c": "mean"
    }
    if "soil_moisture" in weather.columns:
        w_agg_dict["soil_moisture"] = "mean"
    if "evapotranspiration" in weather.columns:
        w_agg_dict["evapotranspiration"] = "mean"

    state_weather = (
        weather.groupby(["date", "state"])
        .agg(w_agg_dict)
        .reset_index()
    )
    
    state_level_df = state_daily_df.merge(state_weather, on=["date", "state"], how="left")
    state_level_df["district"] = "State-Average"
    state_level_df["market"] = "State-Aggregate"
    
    state_level_path = "data/processed/unified_state_daily_modelling.parquet"
    state_level_df.to_parquet(state_level_path, index=False)
    print(f"Saved State Level dataset ({len(state_level_df)} rows) to {state_level_path}")
    
    return market_level_df, state_level_df

def run_feature_engineering_phase(modelling_df: pd.DataFrame, dfs: dict):
    """Step 4: Execute Feature Engineering with strict Anti-Leakage controls."""
    print("=== PHASE 1: FEATURE ENGINEERING PHASE ===")
    
    df = modelling_df.copy()
    
    # A. Price Features
    df = add_price_features(df, price_col="price_rs_per_qtl", group_cols=["state", "district", "market"])
    
    # B. Arrival Features
    df = add_arrival_features(df, arrival_col="arrival_mt", group_cols=["state", "district", "market"])
    
    # C. Weather Features
    df = add_weather_features(df, group_cols=["state", "district"])
    
    # D. Government Features (MSP, Procurement, Stock)
    df = add_government_features(
        df,
        msp_df=dfs.get("msp"),
        procurement_df=dfs.get("procurement"),
        stock_df=dfs.get("stock")
    )
    
    # E. Seasonal Features
    df["date_dt"] = pd.to_datetime(df["date"])
    df["month"] = df["date_dt"].dt.month
    df["quarter"] = df["date_dt"].dt.quarter
    
    def get_season(m):
        if m in [6, 7, 8, 9, 10, 11]:
            return "Kharif"
        elif m in [12, 1, 2, 3, 4]:
            return "Rabi"
        else:
            return "Summer"
            
    df["season"] = df["month"].apply(get_season)
    df["harvest_period"] = df["month"].apply(lambda m: 1 if m in [10, 11, 12, 1, 4, 5] else 0)
    
    df = df.drop(columns=["date_dt"], errors="ignore")
    
    output_path = "data/processed/feature_engineered_modelling_dataset.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved Feature Engineered Dataset ({len(df)} rows, {len(df.columns)} columns) to {output_path}")
    return df

def save_split_metadata(df: pd.DataFrame):
    """Step 5: Generate and save chronological Train/Validation/Test split metadata."""
    df["year"] = pd.to_datetime(df["date"]).dt.year
    
    train_mask = df["year"] == 2024
    val_mask = df["year"] == 2025
    test_mask = df["year"] == 2026
    
    split_meta = {
        "train_period": {"start": "2024-01-01", "end": "2024-12-31", "records": int(train_mask.sum())},
        "validation_period": {"start": "2025-01-01", "end": "2025-12-31", "records": int(val_mask.sum())},
        "test_period": {"start": "2026-01-01", "end": "2026-07-31", "records": int(test_mask.sum())},
        "total_records": len(df),
        "split_method": "Chronological (No Data Leakage)"
    }
    
    meta_path = "data/metadata/train_val_test_split.json"
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    print(f"Saved Train/Val/Test Split Metadata to {meta_path}")

def main():
    run_ingestion_phase()
    dfs = run_cleaning_phase()
    market_level_df, state_level_df = build_merged_modelling_datasets(dfs)
    fe_df = run_feature_engineering_phase(market_level_df, dfs)
    save_split_metadata(fe_df)
    print("\nPhase 1 Pipeline Execution Successfully Complete!")

if __name__ == "__main__":
    main()
