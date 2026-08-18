import os
import json
import pandas as pd
import numpy as np

from src.cleaning.clean_production import standardize_state

def clean_market_data(raw_json_path: str, output_parquet_path: str, commodity_name: str = "Rice") -> pd.DataFrame:
    """
    Clean raw market price and arrival data from AGMARKNET.
    Ensures Rice and Paddy remain completely separate datasets.
    """
    if not os.path.exists(raw_json_path):
        print(f"[clean_market] File not found: {raw_json_path}")
        return pd.DataFrame()

    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    records = raw_data.get("records", [])
    if not records:
        print(f"[clean_market] No records found in {raw_json_path}")
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    
    # 1. Standardize Dates (YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["arrival_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    
    # 2. Standardize Strings (State, District, Market, Commodity)
    df["state"] = df["state"].astype(str).apply(standardize_state)
    df["district"] = df["district"].astype(str).str.strip().str.title()
    df["market"] = df["market"].astype(str).str.strip().str.title()
    df["commodity"] = commodity_name
    
    # 3. Deduplicate Business Key: Date + State + District + Market + Commodity
    df = df.drop_duplicates(subset=["date", "state", "district", "market", "commodity"], keep="first")
    
    # 4. Standardize Numeric Columns (Price & Arrivals)
    numeric_cols = ["min_price", "max_price", "modal_price", "arrivals"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # 5. Units & Conversions
    # Keep original units
    df["arrival_unit"] = df.get("arrival_unit", "Tonnes")
    df["arrival_mt"] = df["arrivals"] # All AGMARKNET arrival records in our ingestion are in Tonnes/MT
    df["price_rs_per_qtl"] = df["modal_price"]
    
    # 6. Sanity Checks & Outlier Flagging
    df["is_suspicious"] = False
    invalid_price = (df["price_rs_per_qtl"].notna()) & ((df["price_rs_per_qtl"] <= 500) | (df["price_rs_per_qtl"] >= 15000))
    invalid_range = (df["min_price"].notna()) & (df["max_price"].notna()) & (df["min_price"] > df["max_price"])
    invalid_arrivals = (df["arrival_mt"].notna()) & (df["arrival_mt"] < 0)
    
    df.loc[invalid_price | invalid_range | invalid_arrivals, "is_suspicious"] = True
    
    # 7. Metadata Tags
    df["primary_source"] = "AGMARKNET (Government of India)"
    df["secondary_source"] = "data.gov.in"
    df["verification_status"] = "Verified"
    df["data_status"] = "Derived"
    df["notes"] = f"Processed {commodity_name} market prices and arrivals."
    
    # Select & Order output columns
    output_cols = [
        "date", "state", "district", "market", "commodity",
        "arrival_mt", "arrival_unit", "min_price", "max_price", "price_rs_per_qtl",
        "primary_source", "secondary_source", "verification_status", "data_status",
        "is_suspicious", "notes"
    ]
    df_clean = df[output_cols].sort_values(by=["state", "district", "market", "date"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df_clean.to_parquet(output_parquet_path, index=False)
    print(f"[clean_market] Saved {len(df_clean)} rows of {commodity_name} to {output_parquet_path}")
    return df_clean

if __name__ == "__main__":
    clean_market_data("data/raw/market/rice_market_raw.json", "data/processed/rice_market.parquet", "Rice")
    clean_market_data("data/raw/market/paddy_market_raw.json", "data/processed/paddy_market.parquet", "Paddy (Dhan)")
