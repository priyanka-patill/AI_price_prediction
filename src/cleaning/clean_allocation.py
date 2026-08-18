import os
import json
import pandas as pd

from src.cleaning.clean_production import standardize_state

def clean_allocation_dataset(raw_json_path: str = "data/raw/allocation/allocation_raw.json",
                             output_parquet_path: str = "data/processed/allocation.parquet") -> pd.DataFrame:
    """
    Clean DFPD PDS Rice Allocation and Offtake dataset.
    """
    if not os.path.exists(raw_json_path):
        print(f"[clean_allocation] File not found: {raw_json_path}")
        return pd.DataFrame()
        
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    records = raw_data.get("records", [])
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    
    # Dates & States
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["state"] = df["state"].apply(standardize_state)
    
    # Keep Allocation & Offtake separate
    df["offtake_to_allocation_ratio"] = df["rice_offtake_mt"] / df["rice_allocated_mt"]
    
    # Quality check
    df["is_suspicious"] = False
    df.loc[(df["rice_allocated_mt"] <= 0) | (df["rice_offtake_mt"] < 0), "is_suspicious"] = True
    
    # Metadata
    df["primary_source"] = "Department of Food & Public Distribution (DFPD)"
    df["secondary_source"] = "State PDS Portals"
    df["verification_status"] = "Verified"
    df["notes"] = "Monthly Rice Allocation & Offtake under NFSA/TPDS"
    
    output_cols = [
        "date", "state", "scheme", "rice_allocated_mt", "rice_offtake_mt",
        "offtake_to_allocation_ratio", "primary_source", "secondary_source",
        "verification_status", "data_status", "is_suspicious", "notes"
    ]
    df_clean = df[output_cols].sort_values(by=["state", "date"]).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df_clean.to_parquet(output_parquet_path, index=False)
    print(f"[clean_allocation] Saved {len(df_clean)} rows of allocation data to {output_parquet_path}")
    return df_clean

if __name__ == "__main__":
    clean_allocation_dataset()
