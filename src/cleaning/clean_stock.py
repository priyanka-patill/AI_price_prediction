import os
import json
import pandas as pd

def clean_stock_dataset(raw_json_path: str = "data/raw/stock/stock_raw.json",
                        output_parquet_path: str = "data/processed/stock.parquet") -> pd.DataFrame:
    """
    Clean Central Pool Rice Stock and Buffer Norms dataset.
    """
    if not os.path.exists(raw_json_path):
        print(f"[clean_stock] File not found: {raw_json_path}")
        return pd.DataFrame()
        
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    records = raw_data.get("records", [])
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    
    # Standardize Dates
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    # Calculate Stock vs Buffer Ratio (%)
    # Formula: Stock_vs_Buffer_Percent = (Rice_Stock_MT / Buffer_Norm_MT) * 100
    df["stock_vs_buffer_percent"] = (df["rice_stock_mt"] / df["buffer_norm_mt"]) * 100.0
    
    # Suspicious flag
    df["is_suspicious"] = False
    df.loc[(df["rice_stock_mt"] <= 0) | (df["buffer_norm_mt"] <= 0), "is_suspicious"] = True
    
    # Metadata
    df["primary_source"] = "Food Corporation of India (FCI)"
    df["secondary_source"] = "DFPD Monthly Foodgrain Bulletin"
    df["verification_status"] = "Verified"
    df["notes"] = "Central Pool Rice Stock vs Quarterly Buffer Norms"
    
    output_cols = [
        "date", "stock_type", "rice_stock_mt", "buffer_norm_mt",
        "stock_vs_buffer_percent", "primary_source", "secondary_source",
        "verification_status", "data_status", "is_suspicious", "notes"
    ]
    df_clean = df[output_cols].sort_values(by="date").reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
    df_clean.to_parquet(output_parquet_path, index=False)
    print(f"[clean_stock] Saved {len(df_clean)} rows of buffer stock data to {output_parquet_path}")
    return df_clean

if __name__ == "__main__":
    clean_stock_dataset()
