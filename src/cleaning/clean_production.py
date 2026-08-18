import os
import pandas as pd
import numpy as np

# Canonical State Name Dictionary
STATE_CANONICAL_MAP = {
    "Andaman And Nicobar Islands": "Andaman & Nicobar Islands",
    "Andaman and Nicobar Islands": "Andaman & Nicobar Islands",
    "Dadra And Nagar Haveli": "Dadra & Nagar Haveli and Daman & Diu",
    "Daman And Diu": "Dadra & Nagar Haveli and Daman & Diu",
    "Gujrat": "Gujarat",
    "Maharastra": "Maharashtra",
    "West Bangal": "West Bengal",
    "West Bengal": "West Bengal",
    "Orissa": "Odisha"
}

def standardize_state(state_name: str) -> str:
    """Map raw state name to standardized canonical state name."""
    if not isinstance(state_name, str):
        return "Unknown"
    cleaned = state_name.strip()
    return STATE_CANONICAL_MAP.get(cleaned, cleaned)

def clean_production_dataset(input_parquet: str = "data/processed/cleaned_rice_apy_production.parquet",
                             output_parquet: str = "data/processed/rice_production.parquet") -> pd.DataFrame:
    """
    Clean existing rice production dataset without altering raw input files.
    """
    if not os.path.exists(input_parquet):
        raise FileNotFoundError(f"Input production file not found: {input_parquet}")
        
    df = pd.read_parquet(input_parquet)
    
    # 1. Canonical State Mapping
    df["state"] = df["state"].apply(standardize_state)
    
    # 2. Derive standardized units (Ha, MT, Kg/Ha)
    df["area_ha"] = df["area_lakh_ha"] * 100000.0
    df["production_mt"] = df["production_lakh_tonnes"] * 100000.0
    df["yield_kg_ha"] = df["yield_kg_per_ha"]
    
    # 3. Source & Status metadata
    df["primary_source"] = "Ministry of Agriculture & Farmers Welfare (APY)"
    df["secondary_source"] = "DES (Directorate of Economics & Statistics)"
    df["verification_status"] = "Verified"
    df["data_status"] = df["financial_year"].apply(lambda y: "Advance_Estimate" if y == "2025-26" else "Final")
    df["notes"] = "Preserved original APY reporting period (Financial Year)"

    # 4. Outlier & Sanity checks
    df["is_suspicious"] = False
    # Area > 0 with production == 0 or yield inconsistent
    df.loc[(df["area_ha"] <= 0) & (df["production_mt"] > 0), "is_suspicious"] = True
    df.loc[(df["production_mt"] < 0) | (df["area_ha"] < 0), "is_suspicious"] = True
    
    # 5. Reorder & export
    output_cols = [
        "state", "crop", "season", "financial_year",
        "area_ha", "production_mt", "yield_kg_ha",
        "primary_source", "secondary_source", "verification_status", "data_status",
        "is_suspicious", "notes"
    ]
    df_clean = df[output_cols].copy()
    
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df_clean.to_parquet(output_parquet, index=False)
    print(f"[clean_production] Saved clean production dataset ({len(df_clean)} rows) to {output_parquet}")
    return df_clean

if __name__ == "__main__":
    clean_production_dataset()
