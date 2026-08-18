import os
import re
import numpy as np
import pandas as pd

def clean_state_crop(val):
    if pd.isna(val):
        return np.nan
    # Remove control/non-printable characters
    val_str = str(val)
    val_str = ''.join(c for c in val_str if c.isprintable())
    # Strip standard and unicode whitespaces, collapse double spaces
    val_str = re.sub(r'[\s\xa0\u200b\ufeff\u200e\u200f]+', ' ', val_str).strip()
    if val_str == '':
        return np.nan
    return val_str

def main():
    # Define paths
    raw_file_path = "data/raw/Final-Estimate-of-Area,-Production-&-Yield-for-Rice.xlsx"
    processed_dir = "data/processed"
    output_file_path = os.path.join(processed_dir, "cleaned_rice_apy_production.parquet")
    
    print(f"Loading raw Excel file: {raw_file_path}")
    
    # 1. Load the raw Excel sheet 'Data' starting from row index 5
    df = pd.read_excel(raw_file_path, sheet_name="Data", header=None, skiprows=5)
    
    # 2. Replace empty strings, spaces, or non-printable characters with np.nan BEFORE calling ffill()
    df[0] = df[0].apply(clean_state_crop)
    df[1] = df[1].apply(clean_state_crop)
    
    df[0] = df[0].ffill()
    df[1] = df[1].ffill()
    
    # 4. Extract data rows (starts from index 2 in skiprows=5 loaded df)
    data_df = df.iloc[2:].copy()
    
    # Clean Season column (column 2)
    data_df[2] = data_df[2].apply(clean_state_crop)
    
    # Filter out summary rows where Season == 'Total' or State == 'All India'
    # Also filter out rows where Season is null (footnote rows)
    data_df = data_df[data_df[2].notna()]
    data_df = data_df[data_df[2] != "Total"]
    data_df = data_df[data_df[1] != "All India"]
    
    # 3. Reshape/melt the table across all 5 years (2021-22 through 2025-26)
    # Columns 0-2: Crop, State, Season
    # Columns 3-7: Area (in Lakh Ha) for years [2021-22, 2022-23, 2023-24, 2024-25, 2025-26]
    # Columns 8-12: Production (in Lakh Tonnes) for years [2021-22, 2022-23, 2023-24, 2024-25, 2025-26]
    # Columns 13-17: Yield (in Kg/Ha) for years [2021-22, 2022-23, 2023-24, 2024-25, 2025-26]
    records = []
    years = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    
    for i, year in enumerate(years):
        year_df = data_df[[0, 1, 2, 3+i, 8+i, 13+i]].copy()
        year_df.columns = [
            "crop",
            "state",
            "season",
            "area_lakh_ha",
            "production_lakh_tonnes",
            "yield_kg_per_ha",
        ]
        year_df["financial_year"] = year
        records.append(year_df)
        
    long_df = pd.concat(records, ignore_index=True)
    
    # Standardize string fields
    long_df["state"] = long_df["state"].astype(str).str.strip().str.title()
    long_df["crop"] = long_df["crop"].fillna("Rice").astype(str).str.strip().str.title()
    long_df["season"] = long_df["season"].astype(str).str.strip().str.title()
    long_df["financial_year"] = long_df["financial_year"].astype(str).str.strip()
    
    # 5. Clean all numeric fields (coerce missing/dashes/strings to standard float or NaN)
    for col in ["area_lakh_ha", "production_lakh_tonnes", "yield_kg_per_ha"]:
        # Coerce non-numeric markers to numeric/NaN
        long_df[col] = pd.to_numeric(
            long_df[col].astype(str).str.replace(",", "").str.strip(),
            errors="coerce"
        )
        # Fill missing season entries (NaN) with 0.0 since Rice is not grown in that season for that state
        long_df[col] = long_df[col].fillna(0.0)
        
    # Reorder columns to target schema:
    # state, crop, season, financial_year, area_lakh_ha, production_lakh_tonnes, yield_kg_per_ha
    long_df = long_df[[
        "state",
        "crop",
        "season",
        "financial_year",
        "area_lakh_ha",
        "production_lakh_tonnes",
        "yield_kg_per_ha",
    ]]
    
    # 6. Verify that NO row in state is null or blank
    assert long_df["state"].notna().all(), "Assertion failed: State column contains null values!"
    assert (long_df["state"].str.strip() != "").all(), "Assertion failed: State column contains blank strings!"
    
    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save the cleaned dataset to parquet
    print(f"Saving cleaned dataset to: {output_file_path}")
    long_df.to_parquet(output_file_path, index=False)
    
    # Print summary statistics and first 10 rows
    print("\n--- Summary Statistics ---")
    print(long_df.describe())
    
    print("\n--- First 10 Rows ---")
    print(long_df.head(10).to_string())
    
    print(f"\nProcessing complete! Saved {len(long_df)} rows to {output_file_path}.")

if __name__ == "__main__":
    main()
