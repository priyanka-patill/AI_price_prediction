import pytest
import pandas as pd
import numpy as np
from src.cleaning.clean_production import standardize_state, clean_production_dataset
from src.cleaning.clean_market import clean_market_data

def test_standardize_state():
    assert standardize_state("Maharastra") == "Maharashtra"
    assert standardize_state("West Bangal") == "West Bengal"
    assert standardize_state("Gujrat") == "Gujarat"
    assert standardize_state("Punjab") == "Punjab"

def test_clean_production_dataset():
    df = clean_production_dataset()
    assert not df.empty
    assert "area_ha" in df.columns
    assert "production_mt" in df.columns
    assert "yield_kg_ha" in df.columns
    assert (df["state"] != "Maharastra").all()
    assert (df["state"] != "West Bangal").all()

def test_clean_market_data_deduplication():
    # Ingest and clean sample rice market data
    df = clean_market_data("data/raw/market/rice_market_raw.json", "data/processed/rice_market.parquet", "Rice")
    assert not df.empty
    # Verify no business key duplicates
    duplicates = df.duplicated(subset=["date", "state", "district", "market", "commodity"]).sum()
    assert duplicates == 0
    assert "price_rs_per_qtl" in df.columns
    assert "arrival_mt" in df.columns
