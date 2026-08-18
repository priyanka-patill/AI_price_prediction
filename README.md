# AI-Enabled Predictive Price Intelligence & Buffer Stock Decision Support System

## Phase 1: Data Integration & Feature Engineering (with Open-Meteo Weather)

This repository contains the complete Phase 1 implementation for building a clean, reliable, traceable, ML-ready Indian rice time-series dataset.

---

## 1. Project Directory Structure

```text
AI_price_prediction/
├── config/
│   ├── sources.yaml                # API endpoints, resource IDs, parameters, status
│   └── features.yaml               # Feature lag windows, rolling periods, split metadata
├── data/
│   ├── raw/                        # Immutable raw API responses and original datasets
│   │   ├── production/
│   │   ├── market/
│   │   ├── weather/                # Contains openmeteo_weather_raw.json
│   │   ├── procurement/
│   │   ├── msp/
│   │   ├── stock/
│   │   └── allocation/
│   ├── processed/                  # Cleaned, standardized Parquet datasets
│   │   ├── rice_production.parquet
│   │   ├── rice_market.parquet
│   │   ├── paddy_market.parquet
│   │   ├── weather.parquet
│   │   ├── procurement.parquet
│   │   ├── msp.parquet
│   │   ├── stock.parquet
│   │   ├── allocation.parquet
│   │   ├── unified_market_daily_modelling.parquet
│   │   ├── unified_state_daily_modelling.parquet
│   │   └── feature_engineered_modelling_dataset.parquet
│   └── metadata/                   # Quality reports, verification logs, split metadata
│       ├── location_coordinates.csv # Open-Meteo geocoded district coordinates cache
│       ├── api_sources.json
│       ├── api_schema.json
│       ├── data_quality_report.json
│       ├── data_quality_report.csv
│       ├── verification_log.csv
│       └── train_val_test_split.json
├── src/
│   ├── ingestion/                  # Reusable Python API clients
│   │   ├── market_api.py
│   │   ├── weather_api.py          # Open-Meteo Geocoding & Historical Weather API Client
│   │   ├── procurement_api.py
│   │   ├── stock_api.py
│   │   └── allocation_api.py
│   ├── cleaning/                   # Cleaning, deduplication, unit conversions
│   │   ├── clean_production.py
│   │   ├── clean_market.py
│   │   ├── clean_weather.py        # Process Open-Meteo weather fields
│   │   ├── clean_procurement.py
│   │   ├── clean_stock.py
│   │   ├── clean_allocation.py
│   │   └── clean_msp.py
│   ├── feature_engineering/        # Lags, rolling stats, anti-leakage shift(1)
│   │   ├── price_features.py
│   │   ├── arrival_features.py
│   │   ├── production_features.py
│   │   ├── weather_features.py      # Rainfall, temperature, soil moisture, evapotranspiration
│   │   └── government_features.py
│   └── pipeline/                   # Pipeline execution & validation
│       ├── build_dataset.py
│       └── validate_dataset.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_data_quality.ipynb
├── tests/
│   ├── test_api.py
│   ├── test_cleaning.py
│   ├── test_features.py
│   ├── test_leakage.py
│   └── test_weather_openmeteo.py   # Geocoding & Weather unit tests
├── requirements.txt
└── README.md
```

---

## 2. Integrated Data Sources & APIs

| Data Category | Primary Source | Resource / API Endpoint | Temporal Coverage | Geographic Coverage |
| :--- | :--- | :--- | :--- | :--- |
| **Rice Production** | Ministry of Agriculture & Farmers Welfare (APY) | Original APY Final Estimates (`data/processed/cleaned_rice_apy_production.parquet`) | 2021-22 to 2025-26 | 38 States & UTs |
| **Market Prices & Arrivals** | AGMARKNET / Government of India | `api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070` | 2024-01-01 to 2026-07-31 | State, District, Market APMCs |
| **Weather Data** | Open-Meteo Historical Weather API | `archive-api.open-meteo.com/v1/archive` | 2024-01-01 to 2026-07-31 | Mapped District Coordinates |
| **District Geocoding** | Open-Meteo Geocoding API | `geocoding-api.open-meteo.com/v1/search` | Cached in `location_coordinates.csv` | Mapped District Coordinates |
| **Government Procurement** | Dept. of Food & Public Distribution (DFPD) / FCI | DFPD KMS Procurement Bulletins | KMS 2021-22 to 2025-26 | Major Paddy Procuring States |
| **Minimum Support Price (MSP)** | CCEA / DFPD | Ministry of Agriculture MSP Releases | KMS 2021-22 to 2025-26 | All India (Common & Grade A) |
| **Buffer Stock** | Food Corporation of India (FCI) | FCI Central Pool Stock Bulletins | 2024-01-01 to 2026-07-01 | Central Pool |
| **State Allocation & Offtake** | DFPD PDS Portal | DFPD Monthly Foodgrain Allocation | 2024-01-01 to 2026-07-01 | 17 Major States |

---

## 3. Open-Meteo Weather Variables & Features

- **Ingested Variables**:
  - `Rainfall_mm` (`precipitation_sum`)
  - `Temperature_Mean_C` (`temperature_2m_mean`)
  - `Temperature_Max_C` (`temperature_2m_max`)
  - `Temperature_Min_C` (`temperature_2m_min`)
  - `Soil_Moisture` (`soil_moisture_0_to_7cm_mean`)
  - `Evapotranspiration` (`et0_fao_evapotranspiration`)

- **Weather Features**:
  - `rainfall_7d`, `rainfall_30d`, `rainfall_90d`
  - `temperature_mean_7d`, `temperature_mean_30d`
  - `soil_moisture_7d`, `soil_moisture_30d`
  - `evapotranspiration_7d`, `evapotranspiration_30d`
  - `rainfall_deviation`

- **Source Attribution**: Retained as `"Open-Meteo Historical Weather API"`.

---

## 4. Execution & Verification Commands

### Execute Pipeline
```bash
python -m src.pipeline.build_dataset
python -m src.pipeline.validate_dataset
```

### Run Test Suite (with Open-Meteo & Anti-Leakage Verification)
```bash
python -m pytest tests/
```

---

## 5. Data Verification Statuses

- `Verified`: Formally cross-checked against secondary government bulletins (DES, FCI, CCEA).
- `Official_API`: Retrieved directly via official public API endpoints (e.g. Open-Meteo API).
- `Derived`: Scientifically computed features (e.g. ratios, percentage changes, rolling means).
