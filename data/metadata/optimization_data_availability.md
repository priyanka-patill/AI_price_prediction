# Optimization Data Availability Audit

| Variable | Available? | Primary Source | Granularity | Date Coverage | Classification & Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Central Pool Rice Stock** | **AVAILABLE** | Food Corporation of India (FCI) | Monthly National | 2024-01 to 2026-07 | **OFFICIAL DATA**: Monthly stock bulletins in Lakh MT. |
| **Buffer Stock Norms** | **AVAILABLE** | FCI / DFPD Directives | Quarterly National | 2024 to 2026 | **OFFICIAL DATA**: Official quarterly buffer requirement norms (~13.5M MT). |
| **State PDS Allocation & Offtake** | **AVAILABLE** | DFPD / NFSA Monthly Reports | Monthly State Level | 2024-01 to 2026-07 | **OFFICIAL DATA**: Monthly state allocations & offtake quantities (~1.8M-2.2M MT/state). |
| **Paddy Procurement** | **AVAILABLE** | DFPD KMS Bulletins | Annual State Level | KMS 2021-22 to 2025-26 | **OFFICIAL DATA**: Total KMS procurement per state. |
| **Mandi Price Forecasts & Early Warnings** | **AVAILABLE** | Phase 2 LightGBM & Early Warning Engine | Daily Market Level | 2024 to 2026 | **MODEL ESTIMATES**: 7D, 15D, 30D price predictions and warning alert levels. |
| **State Destination Warehouse Capacity** | **UNAVAILABLE** | None | N/A | N/A | **USER-CONFIGURED ASSUMPTION**: Modeled as soft upper bound capacity constraint. |
| **Inter-State Freight / Transportation Costs** | **UNAVAILABLE** | None | State-to-State | N/A | **USER-CONFIGURED ASSUMPTION**: Distance & ₹/MT-km matrix configured in `config/optimization_config.yaml`. |

---

> [!IMPORTANT]
> **Data Integrity Rule**: All decision support recommendations in Phase 3 explicitly distinguish **OFFICIAL DATA** (FCI stock, DFPD allocation) from **MODEL ESTIMATES** (LightGBM price forecasts) and **USER-CONFIGURED ASSUMPTIONS** (transport cost matrix).
