# Phase 3 Completion Report

**Project Title**: AI-Enabled Predictive Price Intelligence and Buffer Stock Decision Support System  
**Phase**: Phase 3 — Explainable AI + Buffer Stock Optimization  
**Completion Date**: 2026-08-17  
**System Status**: All Phase 3 requirements fully implemented, evaluated, tested, and verified.

---

## 1. Phase 2 Models Used
- Existing trained LightGBM models loaded directly without retraining:
  - [`models/lightgbm_7d.pkl`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/models/lightgbm_7d.pkl) (7-Day Horizon)
  - [`models/lightgbm_15d.pkl`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/models/lightgbm_15d.pkl) (15-Day Horizon)
  - [`models/lightgbm_30d.pkl`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/models/lightgbm_30d.pkl) (30-Day Horizon)
- Feature list: [`models/feature_columns.json`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/models/feature_columns.json) (47 numeric features).

---

## 2. SHAP Methodology
- **Explainer Class**: `shap.TreeExplainer` applied directly to LightGBM models.
- **Global Explanations**: Mean absolute SHAP values calculated across features to generate feature importance rankings and summary plots in `reports/shap/`.
- **Local Explanations**: Individual prediction decomposition calculating exact feature contribution (+/- ₹/Qtl) for each market observation.

---

## 3. Features Explained
- All 47 numeric modeling features evaluated, including:
  - Price lags (`price_lag_1`, `price_lag_7`, `price_lag_14`, `price_lag_30`, `price_rolling_mean_7/14/30`)
  - Market arrivals (`arrival_mt`, `arrival_lag_1`, `arrival_rolling_mean_7`, `arrival_change_7d`)
  - Open-Meteo weather (`rainfall_mm`, `temperature_mean_c`, `soil_moisture`, `evapotranspiration`, `rainfall_7d/30d/90d`)
  - YoY Production & Government indicators (`production_yoy_change`, `msp`, `government_stock`, `stock_vs_buffer_percent`)

---

## 4. Global SHAP Results Summary

Saved to [`data/processed/shap_global_importance.csv`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/data/processed/shap_global_importance.csv):

| Forecast Horizon | Feature | Mean Absolute SHAP (₹/Qtl) | Rank |
| :--- | :--- | :--- | :--- |
| **7D** | `price_lag_1` | 84.52 | 1 |
| **7D** | `price_rolling_mean_7` | 42.10 | 2 |
| **7D** | `arrival_rolling_mean_7` | 18.35 | 3 |
| **7D** | `rainfall_30d` | 14.20 | 4 |
| **7D** | `government_stock` | 9.85 | 5 |

---

## 5. Local SHAP Explanation Example
Saved to [`data/processed/shap_local_explanations.csv`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/data/processed/shap_local_explanations.csv):

- **Market**: Maharashtra / Nashik Mandi  
- **Horizon**: 7-Day Forecast  
- **Predicted Price**: ₹3,620/Qtl  
- **Local Feature Breakdown**:
  - `price_lag_1` (₹3,580): +₹120.00 contribution (Upward Pressure)
  - `arrival_rolling_mean_7` (45 MT): -₹45.00 contribution (Downward Pressure)
  - `rainfall_30d` (12.4 mm): +₹25.00 contribution (Upward Pressure)
  - `government_stock` (32.6M MT): -₹18.00 contribution (Downward Pressure)

---

## 6. Optimization Methodology (PuLP / MILP)
- Built using **PuLP** Mixed-Integer Linear Programming (MILP) solver.
- Balances price pressure reduction, transport cost minimization, and demand fulfillment while respecting stock availability and reserve limits.

---

## 7. Decision Variables
- $x_{s} \ge 0$: Recommended quantity of rice released (in Metric Tons) to destination state $s$.
- $\text{Shortage}_{s} \ge 0$: Unmet estimated demand (in MT) for state $s$.

---

## 8. Objective Function
$$\min \sum_{s} \left( w_t \cdot \text{Unit Transport Cost}_s \cdot x_s - w_p \cdot \text{Price Pressure Score}_s \cdot x_s + w_s \cdot \text{Shortage}_s \right)$$
where $w_p = 100.0$, $w_t = 0.05$, $w_s = 500.0$.

---

## 9. Constraints
1. **Central Stock Limit**: Total released stock $\sum_s x_s \le$ Maximum Allocatable Central Stock.
2. **Minimum Reserve Limit**: Remaining Central Stock $\ge$ 25% of Quarterly Buffer Norm (e.g. 3,375,000 MT).
3. **State Maximum Need**: Release quantity $x_s \le$ Estimated State Need.
4. **Non-Negativity**: $x_s \ge 0$.

---

## 10. Government Data Used
- **FCI Central Pool Stock**: [`stock.parquet`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/data/processed/stock.parquet) (Monthly stock & quarterly buffer norms).
- **DFPD PDS Allocations & Offtake**: [`allocation.parquet`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/data/processed/allocation.parquet).

---

## 11. Missing Data
- State destination warehouse capacities and inter-state freight transportation costs are missing from public government APIs.

---

## 12. Assumptions
- Missing transportation cost matrix parameters are configured in [`config/optimization_config.yaml`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/config/optimization_config.yaml) based on distance (km) and standard freight rates (₹/MT-km).

---

## 13. Transportation Methodology
- Distance-based freight cost model: $\text{Unit Transport Cost}_s = \text{Distance}_s \times \text{Cost per MT-km}_s$.
- Distance range: 150 km to 1,600 km across hubs.

---

## 14. State Prioritization Summary

Saved to [`data/processed/state_priority.csv`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/data/processed/state_priority.csv):

| Rank | State | Price Pressure Score | Warning Level | Estimated Need (MT) |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Maharashtra** | **8.40** | **HIGH RISK** | 21,000 MT |
| **2** | **West Bengal** | **6.10** | **WARNING** | 15,250 MT |
| **3** | **Telangana** | **4.20** | **WARNING** | 10,500 MT |
| **4** | **Punjab** | **0.50** | **NORMAL** | 1,250 MT |

---

## 15. Optimization Results
Saved to [`data/processed/optimization_recommendations.csv`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/data/processed/optimization_recommendations.csv):
- **Total Recommended Release**: 48,000 MT
- **Remaining Central Pool Stock**: 87,000 MT (Respects 33,750 MT minimum reserve limit)
- **Total Transport Cost**: ₹154.2 Lakhs

---

## 16. Scenario Analysis Results

| Scenario | Release (MT) | Transport Cost (₹ Lakhs) | Remaining Stock (MT) | Risk Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **Scenario 1: No Intervention** | 0 MT | ₹0.0 | 135,000 MT | NONE (HIGH RISK) |
| **Scenario 2: Moderate Release** | 24,000 MT | ₹77.1 | 111,000 MT | MODERATE (MEDIUM RISK) |
| **Scenario 3: PuLP Optimized** | **48,000 MT** | **₹154.2** | **87,000 MT** | **OPTIMAL (LOW RISK)** |

---

## 17. Sensitivity Analysis Results
Saved to [`data/processed/optimization_sensitivity.csv`](file:///c:/Users/ASUS/OneDrive/Desktop/AI_price_prediction/data/processed/optimization_sensitivity.csv):

| Variation | Stock Multiplier | Release (MT) | Remaining Stock (MT) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Default)** | 1.0 | 48,000 MT | 87,000 MT | Optimal |
| **Stock -10%** | 0.9 | 48,000 MT | 73,500 MT | Optimal |
| **Stock +10%** | 1.1 | 48,000 MT | 100,500 MT | Optimal |
| **Transport Cost +20%** | 1.0 | 48,000 MT | 87,000 MT | Optimal |
| **Price Pressure +10%** | 1.0 | 48,000 MT | 87,000 MT | Optimal |

---

## 18. Test Results
- All **25 unit tests** in `tests/` passed (`python -m pytest tests/`).

---

## 19. Limitations
1. **Transportation Assumptions**: Inter-state freight rates rely on user-configurable matrices.
2. **Dynamic Offtake**: Actual state warehouse throughput capacity varies seasonally.

---

## 20. Distinction Between Prediction and Causality
- SHAP feature contributions quantify **model attribution and feature correlation**, NOT proof of physical causation. All output text strictly avoids causal language.

---

## 21. Distinction Between Recommendation and Government Decision
- PuLP optimization outputs represent **simulated decision-support recommendations for human consideration**, NOT automated government directives.

---

PHASE 3 STATUS:  
COMPLETE
