import pandas as pd
from typing import Dict, Any, Tuple

# Comprehensive Canonical State Name Dictionary
STATE_CANONICAL_MAP: Dict[str, str] = {
    "Andaman And Nicobar Islands": "Andaman & Nicobar Islands",
    "Andaman and Nicobar Islands": "Andaman & Nicobar Islands",
    "A & N Islands": "Andaman & Nicobar Islands",
    "Dadra And Nagar Haveli": "Dadra & Nagar Haveli and Daman & Diu",
    "Daman And Diu": "Dadra & Nagar Haveli and Daman & Diu",
    "Dadra and Nagar Haveli and Daman and Diu": "Dadra & Nagar Haveli and Daman & Diu",
    "Gujrat": "Gujarat",
    "Maharastra": "Maharashtra",
    "West Bangal": "West Bengal",
    "West Bengal": "West Bengal",
    "Orissa": "Odisha",
    "Odisha": "Odisha",
    "Keralam": "Kerala",
    "Kerala": "Kerala",
    "Uttaranchal": "Uttarakhand",
    "Uttarakhand": "Uttarakhand",
    "Chattisgarh": "Chhattisgarh",
    "Chhattishgarh": "Chhattisgarh",
    "Chhattisgarh": "Chhattisgarh",
    "Telengana": "Telangana",
    "Telangana": "Telangana",
    "Pondicherry": "Puducherry",
    "Puducherry": "Puducherry",
    "NCT of Delhi": "Delhi",
    "Delhi": "Delhi",
    "Jammu and Kashmir": "Jammu & Kashmir",
    "Jammu & Kashmir": "Jammu & Kashmir",
    "Tamilnadu": "Tamil Nadu",
    "Tamil Nadu": "Tamil Nadu"
}

# State Center Coordinates & Zoom Levels for Map Centering
STATE_COORDINATES: Dict[str, Dict[str, float]] = {
    "Andhra Pradesh": {"lat": 15.9129, "lon": 79.7400, "zoom": 6.5},
    "Arunachal Pradesh": {"lat": 28.2180, "lon": 94.7278, "zoom": 6.5},
    "Assam": {"lat": 26.2006, "lon": 92.9376, "zoom": 6.5},
    "Bihar": {"lat": 25.0961, "lon": 85.3131, "zoom": 6.5},
    "Chhattisgarh": {"lat": 21.2787, "lon": 81.8661, "zoom": 6.5},
    "Goa": {"lat": 15.2993, "lon": 74.1240, "zoom": 8.5},
    "Gujarat": {"lat": 22.2587, "lon": 71.1924, "zoom": 6.5},
    "Haryana": {"lat": 29.0588, "lon": 76.0856, "zoom": 7.0},
    "Himachal Pradesh": {"lat": 31.1048, "lon": 77.1734, "zoom": 7.0},
    "Jharkhand": {"lat": 23.6102, "lon": 85.2799, "zoom": 6.5},
    "Karnataka": {"lat": 15.3173, "lon": 75.7139, "zoom": 6.0},
    "Kerala": {"lat": 10.8505, "lon": 76.2711, "zoom": 7.0},
    "Madhya Pradesh": {"lat": 22.9734, "lon": 78.6569, "zoom": 6.0},
    "Maharashtra": {"lat": 19.7515, "lon": 75.7139, "zoom": 6.0},
    "Manipur": {"lat": 24.6637, "lon": 93.9063, "zoom": 7.5},
    "Meghalaya": {"lat": 25.4670, "lon": 91.3662, "zoom": 7.5},
    "Mizoram": {"lat": 23.1645, "lon": 92.9376, "zoom": 7.5},
    "Nagaland": {"lat": 26.1584, "lon": 94.5624, "zoom": 7.5},
    "Odisha": {"lat": 20.9517, "lon": 85.0985, "zoom": 6.5},
    "Punjab": {"lat": 31.1471, "lon": 75.3412, "zoom": 7.0},
    "Rajasthan": {"lat": 27.0238, "lon": 74.2179, "zoom": 6.0},
    "Sikkim": {"lat": 27.5330, "lon": 88.5122, "zoom": 8.0},
    "Tamil Nadu": {"lat": 11.1271, "lon": 78.6569, "zoom": 6.5},
    "Telangana": {"lat": 18.1124, "lon": 79.0193, "zoom": 6.5},
    "Tripura": {"lat": 23.9408, "lon": 91.9882, "zoom": 8.0},
    "Uttar Pradesh": {"lat": 26.8467, "lon": 80.9462, "zoom": 6.0},
    "Uttarakhand": {"lat": 30.0668, "lon": 79.0193, "zoom": 7.0},
    "West Bengal": {"lat": 22.9868, "lon": 87.8550, "zoom": 6.5},
    "Andaman & Nicobar Islands": {"lat": 11.7401, "lon": 92.6586, "zoom": 6.0},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "zoom": 10.0},
    "Dadra & Nagar Haveli and Daman & Diu": {"lat": 20.1809, "lon": 73.0169, "zoom": 9.0},
    "Delhi": {"lat": 28.7041, "lon": 77.1025, "zoom": 9.5},
    "Jammu & Kashmir": {"lat": 33.7782, "lon": 76.5762, "zoom": 6.5},
    "Ladakh": {"lat": 34.1526, "lon": 77.5771, "zoom": 6.0},
    "Lakshadweep": {"lat": 10.5667, "lon": 72.6417, "zoom": 8.0},
    "Puducherry": {"lat": 11.9416, "lon": 79.8083, "zoom": 9.0}
}

DEFAULT_INDIA_CENTER = {"lat": 22.5937, "lon": 78.9629, "zoom": 4.0}

def standardize_state(state_name: str) -> str:
    """Map raw state name to standardized canonical state name."""
    if not isinstance(state_name, str) or not state_name.strip():
        return "Unknown"
    cleaned = state_name.strip()
    return STATE_CANONICAL_MAP.get(cleaned, cleaned)

def get_state_center(state_name: str = None) -> Dict[str, float]:
    """Get latitude, longitude, and zoom level for a given state name."""
    if not state_name or state_name == "All States":
        return DEFAULT_INDIA_CENTER
    norm_st = standardize_state(state_name)
    return STATE_COORDINATES.get(norm_st, DEFAULT_INDIA_CENTER)
