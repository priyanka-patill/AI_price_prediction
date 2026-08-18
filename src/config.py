import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env at project root
ROOT_DIR = Path(__file__).resolve().parent.parent
env_file = ROOT_DIR / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()

# Central Environment Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").strip()
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "").strip()

# Helper function to get live API Key
def get_api_key() -> str:
    key = os.getenv("DATA_GOV_API_KEY", "").strip()
    if key and key != "your_real_data_gov_api_key_here" and key != "your_agmarknet_api_key_here":
        return key
    return ""

# Load sources.yaml as Single Source of Truth for APIs & Resource IDs
SOURCES_YAML = ROOT_DIR / "config" / "sources.yaml"

def load_sources_config() -> dict:
    if SOURCES_YAML.exists():
        with open(SOURCES_YAML, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

sources_config = load_sources_config()
agmarknet_config = sources_config.get("agmarknet", {})

AGMARKNET_BASE_URL = agmarknet_config.get("base_url", "https://api.data.gov.in/resource")
AGMARKNET_RESOURCE_ID = agmarknet_config.get("resource_id_prices", "9ef84268-d588-465a-a308-a864a43d0070")
AGMARKNET_COMMODITIES = agmarknet_config.get("commodities", {"rice": "Rice", "paddy": "Paddy(Common)"})
