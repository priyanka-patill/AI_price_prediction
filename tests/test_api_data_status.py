import pytest
import os
from unittest.mock import patch, MagicMock
from starlette.testclient import TestClient
from src.api.main import app
from src.ingestion.market_api import AgmarknetClient
from src.ingestion.weather_api import WeatherApiClient
from src.ingestion.data_status import status_tracker

client = TestClient(app)

def test_agmarknet_client_normalization_and_validation():
    """Test 10: Normalization layer maps raw AGMARKNET capitalized field names to canonical schema."""
    raw_api_record = {
        "State": "Andhra Pradesh",
        "District": "East Godavari",
        "Market": "Rajahmundry",
        "Commodity": "Rice",
        "Variety": "Other",
        "Arrival_Date": "18/08/2026",
        "Min_Price": "3400",
        "Max_Price": "3800",
        "Modal_Price": "3600",
        "Arrivals": "150"
    }
    m_client = AgmarknetClient()
    norm = m_client.normalize_record(raw_api_record, data_source_status="LIVE")
    assert norm["state"] == "Andhra Pradesh"
    assert norm["district"] == "East Godavari"
    assert norm["market"] == "Rajahmundry"
    assert norm["commodity"] == "Rice"
    assert norm["arrival_date"] == "18/08/2026"
    assert norm["modal_price"] == 3600.0
    assert norm["min_price"] == 3400.0
    assert norm["max_price"] == 3800.0
    assert norm["data_source_status"] == "LIVE"

def test_weather_client_and_validation():
    w_client = WeatherApiClient()
    raw = w_client.fetch([("Punjab", "Ludhiana")], start_date="2026-01-01", end_date="2026-01-05")
    assert w_client.validate_response(raw)
    df = w_client.return_dataframe(raw)
    assert not df.empty

def test_api_data_status_endpoint():
    resp = client.get("/api/data-status")
    assert resp.status_code == 200
    data = resp.json()
    assert "mandi_source" in data
    assert "mandi_status" in data
    assert "weather_status" in data
    assert "fallback_used" in data

def test_test_1_missing_api_key():
    """Test 1: Missing API key returns FALLBACK status with error mentioning missing."""
    with patch("src.ingestion.market_api.get_api_key", return_value=""):
        m_client = AgmarknetClient()
        m_client.fetch("Rice", 10)
        st = status_tracker.to_dict()
        assert st["mandi_status"] == "FALLBACK"
        assert st["mandi_error_reason"] is not None
        assert "missing" in st["mandi_error_reason"].lower()

def test_test_2_http_200_valid_records():
    """Test 2: HTTP 200 + valid records returns LIVE status with records_received > 0."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "records": [
            {
                "State": "Punjab",
                "District": "Ludhiana",
                "Market": "Ludhiana Mandi",
                "Commodity": "Rice",
                "Arrival_Date": "2026-08-18",
                "Modal_Price": "3500"
            }
        ]
    }
    with patch("src.ingestion.market_api.get_api_key", return_value="valid_mock_key_56_chars_length_long_test_string"):
        with patch("requests.get", return_value=mock_resp):
            m_client = AgmarknetClient()
            res = m_client.fetch("Rice", 10)
            st = status_tracker.to_dict()
            assert st["mandi_status"] == "LIVE"
            assert st["mandi_records_received"] > 0
            assert st["mandi_error_reason"] is None
            assert len(res.get("records", [])) > 0

def test_test_3_http_200_zero_records():
    """Test 3: HTTP 200 + zero records returns FALLBACK status with clear diagnostic."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"records": []}
    with patch("src.ingestion.market_api.get_api_key", return_value="valid_mock_key_56_chars_length_long_test_string"):
        with patch("requests.get", return_value=mock_resp):
            m_client = AgmarknetClient()
            m_client.fetch("Rice", 10)
            st = status_tracker.to_dict()
            assert st["mandi_status"] == "FALLBACK"
            assert st["mandi_error_reason"] is not None
            assert "0 records" in st["mandi_error_reason"] or "no records" in st["mandi_error_reason"].lower()

def test_test_4_http_401_unauthorized():
    """Test 4: HTTP 401 Unauthorized returns FALLBACK with reason containing 401."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": "Unauthorized"}
    with patch("src.ingestion.market_api.get_api_key", return_value="invalid_key"):
        with patch("requests.get", return_value=mock_resp):
            m_client = AgmarknetClient()
            m_client.fetch("Rice", 10)
            st = status_tracker.to_dict()
            assert st["mandi_status"] == "FALLBACK"
            assert st["mandi_error_reason"] is not None
            assert "401" in st["mandi_error_reason"]

def test_test_5_http_403_forbidden():
    """Test 5: HTTP 403 Forbidden returns FALLBACK with reason containing 403."""
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.json.return_value = {"error": "Key not authorised"}
    with patch("src.ingestion.market_api.get_api_key", return_value="invalid_key"):
        with patch("requests.get", return_value=mock_resp):
            m_client = AgmarknetClient()
            m_client.fetch("Rice", 10)
            st = status_tracker.to_dict()
            assert st["mandi_status"] == "FALLBACK"
            assert st["mandi_error_reason"] is not None
            assert "403" in st["mandi_error_reason"]

def test_test_6_http_404_not_found():
    """Test 6: HTTP 404 Not Found returns FALLBACK with reason containing 404."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {"error": "Resource Not Found"}
    with patch("src.ingestion.market_api.get_api_key", return_value="valid_key"):
        with patch("requests.get", return_value=mock_resp):
            m_client = AgmarknetClient()
            m_client.fetch("Rice", 10)
            st = status_tracker.to_dict()
            assert st["mandi_status"] == "FALLBACK"
            assert st["mandi_error_reason"] is not None
            assert "404" in st["mandi_error_reason"]

def test_test_7_http_429_rate_limit():
    """Test 7: HTTP 429 Rate Limit returns FALLBACK with reason containing 429."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.json.return_value = {"error": "Rate limit exceeded"}
    with patch("src.ingestion.market_api.get_api_key", return_value="valid_key"):
        with patch("requests.get", return_value=mock_resp):
            m_client = AgmarknetClient()
            m_client.fetch("Rice", 10)
            st = status_tracker.to_dict()
            assert st["mandi_status"] == "FALLBACK"
            assert st["mandi_error_reason"] is not None
            assert "429" in st["mandi_error_reason"]

def test_test_8_network_timeout():
    """Test 8: Network timeout returns FALLBACK with reason containing timeout."""
    import requests
    with patch("src.ingestion.market_api.get_api_key", return_value="valid_key"):
        with patch("requests.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
            m_client = AgmarknetClient()
            m_client.fetch("Rice", 10)
            st = status_tracker.to_dict()
            assert st["mandi_status"] == "FALLBACK"
            assert st["mandi_error_reason"] is not None
            assert "timeout" in st["mandi_error_reason"].lower()

def test_test_9_successful_sync_and_persistence():
    """Test 9: POST /api/sync-realtime-data updates status to LIVE and market-overview uses live data."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "records": [
            {
                "State": "Punjab",
                "District": "Ludhiana",
                "Market": "Ludhiana Mandi",
                "Commodity": "Rice",
                "Arrival_Date": "2026-08-18",
                "Modal_Price": "3600"
            }
        ]
    }
    with patch("src.ingestion.market_api.get_api_key", return_value="valid_sync_key"):
        with patch("requests.get", return_value=mock_resp):
            sync_resp = client.post("/api/sync-realtime-data", json={"commodity": "Rice", "limit": 10})
            assert sync_resp.status_code == 200
            assert sync_resp.json()["status"] == "success"
            
            status_resp = client.get("/api/data-status")
            assert status_resp.json()["mandi_status"] == "LIVE"
            
            market_resp = client.get("/api/market-overview")
            assert market_resp.json()["data_source_status"] == "LIVE"
            assert market_resp.json()["current_price"] == 3600.0
