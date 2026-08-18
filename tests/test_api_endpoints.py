import pytest
from starlette.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_api_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["phase"] == "4"

def test_api_market_overview_endpoint():
    response = client.get("/api/market-overview")
    assert response.status_code == 200
    data = response.json()
    assert "current_price" in data
    assert "predicted_price_7d" in data

def test_api_forecast_endpoint():
    response = client.get("/api/forecast?horizon=7")
    assert response.status_code == 200
    data = response.json()
    assert data["horizon"] == 7
    assert "historical" in data
    assert "forecast" in data

def test_api_risk_endpoints():
    r1 = client.get("/api/risk")
    assert r1.status_code == 200
    assert isinstance(r1.json(), list)
    
    r2 = client.get("/api/risk-map")
    assert r2.status_code == 200
    assert isinstance(r2.json(), list)

def test_api_explainability_endpoint():
    response = client.get("/api/explain?horizon=7")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "disclaimer" in data

def test_api_optimization_endpoints():
    r1 = client.get("/api/optimization")
    assert r1.status_code == 200
    assert isinstance(r1.json(), list)
    
    r2 = client.get("/api/scenarios")
    assert r2.status_code == 200
    assert isinstance(r2.json(), list)

def test_api_scenario_simulator_endpoint():
    res = client.get("/api/scenario-simulator?base_price=3000&ceiling_price=3300&simulated_release_mt=8000&scenario=worst_case")
    assert res.status_code == 200
    data = res.json()
    assert data["scenario"] == "worst_case"
    assert "release_range" in data
    assert "mitigated_trajectory" in data
    assert "sections" in data
    assert len(data["sections"]) == 4

