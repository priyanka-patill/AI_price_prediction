import pytest
from src.ingestion.market_api import AgmarknetClient
from src.ingestion.weather_api import WeatherApiClient
from src.ingestion.procurement_api import ProcurementClient
from src.ingestion.stock_api import BufferStockClient
from src.ingestion.allocation_api import AllocationClient

def test_agmarknet_client_rice():
    client = AgmarknetClient()
    raw = client.fetch(commodity="Rice", limit=10)
    assert client.validate_response(raw)
    df = client.return_dataframe(raw.get("records", []))
    assert not df.empty
    assert "modal_price" in df.columns
    assert "arrivals" in df.columns

def test_agmarknet_client_paddy():
    client = AgmarknetClient()
    raw = client.fetch(commodity="Paddy", limit=10)
    assert client.validate_response(raw)
    df = client.return_dataframe(raw.get("records", []))
    assert not df.empty
    assert df["commodity"].str.contains("Paddy", case=False).all()

def test_weather_client():
    client = WeatherApiClient()
    raw = client.fetch("2024-01-01", "2024-01-05")
    assert client.validate_response(raw)
    df = client.return_dataframe(raw)
    assert not df.empty
    assert "rainfall_mm" in df.columns

def test_procurement_client():
    client = ProcurementClient()
    raw = client.fetch()
    assert client.validate_response(raw)
    df = client.return_dataframe(raw)
    assert not df.empty
    assert "paddy_procured_mt" in df.columns
    assert "rice_equivalent_mt" in df.columns

def test_buffer_stock_client():
    client = BufferStockClient()
    raw = client.fetch()
    assert client.validate_response(raw)
    df = client.return_dataframe(raw)
    assert not df.empty
    assert "stock_vs_buffer_percent" in df.columns

def test_allocation_client():
    client = AllocationClient()
    raw = client.fetch()
    assert client.validate_response(raw)
    df = client.return_dataframe(raw)
    assert not df.empty
    assert "rice_allocated_mt" in df.columns
    assert "rice_offtake_mt" in df.columns
