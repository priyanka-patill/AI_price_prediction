import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routes
from src.api.routes import health, market, forecast, risk, explainability, optimization, sync, status

app = FastAPI(
    title="AI-Enabled Predictive Price Intelligence & Buffer Stock API",
    description="REST API backend providing rice price forecasts, early warning signals, SHAP explainability, PuLP buffer stock decision support, data lineage, and real-time live data syncing.",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Streamlit frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(health.router)
app.include_router(market.router)
app.include_router(forecast.router)
app.include_router(risk.router)
app.include_router(explainability.router)
app.include_router(optimization.router)
app.include_router(sync.router)
app.include_router(status.router)

@app.on_event("startup")
def startup_event():
    try:
        from src.ingestion.data_status import status_tracker
        status_tracker.reset()
        print("[Startup] System configuration and status tracker initialized cleanly.")
        from src.ingestion.market_api import AgmarknetClient
        print("[Startup] Fetching initial live AGMARKNET mandi data...")
        market_client = AgmarknetClient()
        market_client.fetch(commodity="Rice", limit=500)
    except Exception as e:
        print(f"[Startup] Configuration initialization notice: {e}")

@app.get("/", summary="API Root Endpoint")
def read_root():
    return {
        "message": "AI-Enabled Predictive Price Intelligence and Buffer Stock Decision Support API",
        "documentation": "/docs",
        "health": "/api/health",
        "data_status": "/api/data-status",
        "sync_realtime": "/api/sync-realtime-data"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

