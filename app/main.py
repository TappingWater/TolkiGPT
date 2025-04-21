# app/main.py
from fastapi import FastAPI, HTTPException # Added HTTPException
from contextlib import asynccontextmanager # Use asynccontextmanager for lifespan
from app.api.routes import router as api_router
import app.config as config
# Import your utils module
from app.utils import process_text as utils # Assuming utils is named process_text.py
# Or if it's just utils.py: from app import utils
import os # To get env variables if needed

# --- Application Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Startup ===
    print(f"INFO:     Starting up application in {config.ENVIRONMENT} mode...")

    # 1. Check/Download WordNet (can be done once)
    print("INFO:     Checking WordNet data...")
    utils.check_wordnet()

    # 2. Load NLP Models
    print("INFO:     Loading NLP models...")
    models_loaded = utils.load_models()
    if not models_loaded:
         print("CRITICAL: Failed to load NLP models. Service may be impaired.")
         # Decide if you want to raise an error and stop startup

    # 3. Initialize Neo4j Driver (using utils.setup_neo4j)
    print("INFO:     Initializing Neo4j connection...")
    db_connected = utils.setup_neo4j(
        uri=config.NEO4J_URI, # Get from config
        user=config.NEO4J_USER, # Get from config
        password=config.NEO4J_PASSWORD # Get from config (ensure it's handled securely)
    )
    if not db_connected:
        print("CRITICAL: Failed to connect to Neo4j. Service may be impaired.")
        # Decide if you want to raise an error and stop startup

    # Optional: Replace config.test_connections() if it only tested Neo4j
    # If config.test_connections() tests other things, keep it or integrate checks here.
    # print("INFO:     Running initial connection tests...")
    # config.test_connections() # Remove or adapt this line

    print("INFO:     Application startup complete.")
    yield # Application runs here
    # === Shutdown ===
    print("INFO:     Shutting down application...")
    # 1. Close Neo4j Driver (using utils.close_neo4j)
    utils.close_neo4j()
    print("INFO:     Application shutdown complete.")


# --- FastAPI App Instance ---
app = FastAPI(
    title="TolkiGPT",
    description="REST API for novel generation",
    version="1.0.0",
    lifespan=lifespan # Use the lifespan manager
)

# --- API Routes ---
app.include_router(api_router)

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "TOLKI GPT HERE!!!"}

# --- Run Command ---
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000