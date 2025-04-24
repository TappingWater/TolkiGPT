# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.db import init_neo4j_driver 
import app.config as config
from app.api.routes import router as api_router
import app.utils as utils

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    utils.log_info("Initializing Neo4j driver...")
    NEO4J_DRIVER = init_neo4j_driver(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD
    )    
    if not NEO4J_DRIVER:
        raise RuntimeError("Neo4j driver failed to initialize")
    utils.log_info("Initializing models")
    utils.initialize_models(
        spacy_model_name=config.DEFAULT_SPACY_MODEL,
        fastcoref_model_name=config.DEFAULT_COREF_MODEL,
        gen_model_name=config.DEFAULT_GEN_MODEL,
        device=config.DEFAULT_INFERENCE_DEVICE
    )
    yield
    
    # Shutdown
    utils.log_info("Closing Neo4j driver...")
    if NEO4J_DRIVER:
        NEO4J_DRIVER.close()

app = FastAPI(lifespan=lifespan)

app.include_router(api_router)