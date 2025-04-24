# app/core/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Searches for .env file in the current directory or parent directories
load_dotenv()

# Application settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Neo4j settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# --- Configuration (can be changed before calling initialization/getters if needed) ---
DEFAULT_SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_lg")
DEFAULT_COREF_MODEL = os.getenv("COREF_MODEL", "biu-nlp/f-coref")
DEFAULT_GEN_MODEL = os.getenv("GEN_MODEL", "mrcedric98/falcon-rw-1b-finetuned")
DEFAULT_INFERENCE_DEVICE = os.getenv("INFERENCE_DEVICE", "cpu")

PYTORCH_MODEL_PATH = os.getenv("PYTORCH_MODEL_PATH")
