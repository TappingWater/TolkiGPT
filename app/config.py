# app/core/config.py
import os
from dotenv import load_dotenv
from app.db.neo4j import test_neo4j_connection

# Load environment variables from .env file
# Searches for .env file in the current directory or parent directories
load_dotenv()

# Application settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Neo4j settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Test connection
def test_connections():
    test_neo4j_connection(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

PYTORCH_MODEL_PATH = os.getenv("PYTORCH_MODEL_PATH")
