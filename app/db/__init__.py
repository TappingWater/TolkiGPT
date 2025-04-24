# Import the desired names from the neo4j module within this package
from .neo4j import (
    NEO4J_DRIVER,          # The global driver variable
    init_neo4j_driver,     # The initialization function
    close_neo4j_driver,    # The closing function
    execute_query,
    generate_neo4j_queries,
    process_relations_for_neo4j,
    insert_relationships,
    get_graph_data_for_book,
    get_driver
    # The query execution function
)

# Optional: Define __all__ to control 'from app.db import *' behavior
# This explicitly lists what gets imported on wildcard imports. It's good practice.
__all__ = [
    "NEO4J_DRIVER",
    "init_neo4j_driver",
    "close_neo4j_driver",
    "execute_query",
    generate_neo4j_queries,
    process_relations_for_neo4j,
    insert_relationships,
    get_driver,
    get_graph_data_for_book
]