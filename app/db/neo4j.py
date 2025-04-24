import os
from neo4j import GraphDatabase, Driver # Ensure Driver is imported for type hints
from neo4j.exceptions import AuthError, ServiceUnavailable
from typing import Optional, List, Dict, Any # For type hints

# Global variable to hold the driver instance
NEO4J_DRIVER: Optional[Driver] = None

def get_driver() -> Optional[Driver]:
    if NEO4J_DRIVER:
        return NEO4J_DRIVER
    else:
        return None

def init_neo4j_driver(uri: str, user: str, password: Optional[str]) -> Optional[Driver]:
    """
    Initializes the Neo4j driver instance and stores it globally.
    Verifies connectivity upon creation.

    Args:
        uri: The connection URI for the Neo4j instance (e.g., "bolt://localhost:7687").
        user: The username for authentication.
        password: The password for authentication (can be None if auth is disabled).

    Returns:
        The initialized Neo4j Driver instance if successful, otherwise None.
    """
    global NEO4J_DRIVER # Declare intent to modify the global variable
    if NEO4J_DRIVER:
        print("INFO: Neo4j driver already initialized.")
        return NEO4J_DRIVER
    try:
        print(f"uri:{uri} user:{user} password:{password}")
        # Attempt to create a driver instance
        driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connectivity and authentication
        driver.verify_connectivity()
        # Store the successfully created driver globally
        NEO4J_DRIVER = driver
        print(f"INFO: Successfully connected to Neo4j at {uri} as user '{user}' and initialized driver.")
        return NEO4J_DRIVER
    except AuthError as e:
        print(f"ERROR: Neo4j authentication failed for user '{user}'. Please check credentials.")
        print(f"Details: {e}")
        NEO4J_DRIVER = None # Ensure global var is None on failure
        return None
    except ServiceUnavailable as e:
        print(f"ERROR: Could not connect to Neo4j at {uri}. Please ensure the server is running and the URI is correct.")
        print(f"Details: {e}")
        NEO4J_DRIVER = None # Ensure global var is None on failure
        return None
    except Exception as e:
        print("ERROR: An unexpected error occurred while initializing the Neo4j driver.")
        print(f"Details: {e}")
        NEO4J_DRIVER = None # Ensure global var is None on failure
        return None

def close_neo4j_driver():
    """
    Closes the globally stored Neo4j driver instance if it exists.
    """
    global NEO4J_DRIVER # Declare intent to modify the global variable
    if NEO4J_DRIVER:
        try:
            NEO4J_DRIVER.close()
            print("INFO: Neo4j driver closed.")
        except Exception as e:
            print(f"ERROR: An error occurred while closing the Neo4j driver: {e}")
        finally:
            NEO4J_DRIVER = None 
    else:
        print("INFO: No active Neo4j driver to close.")

def execute_query(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    database: str = "neo4j" 
) -> Optional[List[Dict[str, Any]]]:
    """
    Executes a given Cypher query using the global driver connection.

    Suitable for read queries or simple auto-commit writes. For complex writes
    requiring explicit transaction control, use session.write_transaction separately.

    Args:
        query: The Cypher query string to execute.
        parameters: An optional dictionary of parameters for the query.
        database: The name of the database to run the query against.

    Returns:
        A list of dictionaries representing the result records if successful,
        otherwise None if an error occurred or the driver isn't initialized.
    """
    if not NEO4J_DRIVER:
        print("ERROR: Neo4j driver is not initialized. Cannot execute query.")
        return None
    records = []
    summary = None
    try:
        # Use try-with-resources for the session
        with NEO4J_DRIVER.session(database=database) as session:
            # Use session.run for generic execution.
            # If run outside an explicit transaction block, it behaves as auto-commit.
            result = session.run(query, parameters or {})

            # Consume the result into a list of dictionaries for easier handling
            # record.data() converts a Record object to a dictionary
            records = [record.data() for record in result]

            # Optionally consume the summary for metadata (counters, etc.)
            summary = result.consume()
            print(f"INFO: Query executed. Summary: NodesCreated={summary.counters.nodes_created}, RelsCreated={summary.counters.relationships_created}, PropsSet={summary.counters.properties_set}")

            return records
    except Exception as e:
        print(f"ERROR: Failed to execute query: {e}")
        print(f"Query: {query}")
        print(f"Parameters: {parameters}")
        return None