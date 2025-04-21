import os
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

def test_neo4j_connection(uri: str, user: str, password: str | None):
    """
    Tests the connection to the Neo4j database using provided credentials.

    Prints a success message or a detailed error message to the console.

    Args:
        uri: The connection URI for the Neo4j instance (e.g., "bolt://localhost:7687").
        user: The username for authentication.
        password: The password for authentication (can be None if auth is disabled).
    """
    driver = None  # Initialize driver to None for the finally block
    try:
        # Attempt to create a driver instance
        # The driver manages connection pooling automatically
        driver = GraphDatabase.driver(uri, auth=(user, password))

        # Verify connectivity and authentication explicitly
        driver.verify_connectivity()

        print(f"Successfully connected to Neo4j at {uri} as user '{user}'.")
    except AuthError as e:
        # Handle authentication errors (wrong user/password)
        print(f"ERROR: Neo4j authentication failed for user '{user}'. Please check credentials.")
        print(f"Details: {e}")
    except ServiceUnavailable as e:
        # Handle connection errors (server down, wrong URI/port)
        print(f"ERROR: Could not connect to Neo4j at {uri}. Please ensure the server is running and the URI is correct.")
        print(f"Details: {e}")
    except Exception as e:
        # Catch any other unexpected errors during driver creation or verification
        print(f"ERROR: An unexpected error occurred while connecting to Neo4j.")
        print(f"Details: {e}")
    finally:
        # Ensure the driver is closed if it was successfully created
        if driver:
            driver.close()
            print("Neo4j driver closed.")