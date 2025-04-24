import os
from neo4j import GraphDatabase, Driver # Ensure Driver is imported for type hints
from neo4j.exceptions import AuthError, ServiceUnavailable
from typing import Optional, List, Dict, Any, Tuple # For type hints

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
    
def process_relations_for_neo4j(
    relations_list: List[Dict[str, Any]],
    ner_dict: Dict[str, str] # Example: {'Cinderella': 'PERSON', 'Pickle': 'ANIMAL'}
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Tuple[str, str]]]]:
    """
    Processes a list of relations, keeping only entities found in ner_dict
    and relationships/attributes connected to them.

    Args:
        relations_list: A list of dictionaries, each representing a relation
                        with "head", "type", and "tail" keys.
        ner_dict: A dictionary mapping known named entities (strings) to their
                  types (e.g., 'PERSON', 'LOCATION'). Used to identify valid entities.

    Returns:
        A tuple containing two dictionaries:
        1. entities: Dict[str, Dict[str, Any]] - Contains only entities found in ner_dict.
        2. relationships: Dict[str, List[Tuple[str, str]]] - Contains only relationships
                          where both head and tail were found in ner_dict.
    """
    entities: Dict[str, Dict[str, Any]] = {}
    relationships: Dict[str, List[Tuple[str, str]]] = {}

    print(f"INFO: Processing {len(relations_list)} relations with filtering...")

    for relation in relations_list:
        head = relation.get("head")
        rel_type = relation.get("type")
        tail = relation.get("tail")

        if not head or not rel_type or tail is None:
             print(f"WARN: Skipping relation due to missing basic data: {relation}")
             continue

        head_is_ner = head in ner_dict

        # --- Primary Filter: Only process if Head is a known NER ---
        if head_is_ner:
            head_type = ner_dict[head]

            # Ensure head entity exists in our 'entities' store
            if head not in entities:
                entities[head] = {"type": head_type, "attributes": {}}
            # Note: No need to update type here, as it's guaranteed by ner_dict check

            # Now check the tail
            tail_is_ner = tail in ner_dict

            if tail_is_ner:
                # Case 1: Relationship between two known entities
                tail_type = ner_dict[tail]
                print(f"  -> Found Valid Relationship: ({head}:{head_type})-[{rel_type}]->({tail}:{tail_type})")

                # Ensure tail entity exists in our 'entities' store
                if tail not in entities:
                     entities[tail] = {"type": tail_type, "attributes": {}}

                # Add relationship
                if rel_type not in relationships:
                    relationships[rel_type] = []
                rel_tuple = (head, tail)
                if rel_tuple not in relationships[rel_type]:
                     relationships[rel_type].append(rel_tuple)
                else:
                     print(f"  -> (Skipping duplicate relationship: {rel_tuple} for type {rel_type})")

            else:
                # Case 2: Attribute of a known head entity
                attribute_key = rel_type
                attribute_value = tail
                print(f"  -> Found Attribute for '{head}' ({head_type}): {attribute_key} = {attribute_value}")
                # Add attribute to the head entity
                entities[head]["attributes"][attribute_key] = attribute_value

        else:
            # Case 3: Head is NOT a known NER. Skip this entire relation.
            print(f"  -> Skipping relation: Head '{head}' not found in NER dictionary.")
            # No entity created for head, no attributes added, no relationship added.

    print(f"INFO: Processing complete.")
    # Recalculate counts based on filtered results
    final_entity_count = len(entities)
    final_relationship_count = sum(len(v) for v in relationships.values())
    print(f"INFO: Kept {final_entity_count} unique entities (found in NER dict).")
    print(f"INFO: Kept {final_relationship_count} relationship instances across {len(relationships)} types (between known entities).")

    return entities, relationships

# --- (Keep the generate_neo4j_queries function exactly as before) ---
def generate_neo4j_queries(
    entities: Dict[str, Dict[str, Any]],
    relationships: Dict[str, List[Tuple[str, str]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Generates Neo4j Cypher MERGE queries from processed entities and relationships.
    (No changes needed here, it works on the data it receives)
    """
    queries_with_params: List[Tuple[str, Dict[str, Any]]] = []
    print("INFO: Generating MERGE queries for nodes...")
    for name, data in entities.items():
        entity_type = data.get("type", "UnknownEntity") # Should always have a type now
        attributes = data.get("attributes", {})
        query = f"MERGE (n:{entity_type} {{name: $name}}) SET n += $props"
        params = {"name": name, "props": attributes}
        queries_with_params.append((query, params))
        print(f"  -> Node Query for '{name}' ({entity_type}) with {len(attributes)} attributes.")

    print("INFO: Generating MERGE queries for relationships...")
    for rel_type, rel_list in relationships.items():
        for head_name, tail_name in rel_list:
            head_type = entities.get(head_name, {}).get("type", "Error") # Should exist
            tail_type = entities.get(tail_name, {}).get("type", "Error") # Should exist
            if head_type == "Error" or tail_type == "Error":
                 print(f"WARN: Skipping relationship query due to missing entity data for ({head_name} or {tail_name}). This shouldn't happen.")
                 continue
            query = f"MATCH (h:{head_type} {{name: $head_name}}) MATCH (t:{tail_type} {{name: $tail_name}}) MERGE (h)-[r:{rel_type}]->(t)"
            params = {"head_name": head_name, "tail_name": tail_name}
            queries_with_params.append((query, params))
            print(f"  -> Relationship Query: ({head_name}:{head_type})-[{rel_type}]->({tail_name}:{tail_type})")

    print(f"INFO: Generated {len(queries_with_params)} total queries.")
    return queries_with_params

def create_or_update_node(
    book_title: str,
    node_name: str,
    node_type: str,
    section: int,
    attributes_to_set: Dict[str, Any],
    database: str = "neo4j"
) -> Optional[Dict[str, Any]]:
    """
    Finds a node by book_title and name, creating it if it doesn't exist.
    Initializes/updates properties including type, book_title, mentions, and others.

    Args:
        book_title: The book title associated with the node.
        node_name: The unique name of the node within the book context.
        node_type: The label (type) for the node (e.g., 'PERSON', 'LOCATION').
        section: The section number where this entity was mentioned/processed.
        attributes_to_set: Dictionary of other attributes to set/update on the node.
                           'mentions', 'name', 'book_title', 'type' will be ignored if present.
        database: The name of the database.

    Returns:
        The properties of the created/updated node, or None if an error occurred.
    """
    print(f"INFO: Creating/Updating node '{node_name}' (Type: {node_type}, Book: '{book_title}', Section: {section})")

    # Prepare properties, ensuring core ones aren't overwritten by accident
    props = attributes_to_set.copy()
    props.pop('mentions', None) # Handled specifically
    props.pop('name', None)     # Used in MERGE condition
    props.pop('book_title', None) # Used in MERGE condition
    props.pop('type', None)       # Handled by label

    # Use MERGE for node creation/finding
    # Use ON CREATE for initial setup
    # Use ON MATCH for updates
    # Note: Setting the label dynamically in MERGE can be tricky. Setting it in ON CREATE is safer.
    # We'll MERGE based on key properties and then ensure the label and type property.
    query = f"""
    MERGE (n {{name: $name, book_title: $book_title}})
    ON CREATE SET
        n:{node_type}, // Set label on create
        n.type = $node_type, // Set type property on create
        n.mentions = [$section], // Initialize mentions list on create
        n += $props // Add other attributes on create
    ON MATCH SET
        n.mentions = CASE // Append section if not present on match
                       WHEN $section IN coalesce(n.mentions, []) THEN n.mentions
                       ELSE coalesce(n.mentions, []) + $section
                     END,
        n += $props, // Merge other attributes on match
        // Ensure label and type property exist even if matched node somehow lacked them
        n:{node_type},
        n.type = $node_type
    RETURN properties(n) as properties
    """

    parameters = {
        "name": node_name,
        "book_title": book_title,
        "node_type": node_type,
        "section": section,
        "props": props
    }

    result_data = execute_query(query, parameters, database)

    if result_data is not None:
        if result_data:
            updated_properties = result_data[0].get("properties", {})
            # print(f"DEBUG: Node '{node_name}' processed. Properties: {updated_properties}") # Optional debug
            return updated_properties
        else:
            # Should not happen with MERGE unless query fails fundamentally
            print(f"ERROR: MERGE query for node '{node_name}' returned no result unexpectedly.")
            return None
    else:
        print(f"ERROR: Query failed during node create/update for '{node_name}'.")
        return None

# --- Relation Processing (Modified) ---

def process_relations_for_neo4j(
    relations_list: List[Dict[str, Any]],
    ner_dict: Dict[str, str],
    book_title: str, # Added
    section: int,    # Added
    database: str = "neo4j" # Added for calling update function
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Tuple[str, str]]]]:
    """
    Processes relations, creating/updating nodes inline via create_or_update_node,
    and collecting relationships to be created later.

    Args:
        relations_list: List of relation dictionaries.
        ner_dict: Dictionary mapping entity names to types.
        book_title: The title of the book being processed.
        section: The section number associated with these relations.
        database: The Neo4j database name.

    Returns:
        A tuple containing:
        1. entities: Dictionary storing basic info (name -> type) for relationship creation.
        2. relationships: Dictionary mapping relationship types to lists of (head, tail) tuples.
    """
    entities: Dict[str, Dict[str, Any]] = {} # Store name -> {type: type} for relationship matching
    relationships: Dict[str, List[Tuple[str, str]]] = {}
    attributes_to_update: Dict[str, Dict[str, Any]] = {} # Store node_name -> {attr: val}

    print(f"INFO: Processing {len(relations_list)} relations for book '{book_title}', section {section}...")

    # --- First Pass: Identify entities, collect attributes, build relationship list ---
    for relation in relations_list:
        head = relation.get("head")
        rel_type = relation.get("type")
        tail = relation.get("tail")

        if not all([head, rel_type, tail is not None]):
            print(f"WARN: Skipping relation due to missing basic data: {relation}")
            continue

        head_is_ner = head in ner_dict
        tail_is_ner = tail in ner_dict

        if head_is_ner:
            head_type = ner_dict[head]
            # Store entity type info
            if head not in entities: entities[head] = {"type": head_type}

            if tail_is_ner:
                tail_type = ner_dict[tail]
                # Store entity type info
                if tail not in entities: entities[tail] = {"type": tail_type}

                # Add to relationship list
                print(f"  -> Found Relationship: ({head}:{head_type})-[{rel_type}]->({tail}:{tail_type})")
                if rel_type not in relationships: relationships[rel_type] = []
                rel_tuple = (head, tail)
                if rel_tuple not in relationships[rel_type]:
                     relationships[rel_type].append(rel_tuple)
                # else: print(f"  -> (Skipping duplicate relationship: {rel_tuple} for type {rel_type})") # Less verbose

            else:
                # Collect attribute for the head entity
                print(f"  -> Found Attribute for '{head}' ({head_type}): {rel_type} = {tail}")
                if head not in attributes_to_update: attributes_to_update[head] = {}
                attributes_to_update[head][rel_type] = tail
        else:
             print(f"  -> Skipping relation: Head '{head}' not found in NER dictionary.")

    # --- Second Pass: Create/Update Nodes ---
    print(f"INFO: Creating/Updating {len(entities)} identified entities...")
    for node_name, entity_data in entities.items():
        node_type = entity_data["type"]
        # Get collected attributes for this node, or empty dict if none
        node_attributes = attributes_to_update.get(node_name, {})

        # Call the create/update function for each valid entity found
        create_or_update_node(
            book_title=book_title,
            node_name=node_name,
            node_type=node_type,
            section=section,
            attributes_to_set=node_attributes,
            database=database
        )

    print(f"INFO: Relation processing complete for section {section}.")
    return entities, relationships


# --- NEW Relationship Insertion Function ---

def insert_relationships(
    relationships: Dict[str, List[Tuple[str, str]]],
    entities: Dict[str, Dict[str, Any]], # Needed for types if not stored on nodes
    book_title: str,
    database: str = "neo4j"
) -> None:
    """
    Inserts relationships into Neo4j based on the processed data.

    Args:
        relationships: Dictionary mapping relationship types to lists of (head, tail) tuples.
        entities: Dictionary mapping entity names to their types (used for matching).
        book_title: The book title to match nodes within.
        database: The Neo4j database name.
    """
    total_rels_to_insert = sum(len(v) for v in relationships.values())
    if not total_rels_to_insert:
        print("INFO: No relationships to insert.")
        return

    print(f"INFO: Inserting {total_rels_to_insert} relationship instances...")
    inserted_count = 0
    failed_count = 0

    for rel_type, rel_list in relationships.items():
        for head_name, tail_name in rel_list:
            # We match nodes based on name and book_title
            query = f"""
            MATCH (h {{name: $head_name, book_title: $book_title}})
            MATCH (t {{name: $tail_name, book_title: $book_title}})
            MERGE (h)-[r:`{rel_type}`]->(t) // Use backticks for safety
            """
            params = {
                "head_name": head_name,
                "tail_name": tail_name,
                "book_title": book_title
            }
            # Execute query - we don't need the result data, just success/failure
            result = execute_query(query, params, database)
            if result is not None:
                inserted_count += 1
                # print(f"  -> Relationship inserted: ({head_name})-[`{rel_type}`]->({tail_name})") # Verbose
            else:
                failed_count += 1
                print(f"ERROR: Failed to insert relationship: ({head_name})-[`{rel_type}`]->({tail_name})")

    print(f"INFO: Relationship insertion complete. Inserted: {inserted_count}, Failed: {failed_count}")


def get_nodes_by_attribute(
    attribute_key: str,
    attribute_value: Any,
    database: str = "neo4j"
) -> Optional[List[Dict[str, Any]]]:
    """Retrieves nodes that have a specific attribute key-value pair."""
    # ... (implementation as before)
    print(f"INFO: Searching for nodes where '{attribute_key}' = {attribute_value}")
    query_return_props = """
    MATCH (n)
    WHERE n[$key] = $value
    RETURN properties(n) as properties
    """
    parameters = {"key": attribute_key, "value": attribute_value}
    result_data = execute_query(query_return_props, parameters, database)
    if result_data is not None:
        nodes = [record.get("properties", {}) for record in result_data]
        print(f"INFO: Found {len(nodes)} nodes matching the criteria.")
        return nodes
    else:
        print(f"ERROR: Query failed while getting nodes by attribute '{attribute_key}'.")
        return None

def get_node_by_book_and_name(
    book_title: str,
    node_name: str,
    database: str = "neo4j"
) -> Optional[Dict[str, Any]]:
    """Retrieves a single node based on its name and book_title properties."""
    # ... (implementation as before)
    print(f"INFO: Searching for node with name='{node_name}' and book_title='{book_title}'")
    query = """
    MATCH (n {name: $name, book_title: $book_title})
    RETURN properties(n) as properties
    LIMIT 1
    """
    parameters = {"name": node_name, "book_title": book_title}
    result_data = execute_query(query, parameters, database)
    if result_data is not None:
        if result_data:
            node_properties = result_data[0].get("properties", {})
            print(f"INFO: Found node: {node_properties}")
            return node_properties
        else:
            print(f"INFO: Node not found.")
            return None
    else:
        print(f"ERROR: Query failed while getting node by book and name.")
        return None

def get_graph_data_for_book(
    book_title: str,
    database: str = "neo4j"
) -> Optional[List[Dict[str, Any]]]:
    """
    Retrieves all nodes and their relationships within a specific book,
    ordered by mention count (descending) then name (ascending).

    Args:
        book_title: The title of the book to query.
        database: The name of the database.

    Returns:
        A list of dictionaries, each representing a node and its connections,
        ordered by importance (mention count), or None if an error occurs.
        Returns an empty list if the book has no nodes.
        Each dictionary contains:
        - 'node_properties': Properties of the central node.
        - 'mention_count': Size of the 'mentions' array.
        - 'outgoing_rels': List of {'rel_type', 'target_node_properties'}
        - 'incoming_rels': List of {'rel_type', 'source_node_properties'}
    """
    print(f"INFO: Retrieving graph data for book_title='{book_title}'")

    # This query fetches each node once, aggregates its relationships within the same book,
    # calculates mention count, and orders the results.
    query = """
    MATCH (n {book_title: $book_title}) // Find all nodes for the book
    // Calculate mention count safely (handles missing 'mentions' property)
    WITH n, size(coalesce(n.mentions, [])) as mention_count
    // Collect outgoing relationships TO nodes within the same book
    OPTIONAL MATCH (n)-[r]->(m {book_title: $book_title})
    WITH n, mention_count, collect(DISTINCT {rel_type: type(r), target_node_properties: properties(m)}) as outgoing_rels
    // Collect incoming relationships FROM nodes within the same book
    OPTIONAL MATCH (n)<-[l]-(k {book_title: $book_title})
    WITH n, mention_count, outgoing_rels, collect(DISTINCT {rel_type: type(l), source_node_properties: properties(k)}) as incoming_rels
    // Return combined data for each node
    RETURN
        properties(n) as node_properties,
        mention_count,
        // Filter out null relationships that can occur if a node has only incoming or outgoing
        [rel IN outgoing_rels WHERE rel.rel_type IS NOT NULL] as outgoing_rels,
        [rel IN incoming_rels WHERE rel.rel_type IS NOT NULL] as incoming_rels
    ORDER BY mention_count DESC, n.name ASC // Order by importance, then name
    """
    parameters = {"book_title": book_title}

    result_data = execute_query(query, parameters, database)

    if result_data is not None:
        print(f"INFO: Found {len(result_data)} nodes for book '{book_title}'.")
        # Post-process to ensure structure is as expected (optional but good practice)
        processed_results = []
        for record in result_data:
            processed_results.append({
                "node_properties": record.get("node_properties", {}),
                "mention_count": record.get("mention_count", 0),
                "outgoing_rels": record.get("outgoing_rels", []),
                "incoming_rels": record.get("incoming_rels", [])
            })
        return processed_results
    else:
        print(f"ERROR: Query failed while retrieving graph data for book '{book_title}'.")
        return None

