# app/api/routes.py
from typing import List
from fastapi import APIRouter, HTTPException, utils
from app.db.data import StoryInput, TextInput 
import app.db as db
import app.utils as utils

# Create a router instance
router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    Returns a status message indicating the service is running.
    """
    return {"status": "ok", "message": "Service is running"}

@router.post("/generate-paragraph")
async def generate_paragraph(data: TextInput):
    try:
        utils.log_info(f"Received inference request with data: {data}")
        next_paragraph = utils.generate_paragraph(data.text)
        return {"generated_paragraph": next_paragraph}
    except Exception as e:
        utils.log_error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate paragraph.")

@router.post("/extract-entities")
async def extract_entities(data: TextInput):
    entities = utils.extract_entities(data.text)
    return entities

@router.post("/resolve-coreferences")
async def resolve_coreferences_route(data: TextInput):
    """Resolves coreferences in the provided text using FastCoref."""
    try:
        utils.log_info(f"Received coreference resolution request for text: '{data.text[:100]}...'")
        coref_map = utils.resolve_coreferences(data.text)
        utils.log_info(f"Coreference resolution successful. Found {len(coref_map)} mappings.")
        return {"coreference_map": coref_map}
    except Exception as e:
        utils.log_error(f"Coreference resolution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Coreference resolution failed.")


@router.post("/extract-relations")
async def extract_relations_route(data: TextInput):
    """
    Extracts subject-verb-object relations from the text.
    Performs entity recognition and coreference resolution internally first.
    """
    try:
        utils.log_info(f"Received relation extraction request for text: '{data.text[:100]}...'")
        # 1. Extract Entities
        entities = utils.extract_entities(data.text)
        if not entities:
             utils.log_warning("No entities found, relation extraction might yield limited results.")

        # 2. Resolve Coreferences
        coref_map = utils.resolve_coreferences(data.text)

        # 3. Extract Relations using text, coref_map, and entities
        relations = utils.extract_relations(data.text, coref_map)
        utils.log_info(f"Relation extraction successful. Found {len(relations)} relations.")
        processed_entities, processed_relationships = db.process_relations_for_neo4j(relations, utils.extract_entities(data.text), data.book_title, data.book_section)
        db.insert_relationships(processed_relationships, processed_entities, data.book_title)
        return {"relations": relations, "entities": processed_entities }
    except Exception as e:
        utils.log_error(f"Relation extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Relation extraction failed.")
    
@router.get("/get-graph-text")
async def get_graph_as_text(book_title: str):
    """
    Retrieves the knowledge graph for a specific book title from Neo4j
    and returns it as a formatted text string, ordered by node importance (mention count).
    """
    try:
        utils.log_info(f"Received request to get graph text for book: '{book_title}'")

        # 1. Fetch data from Neo4j using the db function
        graph_data = db.get_graph_data_for_book(book_title)

        # 2. Handle cases where data retrieval failed or book not found
        if graph_data is None:
            # This indicates an error during the query execution
            utils.log_error(f"Failed to retrieve graph data for book: '{book_title}' (DB query error).")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve graph data for book '{book_title}'.")
        if not graph_data:
            # This means the query ran fine, but no nodes were found for the book
            utils.log_info(f"No graph data found in database for book: '{book_title}'.")
            # Return 404 Not Found might be appropriate here
            raise HTTPException(status_code=404, detail=f"No graph data found for book title '{book_title}'.")

        # 3. Format the data into text using the utility function
        formatted_text = utils.format_graph_to_text(graph_data) # Use the utility function

        utils.log_info(f"Successfully retrieved and formatted graph text for book: '{book_title}'.")
        # 4. Return the formatted text
        # Return as plain text response
        # from fastapi.responses import PlainTextResponse
        # return PlainTextResponse(content=formatted_text)
        # Or return as JSON if preferred by the client
        return {"book_title": book_title, "graph_summary_text": formatted_text}

    except HTTPException as http_exc:
         raise http_exc # Re-raise FastAPI specific exceptions
    except Exception as e:
        utils.log_error(f"Error getting graph text for '{book_title}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate graph text.")
    
async def _process_and_store_text(text: str, book_title: str, chapter_number: int):
    """
    Helper function to extract entities/relations from text and store them in Neo4j.
    (Implementation provided above)
    """
    try:
        utils.log_info(f"Processing text for storage: Book='{book_title}', Chapter={chapter_number}")
        if not db.get_driver():
             utils.log_error("Database connection not available during text processing."); return False
        entities_in_text_list = utils.extract_entities(text)
        ner_dict_from_text = {entity: "Unknown" for entity in entities_in_text_list}
        utils.log_info(f"Extracted entities for chapter {chapter_number}: {list(ner_dict_from_text.keys())}")
        if not ner_dict_from_text: utils.log_warning(f"No entities found by NER for chapter {chapter_number}.")
        coref_map = utils.resolve_coreferences(text)
        utils.log_info(f"Coreference map generated for chapter {chapter_number}.")
        relations = utils.extract_relations(text, coref_map)
        utils.log_info(f"Raw relations extracted for chapter {chapter_number}: {len(relations)}")
        processed_entities, processed_relationships = db.process_relations_for_neo4j(
            relations, ner_dict_from_text, book_title, chapter_number
        )
        if processed_relationships: db.insert_relationships(processed_relationships, processed_entities, book_title)
        else: utils.log_info(f"No relationships to insert for chapter {chapter_number}.")
        utils.log_info(f"Successfully processed and stored relations for chapter {chapter_number}.")
        return True
    except Exception as e:
        utils.log_error(f"Error processing/storing text for chapter {chapter_number}: {e}", exc_info=True)
        if not db.get_driver(): utils.log_error("Database connection lost during processing.")
        return False
    
@router.post("/generate-story-simple", response_model=dict)
async def generate_story_simple_route(data: StoryInput):
    """
    Generates a story with a specified number of chapters sequentially.
    Starts with an initial prompt, then uses the previously generated chapter text
    as the prompt for the next one. Does NOT interact with the Neo4j graph.
    """
    try:
        utils.log_info(f"Received SIMPLE story generation request for book '{data.book_title}' ({data.num_chapters} chapters)")

        all_chapters_text: List[str] = []
        current_prompt = data.text
        last_successful_chapter = 0

        for chapter_num in range(1, data.num_chapters + 1):
            utils.log_info(f"--- Generating Chapter {chapter_num}/{data.num_chapters} (Simple) for '{data.book_title}' ---")
            utils.log_info(f"Using prompt: '{current_prompt[:100]}...'")

            # --- Generate Text ---
            try:
                # Use the generate_paragraph utility function
                generated_text = utils.generate_paragraph(current_prompt)
                if not generated_text:
                    utils.log_error(f"Simple text generation failed or returned empty for chapter {chapter_num}.")
                    raise HTTPException(status_code=500, detail=f"Text generation failed for chapter {chapter_num}.")
                utils.log_info(f"Generated text for chapter {chapter_num}: '{generated_text[:100]}...'")
            except Exception as gen_e:
                 utils.log_error(f"Exception during simple text generation for chapter {chapter_num}: {gen_e}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Text generation failed for chapter {chapter_num}.")

            # --- Prepare for Next Iteration (No Storage) ---
            all_chapters_text.append(generated_text)
            current_prompt = generated_text # Use generated text as the next prompt
            last_successful_chapter = chapter_num

        # --- Combine and Return Full Story ---
        full_story = "\n\n".join(all_chapters_text)
        utils.log_info(f"Successfully generated {last_successful_chapter} chapters (Simple) for book '{data.book_title}'.")

        return {
            "book_title": data.book_title,
            "generation_type": "simple",
            "num_chapters_generated": last_successful_chapter,
            "full_story_text": full_story
        }

    except HTTPException as http_exc:
        utils.log_error(f"HTTPException during simple story generation: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        utils.log_error(f"Unexpected error in /generate-story-simple route: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during simple story generation.")

@router.post("/generate-story-graph", response_model=dict)
async def generate_story_graph_enhanced_route(data: StoryInput):
    """
    Generates a story with a specified number of chapters sequentially.
    For each chapter after the first, it incorporates a summary of the current
    knowledge graph state into the prompt. Extracts and stores relations
    from each generated chapter into Neo4j.
    """
    try:
        utils.log_info(f"Received GRAPH-ENHANCED story generation request for book '{data.book_title}' ({data.num_chapters} chapters)")

        all_chapters_text: List[str] = []
        gen_text = data.text # This holds the text part of the prompt
        last_successful_chapter = 0

        for chapter_num in range(1, data.num_chapters + 1):
            utils.log_info(f"--- Generating Chapter {chapter_num}/{data.num_chapters} (Graph-Enhanced) for '{data.book_title}' ---")

            # --- Prepare Prompt ---
            graph_summary = ""
            if chapter_num > 1:
                # Fetch and format graph data for chapters after the first
                utils.log_info(f"Retrieving graph summary for book '{data.book_title}' before generating chapter {chapter_num}")
                graph_data = db.get_graph_data_for_book(data.book_title)
                if graph_data is None:
                    # Error fetching graph - stop generation
                    utils.log_error(f"Failed to retrieve graph data for prompt generation (chapter {chapter_num}).")
                    raise HTTPException(status_code=500, detail=f"Failed to retrieve graph data before generating chapter {chapter_num}.")
                elif not graph_data:
                    utils.log_info(f"No graph data found for book '{data.book_title}' yet.")
                    graph_summary = "Current Knowledge Graph: (Empty)"
                else:
                    # Format the graph data into text
                    # Consider summarizing or selecting key info if format_graph_to_text is too verbose
                    graph_summary = utils.format_graph_to_text(graph_data)
                    graph_summary = f"Current Knowledge Graph Summary:\n{graph_summary[:200]}\n---" # Add context marker

            # Combine previous text and graph summary (if available) for the prompt
            combined_prompt = f"""
                You are a author who write books. Here is a graphical summary of the current situation in the book:
                {graph_summary[:200]}.
                I want you to write the next part of your book and continue:
                {gen_text}
                """
            utils.log_info(f"Using combined prompt for chapter {chapter_num}: '{combined_prompt[:150]}...'")

            # --- 1. Generate Text for Current Chapter ---
            try:
                generated_text = utils.generate_paragraph(combined_prompt)
                if not generated_text:
                    utils.log_error(f"Graph-enhanced text generation failed or returned empty for chapter {chapter_num}.")
                    raise HTTPException(status_code=500, detail=f"Text generation failed for chapter {chapter_num}.")
                utils.log_info(f"Generated text for chapter {chapter_num}: '{generated_text[:100]}...'")
            except Exception as gen_e:
                 utils.log_error(f"Exception during graph-enhanced text generation for chapter {chapter_num}: {gen_e}", exc_info=True)
                 raise HTTPException(status_code=500, detail=f"Text generation failed for chapter {chapter_num}.")

            # --- 2. Process and Store Generated Text ---
            storage_success = await _process_and_store_text(generated_text, data.book_title, chapter_num)

            if not storage_success:
                # Stop if storage fails, as the graph state is now inconsistent
                status_code = 503 if not db.get_driver() else 500
                raise HTTPException(status_code=status_code, detail=f"Generated text for chapter {chapter_num}, but failed to process/store relations. Stopping generation.")

            # --- 3. Prepare for Next Iteration ---
            all_chapters_text.append(generated_text)
            last_successful_chapter = chapter_num

        # --- 4. Combine and Return Full Story ---
        full_story = "\n\n".join(all_chapters_text)
        utils.log_info(f"Successfully generated {last_successful_chapter} chapters (Graph-Enhanced) for book '{data.book_title}'.")

        return {
            "book_title": data.book_title,
            "generation_type": "graph-enhanced",
            "num_chapters_generated": last_successful_chapter,
            "full_story_text": full_story
        }

    except HTTPException as http_exc:
        utils.log_error(f"HTTPException during graph-enhanced story generation: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        utils.log_error(f"Unexpected error in /generate-story-graph-enhanced route: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred during graph-enhanced story generation.")

@router.post("/generate-graph-prompt", response_model=dict)
async def generate_graph_prompt_route(data: TextInput):
    """
    Processes the input text (storing its relations in the graph),
    retrieves the current graph state for the book, and formats a prompt
    combining the graph summary and the input text for external generation.
    """
    try:
        utils.log_info(f"Received request to generate graph-enhanced prompt for book '{data.book_title}', chapter {data.book_section}")

        # --- 1. Process and Store Input Text ---
        # Use the provided text and chapter number to update the graph first
        storage_success = await _process_and_store_text(data.text, data.book_title, data.book_section)

        if not storage_success:
            # If storage fails, we cannot reliably generate the prompt based on the graph
            status_code = 503 if not db.get_driver() else 500
            raise HTTPException(status_code=status_code, detail="Failed to process or store input text relations, cannot generate graph-enhanced prompt.")

        utils.log_info(f"Input text processed and stored for chapter {data.book_section}.")

        # --- 2. Retrieve Updated Graph State ---
        utils.log_info(f"Retrieving updated graph summary for book '{data.book_title}'")
        graph_data = db.get_graph_data_for_book(data.book_title)
        graph_summary_text = "Current Knowledge Graph: (Empty)" # Default if retrieval fails or empty

        if graph_data is None:
            # Log error but maybe proceed with empty graph summary? Or fail? Let's proceed.
            utils.log_error(f"Failed to retrieve graph data after update for book '{data.book_title}'. Proceeding with empty summary.")
        elif not graph_data:
            utils.log_info(f"No graph data found for book '{data.book_title}' after update.")
        else:
            # Format the graph data into text
            # Ensure format_graph_to_text utility exists and is imported/defined
            graph_summary_text = utils.format_graph_to_text(graph_data)
            utils.log_info(f"Successfully retrieved and formatted graph summary.")

        # --- 3. Construct the Prompt ---
        # Using the specific format requested by the user
        prompt = f"you are a author. here is a knowledge graph representing your current world: {graph_summary_text}. please continue the story from where you last left off: {data.text}"

        utils.log_info(f"Generated graph-enhanced prompt for book '{data.book_title}'.")

        # --- 4. Return the Prompt ---
        return {
            "book_title": data.book_title,
            "chapter_number_processed": data.book_section,
            "generated_prompt": prompt
        }

    except HTTPException as http_exc:
        # Re-raise FastAPI specific exceptions
        raise http_exc
    except Exception as e:
        utils.log_error(f"Unexpected error in /generate-graph-prompt route: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred while generating the graph-enhanced prompt.")

