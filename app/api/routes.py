# app/api/routes.py
from fastapi import APIRouter, HTTPException, utils

from app.db.models import TextInput
import app.utils.process_text as utils
import app.utils.inference as inf
import os

# Create a router instance
router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    Returns a status message indicating the service is running.
    """
    return {"status": "ok", "message": "Service is running"}

@router.post("/extract-graph")
async def process_and_ingest_text(data: TextInput):
    """
    Receives text, processes it to extract entities and relationships,
    and inserts/updates them into the Neo4j database.
    """
    if not utils.NLP or not utils.FC_MODEL:
        raise HTTPException(status_code=503, detail="NLP models are not available.")
    if not utils.NEO4J_DRIVER:
         raise HTTPException(status_code=503, detail="Database connection is not available.")

    try:
        success = utils.process_text_and_insert_graph(
            text=data.text,
            book_title=data.book_title,
            book_section=data.book_section
            # Uses the global driver setup during startup
        )

        if success:
            return {"message": "Text processed and graph updated successfully (or no updates needed)."}
        else:
            # Log the error on the server side via utils.py logging
            raise HTTPException(status_code=500, detail="Failed to process text or update graph. Check server logs.")

    except Exception as e:
        # Log the exception details on the server
        utils.log.exception(f"Unhandled error during /process-text/ endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
    
    
 # Adjust path as needed

@router.post("/generate-paragraph")
async def inference_paragraph(data: TextInput):
    from app.utils.inference import generate_next_paragraph, load_model

    tokenizer, model, device = load_model("mrcedric98/falcon-rw-1b-finetuned") 
    
    try:
        print(f"Received inference request with data: {data}")
        next_paragraph = generate_next_paragraph(data.text, tokenizer, model, device)
        print("cedced_",data.text)
        return {"generated_paragraph": next_paragraph}
    except Exception as e:
        utils.log.exception(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate paragraph.")
    

