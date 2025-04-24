# app/api/routes.py
from fastapi import APIRouter, HTTPException, utils
from app.db.data import TextInput 
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
        print(f"Received inference request with data: {data}")
        next_paragraph = utils.generate_paragraph(data.text)
        return {"generated_paragraph": next_paragraph}
    except Exception as e:
        utils.log_error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate paragraph.")
