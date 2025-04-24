from typing import Optional
from pydantic import BaseModel

# This class likely represents a text input field in a Python application.
class TextInput(BaseModel):
    text: str
    book_title: Optional[str] = None
    book_section: Optional[str] = None