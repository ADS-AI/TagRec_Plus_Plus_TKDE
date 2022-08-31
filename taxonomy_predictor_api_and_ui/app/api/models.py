from typing import List
from pydantic import BaseModel

class Taxonomies(BaseModel):
    taxonomy: str
    confidence: float

class LearningContent(BaseModel):
    content: str