from typing import List
from pydantic import BaseModel


class UploadResponse(BaseModel):
    student_code: str
    mean_embedding: List[float]
    stack_embeddings: List[List[float]]