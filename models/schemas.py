from pydantic import BaseModel, Field

class Quote(BaseModel):
    quote: str = Field(..., description="The exact quotable sentence or phrase.")
    context: str = Field(..., description="Brief context or explanation from the surrounding text.")

class Model(BaseModel):
    name: str = Field(..., description="The name of the model, framework, or methodology.")
    description: str = Field(..., description="Detailed description or explanation of the model.")

class Summary(BaseModel):
    key_points: list[str] = Field(..., description="List of key points from the chunk.")
    tone: str = Field(..., description="The overall tone or style of the chunk (e.g., philosophical, technical).")