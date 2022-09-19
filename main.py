import uvicorn
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from mlm import MaskedLanguageMasking

app = FastAPI(
    title="Masked Language Modeling",
    description="Masked Language Modeling API",
    version="1.0.0"
)

tags_metadata = [
    {
        "name": "MLM",
        "description": "MLM via Transformers",
    }
]

mlm_generator = MaskedLanguageMasking()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


class MLMResponse(BaseModel):
    unmaskedSequence: dict = Field(..., title="Predicted masked tokens")


@app.post("/mlm/", tags=["mlm"])
async def classify(text: str):

    # Run inference
    output = mlm_generator.generate_masked_tokens(text)

    return output
