from fastapi import FastAPI
from app.api.taxonomy_prediction import taxonomy_predictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/health')
async def index():
    return {"health": "hello this is taxonomy prediction API"}

app.include_router(taxonomy_predictor)