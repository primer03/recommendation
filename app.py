from fastapi import FastAPI
from model import init_db
from routes import recommend_router

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    await init_db()

app.include_router(recommend_router)