from fastapi import FastAPI
from dotenv import load_dotenv
from helpers.config import get_settings
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager

load_dotenv(".env")

from routes import base,data

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect to the database
    settings = get_settings()
    app.state.db_client = AsyncIOMotorClient(settings.MONGODB_URI)
    app.state.db = app.state.db_client[settings.MONGODB_DATABASE]
    yield
    # Shutdown: close connection (optional)
    app.state.db_client.close()

app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
app.include_router(data.data_router)




