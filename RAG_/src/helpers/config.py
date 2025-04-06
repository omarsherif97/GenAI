from pydantic_settings import BaseSettings,SettingsConfigDict


class Settings(BaseSettings):
    
    APP_NAME:str
    APP_VERSION:str
    FILE_TYPE:list[str]
    FILE_SIZE:int
    FILE_MAX_CHUNK_SIZE:int

    MONGODB_URI:str
    MONGODB_DATABASE:str

    class config:
        env_file = ".env"
        env_file_encoding = "utf-8"


def get_settings():
    return Settings()

