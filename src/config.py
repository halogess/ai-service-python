import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DB_HOST: str = "mysql"
    DB_PORT: int = 3306
    DB_NAME: str = "db_korektor_buku"
    DB_USER: str
    DB_PASSWORD: str
    BASE_PATH: str = "/data/cek-ta"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

_settings = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings