from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    langsmith_api_key: str = Field(..., env="LANGSMITH_API_KEY")
    upload_directory: str = Field("/tmp/studybot", env="UPLOAD_DIRECTORY")
    neo4j_uri: str = Field("/tmp/studybot", env="NEO4J_URI")
    neo4j_username: str = Field("neo4j_username", env="NEO4J_USERNAME")
    neo4j_password: str = Field("neo4j_password", env="NEO4J_PASSWORD")
    inference_model: str = Field(..., env="INFERENCE_MODEL")
    embeddings_model: str = Field(..., env="EMBEDDINGS_MODEL")
    api_endpoint: str = Field(..., env="API_ENDPOINT")


settings = Settings()
