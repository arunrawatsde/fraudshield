"""All settings come from environment variables (or a .env file), so no keys live in code."""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "fraud_model.joblib"

load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str
    embedding_model: str
    embedding_dim: int
    pinecone_api_key: str
    pinecone_index: str
    pinecone_cloud: str
    pinecone_region: str
    langfuse_public_key: str
    langfuse_secret_key: str
    flag_threshold: float
    max_tool_rounds: int

    @property
    def use_pinecone(self) -> bool:
        return bool(self.pinecone_api_key and self.openai_api_key)

    @property
    def use_langfuse(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    env = os.getenv
    return Settings(
        openai_api_key=env("OPENAI_API_KEY", ""),
        openai_model=env("OPENAI_MODEL", "gpt-4.1-mini"),
        embedding_model=env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_dim=int(env("OPENAI_EMBEDDING_DIM", "1536")),
        pinecone_api_key=env("PINECONE_API_KEY", ""),
        pinecone_index=env("PINECONE_INDEX", "fraudshield-playbook"),
        pinecone_cloud=env("PINECONE_CLOUD", "aws"),
        pinecone_region=env("PINECONE_REGION", "us-east-1"),
        langfuse_public_key=env("LANGFUSE_PUBLIC_KEY", ""),
        langfuse_secret_key=env("LANGFUSE_SECRET_KEY", ""),
        flag_threshold=float(env("FLAG_THRESHOLD", "0.2")),
        max_tool_rounds=int(env("MAX_TOOL_ROUNDS", "6")),
    )
