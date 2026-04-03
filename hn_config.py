import os

from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hn-semantic-search")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "stories")

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.4-mini")

PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

HN_SEARCH_BASE = "https://hn.algolia.com/api/v1"


def validate_required_env() -> None:
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY")
    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY")
