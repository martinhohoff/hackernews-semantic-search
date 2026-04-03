from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from hn_config import (
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_REGION,
    validate_required_env,
)


validate_required_env()

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


def get_embedding_dimension() -> int:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=["dimension probe"],
    )
    return len(response.data[0].embedding)


def ensure_index(index_name: str):
    existing = pinecone_client.list_indexes().names()
    if index_name not in existing:
        dimension = get_embedding_dimension()
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
    return pinecone_client.Index(index_name)
