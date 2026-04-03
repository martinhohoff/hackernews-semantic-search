import os
import time
import html
import argparse
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec


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

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY")


openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


def clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def to_unix_seconds(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def get_embedding_dimension() -> int:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=["dimension probe"],
    )
    return len(response.data[0].embedding)


def ensure_index(index_name: str):
    existing = pc.list_indexes().names()
    if index_name not in existing:
        dimension = get_embedding_dimension()
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
    return pc.Index(index_name)


def embed_texts(texts: List[str]) -> List[List[float]]:
    cleaned = [clean_text(t) or " " for t in texts]
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=cleaned,
    )
    return [item.embedding for item in response.data]


def fetch_hn_stories(
    max_stories: int = 2000,
    min_points: int = 50,
    months_back: int = 12,
    hits_per_page: int = 100,
    polite_sleep_s: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Fetch recent HN stories from Algolia, sorted by date, filtered by points.
    """
    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(days=30 * months_back)
    start_ts = to_unix_seconds(start_dt)

    stories: List[Dict[str, Any]] = []
    seen_ids = set()
    page = 0

    while len(stories) < max_stories:
        params = {
            "tags": "story",
            "hitsPerPage": hits_per_page,
            "page": page,
            "numericFilters": f"points>={min_points},created_at_i>={start_ts}",
        }

        response = requests.get(
            f"{HN_SEARCH_BASE}/search_by_date",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()

        hits = payload.get("hits", [])
        if not hits:
            break

        added_this_page = 0
        for hit in hits:
            object_id = str(hit.get("objectID", "")).strip()
            if not object_id or object_id in seen_ids:
                continue

            title = clean_text(hit.get("title") or hit.get("story_title"))
            story_text = clean_text(hit.get("story_text"))
            url = clean_text(hit.get("url"))
            author = clean_text(hit.get("author"))
            created_at = hit.get("created_at")
            created_at_i = hit.get("created_at_i")
            points = int(hit.get("points") or 0)
            num_comments = int(hit.get("num_comments") or 0)

            if not title and not story_text:
                continue

            seen_ids.add(object_id)
            stories.append(
                {
                    "id": f"story-{object_id}",
                    "object_id": object_id,
                    "title": title,
                    "story_text": story_text,
                    "url": url,
                    "author": author,
                    "created_at": created_at,
                    "created_at_i": created_at_i,
                    "points": points,
                    "num_comments": num_comments,
                }
            )
            added_this_page += 1

            if len(stories) >= max_stories:
                break

        print(
            f"Fetched page={page} | hits={len(hits)} | added={added_this_page} | total={len(stories)}"
        )

        page += 1
        if page >= payload.get("nbPages", 0):
            break

        time.sleep(polite_sleep_s)

    return stories[:max_stories]


def story_to_document(story: Dict[str, Any]) -> str:
    parts = [
        f"Title: {story.get('title', '')}",
        f"Author: {story.get('author', '')}",
        f"Points: {story.get('points', 0)}",
        f"Comments: {story.get('num_comments', 0)}",
        f"URL: {story.get('url', '')}",
        f"Story text: {story.get('story_text', '')}",
    ]
    return "\n".join(part for part in parts if clean_text(part))


def upsert_stories(
    index,
    stories: List[Dict[str, Any]],
    namespace: str = NAMESPACE,
    batch_size: int = 100,
) -> None:
    total = 0

    for start in range(0, len(stories), batch_size):
        batch = stories[start : start + batch_size]
        docs = [story_to_document(story) for story in batch]
        embeddings = embed_texts(docs)

        vectors = []
        for story, emb, doc in zip(batch, embeddings, docs):
            metadata = {
                "kind": "story",
                "object_id": story["object_id"],
                "title": story.get("title", ""),
                "url": story.get("url", ""),
                "author": story.get("author", ""),
                "points": int(story.get("points", 0)),
                "num_comments": int(story.get("num_comments", 0)),
                "created_at": story.get("created_at", ""),
                "created_at_i": int(story.get("created_at_i") or 0),
                "text": doc,
            }

            vectors.append(
                {
                    "id": story["id"],
                    "values": emb,
                    "metadata": metadata,
                }
            )

        index.upsert(vectors=vectors, namespace=namespace)
        total += len(vectors)
        print(f"Upserted {total}/{len(stories)}")

    print(f"Done. Upserted {total} stories.")


def build_filter(
    min_points: Optional[int] = None,
    months_back: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    clauses = []

    if min_points is not None:
        clauses.append({"points": {"$gte": int(min_points)}})

    if months_back is not None:
        start_dt = datetime.now(timezone.utc) - timedelta(days=30 * months_back)
        start_ts = to_unix_seconds(start_dt)
        clauses.append({"created_at_i": {"$gte": start_ts}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def semantic_search(
    index,
    query: str,
    namespace: str = NAMESPACE,
    top_k: int = 10,
    min_points: Optional[int] = None,
    months_back: Optional[int] = None,
) -> List[Dict[str, Any]]:
    query_embedding = embed_texts([query])[0]
    metadata_filter = build_filter(min_points=min_points, months_back=months_back)

    search_kwargs = {
        "namespace": namespace,
        "vector": query_embedding,
        "top_k": top_k,
        "include_metadata": True,
    }
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    results = index.query(**search_kwargs)

    matches = []
    for match in results.matches:
        md = match.metadata or {}
        matches.append(
            {
                "id": match.id,
                "score": float(match.score),
                "title": md.get("title", ""),
                "url": md.get("url", ""),
                "author": md.get("author", ""),
                "points": md.get("points", 0),
                "num_comments": md.get("num_comments", 0),
                "created_at": md.get("created_at", ""),
                "text": md.get("text", ""),
            }
        )

    return matches


def build_answer_prompt(query: str, matches: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for i, m in enumerate(matches, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {i}]",
                    f"Title: {m.get('title', '')}",
                    f"Author: {m.get('author', '')}",
                    f"Points: {m.get('points', 0)}",
                    f"Comments: {m.get('num_comments', 0)}",
                    f"Date: {m.get('created_at', '')}",
                    f"URL: {m.get('url', '')}",
                    f"Content: {m.get('text', '')}",
                ]
            )
        )

    joined_context = "\n\n---\n\n".join(context_blocks)

    return (
        "You are answering questions about Hacker News stories.\n"
        "Use only the provided sources.\n"
        "If the sources are insufficient, say so.\n"
        "Prefer concrete, source-grounded answers over broad claims.\n\n"
        f"User query:\n{query}\n\n"
        f"Retrieved sources:\n{joined_context}"
    )


def answer_with_llm(query: str, matches: List[Dict[str, Any]]) -> str:
    prompt = build_answer_prompt(query, matches)

    response = openai_client.responses.create(
        model=CHAT_MODEL,
        input=prompt,
    )

    return response.output_text.strip()


def run_ingest(args: argparse.Namespace) -> None:
    index = ensure_index(INDEX_NAME)

    stories = fetch_hn_stories(
        max_stories=args.max_stories,
        min_points=args.min_points,
        months_back=args.months_back,
        hits_per_page=args.hits_per_page,
        polite_sleep_s=args.sleep,
    )

    if not stories:
        print("No stories found.")
        return

    upsert_stories(
        index=index,
        stories=stories,
        namespace=NAMESPACE,
        batch_size=args.batch_size,
    )


def run_search(args: argparse.Namespace) -> None:
    index = ensure_index(INDEX_NAME)

    matches = semantic_search(
        index=index,
        query=args.query,
        namespace=NAMESPACE,
        top_k=args.top_k,
        min_points=args.filter_min_points,
        months_back=args.filter_months_back,
    )

    if not matches:
        print("No matches found.")
        return

    print("\nSemantic matches:\n")
    for i, m in enumerate(matches, start=1):
        print(f"{i}. {m['title']}")
        print(f"   score={m['score']:.4f} | points={m['points']} | comments={m['num_comments']}")
        print(f"   author={m['author']} | date={m['created_at']}")
        print(f"   url={m['url']}")
        print()

    if args.answer:
        answer = answer_with_llm(args.query, matches)
        print("\nAnswer:\n")
        print(answer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic Hacker News story search with Pinecone + OpenAI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Fetch HN stories and index them")
    ingest.add_argument("--max-stories", type=int, default=2000)
    ingest.add_argument("--min-points", type=int, default=50)
    ingest.add_argument("--months-back", type=int, default=12)
    ingest.add_argument("--hits-per-page", type=int, default=100)
    ingest.add_argument("--batch-size", type=int, default=100)
    ingest.add_argument("--sleep", type=float, default=0.25)

    search = subparsers.add_parser("search", help="Run semantic search")
    search.add_argument("query", type=str)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--filter-min-points", type=int, default=None)
    search.add_argument("--filter-months-back", type=int, default=None)
    search.add_argument("--answer", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "search":
        run_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
