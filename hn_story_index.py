import html
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from hn_clients import openai_client
from hn_config import CHAT_MODEL, EMBEDDING_MODEL, HN_SEARCH_BASE, NAMESPACE


def clean_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split()).strip()


def to_unix_seconds(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


def embed_texts(texts: List[str]) -> List[List[float]]:
    cleaned = [clean_text(text) or " " for text in texts]
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


def average_story_chars(stories: List[Dict[str, Any]]) -> float:
    docs = [story_to_document(story) for story in stories]
    if not docs:
        return 0.0
    return sum(len(doc) for doc in docs) / len(docs)


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
        for story, embedding, doc in zip(batch, embeddings, docs):
            vectors.append(
                {
                    "id": story["id"],
                    "values": embedding,
                    "metadata": {
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
                    },
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
        metadata = match.metadata or {}
        matches.append(
            {
                "id": match.id,
                "score": float(match.score),
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "author": metadata.get("author", ""),
                "points": metadata.get("points", 0),
                "num_comments": metadata.get("num_comments", 0),
                "created_at": metadata.get("created_at", ""),
                "text": metadata.get("text", ""),
            }
        )

    return matches


def print_semantic_matches(matches: List[Dict[str, Any]]) -> None:
    print("\nSemantic matches:\n")
    for i, match in enumerate(matches, start=1):
        print(f"{i}. {match['title']}")
        print(
            f"   score={match['score']:.4f} | points={match['points']} | comments={match['num_comments']}"
        )
        print(f"   author={match['author']} | date={match['created_at']}")
        print(f"   url={match['url']}")
        print()


def build_answer_prompt(query: str, matches: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for i, match in enumerate(matches, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {i}]",
                    f"Title: {match.get('title', '')}",
                    f"Author: {match.get('author', '')}",
                    f"Points: {match.get('points', 0)}",
                    f"Comments: {match.get('num_comments', 0)}",
                    f"Date: {match.get('created_at', '')}",
                    f"URL: {match.get('url', '')}",
                    f"Content: {match.get('text', '')}",
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
