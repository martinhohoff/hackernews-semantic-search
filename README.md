# Hacker News Semantic Search

Semantic search over recent Hacker News stories using OpenAI embeddings and Pinecone.

## Install

```bash
pip install openai pinecone requests python-dotenv
```

Or:

```bash
pip install -r requirements.txt
```

## Environment

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key

PINECONE_INDEX_NAME=hn-semantic-search
PINECONE_NAMESPACE=stories

OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5.4-mini

PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Usage

Ingest stories:

```bash
python hn_semantic.py ingest --max-stories 2000 --min-points 50 --months-back 12
```

Search:

```bash
python hn_semantic.py search "threads about AI replacing junior developers" --top-k 8
```

Search with a grounded answer:

```bash
python hn_semantic.py search "What does HN think about remote work burnout?" --top-k 8 --answer
```

## What It Does

- Pulls recent HN stories from Algolia using `search_by_date`, `tags=story`, and numeric filters for points and date.
- Turns each story into a semantic document using title, author, points, URL, and story text.
- Embeds those documents with `text-embedding-3-small` and stores them in Pinecone.
- Embeds the user's query, retrieves the nearest stories semantically, and optionally asks an LLM to summarize them.
- Supports Pinecone metadata filters at search time, so you can further restrict by points or recency.

## Notes

- This version indexes stories only, which keeps the first version simple and much smaller.
- If you want "what HN thinks" to get better, the next upgrade is adding selected comments as separate vectors.
- Algolia HN search is relevance-oriented and filterable, but not vector/semantic search, so this project adds a different retrieval layer on top.
