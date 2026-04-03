# Hacker News Semantic Search

Semantic search over recent Hacker News stories using OpenAI embeddings and Pinecone.

## Install

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

## Scripts

- `hn_ingest.py`: fetches stories, selects useful comments, estimates ingest cost, asks for confirmation, and upserts to Pinecone
- `hn_search.py`: answers questions using retrieved stories and comments, always prints sources, and can optionally show raw match output
- `hn_costs.py`: standalone what-if cost calculator

## Usage

Ingest stories:

```bash
python hn_ingest.py --max-stories 2000 --min-points 50 --months-back 12
```

The ingest command prints an estimated OpenAI embedding cost and Pinecone write/storage footprint, then asks for confirmation before it upserts anything.

By default it indexes stories plus up to 3 selected comments per story. To keep a story-only index:

```bash
python hn_ingest.py --max-stories 2000 --min-points 50 --months-back 12 --no-comments
```

Skip the confirmation prompt:

```bash
python hn_ingest.py --max-stories 2000 --min-points 50 --months-back 12 --yes
```

Include Pinecone pricing in the estimate:

```bash
python hn_ingest.py \
  --max-stories 2000 \
  --min-points 50 \
  --months-back 12 \
  --pinecone-storage-price-per-gb-month YOUR_STORAGE_PRICE \
  --pinecone-write-price-per-million-wu YOUR_WRITE_PRICE
```

Search:

```bash
python hn_search.py "threads about AI replacing junior developers" --top-k 8
```

The default search output includes a detailed plain-text answer for the CLI, followed by a Sources section.

Search always retrieves the closest Hacker News stories/comments first. The answer model is instructed to answer only when the retrieved material is actually relevant to the query. Otherwise it returns a short “not relevant enough to answer” message and still prints the sources.

Search and also print raw retrieved matches:

```bash
python hn_search.py "What does HN think about remote work burnout?" --top-k 8 --raw-matches
```

Estimate OpenAI and Pinecone costs ahead of time:

```bash
python hn_costs.py --stories 2000 --queries-per-month 5000 --answer-rate 0.25
```

Estimate costs with Pinecone pricing plugged in:

```bash
python hn_costs.py \
  --stories 2000 \
  --queries-per-month 5000 \
  --answer-rate 0.25 \
  --pinecone-storage-price-per-gb-month YOUR_STORAGE_PRICE \
  --pinecone-read-price-per-million-ru YOUR_READ_PRICE \
  --pinecone-write-price-per-million-wu YOUR_WRITE_PRICE
```

## What It Does

- Pulls recent HN stories from Algolia using `search_by_date`, `tags=story`, and numeric filters for points and date.
- Pulls selected comment threads for each story from Algolia's item API and keeps a small number of longer comments per story.
- Turns both stories and selected comments into semantic documents with story context attached.
- Embeds those documents with `text-embedding-3-small` and stores them in Pinecone.
- Embeds the user's query, retrieves the nearest stories or comments semantically, and answers the question from those retrieved sources.
- Supports Pinecone metadata filters at search time, so you can further restrict by points or recency.

## Notes

- This version indexes both stories and a small number of selected comments per story.
- The comment selection is intentionally simple: longer comments are favored, with extra weight for points and replies when available.
- Low-quality and suspicious comments are filtered out before indexing, including obvious spam and prompt-injection-like text.
- Retrieved source text is truncated and treated as untrusted data before it is passed to the answer model.
- Search uses a lightweight relevance check before the final answer step, and falls back to sources-only behavior when the retrieved HN material is not relevant enough.
- Algolia HN search is relevance-oriented and filterable, but not vector/semantic search, so this project adds a different retrieval layer on top.
- `hn_ingest.py` estimates the cost of the pending ingest and asks for confirmation before embedding and upserting.
- `hn_costs.py` is still useful for what-if planning, and uses OpenAI's current public prices for `text-embedding-3-small` and `gpt-5.4-mini` by default while estimating Pinecone usage from vector size, read units, and write units.
- Pinecone pricing varies by plan and can change over time, so the script lets you pass the current storage, read, and write rates as CLI flags when you want dollar totals.
