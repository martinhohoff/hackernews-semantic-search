import argparse

from hn_clients import ensure_index, get_embedding_dimension
from hn_config import INDEX_NAME, NAMESPACE
from hn_costs import (
    DEFAULT_ID_BYTES,
    DEFAULT_METADATA_BYTES,
    DEFAULT_OPENAI_EMBEDDING_PRICE_PER_MILLION,
    calculate_cost_estimate,
    print_cost_estimate,
)
from hn_story_index import average_story_chars, fetch_hn_stories, upsert_stories


def confirm_continue() -> bool:
    answer = input("\nProceed with embedding + Pinecone upsert? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Hacker News stories, estimate ingest cost, and index them in Pinecone."
    )
    parser.add_argument("--max-stories", type=int, default=2000)
    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--months-back", type=int, default=12)
    parser.add_argument("--hits-per-page", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt")
    parser.add_argument("--chars-per-token", type=float, default=4.0)
    parser.add_argument(
        "--openai-embedding-price-per-million",
        type=float,
        default=DEFAULT_OPENAI_EMBEDDING_PRICE_PER_MILLION,
    )
    parser.add_argument("--pinecone-storage-price-per-gb-month", type=float, default=None)
    parser.add_argument("--pinecone-write-price-per-million-wu", type=float, default=None)
    parser.add_argument("--avg-metadata-bytes", type=int, default=DEFAULT_METADATA_BYTES)
    parser.add_argument("--avg-id-bytes", type=int, default=DEFAULT_ID_BYTES)
    return parser


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

    estimate = calculate_cost_estimate(
        stories=len(stories),
        avg_chars_per_story=average_story_chars(stories),
        queries_per_month=0,
        answer_rate=0.0,
        avg_query_chars=0,
        avg_answer_input_chars=0,
        avg_answer_output_chars=0,
        chars_per_token=args.chars_per_token,
        embedding_dimensions=get_embedding_dimension(),
        avg_metadata_bytes=args.avg_metadata_bytes,
        avg_id_bytes=args.avg_id_bytes,
        openai_embedding_price_per_million=args.openai_embedding_price_per_million,
        openai_chat_input_price_per_million=0.0,
        openai_chat_output_price_per_million=0.0,
        pinecone_storage_price_per_gb_month=args.pinecone_storage_price_per_gb_month,
        pinecone_read_price_per_million_ru=None,
        pinecone_write_price_per_million_wu=args.pinecone_write_price_per_million_wu,
    )
    print()
    print_cost_estimate(
        estimate,
        chars_per_token=args.chars_per_token,
        include_monthly_sections=False,
    )

    if not args.yes and not confirm_continue():
        print("Cancelled before embedding and upsert.")
        return

    upsert_stories(
        index=index,
        stories=stories,
        namespace=NAMESPACE,
        batch_size=args.batch_size,
    )


def main() -> None:
    args = build_parser().parse_args()
    run_ingest(args)


if __name__ == "__main__":
    main()
