import argparse
import math
from typing import Dict, Optional


DEFAULT_CHARS_PER_TOKEN = 4.0
DEFAULT_OPENAI_EMBEDDING_PRICE_PER_MILLION = 0.02
DEFAULT_OPENAI_CHAT_INPUT_PRICE_PER_MILLION = 0.75
DEFAULT_OPENAI_CHAT_OUTPUT_PRICE_PER_MILLION = 4.50
DEFAULT_EMBEDDING_DIMENSIONS = 1536
DEFAULT_METADATA_BYTES = 1200
DEFAULT_ID_BYTES = 24


def estimate_tokens_from_chars(char_count: float, chars_per_token: float) -> float:
    return max(char_count / chars_per_token, 0.0)


def monthly_storage_gb(
    stories: int,
    embedding_dimensions: int,
    avg_metadata_bytes: int,
    avg_id_bytes: int,
) -> float:
    bytes_per_record = avg_id_bytes + avg_metadata_bytes + (embedding_dimensions * 4)
    total_bytes = stories * bytes_per_record
    return total_bytes / (1024 ** 3)


def monthly_query_read_units(storage_gb: float, queries_per_month: int) -> float:
    ru_per_query = max(storage_gb, 0.25)
    return ru_per_query * queries_per_month


def monthly_write_units(
    stories: int,
    embedding_dimensions: int,
    avg_metadata_bytes: int,
    avg_id_bytes: int,
) -> float:
    bytes_per_record = avg_id_bytes + avg_metadata_bytes + (embedding_dimensions * 4)
    kb_per_record = bytes_per_record / 1000.0
    return stories * kb_per_record


def dollars_from_per_million(units: float, price_per_million: float) -> float:
    return (units / 1_000_000.0) * price_per_million


def calculate_cost_estimate(
    *,
    stories: int,
    avg_chars_per_story: float,
    queries_per_month: int,
    answer_rate: float,
    avg_query_chars: float,
    avg_answer_input_chars: float,
    avg_answer_output_chars: float,
    chars_per_token: float,
    embedding_dimensions: int,
    avg_metadata_bytes: int,
    avg_id_bytes: int,
    openai_embedding_price_per_million: float,
    openai_chat_input_price_per_million: float,
    openai_chat_output_price_per_million: float,
    pinecone_storage_price_per_gb_month: Optional[float],
    pinecone_read_price_per_million_ru: Optional[float],
    pinecone_write_price_per_million_wu: Optional[float],
) -> Dict[str, float]:
    ingest_chars = stories * avg_chars_per_story
    ingest_tokens = estimate_tokens_from_chars(ingest_chars, chars_per_token)
    embedding_cost = dollars_from_per_million(
        ingest_tokens,
        openai_embedding_price_per_million,
    )

    query_embedding_tokens = estimate_tokens_from_chars(
        queries_per_month * avg_query_chars,
        chars_per_token,
    )
    query_embedding_cost = dollars_from_per_million(
        query_embedding_tokens,
        openai_embedding_price_per_million,
    )

    answered_queries = queries_per_month * answer_rate
    answer_input_tokens = estimate_tokens_from_chars(
        answered_queries * avg_answer_input_chars,
        chars_per_token,
    )
    answer_output_tokens = estimate_tokens_from_chars(
        answered_queries * avg_answer_output_chars,
        chars_per_token,
    )
    answer_input_cost = dollars_from_per_million(
        answer_input_tokens,
        openai_chat_input_price_per_million,
    )
    answer_output_cost = dollars_from_per_million(
        answer_output_tokens,
        openai_chat_output_price_per_million,
    )

    storage_gb = monthly_storage_gb(
        stories=stories,
        embedding_dimensions=embedding_dimensions,
        avg_metadata_bytes=avg_metadata_bytes,
        avg_id_bytes=avg_id_bytes,
    )
    write_units = monthly_write_units(
        stories=stories,
        embedding_dimensions=embedding_dimensions,
        avg_metadata_bytes=avg_metadata_bytes,
        avg_id_bytes=avg_id_bytes,
    )
    read_units = monthly_query_read_units(
        storage_gb=storage_gb,
        queries_per_month=queries_per_month,
    )

    pinecone_storage_cost = None
    pinecone_write_cost = None
    pinecone_read_cost = None

    if pinecone_storage_price_per_gb_month is not None:
        pinecone_storage_cost = storage_gb * pinecone_storage_price_per_gb_month
    if pinecone_write_price_per_million_wu is not None:
        pinecone_write_cost = dollars_from_per_million(
            write_units,
            pinecone_write_price_per_million_wu,
        )
    if pinecone_read_price_per_million_ru is not None:
        pinecone_read_cost = dollars_from_per_million(
            read_units,
            pinecone_read_price_per_million_ru,
        )

    openai_monthly_total = query_embedding_cost + answer_input_cost + answer_output_cost
    openai_first_month_total = embedding_cost + openai_monthly_total

    result: Dict[str, float] = {
        "stories": float(stories),
        "queries_per_month": float(queries_per_month),
        "answered_queries": float(answered_queries),
        "answer_rate": float(answer_rate),
        "ingest_tokens": float(ingest_tokens),
        "embedding_cost": float(embedding_cost),
        "query_embedding_tokens": float(query_embedding_tokens),
        "query_embedding_cost": float(query_embedding_cost),
        "answer_input_tokens": float(answer_input_tokens),
        "answer_output_tokens": float(answer_output_tokens),
        "answer_input_cost": float(answer_input_cost),
        "answer_output_cost": float(answer_output_cost),
        "openai_monthly_total": float(openai_monthly_total),
        "openai_first_month_total": float(openai_first_month_total),
        "storage_gb": float(storage_gb),
        "write_units": float(write_units),
        "read_units": float(read_units),
    }
    if pinecone_storage_cost is not None:
        result["pinecone_storage_cost"] = float(pinecone_storage_cost)
    if pinecone_write_cost is not None:
        result["pinecone_write_cost"] = float(pinecone_write_cost)
    if pinecone_read_cost is not None:
        result["pinecone_read_cost"] = float(pinecone_read_cost)
    if (
        pinecone_storage_cost is not None
        and pinecone_write_cost is not None
        and pinecone_read_cost is not None
    ):
        result["pinecone_total"] = float(
            pinecone_storage_cost + pinecone_write_cost + pinecone_read_cost
        )
    return result


def print_cost_estimate(
    estimate: Dict[str, float],
    *,
    chars_per_token: float,
    include_monthly_sections: bool = True,
    pinecone_storage_price_per_gb_month: Optional[float] = None,
    pinecone_read_price_per_million_ru: Optional[float] = None,
    pinecone_write_price_per_million_wu: Optional[float] = None,
) -> None:
    print("HN Semantic Search Cost Estimate")
    print("=" * 32)
    print(f"Stories indexed: {int(round(estimate['stories'])):,}")

    if include_monthly_sections:
        print(f"Queries per month: {int(round(estimate['queries_per_month'])):,}")
        print(
            "Queries with LLM answers: "
            f"{estimate['answered_queries']:,.0f} ({estimate['answer_rate']:.0%})"
        )
    print()

    print("OpenAI")
    print(f"- One-time ingest embedding tokens: {math.ceil(estimate['ingest_tokens']):,}")
    print(f"- One-time ingest embedding cost: ${estimate['embedding_cost']:,.4f}")

    if include_monthly_sections:
        print(
            f"- Monthly query embedding tokens: {math.ceil(estimate['query_embedding_tokens']):,}"
        )
        print(
            f"- Monthly query embedding cost: ${estimate['query_embedding_cost']:,.4f}"
        )
        print(f"- Monthly answer input tokens: {math.ceil(estimate['answer_input_tokens']):,}")
        print(f"- Monthly answer output tokens: {math.ceil(estimate['answer_output_tokens']):,}")
        print(f"- Monthly answer input cost: ${estimate['answer_input_cost']:,.4f}")
        print(f"- Monthly answer output cost: ${estimate['answer_output_cost']:,.4f}")
        print(
            f"- Monthly OpenAI total (after ingest): ${estimate['openai_monthly_total']:,.4f}"
        )
        print(
            f"- First-month OpenAI total (with ingest): ${estimate['openai_first_month_total']:,.4f}"
        )
    print()

    print("Pinecone")
    print(f"- Estimated namespace size: {estimate['storage_gb']:,.4f} GB")
    if include_monthly_sections:
        print(f"- Estimated monthly write units: {math.ceil(estimate['write_units']):,}")
        print(f"- Estimated monthly read units: {math.ceil(estimate['read_units']):,}")
    else:
        print(f"- Estimated write units for this ingest: {math.ceil(estimate['write_units']):,}")

    if "pinecone_storage_cost" in estimate:
        label = "monthly storage cost"
        print(f"- Estimated Pinecone {label}: ${estimate['pinecone_storage_cost']:,.4f}")
    else:
        print("- Estimated Pinecone monthly storage cost: provide --pinecone-storage-price-per-gb-month")

    if "pinecone_write_cost" in estimate:
        prefix = "monthly write cost" if include_monthly_sections else "write cost for this ingest"
        print(f"- Estimated Pinecone {prefix}: ${estimate['pinecone_write_cost']:,.4f}")
    else:
        print("- Estimated Pinecone write cost: provide --pinecone-write-price-per-million-wu")

    if include_monthly_sections:
        if "pinecone_read_cost" in estimate:
            print(f"- Estimated Pinecone monthly read cost: ${estimate['pinecone_read_cost']:,.4f}")
        else:
            print("- Estimated Pinecone monthly read cost: provide --pinecone-read-price-per-million-ru")

        if "pinecone_total" in estimate:
            print(f"- Estimated monthly Pinecone total: ${estimate['pinecone_total']:,.4f}")

    print()
    print("Assumptions")
    print(f"- Token estimate uses {chars_per_token:g} characters per token.")
    print(f"- Pinecone query cost uses max(namespace_size_gb, 0.25) RUs per query.")
    print("- Pinecone write units are approximated as record_size_in_kilobytes per upserted record.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate OpenAI and Pinecone costs for the Hacker News semantic search project."
    )
    parser.add_argument("--stories", type=int, default=2000)
    parser.add_argument("--avg-chars-per-story", type=int, default=1800)
    parser.add_argument("--queries-per-month", type=int, default=5000)
    parser.add_argument("--answer-rate", type=float, default=0.25)
    parser.add_argument("--avg-query-chars", type=int, default=120)
    parser.add_argument("--avg-answer-input-chars", type=int, default=12000)
    parser.add_argument("--avg-answer-output-chars", type=int, default=1600)
    parser.add_argument("--chars-per-token", type=float, default=DEFAULT_CHARS_PER_TOKEN)
    parser.add_argument("--embedding-dimensions", type=int, default=DEFAULT_EMBEDDING_DIMENSIONS)
    parser.add_argument("--avg-metadata-bytes", type=int, default=DEFAULT_METADATA_BYTES)
    parser.add_argument("--avg-id-bytes", type=int, default=DEFAULT_ID_BYTES)
    parser.add_argument(
        "--openai-embedding-price-per-million",
        type=float,
        default=DEFAULT_OPENAI_EMBEDDING_PRICE_PER_MILLION,
    )
    parser.add_argument(
        "--openai-chat-input-price-per-million",
        type=float,
        default=DEFAULT_OPENAI_CHAT_INPUT_PRICE_PER_MILLION,
    )
    parser.add_argument(
        "--openai-chat-output-price-per-million",
        type=float,
        default=DEFAULT_OPENAI_CHAT_OUTPUT_PRICE_PER_MILLION,
    )
    parser.add_argument("--pinecone-storage-price-per-gb-month", type=float, default=None)
    parser.add_argument("--pinecone-read-price-per-million-ru", type=float, default=None)
    parser.add_argument("--pinecone-write-price-per-million-wu", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    estimate = calculate_cost_estimate(
        stories=args.stories,
        avg_chars_per_story=args.avg_chars_per_story,
        queries_per_month=args.queries_per_month,
        answer_rate=args.answer_rate,
        avg_query_chars=args.avg_query_chars,
        avg_answer_input_chars=args.avg_answer_input_chars,
        avg_answer_output_chars=args.avg_answer_output_chars,
        chars_per_token=args.chars_per_token,
        embedding_dimensions=args.embedding_dimensions,
        avg_metadata_bytes=args.avg_metadata_bytes,
        avg_id_bytes=args.avg_id_bytes,
        openai_embedding_price_per_million=args.openai_embedding_price_per_million,
        openai_chat_input_price_per_million=args.openai_chat_input_price_per_million,
        openai_chat_output_price_per_million=args.openai_chat_output_price_per_million,
        pinecone_storage_price_per_gb_month=args.pinecone_storage_price_per_gb_month,
        pinecone_read_price_per_million_ru=args.pinecone_read_price_per_million_ru,
        pinecone_write_price_per_million_wu=args.pinecone_write_price_per_million_wu,
    )
    print_cost_estimate(
        estimate,
        chars_per_token=args.chars_per_token,
        include_monthly_sections=True,
        pinecone_storage_price_per_gb_month=args.pinecone_storage_price_per_gb_month,
        pinecone_read_price_per_million_ru=args.pinecone_read_price_per_million_ru,
        pinecone_write_price_per_million_wu=args.pinecone_write_price_per_million_wu,
    )


if __name__ == "__main__":
    main()
