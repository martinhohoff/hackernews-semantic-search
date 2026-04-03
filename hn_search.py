import argparse

from hn_clients import ensure_index
from hn_config import INDEX_NAME, NAMESPACE
from hn_story_index import (
    answer_with_llm,
    print_sources,
    print_semantic_matches,
    semantic_search,
    should_answer_query,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer questions over indexed Hacker News stories and comments."
    )
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--filter-min-points", type=int, default=None)
    parser.add_argument("--filter-months-back", type=int, default=None)
    parser.add_argument(
        "--raw-matches",
        action="store_true",
        help="Print retrieved matches in addition to the answer",
    )
    return parser


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

    if should_answer_query(args.query, matches):
        answer = answer_with_llm(args.query, matches)
    else:
        answer = "The retrieved Hacker News material is not relevant enough to answer that query."
    print("\nAnswer:\n")
    print(answer)
    print_sources(matches)

    if args.raw_matches:
        print_semantic_matches(matches)


def main() -> None:
    args = build_parser().parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
