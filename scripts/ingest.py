import argparse

from self_rag.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into vector database.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete collections before ingestion.",
    )
    parser.add_argument(
        "--use-hierarchy",
        action="store_true",
        default=True,
        help="Enable hierarchical parent-child chunking (default: True).",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Use flat ingestion (no parent-child hierarchy). Shorthand for --use-hierarchy=False.",
    )
    args = parser.parse_args()

    use_hierarchy = not args.flat  # --flat overrides --use-hierarchy
    run_ingestion(reset=args.reset, use_hierarchy=use_hierarchy)


if __name__ == "__main__":
    main()


# Examples:
# Hierarchical (default):  uv run python scripts/ingest.py --reset
# Flat:                    uv run python scripts/ingest.py --reset --flat
