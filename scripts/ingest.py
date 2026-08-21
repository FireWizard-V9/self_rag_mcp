import argparse

from self_rag.ingestion.pipeline import run_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete this application's child and parent collections before ingestion.",
    )
    args = parser.parse_args()
    run_ingestion(reset=args.reset)


if __name__ == "__main__":
    main()


# Fresh rebuild: uv run python scripts/ingest.py --reset
