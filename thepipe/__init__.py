from __future__ import annotations

import argparse
import os
import warnings
from typing import Optional

from openai import OpenAI

from .core import DEFAULT_AI_MODEL, save_outputs
from .scraper import scrape_directory, scrape_file, scrape_url


def parse_arguments() -> argparse.Namespace:  # noqa: D401
    """Parse CLI flags."""

    parser = argparse.ArgumentParser(
        prog="thepipe",
        description="Universal document/Web scraper with optional OpenAI extraction.",
    )

    parser.add_argument(
        "source",
        help="File path, directory, or URL to scrape.",
    )
    parser.add_argument(
        "-i",
        "--inclusion-pattern",
        dest="inclusion_pattern",
        default=None,
        help="Regex pattern - only files whose full path matches are scraped (applies to directory/zip scraping).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--text-only",
        dest="text_only",
        action="store_true",
        help="Suppress images - output only extracted text.",
    )
    parser.add_argument(
        "--allow-local-urls",
        dest="allow_local_urls",
        action="store_true",
        help="Allow scraping localhost and private-network HTTP(S) URLs. Disabled by default for security.",
    )
    parser.add_argument(
        "--openai-api-key",
        dest="openai_api_key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key. If omitted, env variable OPENAI_API_KEY is used.",
    )
    parser.add_argument(
        "--openai-base-url",
        dest="openai_base_url",
        default="https://api.openai.com/v1",
        help="Base URL for the OpenAI API (default: https://api.openai.com/v1).",
    )
    parser.add_argument(
        "--openai-model",
        dest="openai_model",
        default=DEFAULT_AI_MODEL,
        help=f"Chat/VLM model to use (default: {DEFAULT_AI_MODEL}).",
    )
    parser.add_argument(
        "--ai-extraction",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def create_openai_client(
    *,
    api_key: Optional[str],
    base_url: str,
    enable_vlm: bool,
) -> Optional[OpenAI]:
    if api_key:
        return OpenAI(api_key=api_key, base_url=base_url)

    if enable_vlm:
        warnings.warn(
            "--ai-extraction is deprecated; please use --openai-api-key and "
            "--openai-model (and optionally --openai-base-url) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY"))

    return None


def main() -> None:
    """CLI entry point."""

    args = parse_arguments()
    openai_client = create_openai_client(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url,
        enable_vlm=args.ai_extraction,
    )

    if args.source.startswith(("http://", "https://")):
        chunks = scrape_url(
            args.source,
            verbose=args.verbose,
            openai_client=openai_client,
            model=args.openai_model,
            allow_local_urls=args.allow_local_urls,
        )
    elif os.path.isdir(args.source):
        chunks = scrape_directory(
            dir_path=args.source,
            inclusion_pattern=args.inclusion_pattern,
            verbose=args.verbose,
            openai_client=openai_client,
        )
    elif os.path.isfile(args.source):
        chunks = scrape_file(
            filepath=args.source,
            verbose=args.verbose,
            openai_client=openai_client,
            model=args.openai_model,
        )
    else:
        raise ValueError(f"Invalid source: {args.source}")

    save_outputs(
        chunks=chunks,
        verbose=args.verbose,
        text_only=args.text_only,
        output_folder="thepipe_output",
    )

    if args.verbose:
        print("Scraping complete. Outputs saved to 'thepipe_output/'.")


if __name__ == "__main__":
    main()
