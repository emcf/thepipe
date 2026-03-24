from __future__ import annotations

import argparse
import os
import warnings
from typing import Optional

from openai import OpenAI

from .scraper import scrape_directory, scrape_file, scrape_url
from .core import DEFAULT_AI_MODEL, save_outputs
from .provider import (
    PROVIDER_PRESETS,
    create_provider_client,
    detect_provider,
    get_provider_preset,
)


# Argument parsing
def parse_arguments() -> argparse.Namespace:  # noqa: D401 – imperative is fine here
    """
    Parse CLI flags.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="thepipe",
        description="Universal document/Web scraper with optional LLM extraction.",
    )

    # Required source (file, directory, or URL)
    parser.add_argument(
        "source",
        help="File path, directory, or URL to scrape.",
    )

    # Optional flags
    parser.add_argument(
        "-i",
        "--inclusion-pattern",
        dest="inclusion_pattern",
        default=None,
        help="Regex pattern – only files whose *full path* matches are scraped "
        "(applies to directory/zip scraping).",
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
        help="Suppress images – output only extracted text.",
    )

    # Provider selection
    available_providers = ", ".join(sorted(PROVIDER_PRESETS))
    parser.add_argument(
        "--provider",
        default=None,
        help=f"LLM provider to use ({available_providers}). "
        "Auto-detected from environment variables if omitted.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="API key for the selected provider. "
        "Falls back to the provider's environment variable (e.g. OPENAI_API_KEY, MINIMAX_API_KEY).",
    )

    # OpenAI-related flags (kept for backwards compatibility)
    parser.add_argument(
        "--openai-api-key",
        dest="openai_api_key",
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key.  If omitted, env variable OPENAI_API_KEY is used.",
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

    # Legacy flag (will be removed in future versions)
    parser.add_argument(
        "--ai-extraction",
        action="store_true",
        help=argparse.SUPPRESS,  # hidden but still accepted
    )

    return parser.parse_args()


# OpenAI client factory
def create_openai_client(
    *,
    api_key: Optional[str],
    base_url: str,
    enable_vlm: bool,
) -> Optional[OpenAI]:
    if api_key:
        # Normal path – user gave an explicit key
        return OpenAI(api_key=api_key, base_url=base_url)

    if enable_vlm:
        # Old flag: fall back to env vars
        warnings.warn(
            "--ai-extraction is deprecated; "
            "please use --openai-api-key and --openai-model "
            "(and optionally --openai-base-url) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY"))

    # AI extraction disabled
    return None


def main() -> None:
    """CLI entry point"""
    args = parse_arguments()

    # Determine model and client based on provider selection
    if args.provider or args.api_key:
        # New provider-based path
        provider_name = args.provider or detect_provider()
        preset = get_provider_preset(provider_name)
        client, _ = create_provider_client(
            provider=provider_name,
            api_key=args.api_key,
        )
        model = args.openai_model if args.openai_model != DEFAULT_AI_MODEL else preset.default_model
    else:
        # Legacy OpenAI-only path
        client = create_openai_client(
            api_key=args.openai_api_key,
            base_url=args.openai_base_url,
            enable_vlm=args.ai_extraction,
        )
        model = args.openai_model

    # Delegate scraping based on source type
    if args.source.startswith(("http://", "https://")):
        chunks = scrape_url(
            args.source,
            verbose=args.verbose,
            openai_client=client,
            model=model,
        )
    elif os.path.isdir(args.source):
        chunks = scrape_directory(
            dir_path=args.source,
            inclusion_pattern=args.inclusion_pattern,
            verbose=args.verbose,
            openai_client=client,
        )
    elif os.path.isfile(args.source):
        chunks = scrape_file(
            filepath=args.source,
            verbose=args.verbose,
            openai_client=client,
            model=model,
        )
    else:
        raise ValueError(f"Invalid source: {args.source}")

    # Persist results
    save_outputs(
        chunks=chunks,
        verbose=args.verbose,
        text_only=args.text_only,
        output_folder="thepipe_output",
    )

    if args.verbose:
        print(f"Scraping complete. Outputs saved to 'thepipe_output/'.")


# Entry-point shim
if __name__ == "__main__":
    main()
