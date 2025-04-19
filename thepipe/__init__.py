import os
from .scraper import scrape_file, scrape_url, scrape_directory
from .core import parse_arguments, save_outputs


def main() -> None:
    args = parse_arguments()
    chunks = None
    if args.source.startswith("http"):
        chunks = scrape_url(
            args.source,
            ai_extraction=args.ai_extraction,
            verbose=args.verbose,
        )
    elif os.path.isdir(args.source):
        chunks = scrape_directory(
            args.source,
            include_regex=args.include_regex,
            ai_extraction=args.ai_extraction,
            verbose=args.verbose,
        )
    else:
        chunks = scrape_file(
            args.source,
            ai_extraction=args.ai_extraction,
            verbose=args.verbose,
        )
    save_outputs(chunks=chunks, verbose=args.verbose, text_only=args.text_only)


if __name__ == "__main__":
    main()
