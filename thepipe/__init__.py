import os
from .scraper import scrape_file, scrape_url, scrape_directory
from .chunker import chunk_by_document, chunk_by_page, chunk_by_section, chunk_semantic
from .core import Chunk, calculate_tokens, chunks_to_messages, parse_arguments, save_outputs

def main() -> None:
    args = parse_arguments()
    chunks = None
    if args.source.startswith("http"):
        chunks = scrape_url(args.source, local=args.local, verbose=args.verbose)
    elif os.path.isdir(args.source):
        chunks = scrape_directory(args.source, verbose=args.verbose)
    else:
        chunks = scrape_file(args.source, local=args.local, verbose=args.verbose)
    save_outputs(chunks=chunks, verbose=args.verbose, text_only=args.text_only)

if __name__ == "__main__":
    main()