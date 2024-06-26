from typing import List, Optional
import argparse
import os
from .core import Chunk, calculate_tokens
from . import scraper
from . import chunker

def extract(source: str, match: Optional[List[str]] = None, ignore: str = None, ai_extraction: bool = False, text_only: bool = False, verbose: bool = False, local: bool = False) -> List[Chunk]:
    raise DeprecationWarning("This function is deprecated. Please use scraper.scrape_file or scraper.scrape_url instead.")
    # if its a url, return the url source type
    if source.startswith('http'):
        return scraper.scrape_url(url=source, match=match, ignore=ignore, ai_extraction=ai_extraction, text_only=text_only, verbose=verbose)
    # if it's a directory, return the directory source type
    if os.path.isdir(source) or source in ('.', './'):
        if source in ('.', './'):
            source = os.getcwd()
        return scraper.scrape_directory(dir_path=source, include_regex=match, ignore_regex=ignore, ai_extraction=ai_extraction, text_only=text_only, verbose=verbose, local=False)
    # if it's a file, return the file source type
    return scraper.scrape_file(source=source, match=match, ignore=ignore, ai_extraction=ai_extraction, text_only=text_only, verbose=verbose, local=False)

def save_outputs(chunks: List[Chunk], verbose: bool = False, text_only: bool = False) -> None:
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    text = ""

    # Save the text and images to the outputs directory
    for i, chunk in enumerate(chunks):
        if chunk is None:
            continue
        if chunk.path is not None:
            text += f'{chunk.path}:\n'
        if chunk.texts:
            for chunk_text in chunk.texts:
                text += f'```\n{chunk_text}\n```\n'
        if chunk.images and not text_only:
            for j, image in enumerate(chunk.images):
                image.convert('RGB').save(f'outputs/{i}_{j}.jpg')

    # Save the text
    with open('outputs/prompt.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    
    if verbose:
        print(f"[thepipe] {calculate_tokens(chunks)} tokens saved to outputs folder", status='success')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compress project files into a context prompt.')
    parser.add_argument('source', type=str, help='The source file or directory to compress.')
    parser.add_argument('--include_regex', type=str, default=None, help='List of regex patterns to match in a directory.')
    parser.add_argument('--ignore_regex', type=str, default=None, help='List of regex patterns to ignore in a directory.')
    parser.add_argument('--ai_extraction', action='store_true', help='Use ai_extraction to extract text from images.')
    parser.add_argument('--text_only', action='store_true', help='Extract only text from the source.')
    parser.add_argument('--verbose', action='store_true', help='Print status messages.')
    args = parser.parse_args()
    return args

def main() -> None:
    args = parse_arguments()
    if args.source.startswith('http'):
        chunks = scraper.scrape_url(url=args.source, ai_extraction=args.ai_extraction, text_only=args.text_only)
    elif args.source == '.':
        args.source = os.getcwd()
        chunks = scraper.scrape_directory(dir_path=args.source, include_regex=args.include_regex, ignore_regex=args.ignore_regex, ai_extraction=args.ai_extraction, text_only=args.text_only, verbose=args.verbose)
    elif os.path.isdir(args.source):
        chunks = scraper.scrape_directory(dir_path=args.source, include_regex=args.include_regex, ignore_regex=args.ignore_regex, ai_extraction=args.ai_extraction, text_only=args.text_only, verbose=args.verbose)
    elif os.path.isfile(args.source):
        chunks = scraper.scrape_file(source=args.source, verbose=args.verbose, ai_extraction=args.ai_extraction, text_only=args.text_only)
    else:
        raise FileNotFoundError(f"Source must be a valid URL, file, or directory. Got: {args.source}")
    save_outputs(chunks=chunks, verbose=args.verbose, text_only=args.text_only)

if __name__ == '__main__':
    main()