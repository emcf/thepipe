from typing import List, Optional
import argparse
import os
import warnings
from .core import Chunk, calculate_tokens, chunks_to_messages, HOST_URL
from . import scraper
from . import chunker
import requests

IMAGE_DIR = "images"

def scrape(source: str, match: Optional[str] = None, ai_extraction: Optional[bool] = False, text_only: Optional[bool] = False, verbose: Optional[bool] = False, local: Optional[bool] = False) -> List[Chunk]:
    warnings.warn("This function is deprecated. Please use scrape instead", DeprecationWarning, stacklevel=2)
    chunks = None
    if source.startswith('http'):
        if local:
            chunks = scraper.scrape_url(url=source, match=match, ai_extraction=ai_extraction, text_only=text_only, verbose=verbose)
            messages = chunks_to_messages(chunks)
        else:
            response = requests.post(
                url=f"{HOST_URL}/scrape",
                data={'url': source, 'ai_extraction': ai_extraction, 'text_only': text_only}
            )
            response_json = response.json()
            if 'error' in response_json:
                raise ValueError(f"{response_json['error']}")
            messages = response_json['messages']
    else:
        if local:
            # if it's a file, return the file source type
            chunks = scraper.scrape_file(source=source, ai_extraction=ai_extraction, text_only=text_only, verbose=verbose)
            messages = chunks_to_messages(chunks)
        else:
            with open(source, 'rb') as f:
                response = requests.post(
                    url=f"{HOST_URL}/scrape",
                    files={'file': (source, f)},
                    data={'ai_extraction': ai_extraction, 'text_only': text_only}
                )
            response_json = response.json()
            if 'error' in response_json:
                raise ValueError(f"{response_json['error']}")
            messages = response_json['messages']
    return messages

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
        print(f"[thepipe] {calculate_tokens(chunks)} tokens saved to outputs folder")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compress project files into a context prompt.')
    parser.add_argument('source', type=str, help='The source file or directory to compress.')
    parser.add_argument('--include_regex', type=str, default=None, help='Regex pattern to match in a directory.')
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
        chunks = scraper.scrape_directory(dir_path=args.source, include_regex=args.include_regex, ai_extraction=args.ai_extraction, text_only=args.text_only, verbose=args.verbose)
    elif os.path.isdir(args.source):
        chunks = scraper.scrape_directory(dir_path=args.source, include_regex=args.include_regex, ai_extraction=args.ai_extraction, text_only=args.text_only, verbose=args.verbose)
    elif os.path.isfile(args.source):
        chunks = scraper.scrape_file(source=args.source, verbose=args.verbose, ai_extraction=args.ai_extraction, text_only=args.text_only)
    else:
        raise FileNotFoundError(f"Source must be a valid URL, file, or directory. Got: {args.source}")
    save_outputs(chunks=chunks, verbose=args.verbose, text_only=args.text_only)

if __name__ == '__main__':
    main()