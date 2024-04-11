from typing import List, Dict, Optional
import argparse
import re
import os
from core import Chunk, print_status, count_tokens
import extractor
import compressor
import core

def save_outputs(chunks: List[Chunk], verbose: bool = False, text_only: bool = False) -> None:
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    # Save the chunks to files
    text = ""
    n_images = 0
    for i, chunk in enumerate(chunks):
        if chunk is None:
            continue
        if chunk.text is not None:
            text += f"""{chunk.path}:\n```\n{chunk.text}\n```\n\n"""
        if chunk.image is not None:
            clean_path = chunk.path.replace('/', '_').replace('\\', '_')
            clean_path = re.sub(r"[^a-zA-Z0-9 _]", "", clean_path)
            chunk.image.convert('RGB').save(f'outputs/{clean_path}_{i}.jpg')
            n_images += 1
    # Save the text
    with open('outputs/prompt.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    if verbose:
        print_status(f"Output {len(text)/4} tokens and {n_images} images to 'outputs'", status='success')

def extract(source: str, match: Optional[str] = None, ignore: Optional[str] = None, limit: int = 1e5, verbose: bool = False, ai_extraction: bool = False, text_only: bool = False) -> List[Dict]:
    chunks = extractor.extract_from_source(source=source, match=match, ignore=ignore, limit=limit, ai_extraction=ai_extraction, text_only=text_only, verbose=verbose)
    chunks = compressor.compress_chunks(chunks=chunks, verbose=verbose, limit=limit)
    final_prompt = core.create_messages_from_chunks(chunks)
    if verbose: print_status(f"Successfully created prompt ({count_tokens(chunks)} tokens)", status='success')
    return final_prompt

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compress project files into a context prompt.')
    parser.add_argument('source', type=str, help='The source file or directory to compress.')
    parser.add_argument('--match', type=str, default=None, help='The glob filename pattern to match in the directory. Glob notation, not regex. Only matches filenames, not paths.')
    parser.add_argument('--ignore', type=str, default=None, help='The regex filepath pattern to ignore in the directory. Regex notation, not glob. Matches filenames and paths.')
    parser.add_argument('--limit', type=float, default=1e5, help='The token limit for the compressed project context.')
    parser.add_argument('--ai_extraction', action='store_true', help='Use ai_extraction to extract text from images.')
    parser.add_argument('--text_only', action='store_true', help='Extract only text from the source.')
    parser.add_argument('--quiet', action='store_true', help='Do not print status messages.')
    args = parser.parse_args()
    verbose = not args.quiet
    args.verbose = verbose
    return args

if __name__ == '__main__':
    args = parse_arguments()
    chunks = extractor.extract_from_source(source=args.source, match=args.match, ignore=args.ignore, limit=args.limit, ai_extraction=args.ai_extraction, text_only=args.text_only, verbose=args.verbose)
    chunks = compressor.compress_chunks(chunks=chunks, verbose=args.verbose, limit=args.limit)
    save_outputs(chunks=chunks, verbose=args.verbose, text_only=args.text_only)
