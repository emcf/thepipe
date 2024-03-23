from typing import *
import argparse
import base64
from io import BytesIO
import re
import os
from PIL import Image
from core import Chunk, SourceTypes
import extract
import core
import compress

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_messages_from_chunks(chunks: List[Chunk]) -> List[Dict]:
    messages = []
    for chunk in chunks:
        content = []
        if chunk.text:
            content.append({"type": "text", "text": f"""{chunk.path}:\n```\n{chunk.text}\n```\n"""})
        if chunk.image:
            base64_image = image_to_base64(chunk.image)
            content.append({"type": "text", "text": f"""{chunk.path} image:"""})
            content.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"})
        messages.append({"role": "system", "content": content})
    return messages

def count_tokens(chunks: List[Chunk]) -> int:
    return sum([(len(chunk.path)+len(chunk.text))/4 for chunk in chunks if chunk.text is not None])

def save_outputs(chunks: List[Chunk], verbose: bool = False, text_only: bool = False) -> None:
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    # Save the chunks to files
    text = ""
    for i, chunk in enumerate(chunks):
        if chunk is None:
            continue
        if chunk.text is not None:
            text += f"""{chunk.path}:\n```\n{chunk.text}\n```\n\n"""
        if chunk.image is not None:
            clean_path = chunk.path.replace('/', '_').replace('\\', '_')
            clean_path = re.sub(r"[^a-zA-Z0-9 _]", "", clean_path)
            chunk.image.save(f'outputs/{clean_path}_{i}.jpg')
    # Save the text
    with open(f'outputs/prompt.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    if verbose: core.print_status(f"Output {len(text)/4} tokens to 'outputs/prompt.txt'", status='success')

def create_prompt_from_source(source_string: str, match: Optional[str] = None, ignore: Optional[str] = None, limit: int = 1e5, verbose: bool = False, mathpix: bool = False, text_only: bool = False) -> List[Dict]:
    chunks = extract.extract_from_source(source_string=source_string, match=match, ignore=ignore, limit=limit, mathpix=mathpix, text_only=text_only, verbose=verbose)
    # If the prompt is too long, compress the chunks
    for cycles in range(3):
        if count_tokens(chunks) <= args.limit:
            break
        if verbose: core.print_status(f"Compressing prompt ({count_tokens(chunks)} tokens / {args.limit} limit)", status='info')
        chunks = compress.compress_chunks(chunks)
    final_prompt = create_messages_from_chunks(chunks)
    if verbose: core.print_status(f"Successfully created prompt ({count_tokens(chunks)} tokens)", status='success')
    return final_prompt

if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Compress project files into a context for NanoGPT.')
    parser.add_argument('source', type=str, help='The source file or directory to compress.')
    parser.add_argument('--match', type=str, default=None, help='The glob filename pattern to match in the directory. Glob notation, not regex. Only matches filenames, not paths.')
    parser.add_argument('--ignore', type=str, default=None, help='The regex filepath pattern to ignore in the directory. Regex notation, not glob. Matches filenames and paths.')
    parser.add_argument('--limit', type=float, default=1e5, help='The token limit for the compressed project context.')
    parser.add_argument('--mathpix', action='store_true', help='Use Mathpix to extract text from images.')
    parser.add_argument('--text_only', action='store_true', help='Extract only text from the source.')
    parser.add_argument('--quiet', action='store_true', help='Do not print status messages.')
    args = parser.parse_args()
    verbose = not args.quiet
    # Make context from source
    chunks = extract.extract_from_source(source_string=args.source, match=args.match, ignore=args.ignore, limit=args.limit, mathpix=args.mathpix, text_only=args.text_only, verbose=verbose)
    # If the prompt is too long, compress the chunks
    for cycles in range(3):
        if count_tokens(chunks) <= args.limit:
            break
        if verbose: core.print_status(f"Compressing prompt ({count_tokens(chunks)} tokens / {args.limit} limit)", status='info')
        chunks = compress.compress_chunks(chunks)
    if count_tokens(chunks) > args.limit:
        if verbose: core.print_status(f"Failed to compress prompt within limit, continuing", status='error')
    # Save the outputs
    save_outputs(chunks=chunks, verbose=verbose, text_only=args.text_only)
