import json
import shutil
import subprocess
import tempfile
from typing import List, Optional
import os
from .core import Chunk, SourceTypes, print_status, count_tokens
from .thepipe import count_tokens
from PIL import Image

CTAGS_EXECUTABLE_PATH = "C:\ctags.exe" if os.name == 'nt' else "ctags-universal"
CTAGS_LANGUAGES = {'py': "Python", "cpp": "C++", "c": "C"}
CTAGS_OUTPUT_FILE = 'ctags_output.json'
MAX_COMPRESSION_ATTEMPTS = 25

def compress_with_ctags(chunk: Chunk, extension: str) -> Chunk:
    if chunk.text is None:
        return Chunk(path=chunk.path, text=chunk.text, image=chunk.image, source_type=SourceTypes.UNCOMPRESSIBLE_CODE)
    language = CTAGS_LANGUAGES[extension]
    tmp_dir = tempfile.mkdtemp()
    try:
        file_path = os.path.join(tmp_dir, "tempfile")+'.'+extension
        with open(file_path, 'w', encoding='utf-8') as tmp_file:
            tmp_file.write(chunk.text)
        # need custom options for ctags to work with typescript
        cmd = [
                CTAGS_EXECUTABLE_PATH,
                f"--languages={language}",
                "--output-format=json",
                "-f", "-",
                file_path
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Error running ctags: {result.stderr}")
        # write output to file
        with open(CTAGS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        # Process the JSON output
        ctag_matches = []
        for line in result.stdout.strip().splitlines():
            tag = json.loads(line)
            if 'pattern' in tag:
                pattern_without_regex = tag['pattern'][2:-2]
                ctag_matches.append(pattern_without_regex)
    finally:
        shutil.rmtree(tmp_dir)
    # remove the json file
    if os.path.exists(CTAGS_OUTPUT_FILE):
        os.remove(CTAGS_OUTPUT_FILE)
    ctags_skeleton = '\n'.join(ctag_matches)
    return Chunk(path=chunk.path, text=ctags_skeleton, image=chunk.image, source_type=SourceTypes.UNCOMPRESSIBLE_CODE)

def compress_with_llmlingua(chunk: Chunk) -> Chunk:
    # import only if needed
    from llmlingua import PromptCompressor
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    llm_lingua = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", use_llmlingua2=True, device_map=device)
    # Compress the text with llmlingua
    new_chunk_text = ""
    WINDOW_SIZE = 500
    for i in range(0, len(chunk.text), WINDOW_SIZE):
        window_text = chunk.text[i:i+WINDOW_SIZE]
        result = llm_lingua.compress_prompt(window_text, rate=0.5)
        new_window_text = result['compressed_prompt']
        new_chunk_text += new_window_text
    new_chunk = Chunk(path=chunk.path, text=new_chunk_text, image=chunk.image, source_type=chunk.source_type)
    return new_chunk

# uses https://platform.openai.com/docs/guides/vision
def calculate_image_tokens(image: Image.Image, detail: str = "auto") -> int:
    width, height = image.size
    if detail == "low":
        return 85
    elif detail == "high":
        # High detail calculation
        width, height = min(width, 2048), min(height, 2048)
        short_side = min(width, height)
        scale = 768 / short_side
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)
        tiles = (scaled_width // 512) * (scaled_height // 512)
        return 170 * tiles + 85
    else:  # auto
        if width <= 512 and height <= 512:
            return 85
        else:
            return calculate_image_tokens(image, detail="high")

def calculate_tokens(chunk: Chunk) -> int:
    text_tokens = len(chunk.text)/4 if chunk.text else 0
    image_tokens = calculate_image_tokens(chunk.image) if chunk.image else 0
    return max(text_tokens, image_tokens)

def compress_chunks(chunks: List[Chunk], verbose: bool = False, limit: Optional[int] = None) -> List[Chunk]:
    new_chunks = chunks
    for _ in range(min(MAX_COMPRESSION_ATTEMPTS, len(chunks))):
        if count_tokens(new_chunks) <= limit:
            break
        if verbose: print_status(f"Compressing prompt ({count_tokens(chunks)} tokens / {limit} limit)", status='info')
        new_chunks = []
        chunk_with_most_tokens = max(chunks, key=calculate_tokens)
        for chunk in chunks:
            # if not longest, skip
            if chunk != chunk_with_most_tokens:
                new_chunks.append(chunk)
                continue
            new_chunk = None
            if chunk is None or chunk.text is None:
                new_chunk = chunk
            elif chunk.source_type == SourceTypes.COMPRESSIBLE_CODE:
                extension = chunk.path.split('.')[-1]
                new_chunk = compress_with_ctags(chunk, extension=extension)
            elif chunk.source_type in {SourceTypes.PLAINTEXT, SourceTypes.PDF, SourceTypes.DOCX, SourceTypes.PPTX, SourceTypes.URL}:
                new_chunk = compress_with_llmlingua(chunk)
            else:
                # if the chunk is not compressible, keep the original text
                new_chunk = chunk
            if new_chunk.image is not None:
                # resize image to half its current res
                new_res = (new_chunk.image.width//2, new_chunk.image.height//2)
                new_chunk.image = new_chunk.image.resize(new_res)
            new_chunks.append(new_chunk)
    if count_tokens(new_chunks) > limit and verbose: 
        print_status("Failed to compress within limit, continuing", status='error')
    return new_chunks
