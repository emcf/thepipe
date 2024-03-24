import json
import shutil
import subprocess
import tempfile
from typing import *
import os
from core import Chunk, SourceTypes
from llmlingua import PromptCompressor
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
llm_lingua = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", use_llmlingua2=True, device_map=device)
CTAGS_LANGUAGES = {'py': "Python", 'ts': "Typescript", "tsx": "Typescript", "cpp": "C++", "c": "C"}

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
        if language == "Typescript":
            cmd = [
                "./ctags",
                f"--languages={language}",
                "--output-format=json",
                f"--options=./ts.ctags",
                "-f", "-",
                file_path
            ]
        else:
            cmd = [
                "./ctags",
                f"--languages={language}",
                "--output-format=json",
                "-f", "-",
                file_path
            ]     
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Error running ctags: {result.stderr}")
        # write output to file
        with open('ctags_output.json', 'w', encoding='utf-8') as f:
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
    if os.path.exists('ctags_output.json'):
        os.remove('ctags_output.json')
    ctags_skeleton = '\n'.join(ctag_matches)
    return Chunk(path=chunk.path, text=ctags_skeleton, image=chunk.image, source_type=SourceTypes.UNCOMPRESSIBLE_CODE)

def compress_chunks(chunks: List[Chunk]) -> List[Chunk]:
    new_chunks = []
    for chunk in chunks:
        new_chunk = None
        if chunk is None or  chunk.text is None:
            new_chunk = chunk
        elif chunk.source_type == SourceTypes.COMPRESSIBLE_CODE:
            # compress certain compatible codes files with ctags
            extension = chunk.path.split('.')[-1]
            new_chunk = compress_with_ctags(chunk, extension=extension)
        elif chunk.source_type in {SourceTypes.PLAINTEXT, SourceTypes.PDF, SourceTypes.DOCX, SourceTypes.PPTX}:
            # Compress the text with llmlingua
            new_chunk_text = ""
            WINDOW_SIZE = 500
            for i in range(0, len(chunk.text), WINDOW_SIZE):
                window_text = chunk.text[i:i+WINDOW_SIZE]
                result = llm_lingua.compress_prompt(window_text, rate=0.5)
                new_window_text = result['compressed_prompt']
                new_chunk_text += new_window_text
            new_chunk = Chunk(path=chunk.path, text=new_chunk_text, image=chunk.image, source_type=chunk.source_type)
        elif chunk.source_type == SourceTypes.SPREADSHEET:
            # compress spreadsheets by returning a list of tuples: colname and coltype
            loaded_json = json.loads(chunk.text)
            row_one = loaded_json[0]
            colnames = []
            coltypes = []
            for key, value in row_one.items():
                colnames.append(key)
                coltypes.append(type(value))
            new_chunk_text = "Column names and types: " + str(list(zip(colnames, coltypes)))
            new_chunk = Chunk(path=chunk.path, text=new_chunk_text, image=chunk.image, source_type=chunk.source_type)
        else:
            # if the chunk is not compressible, keep the original text
            new_chunk = chunk
        new_chunks.append(new_chunk)
    return new_chunks
