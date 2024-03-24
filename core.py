from typing import *
from PIL import Image
from colorama import Style, Fore
from enum import Enum

class SourceTypes(Enum):
    DIR = "directory"
    UNCOMPRESSIBLE_CODE = "code (cannot compress with ctags)"
    COMPRESSIBLE_CODE = "code (can compress with ctags)"
    PLAINTEXT = "plaintext"
    PDF = "pdf"
    IMAGE = "image"
    SPREADSHEET = "spreadsheet"
    IPYNB = "ipynb"
    DOCX = "docx"
    PPTX = "pptx"
    URL = "website"
    GITHUB = "github repository"
    ZIP = "zip"

class Chunk:
    def __init__(self, path: str, text: Optional[str] = None, image: Optional[Image.Image] = None, source_type: Optional[SourceTypes] = None):
        self.path = path
        self.text = text
        self.image = image
        self.source_type = source_type

def print_status(text: str, status: str) -> None:
    message = (Fore.GREEN + f"{text}") if status == 'success' else ((Fore.YELLOW + f"{text}...") if status == 'info' else (Fore.RED + f"{text}"))
    print(Style.RESET_ALL + message + Style.RESET_ALL)

def count_tokens(chunks: List[Chunk]) -> int:
    return sum([(len(chunk.path)+len(chunk.text))/4 for chunk in chunks if chunk.text is not None])
