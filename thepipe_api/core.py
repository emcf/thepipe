import base64
from io import BytesIO
from typing import Dict, List, Optional
from PIL import Image
from colorama import Style, Fore
from enum import Enum

API_URL = "https://thepipe.up.railway.app/extract"

class SourceTypes(Enum):
    DIR = "directory"
    UNCOMPRESSIBLE_CODE = "code"
    COMPRESSIBLE_CODE = "code "
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
    if status == 'success':
        message = Fore.GREEN + f"{text}"
    elif status == 'info':
        message = Fore.YELLOW + f"{text}..."
    elif status == 'error':
        message = Fore.RED + f"{text}"
    print(Style.RESET_ALL + message + Style.RESET_ALL)

def count_tokens(chunks: List[Chunk]) -> int:
    return sum([((len(chunk.path) if chunk.path else 0) + (len(chunk.text) if chunk.text else 0))/4 for chunk in chunks])

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_chunks_from_messages(messages: List[Dict]) -> List[Chunk]:
    chunks = []
    for message in messages:
        text = None
        image = None
        for content in message['content']:
            if content['type'] == 'text':
                text = content['text']
            elif content['type'] == 'image_url':
                # base64 image
                base64_string = content['image_url']['url'].split(",")[1]
                image_data = base64.b64decode(base64_string)
                image = Image.open(BytesIO(image_data))
        chunks.append(Chunk(path=None, text=text, image=image))
    return chunks

def create_messages_from_chunks(chunks: List[Chunk]) -> List[Dict]:
    messages = []
    for chunk in chunks:
        content = []
        if chunk.text:
            content.append({"type": "text", "text": f"""{chunk.path}:\n```\n{chunk.text}\n```\n"""})
        if chunk.image:
            base64_image = image_to_base64(chunk.image)
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        messages.append({"role": "user", "content": content})
    return messages
