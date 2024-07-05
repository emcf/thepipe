import base64
from io import BytesIO
from typing import Dict, List, Optional, Union
from PIL import Image
from colorama import Style, Fore
from llama_index.core.schema import Document, ImageDocument

class Chunk:
    def __init__(self, path: Optional[str] = None, texts: Optional[List[str]] = [], images: Optional[List[Image.Image]] = [], audios: Optional[List] = [], videos: Optional[List] = []):
        self.path = path
        self.texts = texts
        self.images = images
        self.audios = audios
        self.videos = videos

    def to_llamaindex(self) -> List[Union[Document, ImageDocument]]:
        document_text = "\n".join(self.texts)
        if len(self.images) > 0:
            return [ImageDocument(text=document_text, image=image) for image in self.images]
        else:
            return [Document(text=document_text)]
        
    def to_message(self) -> Dict:
        content = []
        if self.texts:
            for text in self.texts:
                content.append({"type": "text", "text": {"content": text}})
        if self.images:
            for image in self.images:
                base64_image = image_to_base64(image)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        return {"role": "user", "content": content}
    
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

def calculate_tokens(chunks: List[Chunk]) -> int:
    n_tokens = 0
    for chunk in chunks:
        for text in chunk.texts:
            n_tokens += len(text)/4
        for image in chunk.images:
            n_tokens += calculate_image_tokens(image)
    return n_tokens

def image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    if image.mode == 'RGBA' or image.mode == 'P':
        image = image.convert('RGB')
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def chunks_to_messsages(chunks: List[Chunk]) -> List[Dict]:
    return [chunk.to_message() for chunk in chunks]