import base64
from io import BytesIO
import os
import time
from typing import Dict, List, Optional, Union
from PIL import Image
from llama_index.core.schema import Document, ImageDocument

HOST_URL = os.getenv("THEPIPE_API_URL", "http://localhost:5000")

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
        
    def to_message(self, host_images: bool = False, max_resolution : Optional[int] = None) -> Dict:
        message = {"role": "user", "content": []}
        if self.texts:
            prompt = "\n```\n" + '\n'.join(self.texts) + "\n```\n" 
            message["content"].append({"type": "text", "text": prompt})
        for image in self.images:
            image_url = make_image_url(image, host_images, max_resolution)
            message["content"].append({"type": "image_url", "image_url": image_url})
        return message

def make_image_url(image: Image.Image, host_images: bool = False, max_resolution: Optional[int] = None) -> str:
    if max_resolution:
        width, height = image.size
        if width > max_resolution or height > max_resolution:
            scale = max_resolution / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height))
    if host_images:
        if not os.path.exists("images"):
            os.makedirs("images")
        image_id = f"{time.time_ns()}.jpg"
        image_path = os.path.join("images", image_id)
        if image.mode in ('P', 'RGBA'):
            image = image.convert('RGB')
        image.save(image_path)
        return f"{HOST_URL}/images/{image_id}"
    else:
        buffered = BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

def calculate_image_tokens(image: Image.Image, detail: str = "auto") -> int:
    width, height = image.size
    if detail == "low":
        return 85
    elif detail == "high":
        width, height = min(width, 2048), min(height, 2048)
        short_side = min(width, height)
        scale = 768 / short_side
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)
        tiles = (scaled_width // 512) * (scaled_height // 512)
        return 170 * tiles + 85
    else:
        if width <= 512 and height <= 512:
            return 85
        else:
            return calculate_image_tokens(image, detail="high")

def calculate_tokens(chunks: List[Chunk]) -> int:
    n_tokens = 0
    for chunk in chunks:
        for text in chunk.texts:
            n_tokens += len(text) / 4
        for image in chunk.images:
            n_tokens += calculate_image_tokens(image)
    return int(n_tokens)

def chunks_to_messages(chunks: List[Chunk]) -> List[Dict]:
    return [chunk.to_message() for chunk in chunks]
