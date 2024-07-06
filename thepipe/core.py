import argparse
import base64
from io import BytesIO
import json
import os
import time
from typing import Dict, List, Optional, Union
import requests
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
    
    def to_json(self, host_images: bool = False) -> str:
        data = {
            'path': self.path,
            'texts': self.texts,
            'images': [make_image_url(image=image, host_images=host_images) for image in self.images],
            'audios': self.audios,
            'videos': self.videos,
        }
        return json.dumps(data)
    
    @staticmethod
    def from_json(json_str: str, host_images: bool = False) -> 'Chunk':
        data = json.loads(json_str)
        images = []
        for image_str in data['images']:
            if host_images:
                image_data = requests.get(image_str).content
                image = Image.open(BytesIO(image_data))
                images.append(image)
            else:
                remove_prefix = image_str.replace("data:image/jpeg;base64,", "")
                image_data = base64.b64decode(remove_prefix)
                image = Image.open(BytesIO(image_data))
                images.append(image)
        return Chunk(
            path=data['path'],
            texts=data['texts'],
            images=images,
            audios=data['audios'],
            videos=data['videos'],
        )
    
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
