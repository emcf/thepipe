import argparse
import base64
from io import BytesIO
import json
import os
import re
import time
from typing import Dict, List, Optional, Union
import requests
from PIL import Image
from llama_index.core.schema import Document, ImageDocument
import weakref

HOST_IMAGES = os.getenv("HOST_IMAGES", "false").lower() == "true"
HOST_URL = os.getenv("THEPIPE_API_URL", "https://thepipe-api.up.railway.app")
THEPIPE_API_KEY = os.getenv("THEPIPE_API_KEY", None)

class Chunk:
    def __init__(self, path: Optional[str] = None, texts: Optional[List[str]] = [], images: Optional[List[Image.Image]] = [], audios: Optional[List] = [], videos: Optional[List] = []):
        self.path = path
        self.texts = texts
        self.images = []
        for img in (images or []):
            if isinstance(img, weakref.ReferenceType):
                self.images.append(img)
            else:
                self.images.append(weakref.ref(img))
        self.audios = audios
        self.videos = videos

    def get_valid_images(self):
        return [img() for img in self.images if img() is not None]

    def to_llamaindex(self) -> List[Union[Document, ImageDocument]]:
        document_text = "\n".join(self.texts)
        if len(self.images) > 0:
            return [ImageDocument(text=document_text, image=image) for image in self.images]
        else:
            return [Document(text=document_text)]
        
    def to_message(self, host_images: bool = False, max_resolution: Optional[int] = None) -> Dict:
        message_text = ""
        message = {"role": "user", "content": []}
        image_urls = [make_image_url(image, host_images, max_resolution) for image in self.images]
        if self.texts:
            img_index = 0
            for text in self.texts:
                if host_images:
                    def replace_image(match):
                        nonlocal img_index
                        if img_index < len(image_urls):
                            url = image_urls[img_index]
                            img_index += 1
                            return f"![image]({url})"
                        return match.group(0)  # If we run out of images, leave the original text
                    # Replace markdown image references with hosted URLs
                    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', replace_image, text)  
                message_text += text + "\n\n"
            # clean up, add to message
            message_text = re.sub(r'\n{3,}', '\n\n', message_text).strip()
            # Wrap the text in a path html block if it exists
            if self.path:
                message_text = f'<Document path="{self.path}">\n{message_text}\n</Document>' 
            message["content"].append({"type": "text", "text": message_text})
        # Add remaining images that weren't referenced in the text
        for image_url in image_urls:
            message["content"].append({"type": "image_url", "image_url": image_url})
        
        return message
    
    def to_json(self, host_images: bool = False) -> Dict:
        data = {
            'path': self.path,
            'texts': [text.strip() for text in self.texts],
            'images': [make_image_url(image=image, host_images=host_images) for image in self.images],
            'audios': self.audios,
            'videos': self.videos,
        }
        return data
    
    @staticmethod
    def from_json(data: Dict, host_images: bool = False) -> 'Chunk':
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
            texts=[text.strip() for text in data['texts']],
            images=images,
            audios=data['audios'],
            videos=data['videos'],
        )
    
    def __repr__(self):
        return f"Chunk(path={self.path}, texts={len(self.texts)} items, images={len(self.images)} items)"

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
            n_tokens += len(text) // 4  # Rough estimate: 1 token ≈ 4 characters
        for image in chunk.get_valid_images():
            try:
                n_tokens += calculate_image_tokens(image)
            except Exception as e:
                print(f"[thepipe] Error calculating tokens for an image: {str(e)}")
                # Add a default token count for failed images
                n_tokens += 85  # Minimum token count for an image
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
            for j, image in enumerate(chunk.get_valid_images()):
                try:
                    image.convert('RGB').save(f'outputs/{i}_{j}.jpg')
                except Exception as e:
                    if verbose:
                        print(f"[thepipe] Error saving image at index {j} in chunk {i}: {str(e)}")

    # Save the text
    with open('outputs/prompt.txt', 'w', encoding='utf-8') as file:
        file.write(text)
    
    if verbose:
        try:
            # Attempt to calculate tokens using the original method
            token_count = calculate_tokens(chunks)
            print(f"[thepipe] Approximately {token_count} tokens saved to outputs folder")
        except Exception as e:
            # If the original method fails, fall back to a simpler estimation
            total_chars = sum(len(chunk_text) for chunk in chunks for chunk_text in chunk.texts)
            estimated_tokens = total_chars // 4  # Rough estimate: 1 token ≈ 4 characters
            print(f"[thepipe] Error calculating exact tokens: {str(e)}")
            print(f"[thepipe] Estimated {estimated_tokens} tokens saved to outputs folder (based on character count)")

        print(f"[thepipe] Outputs saved to 'outputs' folder")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compress project files into a context prompt.')
    parser.add_argument('source', type=str, help='The source file or directory to compress.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--include_regex', type=str, nargs='?', const='.*', default=None, 
                       help='Regex pattern to match in a directory. Use quotes for patterns with special characters.')
    group.add_argument('--include_pattern', type=str, nargs='?', const='*', default=None, 
                       help='Glob pattern to match files in a directory (e.g., "*.tsx"). Use quotes for patterns with special characters.')    
    parser.add_argument('--ai_extraction', action='store_true', help='Use ai_extraction to extract text from images.')
    parser.add_argument('--text_only', nargs='?', const='default', default=None, 
                        choices=['default', 'transcribe', 'ai', 'uploaded'],
                        help='Extract only text from the source. Options: default (try all methods), transcribe (force local transcription), ai (prefer AI-generated), uploaded (prefer uploaded)')
    parser.add_argument('--verbose', action='store_true', help='Print status messages.')
    parser.add_argument('--local', action='store_true', help='Use local processing instead of API.')
    parser.add_argument('--options', type=str, help='JSON string of type-specific options')
    args = parser.parse_args()
    
    if args.options:
        try:
            args.options = json.loads(args.options)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in options")
            exit(1)
    
    return args