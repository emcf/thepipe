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

HOST_IMAGES = os.getenv("HOST_IMAGES", "false").lower() == "true"
HOST_URL = os.getenv("THEPIPE_API_URL", "https://thepipe-api.up.railway.app")
THEPIPE_API_KEY = os.getenv("THEPIPE_API_KEY", None)


class Chunk:
    def __init__(
        self,
        path: Optional[str] = None,
        text: Optional[str] = "",
        images: Optional[List[Image.Image]] = [],
        audios: Optional[List] = [],
        videos: Optional[List] = [],
    ):
        self.path = path
        self.text = text
        self.images = images
        self.audios = audios
        self.videos = videos

    def to_llamaindex(self) -> Union[List[Document], List[ImageDocument]]:
        document_text = self.text if self.text else ""
        metadata = {"filepath": self.path} if self.path else {}

        # If we have PIL Image objects in self.images, convert them to base64 strings
        if self.images:
            image_docs: List[ImageDocument] = []
            for img in self.images:
                # Encode the image to PNG (or use its original format if available)
                buffer = BytesIO()
                fmt = img.format or "PNG"
                img.save(buffer, format=fmt)
                img_bytes = buffer.getvalue()

                # Base64‑encode and build MIME type
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                image_docs.append(
                    ImageDocument(
                        text=document_text,
                        image=img_b64,
                        extra_info=metadata,
                    )
                )
            return image_docs

        # Fallback to plain text Document
        return [Document(text=document_text, extra_info=metadata)]

    def to_message(
        self,
        text_only: bool = False,
        host_images: bool = False,
        max_resolution: Optional[int] = None,
        include_paths: Optional[bool] = False,
    ) -> Dict:
        message_text = ""
        message = {"role": "user", "content": []}
        image_urls = (
            [
                make_image_url(image, host_images, max_resolution)
                for image in self.images
            ]
            if self.images and not text_only
            else []
        )
        img_index = 0
        text = self.text if self.text else ""
        if host_images:

            def replace_image(match):
                nonlocal img_index
                if img_index < len(image_urls):
                    url = image_urls[img_index]
                    img_index += 1
                    return f"![image]({url})"
                return match.group(
                    0
                )  # If we run out of images, leave the original text

            # Replace markdown image references with hosted URLs
            text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", replace_image, text)
        message_text += text + "\n\n"
        # clean up, add to message
        message_text = re.sub(r"\n{3,}", "\n\n", message_text).strip()
        # Wrap the text in a path html block if it exists
        if include_paths and self.path:
            message_text = f'<Document path="{self.path}">\n{message_text}\n</Document>'
        message["content"].append({"type": "text", "text": message_text})

        # Add remaining images that weren't referenced in the text
        for image_url in image_urls:
            message["content"].append({"type": "image_url", "image_url": image_url})

        return message

    def to_json(self, host_images: bool = False, text_only: bool = False) -> Dict:
        data = {
            "path": self.path,
            "text": self.text.strip() if self.text else "",
            "images": (
                [
                    make_image_url(image=image, host_images=host_images)
                    for image in self.images
                    if not text_only
                ]
                if self.images
                else []
            ),
            "audios": self.audios,
            "videos": self.videos,
        }
        return data

    @staticmethod
    def from_json(data: Dict, host_images: bool = False) -> "Chunk":
        images = []
        if "images" in data:
            for image_str in data["images"]:
                if host_images:
                    image_data = requests.get(image_str).content
                    image = Image.open(BytesIO(image_data))
                    images.append(image)
                else:
                    remove_prefix = image_str.replace("data:image/jpeg;base64,", "")
                    image_data = base64.b64decode(remove_prefix)
                    image = Image.open(BytesIO(image_data))
                    images.append(image)
        text = data["text"].strip() if "text" in data else None
        return Chunk(
            path=data["path"],
            text=text,
            images=images,
            # audios=data['audios'],
            # videos=data['videos'],
        )


def make_image_url(
    image: Image.Image, host_images: bool = False, max_resolution: Optional[int] = None
) -> str:
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
        if image.mode in ("P", "RGBA"):
            image = image.convert("RGB")
        image.save(image_path)
        return f"{HOST_URL}/images/{image_id}"
    else:
        buffered = BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
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


def calculate_tokens(chunks: List[Chunk], text_only: bool = False) -> int:
    n_tokens = 0
    for chunk in chunks:
        if chunk.text:
            n_tokens += len(chunk.text) / 4
        if chunk.images:
            for image in chunk.images:
                n_tokens += calculate_image_tokens(image)
    return int(n_tokens)


def chunks_to_messages(
    chunks: List[Chunk],
    text_only: bool = False,
    host_images: bool = False,
    max_resolution: Optional[int] = None,
    include_paths: Optional[bool] = False,
) -> List[Dict]:
    return [
        chunk.to_message(
            text_only=text_only,
            host_images=host_images,
            max_resolution=max_resolution,
            include_paths=include_paths,
        )
        for chunk in chunks
    ]


def save_outputs(
    chunks: List[Chunk], verbose: bool = False, text_only: bool = False
) -> None:
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    text = ""
    # Save the text and images to the outputs directory
    for i, chunk in enumerate(chunks):
        if chunk is None:
            continue
        if chunk.path is not None:
            text += f"{chunk.path}:\n"
        if chunk.text:
            text += f"```\n{chunk.text}\n```\n"
        if not text_only and chunk.images:
            for j, image in enumerate(chunk.images):
                image.convert("RGB").save(f"outputs/{i}_{j}.jpg")
    # Save the text
    with open("outputs/prompt.txt", "w", encoding="utf-8") as file:
        file.write(text)
    if verbose:
        print(f"[thepipe] {calculate_tokens(chunks)} tokens saved to outputs folder")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress project files into a context prompt."
    )
    parser.add_argument(
        "source", type=str, help="The source file or directory to compress."
    )
    parser.add_argument(
        "--include_regex",
        type=str,
        default=None,
        help="Regex pattern to match in a directory.",
    )
    parser.add_argument(
        "--ai_extraction",
        action="store_true",
        help="Use ai_extraction to extract text from images.",
    )
    parser.add_argument("--text_only", action="store_true", help="Only store text.")
    parser.add_argument("--verbose", action="store_true", help="Print status messages.")
    parser.add_argument("--local", action="store_true", help="Print status messages.")
    args = parser.parse_args()
    return args
