import base64
from io import BytesIO
import json
import os
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union
import requests
from PIL import Image
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import ReasoningEffort
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

try:  # Optional LlamaIndex dependency
    from llama_index.core.schema import Document as _LlamaDocument
    from llama_index.core.schema import ImageDocument as _LlamaImageDocument
except ImportError:  # pragma: no cover - handled dynamically in helpers below
    _LlamaDocument = None  # type: ignore[assignment]
    _LlamaImageDocument = None  # type: ignore[assignment]

# Re-export for backwards compatibility (may be ``None`` when not installed)
Document = _LlamaDocument  # type: ignore[assignment]
ImageDocument = _LlamaImageDocument  # type: ignore[assignment]

# LLM provider info, defaults to openai
DEFAULT_AI_MODEL = os.getenv("DEFAULT_AI_MODEL", "gpt-5.6-luna")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# for persistent images via filehosting: images are saved to ./images and
# referenced as {HOST_URL}/images/{id}.jpg, so HOST_URL must serve that folder
HOST_IMAGES = os.getenv("HOST_IMAGES", "false").lower() == "true"
HOST_URL = os.getenv("HOST_URL", "").rstrip("/")
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "75"))

# max simultaneous LLM requests per process
LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "16"))


class _NoLimit:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


llm_slots = (
    threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)
    if LLM_MAX_CONCURRENCY > 0
    else _NoLimit()
)


def llm_parse(
    openai_client: OpenAI,
    model: str,
    messages: Iterable[ChatCompletionMessageParam],
    response_format: Type[T],
    reasoning_effort: Optional[ReasoningEffort] = None,
) -> Tuple[T, str]:
    """Structured-output call, gated by ``llm_slots``. Returns ``(parsed, raw_content)``."""
    with llm_slots:
        if reasoning_effort:
            completion = openai_client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
            )
        else:
            completion = openai_client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=response_format,
            )
    message = completion.choices[0].message
    if message.parsed is None:
        raise RuntimeError(f"LLM did not return a parsable response: {message.refusal}")
    return message.parsed, message.content or ""


def prepare_image(image: Image.Image) -> Image.Image:
    """Return an in-memory copy of ``image`` with its underlying resources closed."""

    try:
        image.load()
    except Exception:
        pass

    try:
        prepared_image = image.copy()
    except Exception:
        return image

    try:
        image.close()
    except Exception:
        pass

    return prepared_image


def _ensure_llama_index() -> Tuple["Document", "ImageDocument"]:
    """Import LlamaIndex lazily and provide a helpful error message if missing."""

    global _LlamaDocument, _LlamaImageDocument

    if _LlamaDocument is not None and _LlamaImageDocument is not None:
        return _LlamaDocument, _LlamaImageDocument  # type: ignore[return-value]

    try:
        from llama_index.core.schema import Document as doc_cls
        from llama_index.core.schema import ImageDocument as image_doc_cls
    except ImportError as exc:  # pragma: no cover - exercised via has_llama_index
        raise ImportError(
            "LlamaIndex support is optional. Install it with "
            "`pip install thepipe-api[llama-index]` to use `Chunk.to_llamaindex`."
        ) from exc

    _LlamaDocument, _LlamaImageDocument = doc_cls, image_doc_cls
    return doc_cls, image_doc_cls  # type: ignore[return-value]


def has_llama_index() -> bool:
    """Return ``True`` when the optional LlamaIndex dependency is available."""

    try:
        _ensure_llama_index()
    except ImportError:
        return False
    return True


class Chunk:
    """
    A unit of scraped content.

    ``metadata`` records provenance. Scrapers set only the keys they know:

    - ``page``        1-based PDF page
    - ``slide``       1-based PPTX slide
    - ``cell``        0-based notebook cell, with ``cell_type``
    - ``row``         0-based spreadsheet row
    - ``block``       0-based DOCX body block, with ``block_type`` (paragraph/table)
    - ``start``/``end``  seconds into an audio/video file
    - ``archive_member``  path of the file inside a scraped ``.zip`` (``path`` is the zip)
    - ``model``       VLM used to produce the text, when one was
    - ``figures``     ``[{"description": str, "bbox": [x0, y0, x1, y1]}]`` in page
                      fractions (0-1, top-left origin), aligned with ``images``
    - ``section``     section title, set by section-aware chunkers

    When chunkers merge chunks, equal values collapse, differing scalars become a
    list (e.g. ``page: [3, 4]``) and lists concatenate.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        text: Optional[str] = None,
        images: Optional[Iterable[Image.Image]] = None,
        audios: Optional[Iterable] = None,
        videos: Optional[Iterable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.path = path
        self.text = text or ""
        self.images = [prepare_image(image) for image in images] if images else []
        self.audios = list(audios) if audios else []
        self.videos = list(videos) if videos else []
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}

    def __repr__(self) -> str:
        parts = []
        if self.path is not None:
            parts.append(f"path={self.path!r}")
        if self.text:
            # Show a concise preview of the text
            snippet = self.text.replace("\n", " ")
            if len(snippet) > 50:
                snippet = snippet[:47] + "..."
            parts.append(f"text_snippet={snippet!r}")
        if self.images:
            parts.append(f"images_count={len(self.images)}")
        if self.audios:
            parts.append(f"audios_count={len(self.audios)}")
        if self.videos:
            parts.append(f"videos_count={len(self.videos)}")
        if self.metadata:
            parts.append(f"metadata={self.metadata!r}")
        content = ", ".join(parts) or "empty"
        return f"Chunk({content})"

    def __str__(self) -> str:
        return self.__repr__()

    def to_llamaindex(self) -> Union[List["Document"], List["ImageDocument"]]:
        DocumentCls, ImageDocumentCls = _ensure_llama_index()
        document_text = self.text if self.text else ""
        metadata = {**({"filepath": self.path} if self.path else {}), **self.metadata}

        # If we have PIL Image objects in self.images, convert them to base64 strings
        if self.images:
            image_docs: List[ImageDocument] = []
            for img in self.images:
                # Encode the image to JPEG (or use its original format if available)
                buffer = BytesIO()
                fmt = img.format or "JPEG"
                img = img.convert("RGB")  # ensure RGB
                img.save(buffer, format=fmt)
                img_bytes = buffer.getvalue()

                # Base64‑encode and build MIME type
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                image_docs.append(
                    ImageDocumentCls(
                        text=document_text,
                        image=img_b64,
                        extra_info=metadata,
                    )
                )
            return image_docs

        # Fallback to plain text Document
        return [DocumentCls(text=document_text, extra_info=metadata)]

    def to_message(
        self,
        text_only: bool = False,
        host_images: bool = False,
        max_resolution: Optional[int] = None,
        include_paths: Optional[bool] = False,
    ) -> ChatCompletionUserMessageParam:
        message_text = ""
        content: List[ChatCompletionContentPartParam] = []
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
        content.append({"type": "text", "text": message_text})

        # Add remaining images that weren't referenced in the text
        for image_url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        return {"role": "user", "content": content}

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
            "metadata": self.metadata,
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
            metadata=data.get("metadata"),
            # audios=data['audios'],
            # videos=data['videos'],
        )


def merge_metadata(chunks: Iterable[Chunk]) -> Dict[str, Any]:
    """Combine provenance of several chunks: equal values collapse, differing scalars become a list, lists concatenate."""
    merged: Dict[str, Any] = {}
    for chunk in chunks:
        for key, value in chunk.metadata.items():
            if key not in merged:
                merged[key] = list(value) if isinstance(value, list) else value
            elif isinstance(value, list):
                merged[key] = merged[key] + value
            elif merged[key] != value:
                existing = merged[key] if isinstance(merged[key], list) else [merged[key]]
                if value not in existing:
                    merged[key] = existing + [value]
    return merged


def make_image_url(
    image: Image.Image, host_images: bool = False, max_resolution: Optional[int] = None
) -> str:
    if max_resolution:
        width, height = image.size
        if width > max_resolution or height > max_resolution:
            scale = max_resolution / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
    if image.mode != "RGB":
        image = image.convert("RGB")
    if host_images:
        if not HOST_URL:
            raise ValueError("HOST_URL must be set to host images (HOST_IMAGES=true).")
        os.makedirs("images", exist_ok=True)
        image_id = f"{time.time_ns()}.jpg"
        image.save(os.path.join("images", image_id), format="JPEG", quality=JPEG_QUALITY)
        return f"{HOST_URL}/images/{image_id}"
    else:
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=JPEG_QUALITY)
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
        if chunk.images and not text_only:
            for image in chunk.images:
                n_tokens += calculate_image_tokens(image)
    return int(n_tokens)


def chunks_to_messages(
    chunks: List[Chunk],
    text_only: bool = False,
    host_images: bool = False,
    max_resolution: Optional[int] = None,
    include_paths: Optional[bool] = False,
) -> List[ChatCompletionUserMessageParam]:
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
    chunks: List[Chunk],
    output_folder: str,
    verbose: bool = False,
    text_only: bool = False,
) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
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
                image.convert("RGB").save(f"{output_folder}/{i}_{j}.jpg")
    # Save the text
    with open(f"{output_folder}/prompt.txt", "w", encoding="utf-8") as file:
        file.write(text)
    if verbose:
        print(f"[thepipe] {calculate_tokens(chunks)} tokens saved to {output_folder}")
