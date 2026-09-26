from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from io import BytesIO, StringIO
import math
import re
import fnmatch
import os
import subprocess
import tempfile
import threading
from urllib.parse import urlparse
import zipfile
from PIL import Image
import requests
import json
from .core import (
    HOST_IMAGES,
    Chunk,
    make_image_url,
    DEFAULT_AI_MODEL,
    llm_parse,
)
from .chunker import (
    chunk_by_page,
    chunk_by_document,
    chunk_by_section,
    chunk_semantic,
    chunk_by_keywords,
    chunk_by_length,
    chunk_agentic,
)
import mimetypes
import dotenv
from magika import Magika
import markdownify
import pymupdf
from openai import OpenAI
from openai.types.chat import ChatCompletionContentPartParam
from openai.types.shared_params import ReasoningEffort
from pydantic import BaseModel, Field

dotenv.load_dotenv()

FOLDERS_TO_IGNORE = {
    "*node_modules*",
    "*.git*",
    "*venv*",
    "*.vscode*",
    "*pycache*",
    "*.ipynb_checkpoints",
}
FILES_TO_IGNORE = {
    ".gitignore",
    "*.bin",
    # Python compiled files
    "*.pyc",
    "*.pyo",
    "*.pyd",
    # Shared libraries and binaries
    "*.so",
    "*.dll",
    "*.exe",
    # Archives and packages
    "*.tar",
    "*.tar.gz",
    "*.egg-info",
    "package-lock.json",
    "package.json",
    # Lock, log, and metadata files
    "*.lock",
    "*.log",
    "Pipfile.lock",
    "requirements.lock",
    "*.exe",
    "*.dll",
    ".DS_Store",
    "Thumbs.db",
}
USER_AGENT_STRING: str = os.getenv(
    "USER_AGENT_STRING",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
)
MAX_WHISPER_DURATION = int(os.getenv("MAX_WHISPER_DURATION", 600))  # 10 minutes
PDF_RENDER_SCALE = float(os.getenv("PDF_RENDER_SCALE", "1.0"))
DIRECTORY_SCRAPE_WORKERS = int(os.getenv("DIRECTORY_SCRAPE_WORKERS", str(min(8, (os.cpu_count() or 1) * 2))))
PDF_PAGE_WORKERS = int(os.getenv("PDF_PAGE_WORKERS", str((os.cpu_count() or 1) * 2)))
SCRAPE_REASONING_EFFORT = cast(ReasoningEffort, os.getenv("SCRAPE_REASONING_EFFORT", "none"))

SCRAPING_PROMPT = os.getenv(
    "SCRAPING_PROMPT",
    """A document page is given. Please output the entire extracted contents from the page in detailed markdown format.
Your accuracy is very important. Please be careful to not miss any content from the page.
Be sure to correctly output a comprehensive format markdown for all the page contents (including, but not limited to, headers, paragraphs, lists, tables, menus, equations, full text contents, titles, subtitles, appendices, page breaks, columns, footers, page numbers, watermarks, footnotes, captions, annotations, form fields, content controls, signatures, etc.)
Additionally, list the bounding box of every image, figure, diagram, chart, photo, or illustration on the page.
Do NOT include tables in the bounding boxes - tables must be transcribed into the markdown instead.
Coordinates are fractions of the page width and height in the range 0.0 to 1.0, with the origin at the top-left corner.
Each box should tightly enclose one visual element, including its embedded labels but not its caption text.""",
)

# Ignore figure boxes smaller than this fraction of the page along either axis
MIN_FIGURE_FRACTION = 0.02


class FigureBox(BaseModel):
    description: str = Field(description="Short description of the visual element")
    x_min: float = Field(description="Left edge as a fraction of page width (0-1)")
    y_min: float = Field(description="Top edge as a fraction of page height (0-1)")
    x_max: float = Field(description="Right edge as a fraction of page width (0-1)")
    y_max: float = Field(description="Bottom edge as a fraction of page height (0-1)")

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        clamp = lambda v: min(max(v, 0.0), 1.0)
        return clamp(self.x_min), clamp(self.y_min), clamp(self.x_max), clamp(self.y_max)

    @property
    def is_valid(self) -> bool:
        x0, y0, x1, y1 = self.bbox
        return x1 - x0 >= MIN_FIGURE_FRACTION and y1 - y0 >= MIN_FIGURE_FRACTION


class PageExtraction(BaseModel):
    markdown: str = Field(description="Complete markdown transcription of the page")
    figures: List[FigureBox] = Field(
        description="Bounding boxes of images, diagrams, charts, and photos (not tables)"
    )


def crop_figures(page_image: Image.Image, figures: List[FigureBox]) -> List[Image.Image]:
    """Crop each figure box out of a rendered page image."""
    width, height = page_image.size
    return [
        page_image.crop((int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)))
        for x0, y0, x1, y1 in (fig.bbox for fig in figures)
    ]


def _load_whisper():
    try:
        import whisper
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Audio and video transcription requires the optional dependency `openai-whisper`. "
            "Install it with `pip install thepipe-api[audio]` or include the `gpu` extra."
        ) from exc

    return whisper


@lru_cache(maxsize=None)
def _get_whisper_model(name: str = "base"):
    return _load_whisper().load_model(name)


_whisper_lock = threading.Lock()  # whisper.transcribe is not thread-safe


def _transcribe(file_path: str, verbose: bool = False) -> List[Dict[str, Any]]:
    model = _get_whisper_model("base")
    with _whisper_lock:
        result = model.transcribe(audio=file_path, verbose=verbose)
    return cast(List[Dict[str, Any]], result.get("segments", []))


@lru_cache(maxsize=1)
def _get_magika() -> Magika:
    return Magika()


def detect_source_mimetype(source: str) -> str:
    # try to detect the file type by its extension
    _, extension = os.path.splitext(source)
    if extension:
        if extension == ".ipynb":
            # special case for notebooks, mimetypes is not familiar
            return "application/x-ipynb+json"
        guessed_mimetype, _ = mimetypes.guess_type(source)
        if guessed_mimetype:
            return guessed_mimetype
    # if that fails, try AI detection with Magika
    magika = _get_magika()
    with open(source, "rb") as file:
        result = magika.identify_bytes(file.read())
    mimetype = result.output.mime_type
    return mimetype


def _is_within_directory(root_path: str, candidate_path: str) -> bool:
    try:
        return os.path.commonpath([root_path, candidate_path]) == root_path
    except ValueError:
        return False


def scrape_file(
    filepath: str,
    verbose: bool = False,
    chunking_method: Optional[Callable[[List[Chunk]], List[Chunk]]] = chunk_by_page,
    openai_client: Optional[OpenAI] = None,
    model: str = DEFAULT_AI_MODEL,
    include_input_images: bool = True,
    include_output_images: bool = True,
    max_input_image_size: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[Chunk]:
    """
    Scrapes a file and returns a list of Chunk objects containing the text and images extracted from the file.

    Parameters
    ----------
    filepath : str
        The path to the file to scrape.
    verbose : bool, optional
        If ``True``, prints verbose output.
    chunking_method : Callable, optional
        A function to chunk the scraped content. Defaults to chunk_by_page.
    openai_client : OpenAI, optional
        An OpenAI client instance for LLM processing. If provided, uses VLM to scrape PDFs.
    model : str, optional
        The LLM model name to use for processing. Defaults to DEFAULT_AI_MODEL.
    include_input_images : bool, optional
        If ``True``, includes input images in the messages sent to the LLM.
    include_output_images : bool, optional
        If ``True``, includes output images in the returned chunks.
    max_input_image_size : int, optional
        Maximum size in pixels of the largest axis of any image sent to the
        VLM during scraping. For example, ``500`` guarantees every image sent
        to the VLM fits within 500x500 pixels. Does not affect the images in
        the returned chunks. If ``None``, images are sent at their native size.
    max_workers : int, optional
        Thread count for PDF pages (VLM path) or files inside a zip. Defaults to
        ``PDF_PAGE_WORKERS`` / ``DIRECTORY_SCRAPE_WORKERS``.
    Returns
    -------
    List[Chunk]
        A list of Chunk objects containing the scraped content.
    """
    # returns chunks of scraped content from the given file
    scraped_chunks = []
    source_mimetype = detect_source_mimetype(filepath)
    if source_mimetype is None:
        if verbose:
            print(f"[thepipe] Unsupported source type: {filepath}")
        return scraped_chunks
    if verbose:
        print(f"[thepipe] Scraping {source_mimetype}: {filepath}...")
    if source_mimetype == "application/pdf":
        scraped_chunks = scrape_pdf(
            file_path=filepath,
            verbose=verbose,
            model=model,
            openai_client=openai_client,
            include_input_images=include_input_images,
            include_output_images=include_output_images,
            max_input_image_size=max_input_image_size,
            max_workers=max_workers,
        )
    elif (
        source_mimetype
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        scraped_chunks = scrape_docx(
            file_path=filepath,
            verbose=verbose,
            include_output_images=include_output_images,
        )
    elif (
        source_mimetype
        == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    ):
        scraped_chunks = scrape_pptx(
            file_path=filepath,
            verbose=verbose,
            include_output_images=include_output_images,
        )
    elif source_mimetype.startswith("image/"):
        scraped_chunks = scrape_image(file_path=filepath)
    elif (
        source_mimetype in ("text/csv", "application/vnd.ms-excel")
        or source_mimetype
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        scraped_chunks = scrape_spreadsheet(
            file_path=filepath, source_type=source_mimetype
        )
    elif source_mimetype == "application/x-ipynb+json":
        scraped_chunks = scrape_ipynb(
            file_path=filepath,
            verbose=verbose,
            include_output_images=include_output_images,
        )
    elif (
        source_mimetype == "application/zip"
        or source_mimetype == "application/x-zip-compressed"
    ):
        scraped_chunks = scrape_zip(
            file_path=filepath,
            verbose=verbose,
            openai_client=openai_client,
            include_input_images=include_input_images,
            include_output_images=include_output_images,
            max_input_image_size=max_input_image_size,
            max_workers=max_workers,
        )
    elif source_mimetype.startswith("video/"):
        scraped_chunks = scrape_video(
            file_path=filepath,
            verbose=verbose,
            include_output_images=include_output_images,
        )
    elif source_mimetype.startswith("audio/"):
        scraped_chunks = scrape_audio(file_path=filepath, verbose=verbose)
    elif source_mimetype.startswith("text/html"):
        scraped_chunks = scrape_html(
            file_path=filepath,
            verbose=verbose,
            include_output_images=include_output_images,
        )
    elif source_mimetype.startswith("text/"):
        scraped_chunks = scrape_plaintext(file_path=filepath)
    else:
        try:
            scraped_chunks = scrape_plaintext(file_path=filepath)
        except Exception as e:
            if verbose:
                print(f"[thepipe] Error extracting from {filepath}: {e}")
    if verbose:
        if scraped_chunks:
            print(f"[thepipe] Extracted from {filepath}")
        else:
            print(f"[thepipe] No content extracted from {filepath}")
    if chunking_method:
        scraped_chunks = chunking_method(scraped_chunks)
    return scraped_chunks


def scrape_html(
    file_path: str,
    verbose: bool = False,
    include_output_images: bool = True,
) -> List[Chunk]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        html_content = file.read()
    markdown_content = markdownify.markdownify(html_content, heading_style="ATX")
    images = get_images_from_markdown(html_content) if include_output_images else []
    return [Chunk(path=file_path, text=markdown_content, images=images)]


def scrape_plaintext(file_path: str) -> List[Chunk]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()
    return [Chunk(path=file_path, text=text)]


def _collect_directory_files(
    dir_path: str,
    pattern: Optional["re.Pattern[str]"],
    canonical_root: str,
    visited_dirs: set,
    verbose: bool,
) -> List[str]:
    files: List[str] = []
    current_dir = os.path.realpath(dir_path)

    if not _is_within_directory(canonical_root, current_dir):
        if verbose:
            print(f"[thepipe] Skipping path outside root: {dir_path}")
        return files

    if current_dir in visited_dirs:
        if verbose:
            print(f"[thepipe] Skipping already visited directory: {current_dir}")
        return files
    visited_dirs.add(current_dir)

    try:
        for entry in os.scandir(dir_path):
            path = entry.path
            resolved_path = os.path.realpath(path)

            if not _is_within_directory(canonical_root, resolved_path):
                if verbose:
                    print(f"[thepipe] Skipping path outside root: {path}")
                continue

            # skip ignored directories
            if entry.is_dir() and any(
                fnmatch.fnmatch(entry.name, pat) for pat in FOLDERS_TO_IGNORE
            ):
                if verbose:
                    print(f"[thepipe] Skipping ignored directory: {path}")
                continue

            # skip ignored files
            if entry.is_file() and any(
                fnmatch.fnmatch(entry.name, pat) for pat in FILES_TO_IGNORE
            ):
                if verbose:
                    print(f"[thepipe] Skipping ignored file: {path}")
                continue

            if entry.is_file():
                # if include_pattern is set, skip files that don't match
                if pattern and not pattern.search(path):
                    if verbose:
                        print(f"[thepipe] Skipping non-matching file: {path}")
                    continue
                files.append(resolved_path)

            elif entry.is_dir():
                if verbose:
                    print(f"[thepipe] Entering directory: {resolved_path}")
                files.extend(
                    _collect_directory_files(
                        resolved_path, pattern, canonical_root, visited_dirs, verbose
                    )
                )
    except PermissionError as e:
        if verbose:
            print(f"[thepipe] Skipping {dir_path} (permission denied): {e}")

    return files


def scrape_directory(
    dir_path: str,
    inclusion_pattern: Optional[str] = None,
    verbose: bool = False,
    openai_client: Optional[OpenAI] = None,
    model: str = DEFAULT_AI_MODEL,
    include_input_images: bool = True,
    include_output_images: bool = True,
    max_input_image_size: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[Chunk]:
    """
    inclusion_pattern: Optional regex string; only files whose path matches this pattern will be scraped.
    By default, ignores all files in baked-in constants FOLDERS_TO_IGNORE and FILES_TO_IGNORE.
    """
    pattern = re.compile(inclusion_pattern) if inclusion_pattern else None
    canonical_root = os.path.realpath(dir_path)
    files = _collect_directory_files(dir_path, pattern, canonical_root, set(), verbose)
    if not files:
        return []

    def _scrape(path: str) -> List[Chunk]:
        if verbose:
            print(f"[thepipe] Scraping file: {path}")
        return scrape_file(
            filepath=path,
            verbose=verbose,
            openai_client=openai_client,
            model=model,
            include_input_images=include_input_images,
            include_output_images=include_output_images,
            max_input_image_size=max_input_image_size,
        )

    workers = max(1, min(max_workers or DIRECTORY_SCRAPE_WORKERS, len(files)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_scrape, files))
    return [chunk for chunks in results for chunk in chunks]


def scrape_zip(
    file_path: str,
    inclusion_pattern: Optional[str] = None,
    verbose: bool = False,
    openai_client: Optional[OpenAI] = None,
    include_input_images: bool = True,
    include_output_images: bool = True,
    max_input_image_size: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[Chunk]:
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        chunks = scrape_directory(
            dir_path=temp_dir,
            inclusion_pattern=inclusion_pattern,
            verbose=verbose,
            openai_client=openai_client,
            include_input_images=include_input_images,
            include_output_images=include_output_images,
            max_input_image_size=max_input_image_size,
            max_workers=max_workers,
        )
        # temp paths mean nothing to the caller: point at the archive and record the member
        root = os.path.realpath(temp_dir)
        for chunk in chunks:
            member = os.path.relpath(chunk.path or root, root).replace(os.sep, "/")
            chunk.metadata["archive_member"] = member
            chunk.path = file_path
    return chunks


def scrape_pdf(
    file_path: str,
    openai_client: Optional[OpenAI] = None,
    model: str = DEFAULT_AI_MODEL,
    verbose: Optional[bool] = False,
    include_input_images: bool = True,
    include_output_images: bool = True,
    max_input_image_size: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> List[Chunk]:
    """
    max_input_image_size caps the largest axis (in pixels) of page images sent to the
    VLM during scraping. It does not affect the images in the returned chunks.

    With an OpenAI client, each page is transcribed to markdown via structured
    outputs, and any images/diagrams/charts the VLM locates are cropped out of the
    page render and returned as the chunk's images. A page whose LLM call fails
    falls back to its plain text with the error in ``metadata["error"]``.
    """
    chunks: List[Chunk] = []

    # Branch 1 – VLM path (OpenAI client supplied)
    if openai_client is not None:
        with open(file_path, "rb") as fp:
            pdf_bytes = fp.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        num_pages = len(doc)

        if verbose:
            print(
                f"[thepipe] Scraping PDF: {file_path} "
                f"({num_pages} pages) with model {model}"
            )

        def _process_page(page_num: int) -> Chunk:
            page = doc[page_num]
            text = page.get_text()  # type: ignore[attr-defined]

            scale = PDF_RENDER_SCALE
            if max_input_image_size is not None:
                rect = page.rect
                scale = min(scale, max_input_image_size / max(rect.width, rect.height))
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)  # type: ignore[attr-defined]
            page_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            msg_content: List[ChatCompletionContentPartParam] = [
                {"type": "text", "text": f"```\n{text}\n```\n{SCRAPING_PROMPT}"}
            ]
            if include_input_images:
                encoded = make_image_url(
                    page_image,
                    host_images=HOST_IMAGES,
                    max_resolution=max_input_image_size,
                )
                msg_content.append(
                    {"type": "image_url", "image_url": {"url": encoded, "detail": "high"}}
                )

            parsed, _ = llm_parse(
                openai_client,
                model=model,
                messages=[{"role": "user", "content": msg_content}],
                response_format=PageExtraction,
                reasoning_effort=SCRAPE_REASONING_EFFORT,
            )

            figures = [fig for fig in parsed.figures if fig.is_valid]
            images = (
                crop_figures(page_image, figures)
                if include_output_images and figures
                else []
            )
            return Chunk(
                path=file_path,
                text=parsed.markdown.strip(),
                images=images,
                metadata={
                    "page": page_num + 1,
                    "model": model,
                    "figures": [
                        {"description": fig.description, "bbox": list(fig.bbox)}
                        for fig in figures
                    ],
                },
            )

        def _safe_process_page(page_num: int) -> Chunk:
            try:
                return _process_page(page_num)
            except Exception as e:
                if verbose:
                    print(f"[thepipe] Page {page_num + 1} failed, using plain text: {e}")
                text = doc[page_num].get_text().strip()  # type: ignore[attr-defined]
                return Chunk(
                    path=file_path,
                    text=text,
                    metadata={"page": page_num + 1, "model": None, "error": str(e)},
                )

        workers = max(1, max_workers or PDF_PAGE_WORKERS)
        if verbose:
            print(f"[thepipe] Using {workers} threads for PDF extraction")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            chunks = list(executor.map(_safe_process_page, range(num_pages)))

        doc.close()
        return chunks

    # Branch 2 – no OpenAI client – text-only offline mode
    import pymupdf4llm  # local import

    doc = pymupdf.open(file_path)
    # OCR off: it depends on a system Tesseract install and prints to stdout
    md_pages = cast(
        List[Dict[str, Any]],
        pymupdf4llm.to_markdown(doc, page_chunks=True, use_ocr=False),
    )

    for i in range(doc.page_count):
        text = re.sub(r"\n{3,}", "\n\n", md_pages[i]["text"]).strip()

        images: List[Image.Image] = []
        if include_output_images:
            pix = doc[i].get_pixmap(alpha=False)  # type: ignore[attr-defined]  # noqa: E501
            images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))

        chunks.append(
            Chunk(path=file_path, text=text, images=images, metadata={"page": i + 1})
        )

    doc.close()
    return chunks


def get_images_from_markdown(text: str) -> List[Image.Image]:
    image_urls = re.findall(r"!\[.*?\]\((.*?)\)", text)
    images = []
    for url in image_urls:
        extension = os.path.splitext(urlparse(url).path)[1]
        if extension not in {".jpg", ".jpeg", ".png"}:
            # ignore incompatible image extractions
            continue

        try:
            response = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": USER_AGENT_STRING},
            )
            response.raise_for_status()
        except Exception:
            continue

        img = Image.open(BytesIO(response.content))
        images.append(img)
    return images


def scrape_image(file_path: str) -> List[Chunk]:
    img = Image.open(file_path)
    img.load()  # needed to close the file
    chunk = Chunk(path=file_path, images=[img])
    return [chunk]


def scrape_spreadsheet(file_path: str, source_type: str) -> List[Chunk]:
    import pandas as pd

    if source_type in ("text/csv", "application/vnd.ms-excel"):
        df = pd.read_csv(file_path)
    elif (
        source_type
        == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    dicts = df.to_dict(orient="records")
    chunks = []
    for i, item in enumerate(dicts):
        # format each row as json along with the row index
        item["row index"] = i
        item_json = json.dumps(item, indent=4)
        chunks.append(Chunk(path=file_path, text=item_json, metadata={"row": i}))
    return chunks


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(secs):02}.{milliseconds:03}"


def _ffmpeg_duration(file_path: str) -> float:
    """Media duration in seconds, parsed from ffmpeg's probe output (ffmpeg is required by whisper)."""
    probe = subprocess.run(
        ["ffmpeg", "-i", file_path], capture_output=True, text=True, errors="ignore"
    )
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", probe.stderr)
    if not match:
        raise ValueError(f"Could not determine duration of {file_path}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def _ffmpeg_frame(file_path: str, seconds: float) -> Image.Image:
    """Grab a single frame at ``seconds`` as a PIL image."""
    png = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", str(seconds), "-i", file_path,
         "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return Image.open(BytesIO(png))


def scrape_video(
    file_path: str,
    verbose: bool = False,
    include_output_images: bool = True,
) -> List[Chunk]:
    """
    Transcribes the whole video with whisper, then emits one chunk per
    MAX_WHISPER_DURATION window with the transcript segments falling in that
    window and a frame from the window's start.
    """
    duration = _ffmpeg_duration(file_path)
    segments = _transcribe(file_path, verbose=verbose)

    chunks = []
    for i in range(math.ceil(duration / MAX_WHISPER_DURATION)):
        start_time = i * MAX_WHISPER_DURATION
        end_time = min(start_time + MAX_WHISPER_DURATION, duration)

        transcript = "\n".join(
            f"[{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}]  {seg['text']}"
            for seg in segments
            if start_time <= seg["start"] < end_time and seg["text"].strip()
        )
        image = _ffmpeg_frame(file_path, start_time) if include_output_images else None

        if transcript or image:
            chunks.append(
                Chunk(
                    path=file_path,
                    text=transcript or None,
                    images=[image] if image else [],
                    metadata={"start": start_time, "end": end_time},
                )
            )

    return chunks


def scrape_audio(file_path: str, verbose: bool = False) -> List[Chunk]:
    segments = _transcribe(file_path, verbose=verbose)

    transcript: List[str] = []
    for segment in segments:
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        if segment["text"].strip():
            transcript.append(f"[{start} --> {end}]  {segment['text']}")
    # join the formatted transcription into a single string
    transcript_text = "\n".join(transcript)
    metadata = {"start": 0.0, "end": segments[-1]["end"]} if segments else {}
    return [Chunk(path=file_path, text=transcript_text, metadata=metadata)]


def scrape_docx(
    file_path: str,
    verbose: bool = False,
    include_output_images: bool = True,
) -> List[Chunk]:
    from docx import Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph
    import csv
    import io

    # helper function to iterate through blocks in the document
    def iter_block_items(parent):
        if parent.__class__.__name__ == "Document":
            parent_elm = parent.element.body
        elif parent.__class__.__name__ == "_Cell":
            parent_elm = parent._tc
        else:
            raise ValueError("Unsupported parent type")
        # iterate through each child element in the parent element
        for child in parent_elm.iterchildren():
            child_elem_class_name = child.__class__.__name__
            if verbose:
                print(f"[thepipe] Found element in docx: {child_elem_class_name}")
            if child_elem_class_name == "CT_P":
                yield Paragraph(child, parent)
            elif child_elem_class_name == "CT_Tbl":
                yield Table(child, parent)

    # helper function to read tables in the document
    def read_docx_tables(tab):
        vf = StringIO()
        writer = csv.writer(vf)
        for row in tab.rows:
            writer.writerow(cell.text for cell in row.cells)
        vf.seek(0)
        return vf.getvalue()

    # read the document
    document = Document(file_path)
    chunks = []
    image_counter = 0

    # Define namespaces
    nsmap = {
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    }

    try:
        # scrape each block in the document to create chunks
        # A block can be a paragraph, table, or image
        for block_index, block in enumerate(iter_block_items(document)):
            block_texts = []
            block_images = []
            if isinstance(block, Paragraph):
                block_texts.append(block.text)
                # "runs" are the smallest units in a paragraph
                for run in block.runs:
                    if "pic:pic" in run.element.xml and include_output_images:
                        # extract images from the paragraph
                        for pic in run.element.findall(".//pic:pic", nsmap):
                            cNvPr = pic.find(".//pic:cNvPr", nsmap)
                            name_attr = (
                                cNvPr.get("name")
                                if cNvPr is not None
                                else f"image_{image_counter}"
                            )
                            blip = pic.find(".//a:blip", nsmap)
                            if blip is not None:
                                embed_attr = blip.get(
                                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                                )
                                if embed_attr:
                                    image_part = document.part.related_parts[embed_attr]
                                    image_data = BytesIO(image_part._blob)
                                    image = Image.open(image_data)
                                    image.load()
                                    block_images.append(image)
                                    image_counter += 1
            elif isinstance(block, Table):
                table_text = read_docx_tables(block)
                block_texts.append(table_text)
            if block_texts or block_images:
                block_text = "\n".join(block_texts).strip()
                if block_text or block_images:
                    chunks.append(
                        Chunk(
                            path=file_path,
                            text=block_text,
                            images=block_images,
                            metadata={
                                "block": block_index,
                                "block_type": "table" if isinstance(block, Table) else "paragraph",
                            },
                        )
                    )
    except Exception as e:
        raise ValueError(f"Error processing DOCX file {file_path}: {e}")
    return chunks


def scrape_pptx(
    file_path: str,
    verbose: bool = False,
    include_output_images: bool = True,
) -> List[Chunk]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    from pptx.shapes.picture import Picture
    from pptx.shapes.autoshape import Shape as AutoShape

    prs = Presentation(file_path)
    chunks = []
    # iterate through each slide in the presentation
    for slide_index, slide in enumerate(prs.slides):
        slide_texts = []
        slide_images = []
        # iterate through each shape in the slide
        for shape in slide.shapes:
            if shape.has_text_frame:
                auto_shape = cast(AutoShape, shape)
                for paragraph in auto_shape.text_frame.paragraphs:
                    text = paragraph.text
                    if len(slide_texts) == 0:
                        text = "# " + text  # header for first text of a slide
                    slide_texts.append(text)
            # extract images from shapes
            if include_output_images and shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                pic = cast(Picture, shape)
                image_data = pic.image.blob
                image = Image.open(BytesIO(image_data))
                slide_images.append(image)
        # add slide to chunks if it has text or images
        if slide_texts or slide_images:
            text = "\n".join(slide_texts).strip()
            if not include_output_images:
                slide_images = []
            chunks.append(
                Chunk(
                    path=file_path,
                    text=text,
                    images=slide_images,
                    metadata={"slide": slide_index + 1},
                )
            )
    # return all chunks
    return chunks


def scrape_ipynb(
    file_path: str,
    verbose: bool = False,
    include_output_images: bool = True,
) -> List[Chunk]:
    with open(file_path, "r", encoding="utf-8") as file:
        notebook = json.load(file)
    chunks = []
    # parse cells in the notebook
    for cell_index, cell in enumerate(notebook["cells"]):
        texts = []
        images: List[Image.Image] = []
        cell_type = cell["cell_type"]
        # parse cell content based on type
        if verbose:
            print(f"[thepipe] Scraping cell {cell_type} from {file_path}")
        if cell_type == "markdown":
            text = "".join(cell["source"])
            if include_output_images:
                images = get_images_from_markdown(text)
            texts.append(text)
        elif cell_type == "code":
            source = "".join(cell["source"])
            texts.append(source)
            output_texts = []
            # code cells can have outputs
            if "outputs" in cell:
                for output in cell["outputs"]:
                    if (
                        include_output_images
                        and "data" in output
                        and "image/png" in output["data"]
                    ):
                        image_data = output["data"]["image/png"]
                        image = Image.open(BytesIO(base64.b64decode(image_data)))
                        images.append(image)
                    elif "data" in output and "text/plain" in output["data"]:
                        output_text = "".join(output["data"]["text/plain"])
                        output_texts.append(output_text)
            if output_texts:
                texts.extend(output_texts)
        elif cell_type == "raw":
            text = "".join(cell["source"])
            texts.append(text)
        if texts or images:
            text = "\n".join(texts).strip()
            chunks.append(
                Chunk(
                    path=file_path,
                    text=text,
                    images=images,
                    metadata={"cell": cell_index, "cell_type": cell_type},
                )
            )
    return chunks
