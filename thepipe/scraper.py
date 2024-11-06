import base64
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
import io
import math
import re
from typing import  List, Dict, Any, Callable, Optional,Tuple, Generator, Union
import glob
import os
import tempfile
from urllib.parse import urlparse
import zipfile
from PIL import Image
import requests
import json
from .core import HOST_URL, THEPIPE_API_KEY, HOST_IMAGES, Chunk, make_image_url
from .chunker import chunk_by_page, chunk_by_document, chunk_by_section, chunk_semantic, chunk_by_keywords
import tempfile
import mimetypes
import dotenv
import shutil
from magika import Magika
import markdownify
dotenv.load_dotenv()
from enum import Enum, auto
from .enums import YouTubeEnum

# Global variables to hold the imported video modules
yt_dlp = None

def initialize_video_processing():
    global yt_dlp
    if yt_dlp is None:
        import yt_dlp

webvtt = None

def initialize_subtitle_libraries():
    global webvtt
    if webvtt is None:
        import webvtt

@lru_cache(maxsize=1)
def get_subtitle_parser():
    initialize_subtitle_libraries()
    return webvtt

FOLDERS_TO_IGNORE = ['*node_modules.*', '.*venv.*', '.*\.git.*', '.*\.vscode.*', '.*pycache.*']
FILES_TO_IGNORE = ['package-lock.json', '.gitignore', '.*\.bin', '.*\.pyc', '.*\.pyo', '.*\.exe', '.*\.dll', '.*\.ipynb_checkpoints']
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", None)
USER_AGENT_STRING: str = os.getenv("USER_AGENT_STRING", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")
MAX_WHISPER_DURATION = 600 # 10 minutes
TWITTER_DOMAINS = ['https://twitter.com', 'https://www.twitter.com', 'https://x.com', 'https://www.x.com']
GITHUB_DOMAINS = ['https://github.com', 'https://www.github.com']
SCRAPING_PROMPT = os.getenv("EXTRACTION_PROMPT", """An open source document is given. Output the entire extracted contents from the document in detailed markdown format.
Be sure to correctly format markdown for headers, paragraphs, lists, tables, menus, equations, full text contents, etc.
Always reply immediately with only markdown. Do not output anything else.""")
DEFAULT_AI_MODEL = os.getenv("DEFAULT_AI_MODEL", "gpt-4o-mini")
FILESIZE_LIMIT_MB = os.getenv("FILESIZE_LIMIT_MB", 50)
VIDEO_PLATFORMS = {
    "youtube", "youtu", "netflix", "amazon", "hulu", "disneyplus", "vimeo", 
    "twitch", "tiktok", "dailymotion", "vevo", "kick",
    "crunchyroll", "peacocktv", "hbomax", "roku", "pluto", 
    "tubitv", "iqiyi", "v.qq", "youku", "bilibili", "flicknexs", 
    "brightcove", "wistia", "jwplayer", "kaltura", "panopto", "vidyard", 
    "vk", "rutube", "metacafe", "veoh", "ustream", "livestream", "periscope", 
    "mixer", "younow", "smashcast", "niconico", "vlive", "afreecatv", 
    "kakao", "naver", "line", "iflix", "hooq", "viu", "mubi"
}

def is_video_platform(url: str) -> bool:
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return any(platform in domain for platform in VIDEO_PLATFORMS)

class YouTubeMetadata(Enum):
    TITLE = 'title'
    DESCRIPTION = 'description'
    UPLOAD_DATE = 'upload_date'
    UPLOADER = 'uploader'
    VIEW_COUNT = 'view_count'
    LIKE_COUNT = 'like_count'
    DURATION = 'duration'
    TAGS = 'tags'
    CATEGORY = 'categories'
    THUMBNAIL_URL = 'thumbnail'

    @classmethod
    def extract(cls, info: Dict[str, Any]) -> Dict[str, Any]:
        return {field.name.lower(): info.get(field.value, 'N/A') for field in cls}

DEFAULT_METADATA = list(YouTubeMetadata)

def format_metadata(metadata: Dict[str, Any]) -> str:
    metadata_text = "Video Metadata:\n\n"
    for key, value in metadata.items():
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        elif not isinstance(value, str):
            value = str(value)
        metadata_text += f"{key.capitalize()}: {value}\n"
    return metadata_text

def detect_source_type(source: str) -> str:
    # otherwise, try to detect the file type by its extension
    _, extension = os.path.splitext(source)
    if extension:
        if extension == '.ipynb':
            # special case for notebooks, mimetypes is not familiar
            return 'application/x-ipynb+json'
        guessed_mimetype = mimetypes.guess_type(source)[0]
        if guessed_mimetype:
            return guessed_mimetype
    # if that fails, try AI detection with Magika
    magika = Magika()
    with open(source, 'rb') as file:
        result = magika.identify_bytes(file.read())
    mimetype = result.output.mime_type
    return mimetype

def scrape_file(filepath: str, ai_extraction: bool = False, text_only: bool = False, verbose: bool = False, local: bool = False, chunking_method: Optional[Callable] = chunk_by_page, ai_model: Optional[str] = DEFAULT_AI_MODEL, options: Optional[Dict[str, Any]] = None) -> List[Chunk]:

    if not local:
        with open(filepath, 'rb') as f:
            response = requests.post(
                url=f"{HOST_URL}/scrape",
                headers={"Authorization": f"Bearer {THEPIPE_API_KEY}"},
                files={'files': (os.path.basename(filepath), f)},
                data={
                    'text_only': str(text_only).lower(),
                    'ai_extraction': str(ai_extraction).lower(),
                    'chunking_method': chunking_method.__name__,
                    'options': json.dumps(options) if options else None
                }
            )
        if "error" in response.content.decode('utf-8'):
            error_message = json.loads(response.content.decode('utf-8'))['error']
            raise ValueError(f"Error scraping {filepath}: {error_message}")
        response.raise_for_status()
        chunks = []
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if 'result' in data:
                    chunk = Chunk(
                        path=data['result']['source'],
                        texts=[content['text'] for content in data['result']['content'] if content['type'] == 'text'],
                        images=[Image.open(BytesIO(base64.b64decode(content['image_url'].split(',')[1]))) 
                                for content in data['result']['content'] if content['type'] == 'image_url']
                    )
                    chunks.append(chunk)
        return chunks

    # returns chunks of scraped content from any source (file, URL, etc.)
    scraped_chunks = []
    source_type = detect_source_type(filepath)
    if source_type is None:
        if verbose:
            print(f"[thepipe] Unsupported source type: {filepath}")
        return scraped_chunks
    if verbose: 
        print(f"[thepipe] Scraping {source_type}: {filepath}...")
    if source_type == 'application/pdf':
        scraped_chunks = scrape_pdf(file_path=filepath, ai_extraction=ai_extraction, text_only=text_only, verbose=verbose, ai_model=ai_model, options=options)
    elif source_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        scraped_chunks = scrape_docx(file_path=filepath, verbose=verbose, text_only=text_only,)
    elif source_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        scraped_chunks = scrape_pptx(file_path=filepath, verbose=verbose, text_only=text_only,)
    elif source_type.startswith('image/'):
        scraped_chunks = scrape_image(file_path=filepath, text_only=text_only,)
    elif source_type.startswith('application/vnd.ms-excel') or source_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        scraped_chunks = scrape_spreadsheet(file_path=filepath, source_type=source_type,)
    elif source_type == 'application/x-ipynb+json':
        scraped_chunks = scrape_ipynb(file_path=filepath, verbose=verbose, text_only=text_only,)
    elif source_type == 'application/zip' or source_type == 'application/x-zip-compressed':
        scraped_chunks = scrape_zip(file_path=filepath, verbose=verbose, ai_extraction=ai_extraction, text_only=text_only, local=local,)
    elif source_type.startswith('video/'):
        scraped_chunks = scrape_video(file_path=filepath, verbose=verbose, text_only=text_only, options=options)
    elif source_type.startswith('audio/'):
        scraped_chunks = scrape_audio(file_path=filepath, verbose=verbose, options=options)
    elif source_type.startswith('text/html'):
        scraped_chunks = scrape_html(file_path=filepath, verbose=verbose, text_only=text_only)
    elif source_type.startswith('text/'):
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
    scraped_chunks = chunking_method(scraped_chunks)
    return scraped_chunks

def scrape_html(file_path: str, verbose: bool = False, text_only: bool = False) -> List[Chunk]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        html_content = file.read()
    markdown_content = markdownify.markdownify(html_content, heading_style="ATX")
    if text_only:
        return [Chunk(path=file_path, texts=[markdown_content])]
    images = get_images_from_markdown(html_content)
    return [Chunk(path=file_path, texts=[markdown_content], images=images)]

def scrape_plaintext(file_path: str) -> List[Chunk]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        text = file.read()
    return [Chunk(path=file_path, texts=[text])]

def scrape_directory(dir_path: str, include_regex: Optional[str] = None, include_patterns: Optional[List[str]] = None, verbose: bool = False, ai_extraction: bool = False, text_only: bool = False, local: bool = False, options: Optional[Dict[str, Any]] = None) -> List[Chunk]:
    extraction = []
    
    if include_patterns is not None:
        # Use glob patterns
        all_files = []
        for pattern in include_patterns:
            pattern_path = os.path.join(dir_path, '**', pattern)
            all_files.extend(glob.glob(pattern_path, recursive=True))
    elif include_regex is not None:
        # Use regex
        all_files = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if re.search(include_regex, file_path, re.IGNORECASE):
                    all_files.append(file_path)
    else:
        # Neither pattern nor regex specified, include all files
        all_files = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                all_files.append(os.path.join(root, file))
    
    # Ensure we're only dealing with files
    all_files = [f for f in all_files if os.path.isfile(f)]
    
    if verbose:
        print(f"[thepipe] Found {len(all_files)} files to process in {dir_path}")
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            lambda file_path: scrape_file(
                filepath=file_path, 
                ai_extraction=ai_extraction, 
                text_only=text_only, 
                verbose=verbose, 
                local=local,
                options=options
            ), 
            all_files
        )
        for result in results:
            extraction.extend(result)
    
    return extraction

def scrape_zip(file_path: str, include_regex: Optional[str] = None, verbose: bool = False, ai_extraction: bool = False, text_only: bool = False, local: bool = False) -> List[Chunk]:
    chunks = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        chunks =scrape_directory(dir_path=temp_dir, include_regex=include_regex, verbose=verbose, ai_extraction=ai_extraction, text_only=text_only, local=local)
    return chunks

def scrape_pdf(file_path: str, ai_extraction: Optional[bool] = False, text_only: Optional[bool] = False, ai_model: Optional[str] = DEFAULT_AI_MODEL, verbose: Optional[bool] = False) -> List[Chunk]:    
    chunks = []
    MAX_PAGES = 128

    if ai_extraction:
        from collections import OrderedDict
        import concurrent.futures
        import fitz
        from openai import OpenAI

        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            num_pages = len(doc)

            if num_pages > MAX_PAGES:
                return f"Error: PDF has {num_pages} pages (max is {MAX_PAGES} for AI extraction)."

            openrouter_client = OpenAI(
                base_url=os.environ.get("LLM_SERVER_BASE_URL"),
                api_key=os.environ["LLM_SERVER_API_KEY"],
            )

            def process_page(page_num):
                page = doc[page_num]
                text = page.get_text()
                pix = page.get_pixmap()
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": make_image_url(image, host_images=HOST_IMAGES)},
                            {"type": "text", "text": f"```{text}```\n{SCRAPING_PROMPT}"},
                        ]
                    },
                ]
                response = openrouter_client.chat.completions.create(
                    model=ai_model,
                    messages=messages,
                    temperature=0
                )
                try:
                    llm_response = response.choices[0].message.content.strip()
                    
                    # remove markdown codeboxes if they are present
                    if llm_response.startswith("```markdown"):
                        llm_response = llm_response[len("```markdown"):]
                    elif llm_response.startswith("```"):
                        llm_response = llm_response[len("```"):]
                    if llm_response.endswith("```"):
                        llm_response = llm_response[:-len("```")]
                    llm_response = llm_response.strip()

                    return page_num, llm_response, image
                except Exception as e:
                    raise ValueError(f"{e} (unable to read LLM response: {response})")

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_page, page_num) for page_num in range(num_pages)]
                page_results = OrderedDict()
                for future in concurrent.futures.as_completed(futures):
                    page_num, llm_response, image = future.result()
                    page_results[page_num] = (llm_response, image)

            chunks = []
            for page_num in sorted(page_results.keys()):
                llm_response, image = page_results[page_num]
                chunks.append(Chunk(path=file_path, texts=[llm_response], images=[] if text_only else [image]))

            return chunks
    else:
        # if not using AI extraction, for each page, extract markdown and (optionally) full page images
        import fitz
        doc = fitz.open(file_path)
        try:
            import pymupdf4llm
            md_reader = pymupdf4llm.helpers.pymupdf_rag.to_markdown(doc, page_chunks=True)
            for i, page in enumerate(doc):
                text = md_reader[i]["text"]
                # remove excessive newlines
                text = re.sub(r'\n{3,}', '\n\n', text)
                text = text.strip()
                if text_only:
                    chunks.append(Chunk(path=file_path, texts=[text]))
                else:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    chunks.append(Chunk(path=file_path, texts=[text], images=[img]))
            doc.close()
        except:
            # try with default pumupdf, since pymupdf4llm often fails
            for i in range(len(doc)):
                page = doc[i]
                text = page.get_text()
                if text_only:
                    chunks.append(Chunk(path=file_path, texts=[text]))
                else:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    chunks.append(Chunk(path=file_path, texts=[text], images=[img]))
            doc.close()
    return chunks

def get_images_from_markdown(text: str) -> List[Image.Image]:
    image_urls = re.findall(r"!\[.*?\]\((.*?)\)", text)
    images = []
    for url in image_urls:
        extension = os.path.splitext(urlparse(url).path)[1]
        if extension in {'.jpg', '.jpeg', '.png'}:
            img = Image.open(requests.get(url, stream=True).raw)
        else:
            # ignore incompatible image extractions
            continue
        images.append(img)
    return images

def scrape_image(file_path: str, text_only: bool = False) -> List[Chunk]:
    import pytesseract
    img = Image.open(file_path)
    img.load()  # needed to close the file
    chunks = []
    if text_only:
        text = pytesseract.image_to_string(img)
        chunks.append(Chunk(path=file_path, texts=[text]))
    else:
        chunks.append(Chunk(path=file_path, images=[img]))
    return chunks

def scrape_spreadsheet(file_path: str, source_type: str) -> List[Chunk]:
    import pandas as pd
    if source_type == 'application/vnd.ms-excel':
        df = pd.read_csv(file_path)
    elif source_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    dicts = df.to_dict(orient='records')
    chunks = []
    for i, item in enumerate(dicts):
        # format each row as json along with the row index
        item['row index'] = i
        item_json = json.dumps(item, indent=4)
        chunks.append(Chunk(path=file_path, texts=[item_json]))
    return chunks

def ai_extract_webpage_content(url: str, text_only: Optional[bool] = False, verbose: Optional[bool] = False, ai_model: Optional[str] = DEFAULT_AI_MODEL) -> Chunk:
    from playwright.sync_api import sync_playwright
    from openai import OpenAI

    #import modal
    #app_name = "scrape-ui"
    #function_name = "get_ui_layout_preds"
    #fn = modal.Function.lookup(app_name, function_name)
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(user_agent=USER_AGENT_STRING)
        page = context.new_page()
        page.goto(url, wait_until='domcontentloaded')
        
        viewport_height = page.viewport_size['height']
        total_height = page.evaluate("document.body.scrollHeight")
        current_scroll_position = 0
        scrolldowns, max_scrolldowns = 0, 3
        images = []

        while current_scroll_position < total_height and scrolldowns < max_scrolldowns:
            page.wait_for_timeout(1000)
            screenshot = page.screenshot(full_page=False)
            img = Image.open(io.BytesIO(screenshot))
            images.append(img)

            current_scroll_position += viewport_height
            page.evaluate(f"window.scrollTo(0, {current_scroll_position})")
            scrolldowns += 1
            total_height = page.evaluate("document.body.scrollHeight")
        
        browser.close()

    if images:
        # Vertically stack the images
        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)
        stacked_image = Image.new('RGB', (max_width, total_height))
        y_offset = 0
        for img in images:
            stacked_image.paste(img, (0, y_offset))
            y_offset += img.height

        # Process the stacked image with the UI model
        #figures = fn.remote(stacked_image)

        # Process the stacked image with VLM
        openrouter_client = OpenAI(
            base_url=os.environ["LLM_SERVER_BASE_URL"],
            api_key=os.environ["LLM_SERVER_API_KEY"],
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": make_image_url(stacked_image, host_images=HOST_IMAGES)},
                    {"type": "text", "text": SCRAPING_PROMPT},
                ]
            },
        ]
        response = openrouter_client.chat.completions.create(
            model=ai_model,
            messages=messages,
            temperature=0
        )
        llm_response = response.choices[0].message.content
        chunk = Chunk(path=url, texts=[llm_response], images=[stacked_image])
    else:
        raise ValueError("Model received 0 images from webpage")

    return chunk

def extract_page_content(url: str, text_only: bool = False, verbose: bool = False, options: Optional[Dict[str, Any]] = None) -> Chunk:
    from urllib.parse import urlparse
    from bs4 import BeautifulSoup
    from playwright.sync_api import sync_playwright
    import base64
    import requests
    
    texts = []
    images = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(user_agent="USER_AGENT_STRING")
        page = context.new_page()
        page.goto(url, wait_until='domcontentloaded')
        
        # Scroll to the bottom of the page to load dynamic content
        viewport_height = page.viewport_size['height']
        total_height = page.evaluate("document.body.scrollHeight")
        current_scroll_position = 0
        scrolldowns, max_scrolldowns = 0, 20  # Finite to prevent infinite scroll
        
        while current_scroll_position < total_height and scrolldowns < max_scrolldowns:
            page.wait_for_timeout(1000)  # Wait for dynamic content to load
            current_scroll_position += viewport_height
            page.evaluate(f"window.scrollTo(0, {current_scroll_position})")
            scrolldowns += 1
            total_height = page.evaluate("document.body.scrollHeight")
        
        # Extract HTML content
        html_content = page.content()
        
        # Convert HTML to Markdown
        soup = BeautifulSoup(html_content, 'html.parser')
        markdown_content = markdownify.markdownify(str(soup), heading_style="ATX")
        
        # Remove excessive newlines in the markdown
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
        markdown_content = markdown_content.strip()

        texts.append(markdown_content)
        
        if not text_only:
            # Extract images from the page using heuristics
            for img in page.query_selector_all('img'):
                img_path = img.get_attribute('src')
                if not img_path:
                    continue
                if img_path.startswith('data:image'):
                    # Save base64 image to PIL Image
                    decoded_data = base64.b64decode(img_path.split(',')[1])
                    try:
                        image = Image.open(BytesIO(decoded_data))
                        images.append(image)
                    except Exception as e:
                        if verbose: print(f"[thepipe] Ignoring error loading image {img_path}: {e}")
                        continue  # Ignore incompatible image extractions
                else:
                    try:
                        image = Image.open(requests.get(img_path, stream=True).raw)
                        images.append(image)
                    except:
                        if 'https://' not in img_path and 'http://' not in img_path:
                            try:
                                while img_path.startswith('/'):
                                    img_path = img_path[1:]
                                path_with_schema = urlparse(url).scheme + "://" + img_path
                                image = Image.open(requests.get(path_with_schema, stream=True).raw)
                                images.append(image)
                            except:
                                try:
                                    path_with_schema_and_netloc = urlparse(url).scheme + "://" + urlparse(url).netloc + "/" + img_path
                                    image = Image.open(requests.get(path_with_schema_and_netloc, stream=True).raw)
                                    images.append(image)
                                except:
                                    if verbose: print(f"[thepipe] Ignoring error loading image {img_path}")
                                    continue  # Ignore incompatible image extractions
                        else:
                            if verbose: print(f"[thepipe] Ignoring error loading image {img_path}")
                            continue  # Ignore incompatible image extractions
                
        browser.close()
    
    return Chunk(path=url, texts=texts, images=images)

# TODO: deprecate this in favor of Chunk.from_json or Chunk.from_message
def create_chunk_from_data(result: Dict, host_images: bool) -> Chunk:
    texts = [content['text'] for content in result['content'] if content['type'] == 'text']
    
    images = []
    for content in result['content']:
        if content['type'] == 'image_url':
            if host_images:
                # If images are hosted, we keep the URL as is
                images.append(content['image_url'])
            else:
                # If images are not hosted, we decode the base64 string
                image_data = content['image_url'].split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(image_data)))
                images.append(image)
    
    return Chunk(
        path=result['source'],
        texts=texts,
        images=images
    )

def scrape_url(url: str, text_only: bool = False, ai_extraction: bool = False, verbose: bool = False, local: bool = False, chunking_method: Optional[Callable] = chunk_by_page, options: Optional[Dict[str, Any]] = None) -> List[Chunk]:
    if not local:
        endpoint = f"{HOST_URL}/scrape"
        headers = {
            "Authorization": f"Bearer {THEPIPE_API_KEY}"
        }
        data = {
            "text_only": str(text_only).lower(),
            "ai_extraction": str(ai_extraction).lower(),
            "chunking_method": chunking_method.__name__
        }
        if options:
            data["options"] = json.dumps(options)
        data["urls"] = url
        response = requests.post(endpoint, headers=headers, data=data, stream=True)
        if "error" in response.content.decode('utf-8'):
            error_message = json.loads(response.content.decode('utf-8'))['error']
            raise ValueError(f"Error scraping {url}: {error_message}")
        response.raise_for_status()
        results = []
        for line in response.iter_lines():
            if line:
                chunk_data = json.loads(line)
                results.append(chunk_data['result'])
        return results
    if is_video_platform(url):
        return scrape_youtube(url, text_only=text_only, verbose=verbose, options=options)
    elif any(url.startswith(domain) for domain in TWITTER_DOMAINS):
        extraction = scrape_tweet(url=url, text_only=text_only, verbose=verbose, options=options)
    elif any(url.startswith(domain) for domain in GITHUB_DOMAINS):
        extraction = scrape_github(github_url=url, text_only=text_only, ai_extraction=ai_extraction, verbose=verbose, options=options)
    else:
        # Handle other types of content
        parsed_url = urlparse(url)
        file_extension = parsed_url.path.split('.')[-1].lower()
        if file_extension in ['pdf', 'docx', 'txt', 'csv', 'xlsx']:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, os.path.basename(url))
                response = requests.get(url)
                if FILESIZE_LIMIT_MB and int(response.headers['Content-Length']) > FILESIZE_LIMIT_MB * 1024 * 1024:
                    raise ValueError(f"File size exceeds {FILESIZE_LIMIT_MB} MB limit.")
                with open(file_path, 'wb') as file:
                    file.write(response.content)
                extraction = scrape_file(filepath=file_path, ai_extraction=ai_extraction, text_only=text_only, verbose=verbose, local=local, chunking_method=chunking_method, options=options)
        else:
            chunk = extract_page_content(url=url, text_only=text_only, verbose=verbose, options=options)
            chunks = chunking_method([chunk])
            # if no text or images were extracted, return error
            if not any(chunk.texts for chunk in chunks) and not any(chunk.images for chunk in chunks):
                raise ValueError("No content extracted from URL.")
            return chunks
    return extraction
    
def extract_metadata(video_info: Dict[str, Any], metadata_fields: List) -> Dict[str, Any]:
    metadata = {}
    for field in metadata_fields:
        metadata[field] = video_info.get(field, 'N/A')
    return metadata

def format_metadata(metadata: dict) -> str:
    metadata_text = "Video Metadata:\n\n"
    for key, value in metadata.items():
        if isinstance(value, list):
            value = ', '.join(map(str, value))
        elif not isinstance(value, str):
            value = str(value)
        metadata_text += f"{key.capitalize()}: {value}\n"
    return metadata_text

def scrape_video(file_path: str, verbose: bool = False, text_only: bool = False) -> List[Chunk]:
    import whisper
    from moviepy.editor import VideoFileClip

    model = whisper.load_model("base")
    video = VideoFileClip(file_path)
    num_chunks = math.ceil(video.duration / MAX_WHISPER_DURATION)
    chunks = []
    # split the video into chunks of fixed duration
    # here, we transcribe each chunk and extract its frame
    try:
        for i in range(num_chunks):
            start_time = i * MAX_WHISPER_DURATION
            end_time = start_time + MAX_WHISPER_DURATION
            if end_time > video.duration:
                end_time = video.duration
            # get the frame in the middle of the chunk
            frame_time = (start_time + end_time) / 2
            frame = video.get_frame(frame_time)
            image = Image.fromarray(frame)
            # save the audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                audio_path = temp_audio_file.name
            audio = video.subclip(start_time, end_time).audio
            transcription = None
            # transcribe it
            if audio is not None:
                audio.write_audiofile(audio_path, codec='pcm_s16le')
                result = model.transcribe(audio=audio_path, verbose=verbose)
                # Format transcription with timestamps
                formatted_transcription = []
                for segment in result['segments']:
                    start = format_timestamp(segment['start'], i, MAX_WHISPER_DURATION)
                    end = format_timestamp(segment['end'], i, MAX_WHISPER_DURATION)
                    formatted_transcription.append(f"[{start} --> {end}]  {segment['text']}")
                transcription = '\n'.join(formatted_transcription)
                os.remove(audio_path)
            texts = [transcription] if transcription else []
            images = [image] if not text_only else []
            if texts or images:
                chunks.append(Chunk(path=file_path, texts=texts, images=images))
    finally:
        video.close()
    return chunks

def scrape_youtube(url: str, text_only: Optional[Union[bool, str]] = None, verbose: bool = False, metadata_fields: Optional[List[YouTubeEnum]] = None, options: Optional[Dict[str, Any]] = None) -> List[Chunk]:
    initialize_video_processing()
    
    ydl_opts = {
        'quiet': not verbose,
        'ignoreerrors': True,
        'extract_flat': 'in_playlist',
        'outtmpl': '%(title)s.%(ext)s',
        'subtitlesformat': 'vtt',
        'skip_download': True,  # Always skip video download initially
    }

    if text_only == 'transcribe':
        ydl_opts.update({
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        })
    else:
        ydl_opts.update({
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-orig'],
        })

    if options and 'youtube' in options:
        ydl_opts.update(YouTubeEnum.process_options(options['youtube'], bool(text_only), verbose))

    chunks = []
    with tempfile.TemporaryDirectory() as temp_dir:
        ydl_opts['outtmpl'] = os.path.join(temp_dir, '%(title)s.%(ext)s')

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if 'entries' in info:  # It's a playlist
                    videos = info['entries']
                else:  # Single video
                    videos = [info]

                for video in videos:
                    video_chunks = process_video(ydl, video, temp_dir, text_only, verbose, metadata_fields)
                    chunks.extend(video_chunks)

        except Exception as e:
            if verbose:
                print(f"[thepipe] Error processing content: {str(e)}")
            chunks.append(Chunk(path=url, texts=[f"Error: Unable to process content. {str(e)}"]))

    return chunks

def process_video(ydl, video_info: Dict[str, Any], temp_dir: str, text_only: Optional[Union[bool, str]], verbose: bool, metadata_fields: Optional[List[YouTubeEnum]] = None) -> List[Chunk]:
    video_chunks = []
    video_url = video_info.get('webpage_url') or video_info.get('url')
    if not video_url:
        if verbose:
            print(f"[thepipe] Skipping video with no URL")
        return video_chunks

    try:
        # Extract metadata
        metadata = YouTubeEnum.extract_metadata(video_info, metadata_fields)
        metadata_chunk = Chunk(path=video_url, texts=[YouTubeEnum.format_metadata(metadata)])
        video_chunks.append(metadata_chunk)

        if text_only == 'transcribe':
            # Directly download audio and transcribe
            ydl.params['skip_download'] = False
            ydl.process_ie_result(video_info, download=True)
            audio_file = find_audio_file(temp_dir, video_info['title'])
            if audio_file:
                transcription_chunks = scrape_audio(audio_file, verbose=verbose)
                video_chunks.extend(transcription_chunks)
            else:
                if verbose:
                    print(f"[thepipe] Failed to download audio for transcription: {video_url}")
                video_chunks.append(Chunk(path=video_url, texts=["No transcription available"]))
        elif text_only in [True, 'ai', 'uploaded']:
            # Try to get subtitles
            ydl.params['skip_download'] = True
            ydl.process_ie_result(video_info, download=True)
            subtitle_files = find_subtitle_files(temp_dir, video_info['title'])

            if subtitle_files:
                for subtitle_file in subtitle_files:
                    subtitle_chunks = clean_subtitles(subtitle_file, video_url, debug=verbose)
                    if subtitle_chunks:
                        video_chunks.extend(subtitle_chunks)
                        break  # Use the first meaningful subtitle file
                    elif verbose:
                        print(f"[thepipe] Subtitle file found but content is not meaningful: {subtitle_file}")

            if len(video_chunks) == 1:  # Only metadata chunk
                if verbose:
                    print(f"[thepipe] No meaningful subtitles found for {text_only} option. Skipping transcription.")
                video_chunks.append(Chunk(path=video_url, texts=[f"No {text_only} subtitles available"]))
        else:  # text_only is False or None
            # Try to get subtitles first
            ydl.params['skip_download'] = True
            ydl.process_ie_result(video_info, download=True)
            subtitle_files = find_subtitle_files(temp_dir, video_info['title'])

            if subtitle_files:
                for subtitle_file in subtitle_files:
                    subtitle_chunks = clean_subtitles(subtitle_file, video_url)
                    if subtitle_chunks:
                        video_chunks.extend(subtitle_chunks)
                        break  # Use the first meaningful subtitle file
                    elif verbose:
                        print(f"[thepipe] Subtitle file found but content is not meaningful: {subtitle_file}")

            # If no meaningful subtitles, download video
            if len(video_chunks) == 1:  # Only metadata chunk
                ydl.params['skip_download'] = False
                ydl.params['format'] = 'bestvideo+bestaudio/best'
                ydl.process_ie_result(video_info, download=True)
                
                video_file = find_video_file(temp_dir, video_info['title'])
                if video_file:
                    if verbose:
                        print(f"[thepipe] Processing video file: {os.path.basename(video_file)}")
                    video_chunks.extend(scrape_video(video_file, verbose=verbose, text_only=False))
                else:
                    video_chunks.append(Chunk(path=video_url, texts=["No video content available"]))

    except Exception as e:
        if verbose:
            print(f"[thepipe] Error processing video {video_url}: {str(e)}")
        video_chunks.append(Chunk(path=video_url, texts=[f"Error: Unable to process video. {str(e)}"]))

    return video_chunks

def clean_subtitles(subtitle_file: str, video_url: str, debug: bool = False) -> List[Chunk]:
    webvtt = get_subtitle_parser()
    captions = webvtt.read(subtitle_file)
    
    cleaned_entries = []
    previous_entry = None
    
    for caption in captions:
        text = re.sub(r'<[^>]+>', '', caption.text)
        words = text.split()
        
        if previous_entry:
            # Check for significant overlap with previous text
            overlap = len(set(previous_entry['text'].split()[-5:]) & set(words)) / min(5, len(words))
            if overlap < 0.5:  # Less than 50% overlap
                cleaned_entries.append(previous_entry)
                previous_entry = {'start': caption.start, 'end': caption.end, 'text': ' '.join(words)}
            else:
                # Merge with previous text, removing duplicates
                merged_words = previous_entry['text'].split() + [word for word in words if word not in previous_entry['text'].split()[-5:]]
                previous_entry['end'] = caption.end
                previous_entry['text'] = ' '.join(merged_words)
        else:
            previous_entry = {'start': caption.start, 'end': caption.end, 'text': ' '.join(words)}
    
    if previous_entry:
        cleaned_entries.append(previous_entry)
    
    chunks = []
    for entry in cleaned_entries:
        formatted_text = f"[{entry['start']} --> {entry['end']}]  {entry['text'].strip()}"
        chunks.append(Chunk(path=video_url, texts=[formatted_text]))
    
    if debug:
        with open("original_transcript.txt", "w", encoding="utf-8") as f:
            for caption in captions:
                f.write(f"[{caption.start} --> {caption.end}] {caption.text}\n")
    
    return chunks

def find_subtitle_files(directory: str, video_title: str) -> List[str]:
    subtitle_files = []
    for file in os.listdir(directory):
        if file.startswith(video_title) and file.endswith('.vtt'):
            subtitle_files.append(os.path.join(directory, file))
    return subtitle_files

def find_audio_file(directory: str, video_title: str) -> Optional[str]:
    for file in os.listdir(directory):
        if file.startswith(video_title) and file.endswith(('.mp3', '.m4a', '.wav')):
            return os.path.join(directory, file)
    return None

def find_video_file(directory: str, video_title: str) -> Optional[str]:
    for file in os.listdir(directory):
        if file.startswith(video_title) and file.endswith(('.mp4', '.webm', '.mkv')):
            return os.path.join(directory, file)
    return None

def is_subtitle_meaningful(subtitle_text: str) -> bool:
    # Remove timestamps and empty lines
    content_lines = [line.strip() for line in subtitle_text.split('\n') 
                     if line.strip() and not line.strip().replace('->', '').replace(':', '').isdigit()]
    
    # Check if there's meaningful content (more than just a few short words)
    return len(content_lines) > 5 and any(len(line.split()) > 3 for line in content_lines)

def scrape_audio(file_path: str, verbose: bool = False, options: Optional[Dict[str, Any]] = None) -> List[Chunk]:
    import whisper
    model = whisper.load_model("base")
    if verbose:
        print(f"[thepipe] Transcribing audio file: {file_path}")
    result = model.transcribe(audio=file_path, verbose=verbose)
    # Format transcription with timestamps
    transcript = []
    for segment in result['segments']:
        start = format_timestamp(segment['start'], 0, 0)
        end = format_timestamp(segment['end'], 0, 0)
        if segment['text'].strip():
            transcript.append(f"[{start} --> {end}]  {segment['text']}")
    # join the formatted transcription into a single string
    transcription_text = '\n'.join(transcript)
    if verbose:
        print(f"[thepipe] Transcription completed for {file_path}")
    return [Chunk(path=file_path, texts=[transcription_text])]

def format_timestamp(seconds: float, chunk_index: int = 0, chunk_duration: int = 0) -> str:
    total_seconds = chunk_index * chunk_duration + seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02}.{milliseconds:03}"

def scrape_github(github_url: str, include_regex: Optional[str] = None, text_only: bool = False, ai_extraction: bool = False, branch: str = 'main', verbose: bool = False) -> List[Chunk]:
    files_contents = []
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable is not set.")
    # make new tempdir for cloned repo
    with tempfile.TemporaryDirectory() as temp_dir:
        # requires git
        os.system(f"git clone {github_url} {temp_dir} --quiet")
        files_contents = scrape_directory(dir_path=temp_dir, include_regex=include_regex, verbose=verbose, ai_extraction=ai_extraction, text_only=text_only, local=True)
    return files_contents

def scrape_docx(file_path: str, verbose: bool = False, text_only: bool = False) -> List[Chunk]:
    from docx import Document
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph
    import csv
    import io
    import weakref

    # helper function to iterate through blocks in the document
    def iter_block_items(parent):
        if parent.__class__.__name__ == 'Document':
            parent_elm = parent.element.body
        elif parent.__class__.__name__ == '_Cell':
            parent_elm = parent._tc
        else:
            raise ValueError("Unsupported parent type")
        # iterate through each child element in the parent element
        for child in parent_elm.iterchildren():
            if child.__class__.__name__ == 'CT_P':
                yield Paragraph(child, parent)
            elif child.__class__.__name__ == 'CT_Tbl':
                yield Table(child, parent)

    # helper function to read tables in the document
    def read_docx_tables(tab):
        vf = io.StringIO()
        writer = csv.writer(vf)
        for row in tab.rows:
            writer.writerow(cell.text for cell in row.cells)
        vf.seek(0)
        return vf.getvalue()

    # read the document
    document = Document(file_path)

    # Define namespaces
    nsmap = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    }
    chunks = []
    image_counter = 0

    try:
        # scrape each block in the document to create chunks
        for block in iter_block_items(document):
            block_texts = []
            block_images = []
            if isinstance(block, Paragraph):
                block_texts.append(block.text)
                if not text_only:
                    # "runs" are the smallest units in a paragraph
                    for run in block.runs:
                        if 'pic:pic' in run.element.xml:
                            # extract images from the paragraph
                            for pic in run.element.findall('.//pic:pic', nsmap):
                                cNvPr = pic.find('.//pic:cNvPr', nsmap)
                                name_attr = cNvPr.get("name") if cNvPr is not None else f"image_{image_counter}"
                                
                                blip = pic.find('.//a:blip', nsmap)
                                if blip is not None:
                                    embed_attr = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                                    if embed_attr:
                                        image_part = document.part.related_parts[embed_attr]
                                        image_data = io.BytesIO(image_part._blob)
                                        image = Image.open(image_data)
                                        image.load()
                                        block_images.append(image)  # Append the image directly, not a weak reference
                                        image_counter += 1
            elif isinstance(block, Table):
                table_text = read_docx_tables(block)
                block_texts.append(table_text)
            if block_texts or block_images:
                chunks.append(Chunk(path=file_path, texts=block_texts, images=block_images))

    finally:
        # Close any open image files
        for chunk in chunks:
            for img_ref in chunk.images:
                img = img_ref() if isinstance(img_ref, weakref.ReferenceType) else img_ref
                if img is not None:
                    try:
                        img.close()
                    except Exception as e:
                        if verbose:
                            print(f"[thepipe] Error closing image: {str(e)}")

    return chunks

def scrape_pptx(file_path: str, verbose: bool = False, text_only: bool = False) -> List[Chunk]:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    prs = Presentation(file_path)
    chunks = []
    # iterate through each slide in the presentation
    for slide in prs.slides:
        slide_texts = []
        slide_images = []
        # iterate through each shape in the slide
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text
                    if len(slide_texts) == 0:
                        text = '# ' + text # header for first text of a slide
                    slide_texts.append(text)
            # extract images from shapes
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE and not text_only:
                image_data = shape.image.blob
                image = Image.open(BytesIO(image_data))
                slide_images.append(image)
        # add slide to chunks if it has text or images
        if slide_texts or slide_images:
            chunks.append(Chunk(path=file_path, texts=slide_texts, images=slide_images))
    # return all chunks
    return chunks

def scrape_ipynb(file_path: str, verbose: bool = False, text_only: bool = False) -> List[Chunk]:
    with open(file_path, 'r', encoding='utf-8') as file:
        notebook = json.load(file)
    chunks = []
    # parse cells in the notebook
    for cell in notebook['cells']:
        texts = []
        images = []
        # parse cell content based on type
        if cell['cell_type'] == 'markdown':
            text = ''.join(cell['source'])
            if not text_only:
                images = get_images_from_markdown(text)
            texts.append(text)
        elif cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            texts.append(source)
            output_texts = []
            # code cells can have outputs
            if 'outputs' in cell:
                for output in cell['outputs']:
                    if 'data' in output and 'image/png' in output['data'] and not text_only:
                        image_data = output['data']['image/png']
                        image = Image.open(BytesIO(base64.b64decode(image_data)))
                        images.append(image)
                    elif 'data' in output and 'text/plain' in output['data']:
                        output_text = ''.join(output['data']['text/plain'])
                        output_texts.append(output_text)
            if output_texts:
                texts.extend(output_texts)
        elif cell['cell_type'] == 'raw':
            text = ''.join(cell['source'])
            texts.append(text)
        if texts or images:
            chunks.append(Chunk(path=file_path, texts=texts, images=images))
    return chunks

def scrape_tweet(url: str, text_only: bool = False, verbose: bool = False) -> List[Chunk]:
    # magic function from https://github.com/vercel/react-tweet/blob/main/packages/react-tweet/src/api/fetch-tweet.ts
    # unofficial, could break at any time
    def get_token(id: str) -> str:
        result = (float(id) / 1e15) * math.pi
        base_36_result = ''
        characters = '0123456789abcdefghijklmnopqrstuvwxyz'
        while result > 0:
            remainder = int(result % (6 ** 2))
            base_36_result = characters[remainder] + base_36_result
            result = (result - remainder) // (6 ** 2)
        base_36_result = re.sub(r'(0+|\.)', '', base_36_result)
        return base_36_result
    tweet_id = url.split('status/')[-1].split('?')[0]
    token = get_token(tweet_id)
    tweet_api_url = "https://cdn.syndication.twimg.com/tweet-result"
    params = {
        "id": tweet_id,
        "language": "en",
        "token": token
    }
    response = requests.get(tweet_api_url, params=params)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch tweet. Status code: {response.status_code}")
    tweet_data = response.json()
    # Extract tweet text
    tweet_text = tweet_data.get("text", "")
    
    chunks = []
    main_chunk = Chunk(path=url, texts=[tweet_text])
    chunks.append(main_chunk)

    if not text_only:
        # Extract images from tweet
        images = []
        if "mediaDetails" in tweet_data:
            for media in tweet_data["mediaDetails"]:
                if media.get("type") == "photo":
                    image_url = media.get("media_url_https")
                    if image_url:
                        image_response = requests.get(image_url)
                        img = Image.open(BytesIO(image_response.content))
                        images.append(img)
                elif media.get("type") == "video":
                    video_url = media.get("video_info", {}).get("variants", [{}])[0].get("url")
                    if video_url:
                        video_chunks = scrape_youtube(video_url, text_only=text_only, verbose=verbose)
                        chunks.extend(video_chunks)

        main_chunk.images = images

    return chunks

