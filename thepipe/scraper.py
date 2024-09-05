import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import io
import math
import re
from typing import  List, Dict, Any, Callable, Optional
import glob
import os
import tempfile
from urllib.parse import urlparse
import zipfile
from PIL import Image
import requests
import json
from .core import HOST_URL, THEPIPE_API_KEY, HOST_IMAGES, Chunk, make_image_url
from .chunker import chunk_by_page
import tempfile
import mimetypes
import dotenv
import shutil
from magika import Magika
dotenv.load_dotenv()
from enum import Enum, auto

FOLDERS_TO_IGNORE = ['*node_modules.*', '.*venv.*', '.*\.git.*', '.*\.vscode.*', '.*pycache.*']
FILES_TO_IGNORE = ['package-lock.json', '.gitignore', '.*\.bin', '.*\.pyc', '.*\.pyo', '.*\.exe', '.*\.dll', '.*\.ipynb_checkpoints']
GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", None)
USER_AGENT_STRING: str = os.getenv("USER_AGENT_STRING", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")
MAX_WHISPER_DURATION = 600 # 10 minutes
TWITTER_DOMAINS = ['https://twitter.com', 'https://www.twitter.com', 'https://x.com', 'https://www.x.com']
YOUTUBE_DOMAINS = ['https://www.youtube.com', 'https://youtube.com']
GITHUB_DOMAINS = ['https://github.com', 'https://www.github.com']
SCRAPING_PROMPT = os.getenv("EXTRACTION_PROMPT", """An open source document is given. Output the entire extracted contents from the document in detailed markdown format.
Be sure to correctly format markdown for headers, paragraphs, lists, tables, menus, equations, full text contents, etc.
Always reply immediately with only markdown. Do not output anything else.""")
DEFAULT_AI_MODEL = os.getenv("DEFAULT_AI_MODEL", "gpt-4o-mini")
FILESIZE_LIMIT_MB = os.getenv("FILESIZE_LIMIT_MB", 50)

class YouTubeMetadata(Enum):
    TITLE = auto()
    DESCRIPTION = auto()
    UPLOAD_DATE = auto()
    UPLOADER = auto()
    VIEW_COUNT = auto()
    LIKE_COUNT = auto()
    DURATION = auto()
    TAGS = auto()
    CATEGORY = auto()
    THUMBNAIL_URL = auto()

DEFAULT_METADATA = [
    YouTubeMetadata.TITLE,
    YouTubeMetadata.DESCRIPTION,
    YouTubeMetadata.UPLOAD_DATE,
    YouTubeMetadata.UPLOADER,
    YouTubeMetadata.VIEW_COUNT,
    YouTubeMetadata.DURATION
]

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

def is_primarily_video_platform(extractor_key: str) -> bool:
    # List of known video platforms
    video_platforms = {
        "youtube", "youtu", "netflix", "amazon", "hulu", "disneyplus", "vimeo", 
        "twitch", "tiktok", "dailymotion", "vevo", "kick",
        "crunchyroll", "peacocktv", "hbomax", "roku", "pluto", 
        "tubitv", "iqiyi", "v.qq", "youku", "bilibili", "flicknexs", 
        "brightcove", "wistia", "jwplayer", "kaltura", "panopto", "vidyard", 
        "vk", "rutube", "metacafe", "veoh", "ustream", "livestream", "periscope", 
        "mixer", "younow", "smashcast", "niconico", "vlive", "afreecatv", 
        "kakao", "naver", "line", "iflix", "hooq", "viu", "mubi"
    }
    return any(platform in extractor_key.lower() for platform in video_platforms)

def detect_url_type(url: str) -> str:
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError
    parsed_url = urlparse(url)
    
    # Check if it's a supported video site
    try:
        with YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            extractor_key = info.get('extractor_key', '').lower()
            
            if is_primarily_video_platform(extractor_key):
                if info.get('_type') == 'playlist':
                    return 'video_playlist'
                elif info.get('_type') in ['video', None]:  # Some extractors don't set _type for single videos
                    return 'video'
            else:
                # It's a website that happens to have embeddable/downloadable content
                return 'website_with_media'
    except DownloadError:
        pass  # Not a supported video site, continue to other checks
    
    # Check for specific file types
    file_extension = parsed_url.path.split('.')[-1].lower()
    if file_extension in ['pdf', 'docx', 'txt', 'csv', 'xlsx']:
        return f'file_{file_extension}'
    
    # If none of the above, assume it's a general website
    return 'website'

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
        scraped_chunks = scrape_docx(file_path=filepath, verbose=verbose, text_only=text_only, options=options)
    elif source_type == 'application/vnd.openxmlformats-officedocument.presentationml.presentation':
        scraped_chunks = scrape_pptx(file_path=filepath, verbose=verbose, text_only=text_only, options=options)
    elif source_type.startswith('image/'):
        scraped_chunks = scrape_image(file_path=filepath, text_only=text_only, options=options)
    elif source_type.startswith('application/vnd.ms-excel') or source_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        scraped_chunks = scrape_spreadsheet(file_path=filepath, source_type=source_type, options=options)
    elif source_type == 'application/x-ipynb+json':
        scraped_chunks = scrape_ipynb(file_path=filepath, verbose=verbose, text_only=text_only, options=options)
    elif source_type == 'application/zip' or source_type == 'application/x-zip-compressed':
        scraped_chunks = scrape_zip(file_path=filepath, verbose=verbose, ai_extraction=ai_extraction, text_only=text_only, local=local, options=options)
    elif source_type.startswith('video/'):
        scraped_chunks = scrape_video(file_path=filepath, verbose=verbose, text_only=text_only, options=options)
    elif source_type.startswith('audio/'):
        scraped_chunks = scrape_audio(file_path=filepath, verbose=verbose, options=options)
    elif source_type.startswith('text/'):
        scraped_chunks = scrape_plaintext(file_path=filepath, options=options)
    else:
        try:
            scraped_chunks = scrape_plaintext(file_path=filepath, options=options)
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

def scrape_plaintext(file_path: str) -> List[Chunk]:
    with open(file_path, 'r', encoding='utf-8') as file:
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
                    temperature=0.1
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

def get_images_from_ipynb_markdown(text: str) -> List[Image.Image]:
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
            temperature=0.1
        )
        llm_response = response.choices[0].message.content
        chunk = Chunk(path=url, texts=[llm_response], images=[stacked_image])
    else:
        raise ValueError("Model received 0 images from webpage")

    return chunk

def extract_page_content(url: str, text_only: bool = False, verbose: bool = False) -> Chunk:
    from urllib.parse import urlparse
    import markdownify
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

def parse_html_to_markdown(html_content):
    from bs4 import BeautifulSoup, NavigableString, Tag
    soup = BeautifulSoup(html_content, 'html.parser')
    markdown_content = []
    # recursively traverse the HTML content and extract text and links
    def traverse_and_extract(element):
        if isinstance(element, NavigableString):
            markdown_content.append(str(element))
        elif isinstance(element, Tag):
            if element.name == 'a' and 'href' in element.attrs:
                link_text = element.get_text()
                link_url = element['href']
                markdown_content.append(f'[{link_text}]({link_url})')
            else:
                for child in element.children:
                    traverse_and_extract(child)
    # extract content from the body tag
    body = soup.body
    if body:
        traverse_and_extract(body)
    return ''.join(markdown_content)

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

def scrape_url(url: str, text_only: bool = False, ai_extraction: bool = False, verbose: bool = False, local: bool = False, chunking_method: Callable = chunk_by_page, options: Optional[Dict[str, Any]] = None) -> List[Chunk]:
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
        response.raise_for_status()
        results = []
        for line in response.iter_lines():
            if line:
                chunk_data = json.loads(line)
                results.append(chunk_data['result'])
        return results

    # For local processing
    url_type = detect_url_type(url)
    
    if verbose:
        print(f"[thepipe] Detected URL type: {url_type}")
    
    if any(url.startswith(domain) for domain in TWITTER_DOMAINS):
        extraction = scrape_tweet(url=url, text_only=text_only, verbose=verbose, options=options)
        return extraction
    elif url_type in ['video', 'video_playlist']:
        extraction = scrape_youtube(url, text_only=text_only, verbose=verbose, options=options)
        return extraction
    elif any(url.startswith(domain) for domain in GITHUB_DOMAINS):
        extraction = scrape_github(github_url=url, text_only=text_only, ai_extraction=ai_extraction, verbose=verbose, options=options)
        return extraction
    elif url_type == 'website_with_media':
        if verbose:
            print("[thepipe] Website contains downloadable media. Scraping entire website in next version.")
        extraction = scrape_youtube(url, text_only=text_only, verbose=verbose, options=options)
        return extraction
    elif url_type.startswith('file_'):
        # if url leads to a file, attempt to download it and scrape it
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, os.path.basename(url))
            response = requests.get(url)
            # verify the ingress/egress with be within limits, if there are any set
            if FILESIZE_LIMIT_MB and int(response.headers['Content-Length']) > FILESIZE_LIMIT_MB * 1024 * 1024:
                raise ValueError(f"File size exceeds {FILESIZE_LIMIT_MB} MB limit.")
            with open(file_path, 'wb') as file:
                file.write(response.content)
            chunks = scrape_file(filepath=file_path, ai_extraction=ai_extraction, text_only=text_only, verbose=verbose, local=local, chunking_method=chunking_method, options=options)
        return chunks
    else:
        # if url leads to web content, scrape it directly
        if ai_extraction:
            chunk = ai_extract_webpage_content(url=url, text_only=text_only, verbose=verbose, options=options)
        else:
            chunk = extract_page_content(url=url, text_only=text_only, verbose=verbose, options=options)
        chunks = chunking_method([chunk])
        return chunks
    
def format_timestamp(seconds, chunk_index, chunk_duration):
    # helper function to format the timestamp.
    total_seconds = chunk_index * chunk_duration + seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02}.{milliseconds:03}"

def extract_metadata(video: dict, metadata_fields: Optional[List[YouTubeMetadata]], verbose: bool) -> dict:
    metadata = {}
    if metadata_fields is not None:
        for field in metadata_fields:
            try:
                if field == YouTubeMetadata.TITLE:
                    metadata['title'] = video.get('title', 'Untitled')
                elif field == YouTubeMetadata.DESCRIPTION:
                    metadata['description'] = video.get('description', '')
                elif field == YouTubeMetadata.UPLOAD_DATE:
                    metadata['upload_date'] = video.get('upload_date', '')
                elif field == YouTubeMetadata.UPLOADER:
                    metadata['uploader'] = video.get('uploader', '')
                elif field == YouTubeMetadata.VIEW_COUNT:
                    metadata['view_count'] = str(video.get('view_count', 'N/A'))
                elif field == YouTubeMetadata.LIKE_COUNT:
                    metadata['like_count'] = str(video.get('like_count', 'N/A'))
                elif field == YouTubeMetadata.DURATION:
                    metadata['duration'] = str(video.get('duration', 'N/A'))
                elif field == YouTubeMetadata.TAGS:
                    metadata['tags'] = video.get('tags', [])
                elif field == YouTubeMetadata.CATEGORY:
                    metadata['category'] = video.get('categories', ['N/A'])[0] if video.get('categories') else 'N/A'
                elif field == YouTubeMetadata.THUMBNAIL_URL:
                    metadata['thumbnail_url'] = video.get('thumbnail', 'N/A')
            except Exception as e:
                if verbose:
                    print(f"[thepipe] Error extracting {field.name} metadata: {str(e)}")
                metadata[field.name.lower()] = 'N/A'
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
    

def scrape_youtube(url: str, text_only: Optional[str] = None, verbose: bool = False, metadata_fields: Optional[List[YouTubeMetadata]] = DEFAULT_METADATA, options: Optional[Dict[str, Any]] = None) -> List[Chunk]:
    import yt_dlp
    from .enums import YouTubeEnum

    subtitle_opts = {
        'outtmpl': {'default': '%(title)s.%(ext)s'},
        'quiet': not verbose,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en,*', 'a.en,a.*'],
        'skip_download': True,
    }

    video_opts = {
        'outtmpl': {'default': '%(title)s.%(ext)s'},
        'quiet': not verbose,
    }

    if text_only == 'transcribe':
        video_opts['format'] = 'bestaudio/best'
        video_opts['postprocessors'] = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }]
    elif text_only in ['ai', 'uploaded']:
        if text_only == 'ai':
            subtitle_opts['subtitleslangs'] = ['a.en,a.*', 'en,*']
        else:  # uploaded
            subtitle_opts['subtitleslangs'] = ['en,*', 'a.en,a.*']
    else:
        video_opts['format'] = 'bestvideo+bestaudio/best'

    # Process additional options
    if options and 'youtube' in options:
        subtitle_opts.update(YouTubeEnum.process_options(options['youtube'], bool(text_only), verbose))
        video_opts.update(YouTubeEnum.process_options(options['youtube'], bool(text_only), verbose))

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            chunks = []

            if text_only == 'transcribe':
                if verbose:
                    print("[thepipe] Downloading audio for transcription.")
                with yt_dlp.YoutubeDL(video_opts) as ydl:
                    ydl.params['outtmpl']['default'] = os.path.join(temp_dir, '%(title)s.%(ext)s')
                    info = ydl.extract_info(url, download=True)
                    video = info['entries'][0] if 'entries' in info else info

                    file = os.listdir(temp_dir)[0]
                    file_path = os.path.join(temp_dir, file)

                    chunks = scrape_audio(file_path=file_path, verbose=verbose)
            else:
                # Try to download subtitles
                with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
                    ydl.params['outtmpl']['default'] = os.path.join(temp_dir, '%(title)s.%(ext)s')
                    ydl.download([url])

                    subtitle_found = False
                    for file in os.listdir(temp_dir):
                        if file.endswith('.vtt'):
                            with open(os.path.join(temp_dir, file), 'r', encoding='utf-8') as f:
                                subtitle_text = f.read()
                            try:
                                cleaned_subtitle_text = clean_subtitles(subtitle_text)
                                chunks.append(Chunk(path=url, texts=[cleaned_subtitle_text]))
                                subtitle_found = True
                                break  # Use the first available subtitle file
                            except Exception as e:
                                if verbose:
                                    print(f"[thepipe] Error cleaning subtitles: {str(e)}")
                                # If cleaning fails, use the original subtitle text
                                chunks.append(Chunk(path=url, texts=[subtitle_text]))
                                subtitle_found = True
                                break

                # If no subtitles found, download video/audio for transcription
                if not subtitle_found:
                    if verbose:
                        print("[thepipe] No subtitles found. Downloading video/audio for transcription.")
                    with yt_dlp.YoutubeDL(video_opts) as ydl:
                        ydl.params['outtmpl']['default'] = os.path.join(temp_dir, '%(title)s.%(ext)s')
                        info = ydl.extract_info(url, download=True)
                        video = info['entries'][0] if 'entries' in info else info

                        file = os.listdir(temp_dir)[0]
                        file_path = os.path.join(temp_dir, file)

                        if file.endswith('.mp3') or file.endswith('.m4a'):
                            chunks = scrape_audio(file_path=file_path, verbose=verbose)
                        else:
                            chunks = scrape_video(file_path=file_path, verbose=verbose, text_only=True)

            # Extract metadata
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                video = info['entries'][0] if 'entries' in info else info
                metadata = extract_metadata(video, metadata_fields, verbose)

            # Add metadata to the first chunk
            if chunks and metadata:
                metadata_text = format_metadata(metadata)
                chunks[0].texts.insert(0, metadata_text)

            return chunks

        except yt_dlp.utils.DownloadError as e:
            if verbose:
                print(f"[thepipe] Error processing YouTube video: {str(e)}")
            return [Chunk(path=url, texts=[f"Error: Unable to process YouTube video. {str(e)}"])]
        except Exception as e:
            if verbose:
                print(f"[thepipe] Unexpected error: {str(e)}")
            return [Chunk(path=url, texts=[f"Error: An unexpected error occurred. {str(e)}"])]
                
def clean_subtitles(subtitle_text: str) -> str:
    lines = subtitle_text.split('\n')
    cleaned_lines: List[Tuple[str, str]] = []
    current_time = ""
    
    for line in lines:
        if re.match(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', line):
            current_time = line.split(' --> ')[0]
        elif line.strip() and not line.startswith('WEBVTT'):
            text = re.sub(r'<[^>]+>', '', line).strip()
            if current_time:  # Only add the line if we have a valid timestamp
                if cleaned_lines and cleaned_lines[-1][1].endswith(text):
                    # If this line is contained entirely in the previous line, skip it
                    continue
                if cleaned_lines and text.startswith(cleaned_lines[-1][1]):
                    # If this line starts with the previous line, extend the previous line
                    cleaned_lines[-1] = (cleaned_lines[-1][0], text)
                else:
                    cleaned_lines.append((current_time, text))

    # Merge lines that are continuations
    merged_lines = []
    buffer = ""
    buffer_time = ""
    for time, text in cleaned_lines:
        if not buffer:
            buffer = text
            buffer_time = time
        elif text.startswith(buffer.split()[-1]):
            buffer += text[len(buffer.split()[-1]):]
        else:
            merged_lines.append(f"[{buffer_time}] {buffer}")
            buffer = text
            buffer_time = time
    
    if buffer:
        merged_lines.append(f"[{buffer_time}] {buffer}")

    return '\n'.join(merged_lines)

def scrape_audio(file_path: str, verbose: bool = False) -> List[Chunk]:
    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio=file_path, verbose=verbose)
    # Format transcription with timestamps
    transcript = []
    for segment in result['segments']:
        start = format_timestamp(segment['start'], 0, 0)
        end = format_timestamp(segment['end'], 0, 0)
        if segment['text'].strip():
            transcript.append(f"[{start} --> {end}]  {segment['text']}")
    # join the formatted transcription into a single string
    return [Chunk(path=file_path, texts=transcript)]

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
                images = get_images_from_ipynb_markdown(text)
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

