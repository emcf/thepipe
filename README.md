<div align="center">
  <a href="https://thepi.pe/">
    <img src="https://rpnutzemutbrumczwvue.supabase.co/storage/v1/object/public/assets/pipeline_small%20(1).png" alt="Pipeline Illustration" style="width:96px; height:72px; vertical-align:middle;">
    <h1>thepi.pe</h1>
  </a>
  <a>
    <img src="https://github.com/emcf/thepipe/actions/workflows/python-ci.yml/badge.svg" alt="python-gh-action">
  </a>
    <a href="https://codecov.io/gh/emcf/thepipe">
    <img src="https://codecov.io/gh/emcf/thepipe/graph/badge.svg?token=OE7CUEFUL9" alt="codecov">
  </a>
  <a href="https://raw.githubusercontent.com/emcf/thepipe/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
  </a>
  <a href="https://thepi.pe/">
    <img src="https://img.shields.io/website?url=https%3A%2F%2Fthepipe.up.railway.app%2F&label=API%20status" alt="Website">
  </a>
  <a href="https://thepi.pe/">
    <img src="https://img.shields.io/badge/API-access-blue" alt="get API">
  </a>
</div>


### Extract markdown and visuals from PDFs URLs, docs, slides, videos, and more, ready for multimodal LLMs. ⚡

thepi.pe is an AI-native scraping engine that generates LLM-ready markdown and visuals from any document, media, or web page. It is built for multimodal language models such as GPT-4o, and works out-of-the-box with any LLM or vector database. thepi.pe is available as a [hosted API](https://thepi.pe), or it can be self-hosted. 

## Features 🌟

- Extract markdown, images, and structured data from any document or web page
- Output works out-of-the-box with all multimodal LLMs and RAG frameworks
- AI filetype detection for missing file extensions and unknown web data
- Quick-start integrations for Twitter, YouTube, GitHub, and more
- GPU-accelerated

## Get started in 5 minutes  🚀

thepi.pe can read a wide range of filetypes and web sources, so it requires a few dependencies. It also requires a strong machine (16GB+ VRAM for optimal PDF & video response times) for AI extraction features. For these reasons, we host a REST API that works out-of-the-box at [thepi.pe](https://thepi.pe).

### Hosted API (Python)

```bash
pip install thepipe-api
setx THEPIPE_API_KEY=your_api_key
```

```python
import thepipe
from openai import OpenAI

# scrape markdown + images
chunks = thepipe.scrape(source="example.pdf")

# call LLM
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=thepipe.chunks_to_messages(chunks),
)
```

### Local Installation


```bash
pip install thepipe-api[local]
```

```python
import thepipe
from openai import OpenAI

# scrape markdown + images
chunks = thepipe.scrape_file(source="example.pdf", local=True)
```

You can also use The Pipe from the command line:
```bash
thepipe path/to/folder --include_regex .*\.tsx
```


## Supported File Types 📚

| Source Type              | Input types                                                    | Multimodal Scraping | Notes |
|--------------------------|----------------------------------------------------------------|---------------------|----------------------|
| Webpage                  | URLs starting with `http`, `https`, `ftp`                      | ✔️                  | Scrapes markdown, images, and tables from web pages |
| PDF                      | `.pdf`                                                          | ✔️                  | Extracts page markdown and page images. Opt-in `ai_extraction` for advanced layout analysis (extracts markdown, LaTeX equations, tables, and figures) |
| Word Document  | `.docx`                                                         | ✔️                  | Extracts text, tables, and images |
| PowerPoint     | `.pptx`                                                         | ✔️                  | Extracts text and images from slides |
| Video                    | `.mp4`, `.mov`, `.wmv`                                          | ✔️                  | Uses Whisper for transcription and extracts frames |
| Audio                    | `.mp3`, `.wav`                                                  | ✔️                  | Uses Whisper for transcription |
| Jupyter Notebook         | `.ipynb`                                                        | ✔️                  | Extracts markdown, code, outputs, and images |
| Spreadsheet              | `.csv`, `.xls`, `.xlsx`                                         | ❌                  | Converts each row to JSON format, including row index for each |
| Plaintext                | `.txt`, `.md`, `.rtf`, etc                                      | ❌                  | Simple text extraction |
| Image                    | `.jpg`, `.jpeg`, `.png`                                    | ✔️                  | Uses pytesseract for OCR in text-only mode |
| ZIP File                 | `.zip`                                                          | ✔️                  | Extracts and processes contained files |
| Directory                | any `path/to/folder`                                            | ✔️                  | Recursively processes all files in directory |
| YouTube Video            | YouTube video URLs starting with `https://youtube.com` or `https://www.youtube.com`.  | ✔️   | Uses pytube for video download and Whisper for transcription. For consistent extraction, you may need to modify your `pytube` installation to send a valid user agent header (see [this issue](https://github.com/pytube/pytube/issues/399)). |
| Tweet                    | URLs starting with `https://twitter.com` or `https://x.com`    | ✔️                  | Uses unofficial API, may break unexpectedly |
| GitHub Repository        | GitHub repo URLs starting with `https://github.com` or `https://www.github.com` | ✔️       | Requires GITHUB_TOKEN environment variable |

## How it works 🛠️

thepi.pe uses computer vision models and heuristics to extract clean content from the source and process it for downstream use with [language models](https://en.wikipedia.org/wiki/Large_language_model), or [vision transformers](https://en.wikipedia.org/wiki/Vision_transformer). The output from thepi.pe is a prompt (a list of messages) containing all content from the source document. The messages returned should look like this:
```json
[
  {
    "role": "user",
    "content": [
      {
        "type": "text",
        "text": "..."
      },
      {
        "type": "image_url",
        "image_url": {
          "url": "data:image/jpeg;base64,..."
        }
      }
    ]
  }
]
```

You can feed these messages directly into the model, or you can use `thepipe_api.chunk_by_page`, `thepipe_api.chunk_by_section`, `thepipe_api.chunk_semantic` to chunk these messages for a vector database such as ChromaDB or a RAG framework (a chunk can be converted to LlamaIndex Document/ImageDocument with `.to_llamaindex`).

> ⚠️ **It is important to be mindful of your model's token limit.**
GPT-4o does not work with too many images in the prompt (see discussion [here](https://community.openai.com/t/gpt-4-vision-maximum-amount-of-images/573110/6)). Large documents should be extracted with `text_only=True` to avoid this issue, or alternatively they can be chunked and saved into a vector database or RAG framework.

# Sponsors

<a href="https://cal.com/emmett-mcf/30min"><img alt="Book us with Cal.com" src="https://cal.com/book-with-cal-dark.svg" /></a>

Thank you to [Cal.com](https://cal.com/) for sponsoring this project. Contact emmett@thepi.pe for sponsorship information.