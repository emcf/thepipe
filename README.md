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
  <a href="https://www.pepy.tech/projects/thepipe-api">
    <img src="https://static.pepy.tech/badge/thepipe-api" alt="PyPI">
  </a>
</div>

## Extract clean data from tricky documents ⚡

thepi.pe is a package that can scrape clean markdown, multimodal media, and structured data from complex documents. It uses vision-language models (VLMs) under the hood for superior output quality, and works out-of-the-box with any LLM, VLM, or vector database. It can extract well-formatted data from a wide range of sources, including PDFs, URLs, Word docs, Powerpoints, Python notebooks, videos, audio, and more.

## Features 🌟

- Scrape clean markdown, tables, and images from any document
- Scrape text, images, video, and audio from any file or URL
- Works out-of-the-box with vision-language models, vector databases, and RAG frameworks
- AI-native file-type detection, layout analysis, and structured data extraction
- Accepts a wide range of sources, including PDFs, URLs, Word docs, Powerpoints, Python notebooks, GitHub repos, videos, audio, and more

## Get started in 5 minutes 🚀

Thepipe can be installed via the command line:

```bash
pip install thepipe-api
```

If you need full functionality with media-rich sources such as webpages, video, and audio, you can choose to install the following dependencies:

```bash
apt-get update && apt-get install -y git ffmpeg
python -m playwright install --with-deps chromium
```

### Default setup (OpenAI)

By default, thepipe uses the [OpenAI API](https://platform.openai.com/docs/overview), so VLM features will work out-of-the-box provided you pass in an OpenAI client.

### Custom VLM server setup (OpenRouter, OpenLLM, etc.)

If you wish to use a local vision-language model or a different cloud provider, you can provide a custom OpenAI client, for example, by setting the base url to `https://openrouter.ai/api/v1` for [OpenRouter](https://openrouter.ai/), or `http://localhost:3000/v1` for a local server such as [OpenLLM](https://github.com/bentoml/OpenLLM). Note that uou must also pass the api key to your non-OpenAI cloud provider into the OpenAI client. You may want to set the `DEFAULT_AI_MODEL` environment variable to specify which model to use for VLM features (for OpenAI, this is defaulted to `gpt-4o`).

### Scraping

```python
from openai import OpenAI
from thepipe.scraper import scrape_file

# create an OpenAI-compatible client
client = OpenAI()

# scrape clean markdown and page images from a PDF
chunks = scrape_file(filepath="paper.pdf", openai_client=client)
```

To use standard scraping without AI scraping features, simply pass in no client:

```python
from thepipe.scraper import scrape_file

# scrape clean markdown and page images from a PDF
chunks = scrape_file(filepath="paper.pdf")
```

### Chunking

To satisfy token-limit constraints, the following chunking methods are available to split the content into smaller chunks.

- `chunk_by_document`: Returns one chunk with the entire content of the file.
- `chunk_by_page`: Returns one chunk for each page (for example: each webpage, PDF page, or PowerPoint slide).
- `chunk_by_length`: Splits chunks by length.
- `chunk_by_section`: Splits chunks by markdown section.
- `chunk_by_keyword`: Splits chunks at keywords.
- `chunk_semantic` (experimental, requires [sentence-transformers](https://pypi.org/project/sentence-transformers/)): Returns chunks split by spikes in semantic changes, with a configurable threshold.
- `chunk_agentic` (experimental, requires [OpenAI](https://pypi.org/project/openai/)): Returns chunks split by an LLM agent that attempts to find semantically meaningful sections.

For example,

```python
from thepipe.scraper import scrape_file
from thepipe.chunker import chunk_by_document, chunk_by_page

# optionally, pass in chunking_method
# chunk_by_document returns one chunk for the entire document
chunks = scrape_file(filepath="paper.pdf", chunking_method=chunk_by_document)

# you can also re-chunk later.
# chunk_by_page returns one chunk for each page (for example: each webpage, PDF page, or PowerPoint slide).
chunks = chunk_by_page(chunks)
```

### OpenAI Chat Integration 🤖

```python
from openai import OpenAI
from thepipe.core import chunks_to_messages

# Initialize OpenAI client
client = OpenAI()

# Use OpenAI-formatted chat messages
messages = [{
  "role": "user",
  "content": [{
      "type": "text",
      "text": "What is the paper about?"
    }]
}]

# Simply add the scraped chunks to the messages
messages += chunks_to_messages(chunks)

# Call LLM
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
)
```

`chunks_to_messages` takes in an optional `text_only` parameter to only output text from the source document. This is useful for downstream use with LLMs that lack multimodal capabilities.

> ⚠️ **It is important to be mindful of your model's token limit.**
> Be sure your prompt is within the token limit of your model. You can use chunking to split your messages into smaller chunks.

### LLamaIndex Integration 🦙

A chunk can be converted to LlamaIndex `Document`/`ImageDocument` with `.to_llamaindex`.

### Structured extraction 🗂️

```python
from thepipe.extract import extract
from openai import OpenAI

client = OpenAI()

schema = {
  "description": "string",
  "amount_usd": "float"
}

results, tokens_used = extract(
    chunks=chunks,
    schema=schema,
    multiple_extractions=True,  # extract multiple rows of data per chunk
    openai_client=client
)
```

## Sponsors

Please consider supporting thepipe by [becoming a sponsor](mailto:emmett@thepi.pe).
Your support helps me maintain and improve the project while helping the open-source community discover your work.

Visit [Cal.com](https://cal.com/) for an open-source scheduling tool that helps you book meetings with ease. It's the perfect solution for busy professionals who want to streamline their scheduling process.

<a href="https://cal.com/emmett-mcf/30min"><img alt="Book us with Cal.com" src="https://cal.com/book-with-cal-dark.svg" /></a>

Looking for enterprise-ready document processing and intelligent automation? Discover how [Trellis AI](https://runtrellis.com/) can streamline your workflows and enhance productivity.

## How it works 🛠️

thepipe uses a combination of computer-vision models and heuristics to scrape clean content from the source and process it for downstream use with [large language models](https://en.wikipedia.org/wiki/Large_language_model), or [vision-language models](https://en.wikipedia.org/wiki/Vision_transformer). You can feed these messages directly into the model, or alternatively you can chunk these messages for downstream storage in a vector database such as ChromaDB, LLamaIndex, or an equivalent RAG framework.

## Supported File Types 📚

| Source                       | Input types                                                                          | Multimodal | Notes                                                                                                                                                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------ | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Webpage                      | URLs starting with `http`, `https`, `ftp`                                            | ✔️         | Scrapes markdown, images, and tables from web pages. AI extraction available by passing an OpenAI client for screenshot analysis                                                                                                              |
| PDF                          | `.pdf`                                                                               | ✔️         | Extracts page markdown and page images. AI extraction available when an OpenAI client is supplied for complex or scanned documents                                                                                                            |
| Word Document                | `.docx`                                                                              | ✔️         | Extracts text, tables, and images                                                                                                                                                                                                             |
| PowerPoint                   | `.pptx`                                                                              | ✔️         | Extracts text and images from slides                                                                                                                                                                                                          |
| Video                        | `.mp4`, `.mov`, `.wmv`                                                               | ✔️         | Uses Whisper for transcription and extracts frames                                                                                                                                                                                            |
| Audio                        | `.mp3`, `.wav`                                                                       | ✔️         | Uses Whisper for transcription                                                                                                                                                                                                                |
| Jupyter Notebook             | `.ipynb`                                                                             | ✔️         | Extracts markdown, code, outputs, and images                                                                                                                                                                                                  |
| Spreadsheet                  | `.csv`, `.xls`, `.xlsx`                                                              | ❌         | Converts each row to JSON format, including row index for each                                                                                                                                                                                |
| Plaintext                    | `.txt`, `.md`, `.rtf`, etc                                                           | ❌         | Simple text extraction                                                                                                                                                                                                                        |
| Image                        | `.jpg`, `.jpeg`, `.png`                                                              | ✔️         | Uses VLM for OCR in text-only mode                                                                                                                                                                                                            |
| ZIP File                     | `.zip`                                                                               | ✔️         | Extracts and processes contained files                                                                                                                                                                                                        |
| Directory                    | any `path/to/folder`                                                                 | ✔️         | Recursively processes all files in directory. Optionally use `inclusion_pattern` to pass regex strings for file inclusion rules.                                                                                                              |
| YouTube Video (known issues) | YouTube video URLs starting with `https://youtube.com` or `https://www.youtube.com`. | ✔️         | Uses pytube for video download and Whisper for transcription. For consistent extraction, you may need to modify your `pytube` installation to send a valid user-agent header (see [this issue](https://github.com/pytube/pytube/issues/399)). |
| Tweet                        | URLs starting with `https://twitter.com` or `https://x.com`                          | ✔️         | Uses unofficial API, may break unexpectedly                                                                                                                                                                                                   |
| GitHub Repository            | GitHub repo URLs starting with `https://github.com` or `https://www.github.com`      | ✔️         | Requires `GITHUB_TOKEN` environment variable                                                                                                                                                                                                  |

## Configuration & Environment

Set these environment variables to control API keys, hosting, and model defaults:

```bash
# If you want longer-term image storage and hosting (saves to ./images and serves via HOST_URL)
export HOST_IMAGES=true

# GitHub token for scraping private/public repos via `scrape_url`
export GITHUB_TOKEN=ghp_...

# Control scraping defaults
export DEFAULT_AI_MODEL=gpt-4o
export FILESIZE_LIMIT_MB=50
```

## CLI Reference

```shell
# Basic usage: scrape a file or URL
thepipe <source> [options]

# Options:
--ai_extraction       Use AI for PDF/image/text extraction  # still accepted for backwards compatibility
--text_only           Only output text (no images)
--inclusion_pattern=REGEX   Only include files matching REGEX when scraping directories
--verbose             Print detailed progress messages
```

## Contributing

We welcome contributions! To get started:

1. Fork the repo and create a feature branch:

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Install dependencies & run tests:

   ```bash
   pip install -r requirements.txt
   python -m unittest discover
   ```

3. Make your changes, format them, and commit them:

   ```bash
   black .
   git add .
   git commit -m "..."
   ```

4. Push to your fork and create a pull request:

   ```bash
   git push origin feature/my-new-feature
   ```

5. Submit a pull request to the main repository.

6. Wait for review and feedback from the maintainers. This may take some time, so please be patient!
