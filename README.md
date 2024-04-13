# <a href="https://thepi.pe/"><img src="https://rpnutzemutbrumczwvue.supabase.co/storage/v1/object/public/assets/pipeline_small%20(1).png" alt="Pipeline Illustration" style="width:96px; height:72px; vertical-align:middle;"> The Pipe</a>
[![codecov](https://codecov.io/gh/emcf/thepipe/graph/badge.svg?token=OE7CUEFUL9)](https://codecov.io/gh/emcf/thepipe) ![python-gh-action](https://github.com/emcf/thepipe/actions/workflows/python-ci.yml/badge.svg) <a href="https://thepi.pe/">![Website](https://img.shields.io/website?url=https%3A%2F%2Fthepipe.up.railway.app%2F&label=API%20status)</a> <a href="https://thepi.pe/">![get API](https://img.shields.io/badge/API-get%20access-blue)</a>

### Prepare PDFs, word docs, slides, web pages and more for Vision-LLMs with one line of code ⚡

The Pipe is a multimodal-first tool for feeding files and web pages into vision language models like GPT-4V, Gemini Pro, and LLaVa. It is best for LLM and RAG applications that require a deep understanding of complex sources. The Pipe is available as a hosted API at [thepi.pe](https://thepi.pe), or it can be set up locally.

## Getting Started  🚀

First, install The Pipe. 
```
pip install thepipe_api
```

Ensure the `THEPIPE_API_KEY` environment variable is set. Don't have an API key yet? [Get one here](https://thepi.pe). Looking to host it yourself locally instead? See the [local installation](#local-installation) section at the bottom.

Now you can extract comprehensive text and visuals from any file:
```python
from thepipe_api import thepipe
chunks = thepipe.extract("example.pdf")
```
Or any website:
```python
chunks = thepipe.extract("https://example.com")
```
Then feed it into GPT-4-Vision:
```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages = chunks,
)
```
The Pipe's output is a list of sensible "chunks", and thus can be used either for storage in a vector database or for direct use as a prompt. Extra features such as data table extraction, bar chart extraction, custom web authentications and more are available in the [API documentation](https://thepi.pe/docs). [LiteLLM](https://github.com/BerriAI/litellm) can be used to easily integrate The Pipe with any LLM provider.

## Features 🌟

- Extracts text and visuals from any file or web page 📚
- Outputs RAG-ready chunks, optimized for multimodal LLMs 🖼️ + 💬
- Can interpret complex PDFs, web apps, markdown, etc 🧠
- Auto-compress prompts exceeding your chosen token limit 📦
- Works with missing file extensions, in-memory data streams 💾
- Works with codebases, URL, git repos, and more 🌐
- Multi-threaded ⚡️

##  How it works 🛠️

The pipe is accessible from the command line or from [Python](https://www.python.org/downloads/). The input source is either a file path, a URL, or a directory (or zip file) path. The pipe will extract information from the source and process it for downstream use with [language models](https://en.wikipedia.org/wiki/Large_language_model), [vision transformers](https://en.wikipedia.org/wiki/Vision_transformer), or [vision-language models](https://arxiv.org/abs/2304.00685). The output from the pipe is a sensible text-based (or multimodal) representation of the extracted information, carefully crafted to fit within context windows for any models from [gemma-7b](https://huggingface.co/google/gemma-7b) to [GPT-4](https://openai.com/gpt-4). It uses a variety of heuristics for optimal performance with vision-language models, including AI filetype detection with [filetype detection](https://opensource.googleblog.com/2024/02/magika-ai-powered-fast-and-efficient-file-type-identification.html), AI [PDF extraction](thepi.pe/docs), efficient [token compression](https://arxiv.org/abs/2403.12968), automatic [image encoding](https://en.wikipedia.org/wiki/Base64), [reranking](https://arxiv.org/abs/2310.06839) for [lost-in-the-middle](https://arxiv.org/abs/2307.03172) effects, and more, all pre-built to work out-of-the-box.

## Supported File Types 📚

| Source Type                           | Input types        | Token Compression 🗜️ | Image Extraction 👁️ | Notes 📌                                                  |
|---------------------------------------|------------------------------------------|-------------------|------------------|---------------------------------------------------------|
| Directory                             | Any `/path/to/directory`                 | ✔️               | ✔️               | Extracts from all files in directory, supports match and ignore patterns |
| Code                                  | `.py`, `.tsx`, `.js`, `.html`, `.css`, `.cpp`, etc | ✔️ (varies)   | ❌               | Combines all code files. `.c`, `.cpp`, `.py` are compressible with ctags, others are not |
| Plaintext                             | `.txt`, `.md`, `.rtf`, etc               | ✔️               | ❌               | Regular text files                                                      |
| PDF                                   | `.pdf`                                  | ✔️               | ✔️    | Extracts text and images of each page; can use AI for extraction of table data and  images within pages |
| Image                                 | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.svg` | ❌                | ✔️              | Extracts images, uses OCR if text_only                        |
| Data Table                           | `.csv`, `.xls`, `.xlsx`             | ✔️                | ❌               | Extracts data from spreadsheets; converts to text representation. For very large datasets, will only extract column names and types         |
| Jupyter Notebook                      | `.ipynb`                                | ❌               | ✔️               | Extracts code, markdown, and images from Jupyter notebooks                                  |
| Microsoft Word Document               | `.docx`                                 | ✔️               | ✔️               | Extracts text and images from Word documents                                        |
| Microsoft PowerPoint Presentation     | `.pptx`                                 | ✔️               | ✔️               | Extracts text and images from PowerPoint presentations                              |
| Website                               | URLs (inputs containing `http`, `https`, `www`, `ftp`)             | ✔️                | ✔️    | Extracts text from web page along with image (or images if scrollable); text-only extraction available          |
| GitHub Repository                     | GitHub repo URLs                         | ✔️               | ✔️                | Extracts from GitHub repositories; supports branch specification         |
| ZIP File                              | `.zip`                                  | ✔️               | ✔️                | Extracts contents of ZIP files; supports nested directory extraction     |

## Local Installation 🛠️

To use The Pipe locally, you will need [playwright](https://github.com/microsoft/playwright), [ctags](https://github.com/universal-ctags/), [pytesseract](https://github.com/h/pytesseract), and the local python requirements, which differ from the more lightweight API requirements. You will also need to use the local version of the requirements file:

```bash
git clone https://github.com/emcf/thepipe
pip install -r requirements_local.txt
```

Tip for windows users: you may need to install the python-libmagic binaries with `pip install python-magic-bin`.

Now you can use The Pipe:
```bash
from thepipe_api import thepipe
chunks = thepipe.extract("example.pdf", local=True)
```

Arguments are:
- `source` (required): can be a file path, a URL, or a directory path.
- `local` (optional): Use the local version of The Pipe instead of the hosted API.
- `match` (optional): Regex pattern to match files in the directory.
- `ignore` (optional): Regex pattern to ignore files in the directory.
- `limit` (optional): The token limit for the output prompt, defaults to 100K. Prompts exceeding the limit will be compressed.
- `ai_extraction` (optional): Extract tables, figures, and math from PDFs using our extractor. Incurs extra costs.
- `text_only` (optional): Do not extract images from documents or websites. Additionally, image files will be represented with OCR instead of as images.