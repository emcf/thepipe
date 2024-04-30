
# <a href="https://thepi.pe/"><img src="https://rpnutzemutbrumczwvue.supabase.co/storage/v1/object/public/assets/pipeline_small%20(1).png" alt="Pipeline Illustration" style="width:96px; height:72px; vertical-align:middle;"> The Pipe</a>
<p>
  <a href="https://github.com/emcf/thepipe/blob/main/README.md">English</a> | <a href="https://github.com/emcf/thepipe/blob/main/README_cn.md">中文</a>
</p>

[![codecov](https://codecov.io/gh/emcf/thepipe/graph/badge.svg?token=OE7CUEFUL9)](https://codecov.io/gh/emcf/thepipe) ![python-gh-action](https://github.com/emcf/thepipe/actions/workflows/python-ci.yml/badge.svg) <a href="https://thepi.pe/">![Website](https://img.shields.io/website?url=https%3A%2F%2Fthepipe.up.railway.app%2F&label=API%20status)</a> <a href="https://thepi.pe/">![get API](https://img.shields.io/badge/API-access-blue)</a> <a href="https://discord.gg/bXfKeGs5qV">![Join discord](https://img.shields.io/discord/1227806200478044274?color=4f69ef&label=Discord&logo=discord&logoColor=ffffff)</a>

### Feed PDFs, web pages, word docs, slides, videos, CSV, and more into Vision-LLMs with one line of code ⚡

The Pipe is a multimodal-first tool for feeding files and web pages into vision-language models such as GPT-4V. It is best for LLM and RAG applications that require a deep understanding of tricky data sources. The Pipe is available as a hosted API at [thepi.pe](https://thepi.pe), or it can be set up locally. 

![Science assistant demo](https://rpnutzemutbrumczwvue.supabase.co/storage/v1/object/public/assets/science_assistantpy2.png)


## Features 🌟

- Extracts text and visuals from files or web pages 📚
- Outputs chunks optimized for multimodal LLMs and RAG frameworks 🖼️
- Interpret complex PDFs, web pages, docs, videos, data, and more 🧠
- Auto-compress prompts exceeding your chosen token limit 📦
- Works even with missing file extensions, in-memory data streams 💾
- Works with codebases, git repos, and custom integrations 🌐
- Multi-threaded ⚡️

## Getting Started  🚀

The Pipe handles a wide array of complex filetypes, and thus has many dependencies that must be installed separately. It also requires a strong machine for good response times. For this reason, we host it as an API that works out-of-the-box. 

First, install The Pipe. 
```
pip install thepipe_api
```

The Pipe is available as a hosted API, or it can be set up locally. An API key is recommended for out-of-the-box functionality (alternatively, see the local installation section). Ensure the `THEPIPE_API_KEY` environment variable is set. Don't have a key yet? [Get one here](https://thepi.pe).

Now you can extract comprehensive text and visuals from any file:
```python
from thepipe_api import thepipe
messages = thepipe.extract("example.pdf")
```
Or websites:
```python
messages = thepipe.extract("https://example.com")
```
Then feed it into GPT-4-Vision:
```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages = messages,
)
```

![Just call OpenAI](https://rpnutzemutbrumczwvue.supabase.co/storage/v1/object/public/assets/IMG_0180.jpg)

You can also use The Pipe from the command line. Here's how to recursively extract from a directory, matching only files containing a substring (in this example, typescript files) and ignore files containing other substrings (in this example, anything in the "tests" folder):
```bash
thepipe path/to/folder --match tsx --ignore tests
```


## Supported File Types 📚

| Source Type                           | Input types        | Token Compression 🗜️ | Image Extraction 👁️ | Notes 📌                                                  |
|---------------------------------------|------------------------------------------|-------------------|------------------|---------------------------------------------------------|
| Directory                             | Any `/path/to/directory`                 | ✔️               | ✔️               | Extracts from all files in directory, supports match and ignore patterns |
| Code                                  | `.py`, `.tsx`, `.js`, `.html`, `.css`, `.cpp`, etc | ✔️ (varies)   | ❌               | Combines all code files. `.c`, `.cpp`, `.py` are compressible with ctags, others are not |
| Plaintext                             | `.txt`, `.md`, `.rtf`, etc               | ✔️               | ❌               | Regular text files                                                      |
| PDF                                   | `.pdf`                                  | ✔️               | ✔️    | Extracts text and images of each page; can use AI for extraction of table data and  images within pages |
| Image                                 | `.jpg`, `.jpeg`, `.png` | ❌                | ✔️              | Extracts images, uses OCR if text_only                        |
| Data Table                           | `.csv`, `.xls`, `.xlsx`             | ✔️                | ❌               | Extracts data from spreadsheets; converts to text representation. For very large datasets, will only extract column names and types         |
| Jupyter Notebook                      | `.ipynb`                                | ❌               | ✔️               | Extracts code, markdown, and images from Jupyter notebooks                                  |
| Microsoft Word Document               | `.docx`                                 | ✔️               | ✔️               | Extracts text and images from Word documents                                        |
| Microsoft PowerPoint Presentation     | `.pptx`                                 | ✔️               | ✔️               | Extracts text and images from PowerPoint presentations                              |
| Video                                 | `.mp4`, `.avi`, `.mov`, `.wmv`     | ✔️               | ✔️                | Extracts frames from video files; supports frame extraction and OCR for text extraction from frames |
| Audio                                 | `.mp3`, `.wav`          | ✔️               | ❌                | Extracts text from audio files; supports speech-to-text conversion        | 
| Website                               | URLs (inputs containing `http`, `https`, `ftp`)             | ✔️                | ✔️    | Extracts text from web page along with image (or images if scrollable); text-only extraction available          |
| GitHub Repository                     | GitHub repo URLs                         | ✔️               | ✔️                | Extracts from GitHub repositories; supports branch specification         |
| YouTube Video                         | YouTube video URLs                      | ✔️               | ✔️                | Extracts text from YouTube videos; supports subtitles extraction          |
| ZIP File                              | `.zip`                                  | ✔️               | ✔️                | Extracts contents of ZIP files; supports nested directory extraction     |

## How it works 🛠️

The input source is either a file path, a URL, or a directory. The pipe will extract information from the source and process it for downstream use with [language models](https://en.wikipedia.org/wiki/Large_language_model), [vision transformers](https://en.wikipedia.org/wiki/Vision_transformer), or [vision-language models](https://arxiv.org/abs/2304.00685). The output from the pipe is a sensible list of multimodal messages representing chunks of the extracted information, carefully crafted to fit within context windows for any models from [gemma-7b](https://huggingface.co/google/gemma-7b) to [GPT-4](https://openai.com/gpt-4). The messages returned should look like this:
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
If you want to feed these messages directly into the model, it is important to be mindful of the token limit.
OpenAI does not allow too many images in the prompt (see discussion [here](https://community.openai.com/t/gpt-4-vision-maximum-amount-of-images/573110/6)), so long files should be extracted with `text_only=True` to avoid this issue, while long text files should either be compressed or embedded in a RAG framework.

The text and images from these messages may also be prepared for a vector database with `thepipe.core.create_chunks_from_messages` or for downstream use with RAG frameworks. [LiteLLM](https://github.com/BerriAI/litellm) can be used to easily integrate The Pipe with any LLM provider. 

It uses a variety of heuristics for optimal performance with vision-language models, including AI filetype detection with [filetype detection](https://opensource.googleblog.com/2024/02/magika-ai-powered-fast-and-efficient-file-type-identification.html), opt-in AI [table, equation, and figure extraction](https://thepi.pe/pricing), efficient [token compression](https://arxiv.org/abs/2403.12968), automatic [image encoding](https://en.wikipedia.org/wiki/Base64), [reranking](https://arxiv.org/abs/2310.06839) for [lost-in-the-middle](https://arxiv.org/abs/2307.03172) effects, and more, all pre-built to work out-of-the-box.

![Demo](https://rpnutzemutbrumczwvue.supabase.co/storage/v1/object/public/assets/grader.py%20(6).png)


## Local Installation 🛠️

If you do not wish to use our API, you are welcome host The Pipe for yourself locally. 
If you choose to do this, you must install a number of dependencies for the code to function correctly, some of which may incur compute costs and/or require a GPU for reasonable performance. Additional installed dependencies are required: pytorch, universal-ctags, playwright, pytesseract, llmlingua, moviepy, and pytube. This installation process will depend on your system and compute capabilities. After installing them, follow these steps for a local setup:

Arguments are:
- `source` (required): can be a file path, a URL, or a directory path.
- `local` (optional): Use the local version of The Pipe instead of the hosted API.
- `match` (optional): Substring to match files in the directory. Regex is not yet supported.
- `ignore` (optional): Substring to ignore files in the directory. Regex is not yet supported.
- `limit` (optional): The token limit for the output prompt, defaults to 100K. Prompts exceeding the limit will be compressed. This may not work as expected with the API, as it is in active development.
- `ai_extraction` (optional): Extract tables, figures, and math from PDFs using our extractor. Incurs extra costs.
- `text_only` (optional): Do not extract images from documents or websites. Additionally, image files will be represented with OCR instead of as images.

# Sponsors

<a href="https://cal.com/emmett-mcf/30min"><img alt="Book us with Cal.com" src="https://cal.com/book-with-cal-dark.svg" /></a>

Thank you to [Cal.com](https://cal.com/) for sponsoring this project. Contact emmett@thepi.pe for sponsorship information.