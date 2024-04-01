# <a href="https://thepi.pe/"><img src="https://rpnutzemutbrumczwvue.supabase.co/storage/v1/object/public/assets/pipeline_small%20(1).png" alt="Pipeline Illustration" style="width:96px; height:72px; vertical-align:middle;"> The Pipe</a>
[![codecov](https://codecov.io/gh/emcf/thepipe/graph/badge.svg?token=OE7CUEFUL9)](https://codecov.io/gh/emcf/thepipe) ![python-gh-action](https://github.com/emcf/thepipe/actions/workflows/python-ci.yml/badge.svg) <a href="https://thepi.pe/">![Website](https://img.shields.io/website?url=https%3A%2F%2Fthepipe.up.railway.app%2F&label=API%20status)</a> <a href="https://thepi.pe/">![get API](https://img.shields.io/badge/API-Apply%20here-blue)</a>

### Prepare any PDF, Word doc, CSV, image, web page, GitHub repo, and more for GPT-4V with one line of code ⚡

The pipe is a multimodal-first tool for flattening unstructured files, directories, and websites into a prompt-ready format for use with large language models. It is built on top of dozens of carefully-crafted heuristics to create sensible text and image prompts from files, directories, web pages, papers, github repos, etc. 

![Demo](https://ngrdaaykhfrmtpodlakn.supabase.co/storage/v1/object/public/assets/demo.gif?t=2024-03-24T19%3A13%3A46.695Z)


## Features 🌟

- Prepare prompts from dozens of complex file types 📄 
- Visual document extraction for complex PDFs, markdown, etc 🧠
- Outputs optimized for multimodal LLMs 🖼️ + 💬
- Auto compresses prompts over your set token limit 📦
- Works with missing file extensions, in-memory data streams 💾
- Works with directories, URL, git repos, and more 🌐
- Multi-threaded ⚡️

If you are hosting the pipe for yourself, you can extract and use the output like this:

```python
import openai
import thepipe
openai_client = openai.OpenAI()
response = openai_client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages = thepipe.extract("example.pdf"),
)
```

## Getting Started 🚀

You can either use the hosted API at [thepi.pe](https://thepi.pe) or run The Pipe locally. The simplest way to use the pipe is to use the hosted API by following the instructions at the [API documentation page](https://thepi.pe/docs).

To use The Pipe locally, you will need [playwright](https://github.com/microsoft/playwright), [ctags](https://github.com/universal-ctags/), [pytesseract](https://github.com/h/pytesseract), and the python requirements:

```bash
git clone https://github.com/emcf/thepipe
pip install -r requirements.txt
```

Tip for windows users: you may need to install the python-libmagic binaries with `pip install python-magic-bin`.

Now you can use The Pipe:
```bash
python thepipe.py path/to/directory
```

This command will process all supported files within the specified directory, compressing any information over the token limit if necessary, and outputting the resulting prompt and images to a folder.

Arguments are:
- The input source (required): can be a file path, a URL, or a directory path.
- `--match` (optional): Regex pattern to match files in the directory.
- `--ignore` (optional): Regex pattern to ignore files in the directory.
- `--limit` (optional): The token limit for the output prompt, defaults to 100K. Prompts exceeding the limit will be compressed.
- `--mathpix` (optional): Extract images, tables, and math from PDFs using [Mathpix](https://docs.mathpix.com/#process-a-pdf).
- `--text_only` (optional): Do not extract images from documents or websites. Additionally, image files will be represented with OCR instead of as images.

You can use the pipe's output with other LLM providers via [LiteLLM](https://github.com/BerriAI/litellm).


##  How it works 🛠️

The pipe is accessible from the command line or from [Python](https://www.python.org/downloads/). The input source is either a file path, a URL, or a directory (or zip file) path. The pipe will extract information from the source and process it for downstream use with [language models](https://en.wikipedia.org/wiki/Large_language_model), [vision transformers](https://en.wikipedia.org/wiki/Vision_transformer), or [vision-language models](https://arxiv.org/abs/2304.00685). The output from the pipe is a sensible text-based (or multimodal) representation of the extracted information, carefully crafted to fit within context windows for any models from [gemma-7b](https://huggingface.co/google/gemma-7b) to [GPT-4](https://openai.com/gpt-4). It uses a variety of heuristics for optimal performance with vision-language models, including AI filetype detection with [filetype detection](https://opensource.googleblog.com/2024/02/magika-ai-powered-fast-and-efficient-file-type-identification.html), AI [PDF extraction](https://mathpix.com), efficient [token compression](https://arxiv.org/abs/2403.12968), automatic [image encoding](https://en.wikipedia.org/wiki/Base64), [reranking](https://arxiv.org/abs/2310.06839) for [lost-in-the-middle](https://arxiv.org/abs/2307.03172) effects, and more, all pre-built to work out-of-the-box.

## Supported File Types 📚

| Source Type                           | Input types        | Token Compression 🗜️ | Image Extraction 👁️ | Notes 📌                                                  |
|---------------------------------------|------------------------------------------|-------------------|------------------|---------------------------------------------------------|
| Directory                             | Any `/path/to/directory`                 | ✔️               | ✔️               | Extracts from all files in directory, supports match and ignore patterns |
| Code                                  | `.py`, `.tsx`, `.js`, `.html`, `.css`, `.cpp`, etc | ✔️ (varies)   | ❌               | Combines all code files. `.c`, `.cpp`, `.py` are compressible with ctags, others are not |
| Plaintext                             | `.txt`, `.md`, `.rtf`, etc               | ✔️               | ❌               | Regular text files                                                      |
| PDF                                   | `.pdf`                                  | ✔️               | ✔️    | Extracts text and images of each page; can use Mathpix for extraction of images within pages |
| Image                                 | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.svg` | ❌                | ✔️              | Extracts images, uses OCR if text_only                        |
| Data Table                           | `.csv`, `.xls`, `.xlsx`             | ✔️                | ❌               | Extracts data from spreadsheets; converts to text representation. For very large datasets, will only extract column names and types         |
| Jupyter Notebook                      | `.ipynb`                                | ❌               | ✔️               | Extracts code, markdown, and images from Jupyter notebooks                                  |
| Microsoft Word Document               | `.docx`                                 | ✔️               | ✔️               | Extracts text and images from Word documents                                        |
| Microsoft PowerPoint Presentation     | `.pptx`                                 | ✔️               | ✔️               | Extracts text and images from PowerPoint presentations                              |
| Website                               | URLs (inputs containing `http`, `https`, `www`, `ftp`)             | ✔️                | ✔️    | Extracts text from web page along with image (or images if scrollable); text-only extraction available          |
| GitHub Repository                     | GitHub repo URLs                         | ✔️               | ✔️                | Extracts from GitHub repositories; supports branch specification         |
| ZIP File                              | `.zip`                                  | ✔️               | ✔️                | Extracts contents of ZIP files; supports nested directory extraction     |
