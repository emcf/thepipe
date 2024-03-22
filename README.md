# <img src="https://rpnutzemutbrumczwvue.supabase.co/storage/v1/object/public/assets/pipeline_small%20(1).png" alt="Pipeline Illustration" style="width:96px; height:72px; vertical-align:middle;"> The Pipe

[![codecov](https://codecov.io/gh/emcf/thepipe/graph/badge.svg?token=KHD1PDOSHF)](https://codecov.io/gh/emcf/thepipe) ![python-gh-action](https://github.com/emcf/thepipe/actions/workflows/python-ci.yml/badge.svg)

The pipe is a tool for feeding complex real-world data into large language models. It is built on top of dozens of carefully-crafted heuristics to create sensible representations from a variety of sources, including code projects, scientific papers, web pages, github repos, data files, databases, and more.

## 🛠️ How it works 

The pipe is accessible from the command line or from [Python](https://www.python.org/downloads/). The input source is either a file path, a URL, or a directory (or zip file) path. The pipe will extract information from the source and process it for downstream use with [LLMs](https://en.wikipedia.org/wiki/Large_language_model). The output from the pipe is a sensible text-based (or multimodal) representation of the extracted information, carefully crafted to fit within context windows for any models from [gemma-7b](https://huggingface.co/google/gemma-7b) to [GPT-4](https://openai.com/gpt-4). It uses a variety of heuristics to optimize the output for LLMs, including [AI-native PDF extraction](https://docs.mathpix.com/#process-a-pdf), [efficient token compression](https://arxiv.org/abs/2403.12968), [code compression with Ctags](https://en.wikipedia.org/wiki/Ctags), automatic [image encoding](https://en.wikipedia.org/wiki/Base64), reranking for [LITM](https://arxiv.org/abs/2307.03172) effects, and more, all pre-built to work out-of-the-box.

## 📂 Supported input sources

### Sources
- 📁 **Project directories** (any directory)
- 🗂️ **Zip / Tarballs** (`.zip`, `.tar`, `.gz`)
- 🔗 **URLs** (any input containing `http` or `www`, or `.url` shortcut file)
- 🐙 **GitHub Repositories** (any input containing `github.com`)
- 🗃️ **Business Database** (any input containing `supabase`)

### Documents
- 📜 **Code files** (`.py`, `.cpp`, `.ts`, `.css`, `.h`, etc.)
- 📚 **PDFs** (`.pdf`) (`.pdf` or any input containing `arxiv`, extract images/tables/math with `--mathpix`)
- 🖼️ **Images** (`.png`, `.jpg`, `.jpeg`, `.gif`)
- 📊 **Spreadsheets** (`.csv`, `.xlsx`)
- 📜 **Configuration files** (`.yaml`, `.json`, `.xml`, `.ini`, `.xaml`, `.cfg`, `.config`)
- 📓 **IPython notebooks** (`.ipynb`)
- 📝 **Word documents** (`.docx`)
- 📊 **Powerpoint presentations** (`.pptx`)



## 🚀 Getting Started

To use The Pipe, simply clone this repository and run

```bash
python thepipe.py --source /path/to/directory --output prompt.txt
```

This command will process all supported files within the specified directory, compressing the information over the token limit if necessary, and outputting the result to `output.txt`.

Arguments are:
- `--source` (required): The input source, can be a file path, a URL, or a directory path.
- `--output` (required): The output file path.
- `--limit` (optional): The token limit for the output, defaults to 64K.
- `--mathpix` (optional): Extract images, tables, and math from PDFs using [Mathpix](https://docs.mathpix.com/#process-a-pdf).
- `--text` (optional): Output text scraped from images instead of [base64](https://en.wikipedia.org/wiki/Base64) encoded images.

Alternatively, to use the pipe from Python:

```python
import openai
import thepipe
openai_client = openai.OpenAI()
response = openai_client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages = thepipe.extract("https://github.com/emcf/thepipe"),
)
response_content = response.choices[0].message.content
print(response_content)
```

## ⚖️ Heuristics

To optimize the output for downstream tasks, the pipe uses a variety of assumptions and heuristics to extract the most important information from the input data, and to format it. Here are some of the most important ones:
- **Optional [Mathpix](https://docs.mathpix.com/#process-a-pdf) PDF extraction**: Optional, extracts images, tables, and math from PDFs.
- **[Ctags](https://en.wikipedia.org/wiki/Ctags) token compression**: When the output prompt is too large, automatically extracts essential code structure (functions, classes, variables, types) and throws away the rest. Useful for high-quality coding under strict token constraints.
- **[LLMLingua](https://arxiv.org/abs/2403.12968) token compression**: When the output prompt is too large, automatically extracts essential tokens, can improve downstream performance by removing noise.
- **[LITM](https://arxiv.org/abs/2307.03172) Reranking**: Reformats the output to minimize the impact of the "lost in the middle" effect to improve downstream performance with LLMs.
- **Image resizing, [base64](https://en.wikipedia.org/wiki/Base64) encoding**: Maximum image dimensions are clipped to 512 pixels and encoded in base64 for easy downstream use with vision language models. Can alternatively output a text description of all images with `--text`, or text scraped from all images with `--scrape`.
- [Unstructured](https://github.com/Unstructured-IO/unstructured) extraction from unknown sources
- **Ignore Rules**: Sensible out-of-the-box ignore rules for common directories and files that are not useful for downstream tasks, such as `node_modules`, `__pycache__`, `.gitignore`, etc. Feel free to customize these for your own use case by modifying `FILES_TO_IGNORE` in `config.py`.

## License 📜

Distributed under the MIT License. See `LICENSE` for more information.

---

Made with ❤️ and Python.