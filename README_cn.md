# <a href="https://thepi.pe/"><img src="https://rpnutzemutbrumczwvue.supabase.co/storage/v1/object/public/assets/pipeline_small%20(1).png" alt="管道插图" style="width:96px; height:72px; vertical-align:middle;"> 管道</a>
<p>
  <a href="https://github.com/emcf/thepipe/blob/main/README.md">English</a> | <a href="https://github.com/emcf/thepipe/blob/main/README_cn.md">中文</a>
</p>

[![codecov](https://codecov.io/gh/emcf/thepipe/graph/badge.svg?token=OE7CUEFUL9)](https://codecov.io/gh/emcf/thepipe) ![python-gh-action](https://github.com/emcf/thepipe/actions/workflows/python-ci.yml/badge.svg) <a href="https://thepi.pe/">![网站](https://img.shields.io/website?url=https%3A%2F%2Fthepipe.up.railway.app%2F&label=API%20状态)</a> <a href="https://thepi.pe/">![获取 API](https://img.shields.io/badge/API-获取访问权限-blue)</a>

### 用一行代码为视觉-语言模型准备 PDF、Word 文档、幻灯片、网页等 ⚡

管道是一个以多模态为首的工具，用于将文件和网页输入到如 GPT-4V 等视觉-语言模型中。它最适合需要深入理解复杂数据源的 LLM 和 RAG 应用。管道可作为托管 API 在 [thepi.pe](https://thepi.pe) 上使用，或者可以在本地设置。

## 开始使用 🚀

首先，安装管道。
```
pip install thepipe_api
```

确保设置了 `THEPIPE_API_KEY` 环境变量。还没有 API 密钥？[在这里获取](https://thepi.pe)。想要在本地自行操作？请参阅本地安装部分。

现在您可以从任何文件中提取全面的文本和视觉内容：
```python
from thepipe_api import thepipe
chunks = thepipe.extract("example.pdf")
```
或任何网站：
```python
chunks = thepipe.extract("https://example.com")
```
然后将其输入到 GPT-4-Vision：
```python
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages = chunks,
)
```
管道的输出是一系列合理的“块”，因此可以用于向量数据库的存储或直接用作提示。在 [API 文档](https://thepi.pe/docs) 中提供了数据表提取、条形图提取、自定义网页认证等额外功能。[LiteLLM](https://github.com/BerriAI/litellm) 可用于轻松将管道与任何 LLM 提供商集成。

## 特性 🌟

- 从任何文件或网页提取文本和视觉内容 📚
- 输出为 RAG 准备的块，优化了多模态 LLMs 🖼️ + 💬
- 能解读复杂的 PDF、网页应用、markdown 等 🧠
- 自动压缩超出您选择的令牌限制的提示 📦
- 支持缺失文件扩展名、内存数据流 💾
- 支持代码库、URL、git 仓库等 🌐
- 多线程 ⚡️

## 工作原理 🛠️

管道可以通过命令行或 [Python](https://www.python.org/downloads/) 访问。输入源可以是文件路径、URL 或目录（或 zip 文件）路径。管道将从源中提取信息，并为与 [语言模型](https://en.wikipedia.org/wiki/Large_language_model)、[视觉变换器](https://en.wikipedia.org/wiki/Vision_transformer) 或 [视觉-语言模型](https://arxiv.org/abs/2304.00685) 的下游使用处理信息。管道的输出是提取信息的合理文本（或多模态）表示，精心制作以适应从 [gemma-7b](https://huggingface.co/google/gemma-7b) 到 [GPT-4](https://openai.com/gpt-4) 的任何模型的上下文窗口。它使用各种启发式方法以最佳性能与视觉-语言模型配合使用，包括 AI 文件类型检测、AI [PDF 提取](thepi.pe/docs)、高效 [令牌压缩](https://arxiv.org/abs/2403.12968)、自动 [图像编码](https://en.wikipedia.org/wiki/Base64)、[重排](https://arxiv.org/abs/2310.06839) 以解决 [中间丢失](https://arxiv.org/abs/2307.03172) 效应等，所有这些都预先构建好，开箱即用。

## 支持的文件类型 📚

| 源类型                               | 输入类型                              | 令牌压缩 🗜️ | 图像提取 👁️ | 备注 📌                                                  |
|---------------------------------------|------------------------------------------|-------------------|------------------|---------------------------------------------------------|
| 目录                                 | 任何 `/path/to/directory`                 | ✔️               | ✔️               | 从目录中的所有文件提取，支持匹配和忽略模式 |
| 代码                                  | `.py`, `.tsx`, `.js`, `.html`, `.css`, `.cpp` 等 | ✔️ (变化)   | ❌               | 合并所有代码文件。`.c`, `.cpp`, `.py` 可以使用 ctags 压缩，其他则不行 |
| 纯文本                             | `.txt`, `.md`, `.rtf` 等               | ✔️               | ❌               | 普通文本文件                                                      |
| PDF                                   | `.pdf`                                  | ✔️               | ✔️    | 提取每页的文本和图像；可以使用 AI 提取表格数据和页面内图像 |
| 图像                                 | `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.tiff`, `.webp`, `.svg` | ❌                | ✔️              | 提取图像，如果仅文本则使用 OCR                        |
| 数据表                           | `.csv`, `.xls`, `.xlsx`             | ✔️                | ❌               | 从电子表格中提取数据；转换为文本表示。对于非常大的数据集，将仅提取列名和类型         |
| Jupyter 笔记本                      | `.ipynb`                                | ❌               | ✔️               | 从 Jupyter 笔记本提取代码、markdown 和图像                                  |
| Microsoft Word 文档               | `.docx`                                 | ✔️               | ✔️               | 从 Word 文档提取文本和图像                                        |
| Microsoft PowerPoint 演示文稿     | `.pptx`                                 | ✔️               | ✔️               | 从 PowerPoint 演示文稿提取文本和图像                              |
| 网站                               | URLs (包含 `http`, `https`, `www`, `ftp`)             | ✔️                | ✔️    | 从网页提取文本及图像（如果可滚动则为多图像）；可提供仅文本提取          |
| GitHub 仓库                     | GitHub 仓库 URLs                         | ✔️               | ✔️                | 从 GitHub 仓库提取；支持分支指定         |
| ZIP 文件                              | `.zip`                                  | ✔️               | ✔️                | 提取 ZIP 文件内容；支持嵌套目录提取     |

## 本地安装 🛠️

要在本地使用管道，您需要 [playwright](https://github.com/microsoft/playwright)、[ctags](https://github.com/universal-ctags/)、[pytesseract](https://github.com/h/pytesseract) 以及与更轻量级 API 要求不同的本地 python 要求。您还需要使用本地版本的要求文件：

```bash
git clone https://github.com/emcf/thepipe
pip install -r requirements_local.txt
```

Windows 用户提示：您可能需要使用 `pip install python-magic-bin` 安装 python-libmagic 二进制文件。

现在您可以使用管道了：
```bash
from thepipe_api import thepipe
chunks = thepipe.extract("example.pdf", local=True)
```

参数有：
- `source` (必需)：可以是文件路径、URL 或目录路径。
- `local` (可选)：使用管道的本地版本而不是托管 API。
- `match` (可选)：正则表达式模式，用于匹配目录中的文件。
- `ignore` (可选)：正则表达式模式，用于忽略目录中的文件。
- `limit` (可选)：输出提示的令牌限制，默认为 100K。超出限制的提示将被压缩。
- `ai_extraction` (可选)：使用我们的提取器从 PDF 中提取表格、图形和数学内容。会产生额外成本。
- `text_only` (可选)：不从文档或网站中提取图像。此外，图像文件将以 OCR 而非图像形式表示。