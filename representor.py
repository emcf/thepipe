import tempfile
import os
import shutil
import subprocess
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from colorama import Fore, Style
import cssutils
import json
import os
import base64
from config import CTAGS_COMPATIBLE_EXTENSIONS, LLMLINGUA_COMPATIBLE_EXTENSIONS
from llmlingua import PromptCompressor

language_map = {
    '.py': 'Python',
    '.c': 'C',
    '.h': 'C',
    '.cpp': 'C++',
    '.js': 'JavaScript',
    '.jsx': 'JavaScript',
    '.ts': 'TypeScript',
    '.tsx': 'TypeScript',
    '.html': 'HTML',
    '.css': 'CSS',
    '.java': 'Java',
    '.php': 'PHP',
    '.rb': 'Ruby',
    '.sh': 'Bash',
    '.xml': 'XML'
}

def get_token_count(text):
    # heuristic, since tokenizers are slow
    CHARS_PER_TOKEN = 4
    if text is None: return 0
    if text.strip() == '': return 0
    return int(len(text)/CHARS_PER_TOKEN)

def create_compressed_project_context(files, token_limit, verbose=False):
    def create_project_context(files):
        context = {'text': "", 'images': []}
        for file in files:
            if isinstance(file.content, str):
                context['text'] += f'{file.filename}:\n'
                context['text'] += f'```\n{file.content}\n```\n'
            elif isinstance(file.content, bytes):
                img_str = base64.b64encode(file.content).decode('utf-8')
                context['images'].append(img_str)
        return context
    attempts = 0
    max_attempts = 10
    k_files_to_compress = 3
    project_context = create_project_context(files)
    token_count = get_token_count(project_context['text'])
    # Compress the project context until it is within the token limit
    while get_token_count(project_context['text']) > token_limit:
        if attempts >= max_attempts:
            print(Style.RESET_ALL + Fore.RED + f"Failed to compress project within {max_attempts} attempts. Returning the uncompressed context." + Style.RESET_ALL)
            return project_context
        if verbose: print(Style.RESET_ALL + Fore.YELLOW + f"Compressing {len(files)} file project ({token_count} tokens) to under {token_limit} tokens..." + Style.RESET_ALL)
        token_count_per_file = {file.filename: get_token_count(file.content) for file in files if isinstance(file.content, str)}
        top_largest_files = sorted(token_count_per_file, key=token_count_per_file.get, reverse=True)[:k_files_to_compress]
        if verbose: print(Style.RESET_ALL + Fore.YELLOW + f"Compressing largest {k_files_to_compress} files: {top_largest_files}..." + Style.RESET_ALL)
        # Summarize top k largest files using multithreading
        files_to_summarize = [file for file in files if file.filename in top_largest_files]
        for file in files_to_summarize:
            extension = '.' + file.filename.split('.')[-1]
            file.content = summarize_content(file.content, extension)
        # Update the project context with the summarized content
        project_context = create_project_context(files)
        attempts += 1
        if verbose: print(Style.RESET_ALL + Fore.GREEN + f"Compressed project ✔️" + Style.RESET_ALL)
    return project_context

def get_ctags_outline(code, extension, language="Python"):
    tmp_dir = tempfile.mkdtemp()
    try:
        file_path = os.path.join(tmp_dir, "tempfile" + extension)
        with open(file_path, 'w', encoding='utf-8') as tmp_file:
            tmp_file.write(code)
        cmd = [
            "ctags",
            f"--languages={language}",
            "--output-format=json",
            "-f", "-",
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Error running ctags: {result.stderr}")
        
        # Process the JSON output
        outlines = []
        current_class = None
        # The JSON output from ctags is one JSON object per line, so we need to parse each line individually
        for line in result.stdout.strip().splitlines():
            tag = json.loads(line)
            if tag['kind'] == 'class':
                current_class = tag['name']
                outlines.append("class " + tag['name'] + ':')
            elif tag['kind'] in ('function', 'member'):
                # Indent member functions if they belong to a class
                prefix = '    ' if ('scope' in tag and tag['scope']==current_class) else ''
                signature = tag['pattern'].strip('/^').rstrip('$/;').rstrip(' {').rstrip('{').strip()
                outlines.append(prefix + signature)
            # Reset current class if we encounter a function that is not a member of a class
            if tag['kind'] == 'function' and current_class and 'scope' not in tag:
                current_class = None
    finally:
        shutil.rmtree(tmp_dir)
    return '\n'.join(outlines)

walk_str = ''
def get_html_outline(html_content):
    global walk_str
    soup = BeautifulSoup(html_content, 'html.parser')
    walk_str = ''
    def walk(node, level=0):
        global walk_str
        if isinstance(node, NavigableString):
            text = node.strip()
            if text:
                walk_str += (' ' * level + text) +'\n'
        elif isinstance(node, Tag):
            walk_str += (' ' * level + node.name) +'\n'
            for child in node.children:
                walk(child, level + 1)
    walk(soup)
    return walk_str

def get_css_outline(css_content):
    outline = ''
    sheet = cssutils.parseString(css_content)
    for rule in sheet:
        if rule.type == rule.STYLE_RULE:
            outline += (rule.selectorText) + '\n'
    return outline

def summarize_content(content, extension):
    try:
        if extension in CTAGS_COMPATIBLE_EXTENSIONS:
            compressed = "Code outline:\n" + get_ctags_outline(content, extension, language=language_map[extension])
        elif extension == '.html':
            compressed = "HTML outline:\n" + get_html_outline(content)
        elif extension == '.css':
            compressed = "CSS definitions:\n" + get_css_outline(content)
        elif extension in LLMLINGUA_COMPATIBLE_EXTENSIONS:
            COMPRESSION_RATIO = 2
            n_tokens = get_token_count(content)
            llm_lingua = PromptCompressor()
            compressed = llm_lingua.compress_prompt(content, instruction="", question="", target_token=n_tokens//COMPRESSION_RATIO)
        else:
            print(Style.RESET_ALL + Fore.YELLOW + f"Unsupported extension {extension}. Returning the original content." + Style.RESET_ALL)
            return content
        return compressed
    except Exception as e:
        print(Style.RESET_ALL + Fore.RED + f"Error summarizing content: {e}, returning the original content." + Style.RESET_ALL)
        return content