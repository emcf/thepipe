import glob
import os
import tempfile
from urllib.parse import unquote, urlparse
import zipfile
from colorama import Fore, Style
import pandas as pd
from PIL import Image
import requests
import json
from config import TXT_EXTENSIONS
import pytesseract
from unstructured.partition.auto import partition
from langchain_community.document_loaders import PlaywrightURLLoader
from config import FILES_TO_IGNORE, GITHUB_TOKEN

class File:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content

def read_data(filename, extension):
    if extension == '.csv':
        df = pd.read_csv(filename)
    elif extension == '.xlsx':
        df = pd.read_excel(filename)
    colnames = list(df.columns)
    coltypes = list(df.dtypes)
    content_string = f'Column names and column types:\n{list(zip(colnames, coltypes))}'
    return content_string

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def read_image(image_path, use_text=False):
    if use_text:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return "This image contains the following text:\n" + text
    else:
        with open(image_path, "rb") as image_file:
            return image_file.read()

def get_content_from_file(filename, use_mathpix = False, use_text = False, verbose = False):
    extension = os.path.splitext(filename)[1].lower()
    if use_mathpix:
        # mathpix call
        return None
    if extension == '.url':
        extracted_url = None
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                if 'URL=' in line:
                    extracted_url = line.split('=')[1]
                    break
        if not extracted_url: 
            return None
        return read_url(extracted_url)
    if extension in {'.csv', '.xlsx'}:
        return read_data(filename, extension=extension)
    elif extension in {'.jpg', '.jpeg', '.png', '.gif', '.webp'}:
        return read_image(filename, use_text=use_text)
    elif extension == '.zip':
        return read_zip(filename, use_text=use_text)
    else:
        try:
            elements = partition(filename)
            file_content = "\n\n".join([str(el) for el in elements])
            return file_content
        except Exception as e:
            if verbose: print(Style.RESET_ALL + Fore.RED + f"Error reading {filename}: {e}" + Style.RESET_ALL)
            if extension in TXT_EXTENSIONS:
                if verbose: print(Style.RESET_ALL + Fore.YELLOW + f"Attempting to re-read as plaintext..." + Style.RESET_ALL)
                file_content = read_text_file(filename)
                if verbose: print(Style.RESET_ALL + Fore.GREEN + f"Success! {filename} read as plaintext." + Style.RESET_ALL)
                return file_content
            raise ValueError(f"Error reading {filename}: {e}")

def assume_user_input_type(path):
    if 'github.com' in path:
        return 'github'
    elif 'http' in path or 'www' in path:
        return 'url'
    elif os.path.isfile(path):
        return 'file'
    elif os.path.isdir(path):
        return 'dir'
    else:
        raise ValueError('Input is not a valid file or directory')

def read_url(url):
    loader = PlaywrightURLLoader(urls=[url]) # must run `playwright install`
    data = loader.load()  
    page_contents = "\n\n".join([str(el.page_content) for el in data])
    return page_contents

# Function to fetch contents from GitHub, fetches from main or branch by default
def read_github_repo(github_url, file_path='', ref='main', substrings_to_ignore=[], verbose=True):
    repo_representation = []
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable is not set.")
    # Function to extract repo details from GitHub URL
    def extract_repo_details(url):
        path = urlparse(url).path
        path_parts = path.strip('/').split('/')
        if len(path_parts) >= 2:
            return path_parts[0], path_parts[1]
        else:
            raise ValueError("Invalid GitHub URL provided.")
    owner, repo = extract_repo_details(github_url)
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{unquote(file_path)}?ref={ref}"
    headers = {
        'Accept': 'application/vnd.github.v3.raw',
        'Authorization': f'token {GITHUB_TOKEN}',
        'X-GitHub-Api-Version': '2022-11-28'
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        response_json = json.loads(response.text)
        for item in response_json:
            if 'path' not in item:
                continue
            path = item['path']
            if any(path.endswith(extension) for extension in FILES_TO_IGNORE):
                continue
            if any(path.startswith(prefix) for prefix in ('~$', '.')):
                continue
            if any(substring in path for substring in substrings_to_ignore):
                continue
            if item['type'] == 'file':
                file_content_request = requests.get(item['download_url'], headers=headers)
                extension = os.path.splitext(path)[1].lower()
                # save a temporary file to read it
                temp_file_path = f"temp{extension}"
                with open(temp_file_path, 'wb') as temp_file:
                    temp_file.write(file_content_request.content)
                    file_content = get_content_from_file(temp_file_path, verbose=verbose)
                os.remove(temp_file_path)
                repo_representation.append(File(path, file_content))
                if verbose: print(Style.RESET_ALL + Fore.GREEN + f"{path} ✔️" + Style.RESET_ALL)
            elif item['type'] == 'dir':
                repo_representation += read_github_repo(github_url, path, ref, substrings_to_ignore, verbose)
        return repo_representation
    else:
        if ref == 'main':
            master_attempt = read_github_repo(github_url, file_path, 'master', substrings_to_ignore, verbose)
            if master_attempt:
                return master_attempt
            else:
                raise Exception(f"GitHub API returned {response.status_code}: {response.text}")
        else:
            raise Exception(f"GitHub API returned {response.status_code}: {response.text}")
        
def get_all_contents(input_source, substrings_to_ignore = [], verbose = False, use_mathpix = False, use_text = False):
    input_type = assume_user_input_type(input_source)
    if input_type == 'github':
        owner_name = input_source.split('github.com/')[-1].split('/')[0]
        repo_name = input_source.split('github.com/')[-1].split('/')[1]
        if verbose: print(Style.RESET_ALL + Fore.YELLOW + f"Reading GitHub repository {owner_name}/{repo_name}..." + Style.RESET_ALL)
        files_with_contents = read_github_repo(input_source, substrings_to_ignore=substrings_to_ignore, verbose=verbose)
    elif input_type == 'url':
        if verbose: print(Style.RESET_ALL + Fore.YELLOW + f"Reading URL {input_source}..." + Style.RESET_ALL)
        page_contents = read_url(input_source)
        files_with_contents = [File(input_source, page_contents)]
    elif input_type == 'file':
        if verbose: print(Style.RESET_ALL + Fore.YELLOW + f"Reading file {input_source}..." + Style.RESET_ALL)
        file_content = get_content_from_file(input_source, use_mathpix=use_mathpix, use_text=use_text, verbose=verbose)
        if isinstance(file_content, list):
            files_with_contents = file_content
        else:
            files_with_contents = [File(input_source, file_content)]
    elif input_type == 'dir':
        all_files = glob.glob(input_source + '/**', recursive=True)
        file_paths = [file for file in all_files if os.path.isfile(file)]
        if verbose: print(Style.RESET_ALL + Fore.YELLOW + f"Reading files in directory {input_source}..." + Style.RESET_ALL)
        # remove files to ignore
        files_with_contents = []
        for file in file_paths:
            if any(file.endswith(extension) for extension in FILES_TO_IGNORE):
                continue
            if any(file.startswith(prefix) for prefix in ('~$', '.')):
                continue
            if any(substring in file for substring in substrings_to_ignore):
                continue
            file_content = get_content_from_file(file, use_mathpix=use_mathpix, use_text=use_text, verbose=verbose)
            if verbose: print(Style.RESET_ALL + Fore.GREEN + f"{file} ✔️" + Style.RESET_ALL)
            files_with_contents.append(File(file, file_content))
    if verbose: print(Style.RESET_ALL + Fore.GREEN + f"Done extracting {input_source} ✔️" + Style.RESET_ALL)
    return files_with_contents

def read_zip(zipfile_path, substrings_to_ignore=[], verbose=False, use_mathpix=False, use_text=False):
    # extract the files
    extracted_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        extracted_files = get_all_contents(temp_dir, substrings_to_ignore=substrings_to_ignore, verbose=verbose, use_mathpix=use_mathpix, use_text=use_text)
    return extracted_files
    