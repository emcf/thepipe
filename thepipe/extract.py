from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from typing import List, Dict, Union, Optional, Tuple, Callable
from thepipe.core import HOST_URL, THEPIPE_API_KEY, Chunk, calculate_tokens
from thepipe.scraper import scrape_url, scrape_file
from thepipe.chunker import chunk_by_document
import requests
import os
from openai import OpenAI

DEFAULT_EXTRACTION_PROMPT = "Extract structured information from the above document according to the following schema: {schema}. Immediately return valid JSON formatted data. If there is missing data, you may use null, but use your reasoning to always fill in every column as best you can. Always immediately return valid JSON."

def extract_json_from_response(llm_response: str) -> Optional[Dict]:
    def clean_response_text(llm_response: str) -> str:
        return llm_response.encode('utf-8', 'ignore').decode('utf-8')
    
    llm_response = llm_response.strip()
    code_block_pattern = r'^```(?:json)?\s*([\s\S]*?)\s*```$'
    match = re.match(code_block_pattern, llm_response, re.MULTILINE | re.DOTALL)
    if match:
        llm_response = match.group(1).strip()
    llm_response = clean_response_text(llm_response)
    try:
        parsed_json = json.loads(llm_response)
        return parsed_json
    except json.JSONDecodeError:
        json_pattern = r'(\[[\s\S]*\]|\{[\s\S]*\})'
        match = re.search(json_pattern, llm_response)
        if match:
            try:
                parsed_json = json.loads(match.group(1))
                return parsed_json
            except json.JSONDecodeError:
                pass

    objects = re.findall(r'\{[^{}]*\}', llm_response)
    if objects:
        valid_objects = []
        for obj in objects:
            try:
                obj = obj.replace('\\', '').replace('\\"', '"')
                valid_objects.append(json.loads(obj))
            except json.JSONDecodeError:
                continue
        if valid_objects:
            return valid_objects if len(valid_objects) > 1 else valid_objects[0]
    
    print("Failed to extract valid JSON.")
    return None

def extract_from_chunk(chunk: Chunk, chunk_index: int, schema: str, ai_model: str, source: str, multiple_extractions: bool, extraction_prompt: str, host_images: bool) -> Tuple[Dict, int]:
    response_dict = {"chunk_index": chunk_index, "source": source}
    tokens_used = 0
    try:
        openrouter_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        messages = [
            chunk.to_message(host_images=host_images),
            {
                "role": "user",
                "content": extraction_prompt.replace("{schema}", schema)
            },
        ]
        response = openrouter_client.chat.completions.create(
            model=ai_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2
        )
        llm_response = response.choices[0].message.content
        input_tokens = calculate_tokens([chunk])
        output_tokens = calculate_tokens([Chunk(texts=[llm_response])])
        tokens_used += input_tokens + output_tokens
        
        llm_response_dict = extract_json_from_response(llm_response)
        if llm_response_dict:
            if multiple_extractions:
                if isinstance(llm_response_dict, dict) and "extraction" in llm_response_dict:
                    response_dict["extraction"] = llm_response_dict["extraction"]
                else:
                    response_dict["extraction"] = [llm_response_dict]
            else:
                if isinstance(llm_response_dict, dict):
                    response_dict.update(llm_response_dict)
                else:
                    raise ValueError(f"Invalid JSON type in LLM response: {llm_response_dict}")
        else:
            raise ValueError(f"Invalid JSON structure in LLM response: {llm_response}")
        
        if not multiple_extractions:
            schema_keys = json.loads(schema).keys()
            for key in schema_keys:
                if key not in response_dict:
                    response_dict[key] = None
    except Exception as e:
        print(f"Error extracting from chunk {chunk_index}: {e}")
        response_dict["error"] = str(e)
    
    return response_dict, tokens_used

def extract(chunks: List[Chunk], schema: str, ai_model: str, multiple_extractions: bool, extraction_prompt: str, host_images: bool) -> Tuple[List[Dict], int]:
    results = []
    total_tokens_used = 0

    with ThreadPoolExecutor() as executor:
        future_to_chunk = {executor.submit(
            extract_from_chunk,
            chunk=chunk,
            chunk_index=i,
            schema=schema,
            ai_model=ai_model,
            source=chunk.path,
            multiple_extractions=multiple_extractions,
            extraction_prompt=extraction_prompt,
            host_images=host_images
        ): i for i, chunk in enumerate(chunks)}

        for future in as_completed(future_to_chunk):
            try:
                result, tokens_used = future.result()
                results.append(result)
                total_tokens_used += tokens_used
            except Exception as e:
                chunk_index = future_to_chunk[future]
                print(f"Chunk {chunk_index} generated an exception: {e}")
                results.append({
                    "chunk_index": chunk_index,
                    "source": chunks[chunk_index].path,
                    "error": str(e)
                })

    results.sort(key=lambda x: x["chunk_index"])
    return results, total_tokens_used

def extract_from_url(
    url: str, 
    schema: Union[str, Dict], 
    ai_model: str = 'google/gemma-2-9b-it', 
    multiple_extractions: bool = False, 
    extraction_prompt: str = DEFAULT_EXTRACTION_PROMPT, 
    host_images: bool = False, 
    text_only: bool = False, 
    ai_extraction: bool = False, 
    verbose: bool = False,
    chunking_method: Callable[[List[Chunk]], List[Chunk]] = chunk_by_document,
    local: bool = False
) -> List[Dict]: #Tuple[List[Dict], int]:
    if isinstance(schema, dict):
        schema = json.dumps(schema)
    if local:
        chunks = scrape_url(url, text_only=text_only, ai_extraction=ai_extraction, verbose=verbose, local=local)
        chunked_content = chunking_method(chunks)
        return extract(chunked_content, schema, ai_model, multiple_extractions, extraction_prompt, host_images)
    else:
        headers = {
            "Authorization": f"Bearer {THEPIPE_API_KEY}"
        }
        data = {
            'urls': [url],
            'schema': schema,
            'ai_model': ai_model,
            'multiple_extractions': str(multiple_extractions).lower(),
            'extraction_prompt': extraction_prompt,
            'host_images': str(host_images).lower(),
            'text_only': str(text_only).lower(),
            'ai_extraction': str(ai_extraction).lower(),
            'chunking_method': chunking_method.__name__
        }
        response = requests.post(f"{HOST_URL}/extract", headers=headers, data=data)
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        results = []
        total_tokens_used = 0
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                result = data['result']
                if 'error' in result:
                    results.append(result)
                else:
                    extracted_data = {
                        'chunk_index': result['chunk_index'],
                        'source': result['source']
                    }
                    if multiple_extractions:
                        extracted_data['extraction'] = result.get('extraction', [])
                    else:
                        extracted_data.update(result)
                        schema_keys = json.loads(schema).keys()
                        for key in schema_keys:
                            if key not in extracted_data:
                                extracted_data[key] = None
                    results.append(extracted_data)
                total_tokens_used += data['tokens_used']
        
        return results#, total_tokens_used

def extract_from_file(
    file_path: str, 
    schema: Union[str, Dict], 
    ai_model: str = 'google/gemma-2-9b-it', 
    multiple_extractions: bool = False, 
    extraction_prompt: str = DEFAULT_EXTRACTION_PROMPT, 
    host_images: bool = False, 
    text_only: bool = False, 
    ai_extraction: bool = False, 
    verbose: bool = False,
    chunking_method: Callable[[List[Chunk]], List[Chunk]] = chunk_by_document,
    local: bool = False
) -> List[Dict]:#Tuple[List[Dict], int]:
    if isinstance(schema, dict):
        schema = json.dumps(schema)
    if local:
        chunks = scrape_file(file_path, ai_extraction=ai_extraction, text_only=text_only, verbose=verbose)
        chunked_content = chunking_method(chunks)
        return extract(chunked_content, schema, ai_model, multiple_extractions, extraction_prompt, host_images)
    else:
        headers = {
            "Authorization": f"Bearer {THEPIPE_API_KEY}"
        }
        data = {
            'schema': schema,
            'ai_model': ai_model,
            'multiple_extractions': str(multiple_extractions).lower(),
            'extraction_prompt': extraction_prompt,
            'host_images': str(host_images).lower(),
            'text_only': str(text_only).lower(),
            'ai_extraction': str(ai_extraction).lower(),
            'chunking_method': chunking_method.__name__
        }
        files = {'files': (os.path.basename(file_path), open(file_path, 'rb'))}
        
        response = requests.post(f"{HOST_URL}/extract", headers=headers, data=data, files=files)
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
        
        results = []
        total_tokens_used = 0
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                result = data['result']
                if 'error' in result:
                    results.append(result)
                else:
                    extracted_data = {
                        'chunk_index': result['chunk_index'],
                        'source': result['source']
                    }
                    if multiple_extractions:
                        extracted_data['extraction'] = result.get('extraction', [])
                    else:
                        extracted_data.update(result)
                        schema_keys = json.loads(schema).keys()
                        for key in schema_keys:
                            if key not in extracted_data:
                                extracted_data[key] = None
                    results.append(extracted_data)
                total_tokens_used += data['tokens_used']
        
        return results#, total_tokens_used