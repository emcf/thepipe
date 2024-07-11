from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from typing import List, Dict, Optional, Tuple
from .core import Chunk, calculate_tokens
import os

DEFAULT_EXTRACTION_PROMPT = "Extract structured information from the above document according to the following schema: {schema}. Immediately return valid JSON formatted data. If there is missing data, you may use null, but use your reasoning to always fill in every column as best you can. Always immediately return valid JSON."

def extract_json_from_response(llm_response: str) -> Optional[Dict]:
    def clean_response_text(llm_response: str) -> str:
        return llm_response.encode('utf-8', 'ignore').decode('utf-8')
    
    llm_response = llm_response.strip()
    # Extract JSON from code block
    code_block_pattern = r'^```(?:json)?\s*([\s\S]*?)\s*```$'
    match = re.match(code_block_pattern, llm_response, re.MULTILINE | re.DOTALL)
    if match:
        llm_response = match.group(1).strip()
    llm_response = clean_response_text(llm_response)
    try:
        parsed_json = json.loads(llm_response)
        return parsed_json
    except json.JSONDecodeError as e:
        # Try to extract JSON by matching syntax
        json_pattern = r'(\[[\s\S]*\]|\{[\s\S]*\})'
        match = re.search(json_pattern, llm_response)
        if match:
            try:
                parsed_json = json.loads(match.group(1))
                return parsed_json
            except json.JSONDecodeError as e:
                pass

    # Additional fallback: Try to extract individual JSON objects
    objects = re.findall(r'\{[^{}]*\}', llm_response)
    if objects:
        valid_objects = []
        for obj in objects:
            try:
                # Replace escaped backslashes and quotes
                obj = obj.replace('\\', '').replace('\\"', '"')
                valid_objects.append(json.loads(obj))
            except json.JSONDecodeError:
                continue
        if valid_objects:
            return valid_objects if len(valid_objects) > 1 else valid_objects[0]
    
    print("Failed to extract valid JSON.")
    return None

def extract_from_chunk(chunk: Chunk, chunk_index: int, schema: str, ai_model: str, source: str, multiple_extractions: bool, extraction_prompt: str, host_images: bool) -> Tuple[Dict, int]:
    from openai import OpenAI # only import if needed
    
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

def extract(chunks: List[Chunk], schema: str, ai_model: str = 'google/gemma-2-9b-it', multiple_extractions: bool = False, extraction_prompt: str = DEFAULT_EXTRACTION_PROMPT, host_images: bool = False) -> Tuple[List[Dict], int]:
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

    # Sort results by chunk_index to maintain original order
    results.sort(key=lambda x: x["chunk_index"])

    return results, total_tokens_used