from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
from typing import Iterable, List, Dict, Union, Optional, Tuple, Callable, cast
from .core import (
    Chunk,
    calculate_tokens,
    DEFAULT_AI_MODEL,
)
from .scraper import scrape_url, scrape_file
from .chunker import (
    chunk_by_page,
    chunk_by_document,
    chunk_by_section,
    chunk_semantic,
    chunk_by_keywords,
    chunk_by_length,
    chunk_agentic,
)
import requests
import os
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

DEFAULT_EXTRACTION_PROMPT = "Extract all the information from the given document according to the following schema: {schema}. Immediately return valid JSON formatted data. If there is missing data, you may use null, but always fill in every column as best you can. Always immediately return valid JSON. You must extract ALL the information available in the entire document."


def extract_json_from_response(llm_response: str) -> Union[Dict, List[Dict], None]:
    def clean_response_text(llm_response: str) -> str:
        return llm_response.encode("utf-8", "ignore").decode("utf-8").strip()

    code_block_pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(code_block_pattern, llm_response, re.MULTILINE | re.DOTALL)
    if match:
        llm_response = match.group(1)
    llm_response = clean_response_text(llm_response)

    try:
        parsed_json = json.loads(llm_response)
        return parsed_json
    except json.JSONDecodeError:
        json_pattern = r"($$[\s\S]*$$|\{[\s\S]*\})"
        match = re.search(json_pattern, llm_response)
        if match:
            try:
                parsed_json = json.loads(match.group(1))
                return parsed_json
            except json.JSONDecodeError:
                pass

    objects = re.findall(r"\{[^{}]*\}", llm_response)
    if objects:
        valid_objects = []
        for obj in objects:
            try:
                obj = obj.replace("\\", "").replace('\\"', '"')
                valid_objects.append(json.loads(obj))
            except json.JSONDecodeError:
                continue
        if valid_objects:
            return valid_objects if len(valid_objects) > 1 else valid_objects[0]
    print(f"[thepipe] Failed to extract valid JSON from LLM response: {llm_response}")
    return None


def extract_from_chunk(
    chunk: Chunk,
    chunk_index: int,
    schema: str,
    ai_model: str,
    source: str,
    multiple_extractions: bool,
    extraction_prompt: str,
    host_images: bool,
    openai_client: OpenAI,
) -> Tuple[Dict, int]:
    response_dict = {"chunk_index": chunk_index, "source": source}
    tokens_used = 0
    try:
        corrected_extraction_prompt = extraction_prompt.replace("{schema}", schema)
        if multiple_extractions:
            corrected_extraction_prompt += """\nIf there are multiple extractions, return each JSON dictionary in a list under the key "extraction". The list should contain each extraction dict (according to the schema) and the entire list should be set to the "extraction" key. Immediately return this extraction JSON object with the "extraction" key mapping to a list containing all the extracted data."""
        else:
            corrected_extraction_prompt += (
                """\nImmediately return the JSON dictionary."""
            )

        messages = [
            chunk.to_message(host_images=host_images),
            {
                "role": "user",
                "content": corrected_extraction_prompt,
            },
        ]

        response = openai_client.chat.completions.create(
            model=ai_model,
            messages=cast(Iterable[ChatCompletionMessageParam], messages),
            response_format={"type": "json_object"},
        )
        llm_response = response.choices[0].message.content
        if not llm_response:
            raise Exception(
                f"Failed to receive a message content from LLM Response: {response}"
            )
        input_tokens = calculate_tokens([chunk])
        output_tokens = calculate_tokens([Chunk(text=llm_response)])
        tokens_used += input_tokens + output_tokens
        try:
            llm_response_dict = extract_json_from_response(llm_response)
            if llm_response_dict:
                if multiple_extractions:
                    if (
                        isinstance(llm_response_dict, dict)
                        and "extraction" in llm_response_dict
                    ):
                        response_dict["extraction"] = llm_response_dict["extraction"]
                    elif isinstance(llm_response_dict, list):
                        response_dict["extraction"] = llm_response_dict
                    else:
                        response_dict["extraction"] = [llm_response_dict]
                else:
                    if isinstance(llm_response_dict, dict):
                        response_dict.update(llm_response_dict)
                    elif isinstance(llm_response_dict, list):
                        response_dict["error"] = (
                            f"Expected a single JSON object but received a list: {llm_response_dict}. Try enabling multiple extractions."
                        )
                    else:
                        response_dict["error"] = (
                            f"Invalid JSON structure in LLM response: {llm_response_dict}"
                        )
            else:
                response_dict["error"] = (
                    f"Failed to extract valid JSON from LLM response: {llm_response}"
                )
        except Exception as e:
            response_dict["error"] = f"Error processing LLM response: {e}"
        if not multiple_extractions:
            schema_keys = (
                json.loads(schema).keys() if isinstance(schema, str) else schema.keys()
            )
            for key in schema_keys:
                if key not in response_dict:
                    response_dict[key] = None
    except Exception as e:
        response_dict = {"chunk_index": chunk_index, "source": source, "error": str(e)}
    return response_dict, tokens_used


def extract(
    chunks: List[Chunk],
    schema: Union[str, Dict],
    ai_model: str = DEFAULT_AI_MODEL,
    multiple_extractions: bool = False,
    extraction_prompt: str = DEFAULT_EXTRACTION_PROMPT,
    host_images: bool = False,
    openai_client: Optional[OpenAI] = None,
) -> Tuple[List[Dict], int]:
    if isinstance(schema, dict):
        schema = json.dumps(schema)

    if openai_client is None:
        raise ValueError(
            "OpenAI client is required for structured extraction. Please provide a valid OpenAI client."
        )

    results = []
    total_tokens_used = 0

    n_threads = (os.cpu_count() or 1) * 2
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_chunk = {
            executor.submit(
                extract_from_chunk,
                chunk=chunk,
                chunk_index=i,
                schema=schema,
                ai_model=ai_model,
                source=chunk.path if chunk.path else "",
                multiple_extractions=multiple_extractions,
                extraction_prompt=extraction_prompt,
                host_images=host_images,
                openai_client=openai_client,
            ): i
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_chunk):
            try:
                result, tokens_used = future.result()
                results.append(result)
                total_tokens_used += tokens_used
            except Exception as e:
                chunk_index = future_to_chunk[future]
                results.append(
                    {
                        "chunk_index": chunk_index,
                        "source": chunks[chunk_index].path,
                        "error": str(e),
                    }
                )

    results.sort(key=lambda x: x["chunk_index"])
    return results, total_tokens_used


def extract_from_url(
    url: str,
    schema: Union[str, Dict],
    ai_model: str = DEFAULT_AI_MODEL,
    multiple_extractions: bool = False,
    extraction_prompt: str = DEFAULT_EXTRACTION_PROMPT,
    host_images: bool = False,
    verbose: bool = False,
    chunking_method: Callable[[List[Chunk]], List[Chunk]] = chunk_by_page,
    openai_client: Optional[OpenAI] = None,
) -> Tuple[List[Dict], int]:
    chunks = scrape_url(
        url,
        verbose=verbose,
        chunking_method=chunking_method,
        openai_client=openai_client,
    )
    extracted_chunks, tokens_used = extract(
        chunks=chunks,
        schema=schema,
        ai_model=ai_model,
        multiple_extractions=multiple_extractions,
        extraction_prompt=extraction_prompt,
        host_images=host_images,
        openai_client=openai_client,
    )
    return extracted_chunks, tokens_used


def extract_from_file(
    file_path: str,
    schema: Union[str, Dict],
    ai_model: str = DEFAULT_AI_MODEL,
    multiple_extractions: bool = False,
    extraction_prompt: str = DEFAULT_EXTRACTION_PROMPT,
    host_images: bool = False,
    verbose: bool = False,
    chunking_method: Callable[[List[Chunk]], List[Chunk]] = chunk_by_page,
    openai_client: Optional[OpenAI] = None,
) -> Tuple[List[Dict], int]:
    chunks = scrape_file(
        file_path,
        verbose=verbose,
        chunking_method=chunking_method,
        openai_client=openai_client,
    )
    extracted_chunks, tokens_used = extract(
        chunks=chunks,
        schema=schema,
        ai_model=ai_model,
        multiple_extractions=multiple_extractions,
        extraction_prompt=extraction_prompt,
        host_images=host_images,
        openai_client=openai_client,
    )
    return extracted_chunks, tokens_used
