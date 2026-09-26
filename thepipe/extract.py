from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple, Type

from openai import OpenAI
from pydantic import BaseModel, create_model

from .chunker import chunk_by_page
from .core import Chunk, DEFAULT_AI_MODEL, calculate_tokens, llm_parse
from .scraper import PDF_PAGE_WORKERS, scrape_file

DEFAULT_EXTRACTION_PROMPT = (
    "Extract all the information from the given document according to the provided schema. "
    "If a value is missing from the document, use null, but always fill in every field as best you can. "
    "You must extract ALL the information available in the entire document."
)


def extract_from_chunk(
    chunk: Chunk,
    chunk_index: int,
    schema: Type[BaseModel],
    model: str,
    multiple_extractions: bool,
    extraction_prompt: str,
    host_images: bool,
    openai_client: OpenAI,
) -> Tuple[Dict, int]:
    result: Dict = {"chunk_index": chunk_index, "source": chunk.path or ""}
    try:
        parsed_model, raw_content = llm_parse(
            openai_client,
            model=model,
            messages=[
                {"role": "system", "content": extraction_prompt},
                chunk.to_message(host_images=host_images),
            ],
            response_format=schema,
        )
        parsed = parsed_model.model_dump()
        if multiple_extractions:
            result["extraction"] = parsed["extraction"]
        else:
            result.update(parsed)
        tokens = calculate_tokens([chunk, Chunk(text=raw_content)])
        return result, tokens
    except Exception as e:
        result["error"] = str(e)
        return result, 0


def extract(
    chunks: List[Chunk],
    schema: Type[BaseModel],
    model: str = DEFAULT_AI_MODEL,
    multiple_extractions: bool = False,
    extraction_prompt: str = DEFAULT_EXTRACTION_PROMPT,
    host_images: bool = False,
    openai_client: Optional[OpenAI] = None,
    max_workers: Optional[int] = None,
) -> Tuple[List[Dict], int]:
    """Extract structured data from each chunk into the given pydantic ``schema``."""
    if openai_client is None:
        raise ValueError("An OpenAI client is required for structured extraction.")
    if multiple_extractions:
        schema = create_model("Extractions", extraction=(List[schema], ...))  # type: ignore[valid-type]

    workers = max(1, max_workers or PDF_PAGE_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                extract_from_chunk,
                chunk,
                i,
                schema,
                model,
                multiple_extractions,
                extraction_prompt,
                host_images,
                openai_client,
            )
            for i, chunk in enumerate(chunks)
        ]
        outcomes = [f.result() for f in as_completed(futures)]

    results = sorted((r for r, _ in outcomes), key=lambda r: r["chunk_index"])
    return results, sum(t for _, t in outcomes)


def extract_from_file(
    file_path: str,
    schema: Type[BaseModel],
    model: str = DEFAULT_AI_MODEL,
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
    return extract(
        chunks,
        schema,
        model=model,
        multiple_extractions=multiple_extractions,
        extraction_prompt=extraction_prompt,
        host_images=host_images,
        openai_client=openai_client,
    )
