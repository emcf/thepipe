import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
from .core import (
    Chunk,
    calculate_tokens,
    merge_metadata,
    DEFAULT_AI_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    llm_parse,
)
import numpy as np
from pydantic import BaseModel
from openai import OpenAI


class Section(BaseModel):
    title: str
    start_line: int
    end_line: int


class SectionList(BaseModel):
    sections: List[Section]


def chunk_by_document(chunks: List[Chunk]) -> List[Chunk]:
    chunks_by_doc = {}
    new_chunks = []
    for chunk in chunks:
        if not chunk.path:
            raise ValueError(
                "Document chunking requires the path attribute to determine the document boundaries"
            )
        if chunk.path not in chunks_by_doc:
            chunks_by_doc[chunk.path] = []
        chunks_by_doc[chunk.path].append(chunk)
    for doc_chunks in chunks_by_doc.values():
        doc_texts = []
        doc_images = []
        for chunk in doc_chunks:
            doc_texts.append(chunk.text)
            doc_images.extend(chunk.images)
        text = "\n".join(doc_texts) if doc_texts else None
        new_chunks.append(
            Chunk(
                path=doc_chunks[0].path,
                text=text,
                images=doc_images,
                metadata=merge_metadata(doc_chunks),
            )
        )
    return new_chunks


def chunk_by_page(chunks: List[Chunk]) -> List[Chunk]:
    # by-page chunking is default behavior
    return chunks


def chunk_by_section(
    chunks: List[Chunk], section_separator: str = "## "
) -> List[Chunk]:
    section_chunks: List[Chunk] = []
    cur_text: Optional[str] = None
    cur_images: List = []
    cur_path: Optional[str] = None
    cur_sources: List[Chunk] = []

    def flush() -> None:
        assert cur_text is not None
        header = cur_text.split("\n", 1)[0]
        metadata = merge_metadata(cur_sources)
        if header.startswith(section_separator):
            metadata["section"] = header[len(section_separator) :].strip()
        section_chunks.append(
            Chunk(
                path=cur_path,
                text=cur_text.rstrip("\n"),
                images=cur_images.copy(),
                metadata=metadata,
            )
        )

    for chunk in chunks:
        # Extract text (always a string or None)
        chunk_text = chunk.text or ""
        # Append images to current section once started
        if cur_text is not None:
            cur_images.extend(chunk.images)
            cur_sources.append(chunk)

        for line in chunk_text.split("\n"):
            if line.startswith(section_separator):
                # New section header found
                if cur_text is not None:
                    flush()
                # Start new section
                cur_text = line + "\n"
                cur_images = []
                cur_path = chunk.path
                cur_sources = [chunk]
            else:
                if cur_text is not None:
                    cur_text += line + "\n"
                else:
                    # Text before any section header: start first section
                    if line.strip():
                        cur_text = line + "\n"
                        cur_path = chunk.path
                        cur_images = []
                        cur_sources = [chunk]

    # Flush last section if present
    if cur_text is not None:
        flush()

    return section_chunks


@lru_cache(maxsize=4)
def _get_embedding_model(model: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised via runtime usage
        raise ImportError(
            "`chunk_semantic` requires the optional dependency `sentence-transformers`. "
            "Install it with `pip install thepipe-api[semantic]` or include the `gpu` extra."
        ) from exc
    return SentenceTransformer(model_name_or_path=model)


def chunk_semantic(
    chunks: List[Chunk],
    model: str = DEFAULT_EMBEDDING_MODEL,
    buffer_size: int = 3,
    similarity_threshold: float = 0.1,
) -> List[Chunk]:
    embedding_model = _get_embedding_model(model)
    # Flatten the chunks into sentences
    sentences = []
    sentence_chunk_map = []
    sentence_path_map = []
    for chunk in chunks:
        chunk_text = chunk.text
        if chunk_text:
            lines = re.split(r"(?<=[.?!])\s+", chunk_text)
            for line in lines:
                sentences.append(line)
                sentence_chunk_map.append(chunk)
                sentence_path_map.append(chunk.path)

    # Compute embeddings
    embeddings = np.array(embedding_model.encode(sentences, convert_to_numpy=True))

    # Create groups based on sentence similarity
    grouped_sentences = []
    current_group = []
    for i, embedding in enumerate(embeddings):
        if not current_group:
            current_group.append(i)
            continue
        # Check similarity with the last sentence in the current group
        # If the similarity is above the threshold, add the sentence to the group
        # Otherwise, start a new group
        a = embedding
        b = embeddings[current_group[-1]]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        similarity = float(np.dot(a, b) / denom) if denom else 0.0
        if similarity >= similarity_threshold:
            current_group.append(i)
        else:
            grouped_sentences.append(current_group)
            current_group = [i]

    if current_group:
        grouped_sentences.append(current_group)

    # Create new chunks based on grouped sentences
    new_chunks = []
    for group in grouped_sentences:
        group_text = "\n".join(sentences[i] for i in group)
        group_images = []
        group_path = sentence_path_map[group[0]]
        seen_images = []
        sources: List[Chunk] = []
        for i in group:
            source = sentence_chunk_map[i]
            if source not in sources:
                sources.append(source)
            for image in source.images:
                if image not in seen_images:
                    group_images.append(image)
                    seen_images.append(image)
        new_chunks.append(
            Chunk(
                path=group_path,
                text=group_text,
                images=group_images,
                metadata=merge_metadata(sources),
            )
        )

    return new_chunks


# starts a new chunk any time a word is found
def chunk_by_keywords(
    chunks: List[Chunk], keywords: List[str] = ["section"]
) -> List[Chunk]:
    new_chunks = []
    current_chunk_text = ""
    current_chunk_images = []
    current_chunk_path = chunks[0].path
    current_sources: List[Chunk] = []
    for chunk in chunks:
        if chunk.images:
            current_chunk_images.extend(chunk.images)
        current_sources.append(chunk)
        lines = chunk.text.split("\n") if chunk.text else []
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in keywords):
                if current_chunk_text:
                    new_chunks.append(
                        Chunk(
                            path=current_chunk_path,
                            text=current_chunk_text,
                            images=current_chunk_images,
                            metadata=merge_metadata(current_sources),
                        )
                    )
                    current_chunk_text = ""
                    current_chunk_images = []
                    current_chunk_path = chunk.path
                    current_sources = [chunk]
            current_chunk_text += line + "\n"
    if current_chunk_text:
        new_chunks.append(
            Chunk(
                path=current_chunk_path,
                text=current_chunk_text,
                images=current_chunk_images,
                metadata=merge_metadata(current_sources),
            )
        )
    return new_chunks


def chunk_by_length(chunks: List[Chunk], max_tokens: int = 10000) -> List[Chunk]:
    new_chunks = []
    for chunk in chunks:
        if calculate_tokens([chunk]) < max_tokens:
            new_chunks.append(chunk)
            continue
        text_halfway_index = len(chunk.text) // 2 if chunk.text else 0
        images_halfway_index = len(chunk.images) // 2 if chunk.images else 0
        if text_halfway_index == 0 and images_halfway_index == 0:
            if not chunk.images:
                # throw error to prevent downstream errors with LLM inference
                raise ValueError(
                    "Chunk cannot be split further. Please increase the max_tokens limit."
                )
            # a lone oversized image: halve its resolution and retry
            image = chunk.images[0]
            halved = Chunk(
                path=chunk.path,
                text=chunk.text,
                images=[image.resize((image.width // 2, image.height // 2))],
                metadata=chunk.metadata,
            )
            new_chunks.extend(chunk_by_length([halved], max_tokens))
            continue
        split_chunks = [
            Chunk(
                path=chunk.path,
                text=chunk.text[:text_halfway_index] if chunk.text else None,
                images=chunk.images[:images_halfway_index] if chunk.images else None,
                metadata=chunk.metadata,
            ),
            Chunk(
                path=chunk.path,
                text=chunk.text[text_halfway_index:] if chunk.text else None,
                images=chunk.images[images_halfway_index:] if chunk.images else None,
                metadata=chunk.metadata,
            ),
        ]
        new_chunks.extend(chunk_by_length(split_chunks, max_tokens))

    return new_chunks


# LLM-based agentic semantic chunking (experimental, openai only)
def chunk_agentic(
    chunks: List[Chunk],
    openai_client: OpenAI,
    model: str = DEFAULT_AI_MODEL,
    max_tokens: int = 50000,
) -> List[Chunk]:
    # 1) Enforce a hard token limit
    chunks = chunk_by_length(chunks, max_tokens=max_tokens)

    # 2) Group by document
    docs: Dict[str, List[Chunk]] = {}
    for c in chunks:
        docs.setdefault(c.path or "__no_path__", []).append(c)

    final_chunks: List[Chunk] = []

    for path, doc_chunks in docs.items():
        # Flatten into numbered lines
        lines: List[str] = []
        line_to_chunk: List[Chunk] = []
        for chunk in doc_chunks:
            texts = (
                chunk.text
                if isinstance(chunk.text, list)
                else ([chunk.text] if chunk.text else [])
            )
            for text in texts:
                for line in text.split("\n"):
                    lines.append(line)
                    line_to_chunk.append(chunk)
        if not lines:
            continue

        numbered = "\n".join(f"{i+1}: {lines[i]}" for i in range(len(lines)))

        # 3) Ask the LLM for structured JSON
        system_prompt = (
            "Divide the following numbered document into semantically cohesive sections. "
            "Return only a single JSON object matching the Pydantic schema `SectionList`, "
            "e.g.:\n"
            "{\n"
            '  "sections": [\n'
            '    {"title": "Introduction", "start_line": 1, "end_line": 5},\n'
            "    ...\n"
            "  ]\n"
            "}\n"
            "Ensure `start_line` and `end_line` are integers, cover every line in order, "
            "and do not overlap or leave gaps."
        )
        user_prompt = numbered

        section_list, _ = llm_parse(
            openai_client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=SectionList,
        )
        sections: List[Section] = section_list.sections

        # build chunks from those sections
        for sec in sections:
            start, end, title = sec.start_line, sec.end_line, sec.title
            # clamp
            start = max(1, min(start, len(lines)))
            end = max(start, min(end, len(lines)))

            sec_lines = lines[start - 1 : end]
            sources: List[Chunk] = []
            seen_imgs = []
            sec_images = []
            for idx in range(start - 1, end):
                source = line_to_chunk[idx]
                if source not in sources:
                    sources.append(source)
                for img in source.images:
                    if img not in seen_imgs:
                        seen_imgs.append(img)
                        sec_images.append(img)

            new_chunk = Chunk(
                path=path if path != "__no_path__" else None,
                text="\n".join(sec_lines),
                images=sec_images,
                metadata={**merge_metadata(sources), "section": title},
            )

            # break further by length if needed
            final_chunks.extend(chunk_by_length([new_chunk], max_tokens=max_tokens))

    return final_chunks
