import re
from typing import Dict, List, Optional, Tuple, Union
from .core import Chunk, calculate_tokens, LLM_SERVER_BASE_URL, LLM_SERVER_API_KEY
from sklearn.metrics.pairwise import cosine_similarity
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
            doc_texts.extend(chunk.text)
            doc_images.extend(chunk.images)
        text = "\n".join(doc_texts) if doc_texts else None
        new_chunks.append(Chunk(path=doc_chunks[0].path, text=text, images=doc_images))
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

    for chunk in chunks:
        # Extract text (always a string or None)
        chunk_text = chunk.text or ""
        # Append images to current section once started
        if cur_text is not None and getattr(chunk, "images", None):
            cur_images.extend(chunk.images)

        for line in chunk_text.split("\n"):
            if line.startswith(section_separator):
                # New section header found
                if cur_text is not None:
                    # Flush previous section
                    section_chunks.append(
                        Chunk(
                            path=cur_path,
                            text=cur_text.rstrip("\n"),
                            images=cur_images.copy(),
                        )
                    )
                # Start new section
                cur_text = line + "\n"
                cur_images = []
                cur_path = chunk.path
            else:
                if cur_text is not None:
                    cur_text += line + "\n"
                else:
                    # Text before any section header: start first section
                    if line.strip():
                        cur_text = line + "\n"
                        cur_path = chunk.path
                        cur_images = []

    # Flush last section if present
    if cur_text is not None:
        section_chunks.append(
            Chunk(path=cur_path, text=cur_text.rstrip("\n"), images=cur_images.copy())
        )

    return section_chunks


def chunk_semantic(
    chunks: List[Chunk],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    buffer_size: int = 3,
    similarity_threshold: float = 0.1,
) -> List[Chunk]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
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
    embeddings = np.array(model.encode(sentences, convert_to_numpy=True))

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
        a = embedding.reshape(1, -1)
        b = embeddings[current_group[-1]].reshape(1, -1)
        similarity = cosine_similarity(a, b)[0, 0]
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
        for i in group:
            for image in sentence_chunk_map[i].images:
                if image not in seen_images:
                    group_images.append(image)
                    seen_images.append(image)
        new_chunks.append(Chunk(path=group_path, text=group_text, images=group_images))

    return new_chunks


# starts a new chunk any time a word is found
def chunk_by_keywords(
    chunks: List[Chunk], keywords: List[str] = ["section"]
) -> List[Chunk]:
    new_chunks = []
    current_chunk_text = ""
    current_chunk_images = []
    current_chunk_path = chunks[0].path
    for chunk in chunks:
        if chunk.images:
            current_chunk_images.extend(chunk.images)
        lines = chunk.text.split("\n") if chunk.text else []
        for line in lines:
            if any(keyword.lower() in line.lower() for keyword in keywords):
                if current_chunk_text:
                    new_chunks.append(
                        Chunk(
                            path=chunk.path,
                            text=current_chunk_text,
                            images=current_chunk_images,
                        )
                    )
                    current_chunk_text = ""
                    current_chunk_images = []
                    current_chunk_path = chunk.path
            current_chunk_text += line + "\n"
    if current_chunk_text:
        new_chunks.append(
            Chunk(
                path=current_chunk_path,
                text=current_chunk_text,
                images=current_chunk_images,
            )
        )
    return new_chunks


def chunk_by_length(chunks: List[Chunk], max_tokens: int = 10000) -> List[Chunk]:
    new_chunks = []
    for chunk in chunks:
        total_tokens = calculate_tokens([chunk])
        if total_tokens < max_tokens:
            new_chunks.append(chunk)
            continue
        text_halfway_index = len(chunk.text) // 2 if chunk.text else 0
        images_halfway_index = len(chunk.images) // 2 if chunk.images else 0
        if text_halfway_index == 0 and images_halfway_index == 0:
            if chunk.images:
                # can't be split further: try to reduce the size of the images
                # by resizing each image to half its size
                new_images = []
                for image in chunk.images:
                    new_width = image.width // 2
                    new_height = image.height // 2
                    resized_image = image.resize((new_width, new_height))
                    new_images.append(resized_image)
            else:
                # throw error to prevent downstream errors with LLM inference
                raise ValueError(
                    "Chunk cannot be split further. Please increase the max_tokens limit."
                )

            return new_chunks
        split_chunks = [
            Chunk(
                path=chunk.path,
                text=chunk.text[:text_halfway_index] if chunk.text else None,
                images=chunk.images[:images_halfway_index] if chunk.images else None,
            ),
            Chunk(
                path=chunk.path,
                text=chunk.text[text_halfway_index:] if chunk.text else None,
                images=chunk.images[images_halfway_index:] if chunk.images else None,
            ),
        ]
        # recursive call
        new_chunks = chunk_by_length(split_chunks, max_tokens)

    return new_chunks


# LLM-based agentic semantic chunking (experimental, openai only)
def chunk_agentic(
    chunks: List[Chunk],
    max_tokens: int = 50000,
) -> List[Chunk]:
    openai_client = OpenAI(
        base_url=LLM_SERVER_BASE_URL,
        api_key=LLM_SERVER_API_KEY,
    )

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

        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=SectionList,
        )

        if not completion.choices[0].message.parsed:
            raise ValueError(
                "LLM did not return a valid response during agentic chunking."
            )

        sections: List[Section] = completion.choices[0].message.parsed.sections

        # build chunks from those sections
        for sec in sections:
            start, end, title = sec.start_line, sec.end_line, sec.title
            # clamp
            start = max(1, min(start, len(lines)))
            end = max(start, min(end, len(lines)))

            sec_lines = lines[start - 1 : end]
            seen_imgs = set()
            sec_images = []
            for idx in range(start - 1, end):
                for img in getattr(line_to_chunk[idx], "images", []):
                    if img not in seen_imgs:
                        seen_imgs.add(img)
                        sec_images.append(img)

            # prepend header
            text_block = "\n".join(sec_lines)
            new_chunk = Chunk(
                path=path if path != "__no_path__" else None,
                text=text_block,
                images=sec_images,
            )

            # break further by length if needed
            final_chunks.extend(chunk_by_length([new_chunk], max_tokens=max_tokens))

    return final_chunks
