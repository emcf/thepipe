import re
from typing import Dict, List, Optional, Tuple, Union
from .core import Chunk, calculate_tokens
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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
    chunks: List[Chunk], section_separator: str = "\n## "
) -> List[Chunk]:
    section_chunks = []
    current_chunk_text = ""
    current_chunk_images = []
    current_chunk_path = None
    # Split the text into sections based on the markdown headers
    for chunk in chunks:
        chunk_text = "\n".join(chunk.text) if chunk.text else None
        if chunk.images:
            current_chunk_images.extend(chunk.images)
        if chunk_text:
            lines = chunk_text.split("\n")
            for line in lines:
                if line.startswith(section_separator):
                    if current_chunk_text:
                        section_chunks.append(
                            Chunk(
                                path=chunk.path,
                                text=current_chunk_text,
                                images=current_chunk_images,
                            )
                        )
                        current_chunk_text = ""
                        current_chunk_images = []
                        if chunk.path:
                            current_chunk_path = chunk.path
                current_chunk_text += line + "\n"
    if current_chunk_text:
        section_chunks.append(
            Chunk(
                path=current_chunk_path,
                text=current_chunk_text,
                images=current_chunk_images,
            )
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
