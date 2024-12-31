import re
from typing import Dict, List, Optional, Tuple
from .core import Chunk, calculate_tokens
from sklearn.metrics.pairwise import cosine_similarity


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
            doc_texts.extend(chunk.texts)
            doc_images.extend(chunk.images)
        new_chunks.append(
            Chunk(path=doc_chunks[0].path, texts=doc_texts, images=doc_images)
        )
    return new_chunks


def chunk_by_page(chunks: List[Chunk]) -> List[Chunk]:
    # by-page chunking is default behavior
    return chunks


def chunk_by_section(chunks: List[Chunk]) -> List[Chunk]:
    section_chunks = []
    current_chunk_text = ""
    current_chunk_images = []
    current_chunk_path = None
    # Split the text into sections based on the markdown headers
    for chunk in chunks:
        chunk_text = "\n".join(chunk.texts) if chunk.texts else None
        if chunk.images:
            current_chunk_images.extend(chunk.images)
        if chunk_text:
            lines = chunk_text.split("\n")
            for line in lines:
                if (
                    line.startswith("# ")
                    or line.startswith("## ")
                    or line.startswith("### ")
                ):
                    if current_chunk_text:
                        section_chunks.append(
                            Chunk(
                                path=chunk.path,
                                texts=[current_chunk_text],
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
                texts=[current_chunk_text],
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
        chunk_text = "\n".join(chunk.texts) if chunk.texts else None
        if chunk_text:
            lines = re.split(r"(?<=[.?!])\s+", chunk_text)
            for line in lines:
                sentences.append(line)
                sentence_chunk_map.append(chunk)
                sentence_path_map.append(chunk.path)

    # Compute embeddings
    embeddings = model.encode(sentences)

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
        similarity = cosine_similarity([embedding], [embeddings[current_group[-1]]])[0][
            0
        ]
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
        group_texts = [sentences[i] for i in group]
        group_images = []
        group_path = sentence_path_map[group[0]]
        seen_images = []
        for i in group:
            for image in sentence_chunk_map[i].images:
                if image not in seen_images:
                    group_images.append(image)
                    seen_images.append(image)
        new_chunks.append(
            Chunk(path=group_path, texts=group_texts, images=group_images)
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
    for chunk in chunks:
        if chunk.images:
            current_chunk_images.extend(chunk.images)
        if chunk.texts:
            chunk_text = "\n".join(chunk.texts)
            lines = chunk_text.split("\n")
            for line in lines:
                if any(keyword.lower() in line.lower() for keyword in keywords):
                    if current_chunk_text:
                        new_chunks.append(
                            Chunk(
                                path=chunk.path,
                                texts=[current_chunk_text],
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
                texts=[current_chunk_text],
                images=current_chunk_images,
            )
        )
    return new_chunks
