import re
from typing import Dict, List, Optional, Tuple
from .core import Chunk, calculate_tokens
from sklearn.metrics.pairwise import cosine_similarity

def chunk_by_page(chunks: List[Chunk]) -> List[Chunk]:
    # by-page chunking is default behavior
    return chunks

def chunk_by_section(chunks: List[Chunk]) -> List[Chunk]:
    section_chunks = []
    current_chunk_text = ""
    current_chunk_images = []
    # Split the text into sections based on the markdown headers
    for chunk in chunks:
        chunk_text = '\n'.join(chunk.texts)
        chunk_images = chunk.images
        lines = chunk_text.split('\n')
        for line in lines:
            if line.startswith('# ') or line.startswith('## ') or line.startswith('### '):
                if current_chunk_text:
                    section_chunks.append(Chunk(texts=[current_chunk_text], images=current_chunk_images))
                    current_chunk_text = ""
                    current_chunk_images = chunk_images
            current_chunk_text += line + '\n'
        current_chunk_images.extend(chunk_images)
    if current_chunk_text:
        section_chunks.append(Chunk(texts=[current_chunk_text], images=current_chunk_images))
    return section_chunks

def chunk_semantic(chunks: List[Chunk], model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', buffer_size: int = 2, similarity_threshold: float = 0.5) -> List[Chunk]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    # Flatten the chunks into sentences
    sentences = []
    sentence_chunk_map = []
    for chunk in chunks:
        chunk_text = '\n'.join(chunk.texts)
        lines = re.split(r'(?<=[.?!])\s+', chunk_text)
        for line in lines:
            sentences.append(line)
            sentence_chunk_map.append(chunk)
    
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
        similarity = cosine_similarity([embedding], [embeddings[current_group[-1]]])[0][0]
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
        for i in group:
            group_images.extend(sentence_chunk_map[i].images)
        new_chunks.append(Chunk(texts=group_texts, images=group_images))
    
    return new_chunks