import unittest
import os
import sys
from typing import List

sys.path.append("..")
import thepipe.chunker as chunker
from thepipe.core import Chunk, calculate_tokens


class test_chunker(unittest.TestCase):
    def setUp(self):
        self.files_directory = os.path.join(os.path.dirname(__file__), "files")
        self.example_markdown_path = os.path.join(self.files_directory, "example.md")
        self.max_tokens_per_chunk = (
            10  # Define an arbitrary max tokens per chunk for testing
        )

    def read_markdown_file(self, file_path: str) -> List[Chunk]:
        with open(file_path, "r") as f:
            text = f.read()
        return [Chunk(path=file_path, text=text)]

    def test_chunk_by_length(self):
        chunks = self.read_markdown_file(self.example_markdown_path)
        chunked_length = chunker.chunk_by_length(
            chunks, max_tokens=self.max_tokens_per_chunk
        )
        # Verify the output
        self.assertIsInstance(chunked_length, list)
        self.assertGreater(len(chunked_length), 0)
        for chunk in chunked_length:
            self.assertIsInstance(chunk, Chunk)
            print(f"Chunk text: {chunk.text}")
            self.assertTrue(any(chunk.text or chunk.images))
            # assert length of text is less than max_tokens
            if chunk.text or chunk.images:
                self.assertLessEqual(
                    calculate_tokens([chunk]), self.max_tokens_per_chunk
                )

    def test_chunk_semantic(self):
        test_sentence = "Computational astrophysics. Numerical astronomy. Bananas."
        chunks = [Chunk(text=test_sentence)]
        chunked_semantic = chunker.chunk_semantic(
            chunks,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            buffer_size=2,
            similarity_threshold=0.5,
        )
        # verify the output
        self.assertIsInstance(chunked_semantic, list)
        self.assertGreater(len(chunked_semantic), 0)
        # verify it split into ['Computational astrophysics.\nNumerical astronomy.', 'Bananas.']
        self.assertEqual(len(chunked_semantic), 2)
        self.assertEqual(
            chunked_semantic[0].text,
            "Computational astrophysics.\nNumerical astronomy.",
        )
        self.assertEqual(
            chunked_semantic[1].text,
            "Bananas.",
        )

    def test_chunk_by_page(self):
        chunks = self.read_markdown_file(self.example_markdown_path)
        chunked_pages = chunker.chunk_by_page(chunks)
        # Verify the output
        self.assertIsInstance(chunked_pages, list)
        self.assertGreater(len(chunked_pages), 0)
        for chunk in chunked_pages:
            self.assertIsInstance(chunk, Chunk)
            self.assertTrue(any(chunk.text or chunk.images))

    def test_chunk_by_section(self):
        chunks = self.read_markdown_file(self.example_markdown_path)
        chunked_sections = chunker.chunk_by_section(chunks)
        self.assertIsInstance(chunked_sections, list)
        self.assertGreater(len(chunked_sections), 0)
        # Verify the output contains chunks with text or images
        for chunk in chunked_sections:
            self.assertIsInstance(chunk, Chunk)
            self.assertTrue(any(chunk.text or chunk.images))

    def test_chunk_by_document(self):
        chunks = self.read_markdown_file(self.example_markdown_path)
        chunked_documents = chunker.chunk_by_document(chunks)
        self.assertIsInstance(chunked_documents, list)
        self.assertEqual(len(chunked_documents), 1)
        # Verify the output contains chunks with text or images
        chunk = chunked_documents[0]
        self.assertIsInstance(chunk, Chunk)
        self.assertTrue(any(chunk.text or chunk.images))


if __name__ == "__main__":
    unittest.main()
