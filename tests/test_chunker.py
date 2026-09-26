import unittest
import os
import sys
from typing import List, cast

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional in tests
    OpenAI = None  # type: ignore[assignment]

try:
    import sentence_transformers  # noqa: F401  # pragma: no cover - optional import

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover - optional import
    HAS_SENTENCE_TRANSFORMERS = False

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

    def test_chunk_by_keywords(self):
        text = "Intro line\nfoo starts here\nmiddle\nbar next\nend"
        chunk = Chunk(path="doc.md", text=text)

        result = chunker.chunk_by_keywords([chunk], keywords=["foo", "bar"])
        # There are 3 segments: before foo, between foo and bar, after bar
        self.assertEqual(len(result), 3)

        # 1st chunk: only the intro
        t0 = cast(str, result[0].text)
        self.assertIn("Intro line", t0)
        self.assertNotIn("foo", t0.lower())

        # 2nd chunk: starts with foo, includes 'middle'
        t1 = cast(str, result[1].text).lower()
        self.assertIn("foo starts here", t1)
        self.assertIn("middle", t1)

        # 3rd chunk: starts with bar, includes 'end'
        t2 = cast(str, result[2].text).lower()
        self.assertIn("bar next", t2)
        self.assertIn("end", t2)

    @unittest.skipIf(OpenAI is None or not os.getenv("OPENAI_API_KEY"), "OpenAI API key required")
    def test_chunk_agentic(self):
        openai_client = OpenAI()
        chunks = self.read_markdown_file(self.example_markdown_path)
        chunked_agentic = chunker.chunk_agentic(chunks, openai_client=openai_client)
        # Verify the output
        self.assertIsInstance(chunked_agentic, list)
        self.assertGreater(len(chunked_agentic), 0)
        # verify there are 3 chunks corresponding to the sections in the example markdown
        self.assertEqual(len(chunked_agentic), 3)
        # Verify the output contains chunks with text or images
        for chunk in chunked_agentic:
            self.assertIsInstance(chunk, Chunk)
            self.assertTrue(chunk.text or chunk.images)

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
            # Verify that the chunk text or images are not none
            self.assertTrue(chunk.text or chunk.images)
            # assert length of text is less than max_tokens
            if chunk.text or chunk.images:
                self.assertLessEqual(
                    calculate_tokens([chunk]), self.max_tokens_per_chunk
                )

    @unittest.skipUnless(HAS_SENTENCE_TRANSFORMERS, "sentence-transformers extra is not installed")
    def test_chunk_semantic(self):
        test_sentence = "Computational astrophysics. Numerical astronomy. Bananas."
        chunks = [Chunk(text=test_sentence)]
        chunked_semantic = chunker.chunk_semantic(
            chunks,
            model="sentence-transformers/all-MiniLM-L6-v2",
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
            self.assertTrue(chunk.text or chunk.images)

    def test_chunk_by_section(self):
        chunks = self.read_markdown_file(self.example_markdown_path)
        chunked_sections = chunker.chunk_by_section(chunks)
        self.assertIsInstance(chunked_sections, list)
        self.assertEqual(len(chunked_sections), 3)
        # Verify the output contains chunks with text or images
        for chunk in chunked_sections:
            self.assertIsInstance(chunk, Chunk)
            self.assertTrue(chunk.text or chunk.images)

    def test_chunkers_carry_provenance(self):
        pages = [
            Chunk(path="d.pdf", text="## A\none", metadata={"page": 1}),
            Chunk(path="d.pdf", text="two\n## B\nthree", metadata={"page": 2}),
        ]
        by_doc = chunker.chunk_by_document(pages)
        self.assertEqual(by_doc[0].metadata, {"page": [1, 2]})

        sections = chunker.chunk_by_section(pages)
        self.assertEqual(sections[0].metadata, {"page": [1, 2], "section": "A"})
        self.assertEqual(sections[1].metadata, {"page": 2, "section": "B"})

        halves = chunker.chunk_by_length([Chunk(text="x" * 400, metadata={"page": 7})], max_tokens=50)
        self.assertGreater(len(halves), 1)
        self.assertTrue(all(h.metadata == {"page": 7} for h in halves))

    def test_chunk_by_section_first_line_and_custom_separator(self):
        # Default separator, with first line as a header
        text1 = "## Alpha\nContent A\n## Beta\nContent B"
        chunk1 = Chunk(text=text1)
        out1 = chunker.chunk_by_section([chunk1])
        self.assertEqual(len(out1), 2)

        # cast .text to str so Pylance stops complaining
        t0 = cast(str, out1[0].text)
        self.assertIn("Alpha", t0)

        t1 = cast(str, out1[1].text)
        self.assertIn("Beta", t1)

        self.assertEqual(out1[0].metadata["section"], "Alpha")
        self.assertEqual(out1[1].metadata["section"], "Beta")

        # Custom separator "### "
        text2 = "### One\nX\n### Two\nY"
        chunk2 = Chunk(text=text2)
        out2 = chunker.chunk_by_section([chunk2], section_separator="### ")
        self.assertEqual(len(out2), 2)

        o0 = cast(str, out2[0].text)
        self.assertIn("One", o0)

        o1 = cast(str, out2[1].text)
        self.assertIn("Two", o1)

    def test_chunk_by_document(self):
        chunks = self.read_markdown_file(self.example_markdown_path)
        chunked_documents = chunker.chunk_by_document(chunks)
        self.assertIsInstance(chunked_documents, list)
        self.assertEqual(len(chunked_documents), 1)
        # Verify the output contains chunks with text or images
        chunk = chunked_documents[0]
        self.assertIsInstance(chunk, Chunk)
        self.assertTrue(chunk.text or chunk.images)


if __name__ == "__main__":
    unittest.main()
