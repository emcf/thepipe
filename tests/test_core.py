import argparse
import base64
import unittest
import os
import sys

sys.path.append("..")
import thepipe.core as core
import thepipe.scraper as scraper
from PIL import Image
from io import BytesIO


class test_core(unittest.TestCase):
    def setUp(self):
        self.files_directory = os.path.join(os.path.dirname(__file__), "files")
        self.outputs_directory = "outputs"

    def tearDown(self):
        # clean up outputs
        if os.path.exists(self.outputs_directory):
            for file in os.listdir(self.outputs_directory):
                os.remove(os.path.join(self.outputs_directory, file))
            os.rmdir(self.outputs_directory)

    def test_chunk_to_llamaindex(self):
        chunk = core.Chunk(path="example.txt", text="Hello, World!")
        llama_index = chunk.to_llamaindex()
        self.assertEqual(type(llama_index), list)
        self.assertEqual(len(llama_index), 1)

    def test_chunks_to_messages(self):
        chunks = scraper.scrape_file(filepath=self.files_directory + "/example.md")
        messages = core.chunks_to_messages(chunks)
        self.assertEqual(type(messages), list)
        for message in messages:
            self.assertEqual(type(message), dict)
            self.assertIn("role", message)
            self.assertIn("content", message)
        # test chunks_to_messages with path included
        messages = core.chunks_to_messages(chunks, include_paths=True)
        for message in messages:
            self.assertIn("example.md", message["content"][0]["text"])

    def test_save_outputs(self):
        chunks = scraper.scrape_plaintext(
            file_path=self.files_directory + "/example.txt"
        )
        core.save_outputs(chunks)
        self.assertTrue(os.path.exists(self.outputs_directory + "/prompt.txt"))
        with open(
            self.outputs_directory + "/prompt.txt", "r", encoding="utf-8"
        ) as file:
            text = file.read()
        self.assertIn("Hello, World!", text)
        # verify with images
        chunks = scraper.scrape_file(filepath=self.files_directory + "/example.jpg")
        core.save_outputs(chunks)
        self.assertTrue(any(".jpg" in f for f in os.listdir(self.outputs_directory)))

    def test_chunk_json(self):
        example_image_path = os.path.join(self.files_directory, "example.jpg")
        image = Image.open(example_image_path)
        chunk = core.Chunk(path="example.md", text="Hello, World!")
        # convert to json
        chunk_json = chunk.to_json()
        # verify it is a dictionary with the expected items
        self.assertEqual(type(chunk_json), dict)
        self.assertIn("text", chunk_json)
        self.assertIn("path", chunk_json)
        # convert back
        chunk = core.Chunk.from_json(chunk_json)
        # verify it is the correct Chunk object
        self.assertEqual(type(chunk), core.Chunk)
        self.assertEqual(chunk.path, "example.md")
        self.assertEqual(chunk.text, "Hello, World!")

    def test_calculate_tokens(self):
        text = "Hello, World!"
        tokens = core.calculate_tokens([core.Chunk(text=text)])
        self.assertAlmostEqual(tokens, 3.25, places=0)

    def test_calculate_image_tokens(self):
        image = Image.open(os.path.join(self.files_directory, "example.jpg"))
        image.load()  # needed to close the file
        tokens = core.calculate_image_tokens(image, detail="auto")
        self.assertAlmostEqual(tokens, 85, places=0)
        tokens = core.calculate_image_tokens(image, detail="low")
        self.assertAlmostEqual(tokens, 85, places=0)
        tokens = core.calculate_image_tokens(image, detail="high")
        self.assertAlmostEqual(tokens, 595, places=0)

    def test_make_image_url(self):
        image = Image.open(os.path.join(self.files_directory, "example.jpg"))
        image.load()  # needed to close the file
        url = core.make_image_url(image, host_images=False)
        # verify it is in the correct format
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        # verify it decodes correctly
        remove_prefix = url.replace("data:image/jpeg;base64,", "")
        image_data = base64.b64decode(remove_prefix)
        image = Image.open(BytesIO(image_data))
        self.assertEqual(image.format, "JPEG")
        # verify it hosts the image correctly
        url = core.make_image_url(image, host_images=True)
        self.assertTrue(url.startswith(core.HOST_URL))
