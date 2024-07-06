import argparse
import base64
import unittest
import os
import sys
sys.path.append('..')
from thepipe import core
from thepipe import scraper
from PIL import Image
from io import BytesIO

class test_core(unittest.TestCase):
    def setUp(self):
        self.files_directory = os.path.join(os.path.dirname(__file__), 'files')
        self.outputs_directory = 'outputs'

    def tearDown(self):
        # clean up outputs
        if os.path.exists(self.outputs_directory):
            for file in os.listdir(self.outputs_directory):
                os.remove(os.path.join(self.outputs_directory, file))
            os.rmdir(self.outputs_directory)

    def test_chunk_to_llamaindex(self):
        chunk = core.Chunk(texts=["Hello, World!"])
        llama_index = chunk.to_llamaindex()
        self.assertEqual(type(llama_index), list)
        self.assertEqual(len(llama_index), 1)
    
    def test_chunks_to_messages(self):
        chunks = scraper.scrape_file(source=self.files_directory+"/example.md", local=True)
        messages = core.chunks_to_messages(chunks)
        self.assertEqual(type(messages), list)
        for message in messages:
            self.assertEqual(type(message), dict)
            self.assertIn('role', message)
            self.assertIn('content', message)

    def test_save_outputs(self):
        chunks = scraper.scrape_plaintext(file_path=self.files_directory+"/example.txt")
        core.save_outputs(chunks)
        self.assertTrue(os.path.exists(self.outputs_directory+"/prompt.txt"))
        with open(self.outputs_directory+"/prompt.txt", 'r', encoding='utf-8') as file:
            text = file.read()
        self.assertIn('Hello, World!', text)
        # verify with images
        chunks = scraper.scrape_file(source=self.files_directory+"/example.jpg", local=True)
        core.save_outputs(chunks)
        self.assertTrue(any('.jpg' in f for f in os.listdir(self.outputs_directory)))

    def test_chunk_json(self):
        chunk = core.Chunk(path="example.md", texts=["Hello, World!"])
        # convert to json
        chunk_json = chunk.to_json()
        # verify it is a dictionary with the expected items
        self.assertEqual(type(chunk_json), dict)
        self.assertIn('texts', chunk_json)
        self.assertIn('path', chunk_json)
        # convert back
        chunk = core.Chunk.from_json(chunk_json)
        # verify it is the correct Chunk object
        self.assertEqual(type(chunk), core.Chunk)
        self.assertEqual(chunk.path, "example.md")
        self.assertEqual(chunk.texts, ["Hello, World!"])

    def test_parse_arguments(self):
        args = core.parse_arguments()
        self.assertEqual(type(args), argparse.Namespace)
        self.assertIn('source', vars(args))
        self.assertIn('include_regex', vars(args))

    def test_calculate_tokens(self):
        text = "Hello, World!"
        tokens = core.calculate_tokens([core.Chunk(texts=[text])])
        self.assertAlmostEqual(tokens, 3.25, places=0)

    def test_calculate_image_tokens(self):
        image = Image.open(os.path.join(self.files_directory, 'example.jpg'))
        image.load() # needed to close the file
        tokens = core.calculate_image_tokens(image)
        self.assertAlmostEqual(tokens, 85, places=0)