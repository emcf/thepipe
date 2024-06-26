import argparse
import base64
import unittest
import os
import sys
sys.path.append('..')
from thepipe_api import core
from thepipe_api import scraper
from thepipe_api import thepipe
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
    
    def test_image_to_base64(self):
        image = Image.open(os.path.join(self.files_directory, 'example.jpg'))
        image.load() # needed to close the file
        base64_string = core.image_to_base64(image)
        self.assertEqual(type(base64_string), str)
        # converting back should be the same
        image_data = base64.b64decode(base64_string)
        decoded_image = Image.open(BytesIO(image_data))
        self.assertEqual(image.size, decoded_image.size)

    def test_to_messages(self):
        chunks = scraper.scrape_file(source=self.files_directory+"/example.md")
        messages = core.to_messages(chunks)
        self.assertEqual(type(messages), list)
        for message in messages:
            self.assertEqual(type(message), dict)
            self.assertIn('role', message)
            self.assertIn('content', message)

    def test_save_outputs(self):
        chunks = scraper.scrape_plaintext(file_path=self.files_directory+"/example.txt")
        thepipe.save_outputs(chunks)
        self.assertTrue(os.path.exists(self.outputs_directory+"/prompt.txt"))
        with open(self.outputs_directory+"/prompt.txt", 'r', encoding='utf-8') as file:
            text = file.read()
        self.assertIn('Hello, World!', text)
        # verify with images
        chunks = scraper.scrape_file(source=self.files_directory+"/example.jpg")
        thepipe.save_outputs(chunks)
        self.assertTrue(any('.jpg' in f for f in os.listdir(self.outputs_directory)))

    def test_parse_arguments(self):
        args = thepipe.parse_arguments()
        self.assertEqual(type(args), argparse.Namespace)
        self.assertIn('source', vars(args))
        self.assertIn('include_regex', vars(args))
        self.assertIn('ignore_regex', vars(args))