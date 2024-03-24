import base64
import unittest
import os
import sys
sys.path.append('..')
import thepipe
from PIL import Image
from io import BytesIO

class test_thepipe(unittest.TestCase):
    def setUp(self):
        self.files_directory = os.path.join(os.path.dirname(__file__), 'files')

    def tearDown(self):
        # clean up outputs
        if os.path.exists('outputs/prompt.txt'):
            os.remove('outputs/prompt.txt')
            os.rmdir('outputs')

    def test_image_to_base64(self):
        image = thepipe.Image.open(os.path.join(self.files_directory, 'example.jpg'))
        image.load() # needed to close the file
        base64_string = thepipe.image_to_base64(image)
        self.assertEqual(type(base64_string), str)
        # converting back should be the same
        image_data = base64.b64decode(base64_string)
        decoded_image = Image.open(BytesIO(image_data))
        self.assertEqual(image.size, decoded_image.size)

    def test_create_messages_from_chunks(self):
        chunks = thepipe.extract.extract_from_source(source_string=self.files_directory)
        messages = thepipe.create_messages_from_chunks(chunks)
        self.assertEqual(type(messages), list)
        for message in messages:
            self.assertEqual(type(message), dict)
            self.assertIn('role', message)
            self.assertIn('content', message)
    
    def test_extract_from_source(self):
        # test extracting examples for all supported file type
        chunks = thepipe.extract.extract_from_source(source_string=self.files_directory)
        self.assertEqual(type(chunks), list)
        for chunk in chunks:
            self.assertEqual(type(chunk), thepipe.core.Chunk)
            self.assertIsNotNone(chunk.path)
            self.assertIsNotNone(chunk.text or chunk.image)

    def test_extract_url(self):
        chunk = thepipe.extract.extract_url('https://en.wikipedia.org/wiki/Piping')
        self.assertEqual(type(chunk), thepipe.core.Chunk)
        self.assertEqual(chunk.path, 'https://en.wikipedia.org/wiki/Piping')
        self.assertIsNotNone(chunk.text)
        self.assertIn('Piping', chunk.text)

    def test_extract_github(self):
        chunks = thepipe.extract.extract_github(github_url='https://github.com/emcf/engshell', branch='main')
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0) # should have some repo contents

    """
    def test_compress_with_llmlingua(self):
        chunks = thepipe.extract.extract_from_source(source_string=self.files_directory+"/example.md")
        new_chunks = thepipe.compress.compress_chunks(chunks)
        # verify that the compressed text is shorter than the original
        old_chunktext = sum([len(chunk.text) for chunk in chunks if chunk.text is not None])
        new_chunktext = sum([len(chunk.text) for chunk in new_chunks if chunk.text is not None])
        self.assertLess(new_chunktext, old_chunktext)
        # verify it still contains vital information
        self.assertIn('markdown', new_chunks[0].text.lower())
        self.assertIn('easy', new_chunks[0].text.lower())

    def test_compress_with_ctags(self):
        chunks = thepipe.extract.extract_from_source(source_string=self.files_directory+"/example.py")
        new_chunks = thepipe.compress.compress_chunks(chunks)
        # verify that the compressed text is shorter than the original
        self.assertLess(len(new_chunks[0].text), len(chunks[0].text))
        # verify it still contains code structure
        self.assertIn('ExampleClass', new_chunks[0].text)
        self.assertIn('greet', new_chunks[0].text)

    def test_save_outputs(self):
        chunks = thepipe.extract.extract_from_source(source_string=self.files_directory+"/example.txt")
        thepipe.save_outputs(chunks)
        self.assertTrue(os.path.exists('outputs/prompt.txt'))
        with open('outputs/prompt.txt', 'r', encoding='utf-8') as file:
            text = file.read()
        self.assertIn('Hello, World!', text)
    """        
