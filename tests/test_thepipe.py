import argparse
import base64
import unittest
import os
import sys
sys.path.append('..')
import thepipe
import core
import extract
import compress
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
        image = Image.open(os.path.join(self.files_directory, 'example.jpg'))
        image.load() # needed to close the file
        base64_string = thepipe.image_to_base64(image)
        self.assertEqual(type(base64_string), str)
        # converting back should be the same
        image_data = base64.b64decode(base64_string)
        decoded_image = Image.open(BytesIO(image_data))
        self.assertEqual(image.size, decoded_image.size)

    def test_create_messages_from_chunks(self):
        chunks = extract.extract_from_source(source_string=self.files_directory)
        messages = thepipe.create_messages_from_chunks(chunks)
        self.assertEqual(type(messages), list)
        for message in messages:
            self.assertEqual(type(message), dict)
            self.assertIn('role', message)
            self.assertIn('content', message)
    
    def test_extract_from_source(self):
        # test extracting examples for all supported file type
        chunks = extract.extract_from_source(source_string=self.files_directory)
        self.assertEqual(type(chunks), list)
        for chunk in chunks:
            self.assertEqual(type(chunk), core.Chunk)
            self.assertIsNotNone(chunk.path)
            self.assertIsNotNone(chunk.text or chunk.image)

    def test_extract_url(self):
        chunk = extract.extract_url('https://en.wikipedia.org/wiki/Piping')
        self.assertEqual(type(chunk), core.Chunk)
        self.assertEqual(chunk.path, 'https://en.wikipedia.org/wiki/Piping')
        self.assertIsNotNone(chunk.text)
        self.assertIn('Piping', chunk.text)

    @unittest.skipUnless(os.environ.get('GITHUB_TOKEN'), "requires GITHUB_TOKEN")
    def test_extract_github(self):
        chunks = extract.extract_github(github_url='https://github.com/emcf/engshell', branch='main')
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0) # should have some repo contents
    
    @unittest.skipUnless(os.environ.get('MATHPIX_APP_ID') and os.environ.get('MATHPIX_APP_KEY'), "requires MATHPIX_APP_ID and MATHPIX_APP_KEY")
    def test_extract_pdf_with_mathpix(self):
        chunks = extract.extract_pdf("tests/files/example.pdf", mathpix=True)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(chunks[0].source_type, core.SourceTypes.PDF)
        self.assertIsNotNone(chunks[1].source_type, core.SourceTypes.IMAGE)
        # verify extraction contains image and text
        self.assertIsNotNone(chunks[0].text)
        self.assertIsNotNone(chunks[1].image)

    def test_compress_spreadsheet(self):
        chunks = extract.extract_from_source(source_string=self.files_directory+"/example.xlsx")
        new_chunks = compress.compress_chunks(chunks=chunks, limit=30)
        self.assertEqual(len(new_chunks), 1)
        # verify that the compressed text is shorter than the original
        self.assertLess(len(new_chunks[0].text.replace("Column names and types: ","")), len(chunks[0].text))
    
    def test_compress_with_llmlingua(self):
        chunks = extract.extract_from_source(source_string=self.files_directory+"/example.md")
        new_chunks = compress.compress_chunks(chunks=chunks, limit=30)
        # verify that the compressed text is shorter than the original
        old_chunktext = sum([len(chunk.text) for chunk in chunks if chunk.text is not None])
        new_chunktext = sum([len(chunk.text) for chunk in new_chunks if chunk.text is not None])
        self.assertLess(new_chunktext, old_chunktext)
        # verify it still contains vital information
        self.assertIn('markdown', new_chunks[0].text.lower())
        self.assertIn('easy', new_chunks[0].text.lower())

    def test_compress_with_ctags(self):
        chunks = extract.extract_from_source(source_string=self.files_directory+"/example.py")
        new_chunks = compress.compress_chunks(chunks=chunks, limit=30)
        # verify that the compressed text is shorter than the original
        self.assertLess(len(new_chunks[0].text), len(chunks[0].text))
        # verify it still contains code structure
        self.assertIn('ExampleClass', new_chunks[0].text)
        self.assertIn('greet', new_chunks[0].text)

    def test_save_outputs(self):
        chunks = extract.extract_from_source(source_string=self.files_directory+"/example.txt")
        thepipe.save_outputs(chunks)
        self.assertTrue(os.path.exists('outputs/prompt.txt'))
        with open('outputs/prompt.txt', 'r', encoding='utf-8') as file:
            text = file.read()
        self.assertIn('Hello, World!', text)

    def test_create_prompt_from_source(self):
        final_prompt = thepipe.create_prompt_from_source(source_string=self.files_directory+"/example.md")
        self.assertEqual(type(final_prompt), list)
        self.assertNotEqual(len(final_prompt), 0)
        self.assertEqual(type(final_prompt[0]), dict)
        # verify it still contains vital information from the markdown file
        self.assertIn('markdown', str(final_prompt).lower())

    def test_parse_arguments(self):
        args = thepipe.parse_arguments()
        self.assertEqual(type(args), argparse.Namespace)
        self.assertIn('source', vars(args))
        self.assertIn('match', vars(args))
        self.assertIn('ignore', vars(args))
        self.assertIn('limit', vars(args))