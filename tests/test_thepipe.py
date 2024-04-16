import argparse
import base64
import unittest
import os
import sys
sys.path.append('..')
from thepipe_api import thepipe
from thepipe_api import core
from thepipe_api import extractor
from thepipe_api import compressor
from PIL import Image
from io import BytesIO

class test_thepipe(unittest.TestCase):
    def setUp(self):
        self.files_directory = os.path.join(os.path.dirname(__file__), 'files')
        self.outputs_directory = 'outputs'

    def tearDown(self):
        # clean up outputs
        if os.path.exists(self.outputs_directory):
            for file in os.listdir(self.outputs_directory):
                os.remove(os.path.join(self.outputs_directory, file))
            os.rmdir(self.outputs_directory)

    def test_image_to_base64(self):
        image = Image.open(os.path.join(self.files_directory, 'example.jpg'))
        image.load() # needed to close the file
        base64_string = core.image_to_base64(image)
        self.assertEqual(type(base64_string), str)
        # converting back should be the same
        image_data = base64.b64decode(base64_string)
        decoded_image = Image.open(BytesIO(image_data))
        self.assertEqual(image.size, decoded_image.size)

    def test_create_messages_from_chunks(self):
        chunks = extractor.extract_from_source(source=self.files_directory)
        messages = core.create_messages_from_chunks(chunks)
        self.assertEqual(type(messages), list)
        for message in messages:
            self.assertEqual(type(message), dict)
            self.assertIn('role', message)
            self.assertIn('content', message)
    
    def test_extract_from_source(self):
        # test extracting examples for all supported file type
        chunks = extractor.extract_from_source(source=self.files_directory, ignore="unknown")
        self.assertEqual(type(chunks), list)
        for chunk in chunks:
            self.assertEqual(type(chunk), core.Chunk)
            self.assertIsNotNone(chunk.path)
            self.assertIsNotNone(chunk.text or chunk.image)
        # ensure unknown sources fail (test has "unknown in filename)
        with self.assertRaises(Exception):
            chunks = extractor.extract_from_source(source=self.files_directory+"/example.unknown")

    def test_extract_from_source_text_only(self):
        # test extracting examples for all supported file type
        chunks = extractor.extract_from_source(source=self.files_directory, text_only=True, ignore="unknown")
        self.assertEqual(type(chunks), list)
        # ensure no images are extracted
        for chunk in chunks:
            self.assertEqual(type(chunk), core.Chunk)
            self.assertIsNone(chunk.image)

    def test_should_ignore(self):
        # test regex ignore
        self.assertTrue(extractor.should_ignore('example.md', ignore='.*\.md'))
        # test file extension ignore
        self.assertTrue(extractor.should_ignore('example.exe', ignore=None))
        # test hidden file ignore
        self.assertTrue(extractor.should_ignore('.gitignore', ignore=None))
        # test directory ignore
        self.assertTrue(extractor.should_ignore('node_modules', ignore=None))
        # test cases that should not be ignored
        self.assertFalse(extractor.should_ignore(self.files_directory+"/example.md", ignore=None))

    def test_extract_url(self):
        # test web page extraction
        chunks = extractor.extract_url('https://en.wikipedia.org/wiki/Piping')
        for chunk in chunks:
            self.assertEqual(type(chunk), core.Chunk)
            self.assertEqual(chunk.path, 'https://en.wikipedia.org/wiki/Piping')
        if chunk.text:
            self.assertIn('pipe', chunk.text)
        # test if at least one image was extracted
        self.assertTrue(any(chunk.image for chunk in chunks))
        # test document url extraction
        chunks = extractor.extract_url('https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf')
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_type, core.SourceTypes.PDF)

    @unittest.skipUnless(os.environ.get('GITHUB_TOKEN'), "requires GITHUB_TOKEN")
    def test_extract_github(self):
        chunks = extractor.extract_github(github_url='https://github.com/emcf/engshell', branch='main')
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0) # should have some repo contents
    
    @unittest.skipUnless(os.environ.get('THEPIPE_API_KEY'), "requires THEPIPE_API_KEY")
    def test_extract_pdf_with_ai_extraction(self):
        chunks = extractor.extract_pdf("tests/files/example.pdf", ai_extraction=True)
        self.assertNotEqual(len(chunks), 0)
        for i in range(len(chunks)):
            if chunks[i].source_type == core.SourceTypes.PDF:
                # verify extraction contains text
                self.assertIsNotNone(chunks[i].text)
                # verify extraction text does not contain image data
                self.assertNotIn('![](', chunks[i].text)
            elif chunks[i].source_type == core.SourceTypes.IMAGE:
                # verify extraction contains image
                self.assertIsNotNone(chunks[i].image)
                
    def test_compress_spreadsheet(self):
        chunks = extractor.extract_from_source(source=self.files_directory+"/example.xlsx")
        new_chunks = compressor.compress_chunks(chunks=chunks, limit=30)
        self.assertEqual(len(new_chunks), 1)
        # verify that the compressed text is shorter than the original
        self.assertLess(len(new_chunks[0].text.replace("Column names and types: ","")), len(chunks[0].text))
    
    def test_compress_with_llmlingua(self):
        chunks = extractor.extract_from_source(source=self.files_directory+"/example.md")
        new_chunks = compressor.compress_chunks(chunks=chunks, limit=30)
        # verify that the compressed text is shorter than the original
        old_chunktext = sum([len(chunk.text) for chunk in chunks if chunk.text is not None])
        new_chunktext = sum([len(chunk.text) for chunk in new_chunks if chunk.text is not None])
        self.assertLess(new_chunktext, old_chunktext)
        # verify it still contains vital information
        self.assertIn('markdown', new_chunks[0].text.lower())
        self.assertIn('easy', new_chunks[0].text.lower())

    def test_compress_with_ctags(self):
        chunks = extractor.extract_from_source(source=self.files_directory+"/example.py")
        new_chunks = compressor.compress_chunks(chunks=chunks, limit=30)
        # verify that the compressed text is shorter than the original
        self.assertLess(len(new_chunks[0].text), len(chunks[0].text))
        # verify it still contains code structure
        self.assertIn('ExampleClass', new_chunks[0].text)
        self.assertIn('greet', new_chunks[0].text)

    def test_save_outputs(self):
        chunks = extractor.extract_from_source(source=self.files_directory+"/example.txt")
        thepipe.save_outputs(chunks)
        self.assertTrue(os.path.exists(self.outputs_directory+"/prompt.txt"))
        with open(self.outputs_directory+"/prompt.txt", 'r', encoding='utf-8') as file:
            text = file.read()
        self.assertIn('Hello, World!', text)
        # test with images
        chunks = extractor.extract_from_source(source=self.files_directory+"/example.jpg")
        thepipe.save_outputs(chunks)
        self.assertTrue(any('.jpg' in f for f in os.listdir(self.outputs_directory)))

    def test_extract(self):
        chunks = thepipe.extract(source=self.files_directory+"/example.md", local=True)
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), dict)
        # verify it still contains vital information from the markdown file
        self.assertIn('markdown', str(chunks).lower())

    def test_extract_api(self):
        # test with markdown file
        chunks = extractor.extract_from_source(source=self.files_directory+"/example.md", local=False)
        # verify it extracted the markdown file
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        self.assertIn('markdown', str(chunks[0].text).lower())
        # test with web page
        chunks = extractor.extract_url('https://en.wikipedia.org/wiki/Piping', local=False)
        # verify it extracted the web page
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        # page should contain the word "Pipe"
        chunk_text = ''.join([chunk.text for chunk in chunks if chunk.text])
        self.assertIn('pipe', chunk_text.lower())

    def test_parse_arguments(self):
        args = thepipe.parse_arguments()
        self.assertEqual(type(args), argparse.Namespace)
        self.assertIn('source', vars(args))
        self.assertIn('match', vars(args))
        self.assertIn('ignore', vars(args))
        self.assertIn('limit', vars(args))