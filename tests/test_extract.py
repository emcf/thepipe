
import setup
import unittest
import os
from thepipe import extract  # Assuming thepipe has an extract function

class TestExtractFunction(unittest.TestCase):

    def setUp(self):
        # Setup path to the 'files' directory for testing
        self.files_directory = os.path.join(os.path.dirname(__file__), 'files')

    def test_py_file(self):
        # Test the extraction from a .py file
        result = extract(os.path.join(self.files_directory, 'example.py'))
        self.assertIsInstance(result, list)
        self.assertIn('Hello, World!', str(result))

    def test_cpp_file(self):
        # Test the extraction from a .cpp file
        result = extract(os.path.join(self.files_directory, 'example.cpp'))
        self.assertIsInstance(result, list)
        self.assertIn('Hello, World!', str(result))

    def test_docx_file(self):
        # Test the extraction from a .docx file
        result = extract(os.path.join(self.files_directory, 'example.docx'))
        self.assertIsInstance(result, list)
        self.assertIn('Hello, World!', str(result))

    def test_pptx_file(self):
        # Test the extraction from a .pptx file
        result = extract(os.path.join(self.files_directory, 'example.pptx'))
        self.assertIsInstance(result, list)
        self.assertIn('Hello, World!', str(result))

    def test_png_file(self):
        # Test the extraction from a .png file
        result = extract(os.path.join(self.files_directory, 'example.png'))
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_png_file_text_extraction(self):
        # Test the extraction from a .png file
        result = extract(os.path.join(self.files_directory, 'example.png'), use_text=True)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn('Example Text', str(result))

    def test_csv_file(self):
        # Test the extraction from a .csv file
        result = extract(os.path.join(self.files_directory, 'example.csv'))
        self.assertIsInstance(result, list)
        # Should contain colname, datatype
        self.assertIn('Column1', str(result))
        self.assertIn('int', str(result))

    def test_pdf_file(self):
        # Test the extraction from a .pdf file
        result = extract(os.path.join(self.files_directory, 'example.pdf'))
        self.assertIsInstance(result, list)
        self.assertIn('Hello, World!', str(result))

    def test_zip_file(self):
        # Test the extraction from a .zip file
        result = extract(os.path.join(self.files_directory, 'example.zip'))
        self.assertIsInstance(result, list)
        self.assertIn('example.txt', str(result)) # should contain extracted file

    def test_ctags_cpp_file(self):
        # Test extraction and compression from a .cpp file using ctags
        result = extract(os.path.join(self.files_directory, 'example.cpp'), limit = 20)
        self.assertIsInstance(result, list)
        self.assertNotIn('Hello, World!', str(result)) # Should not contain the full code

if __name__ == '__main__':
    unittest.main()
