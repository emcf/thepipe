"""
import unittest
import os
from thepipe.extract import extract_from_file, extract_from_url

class TestAPI(unittest.TestCase):
        
    def setUp(self):
        self.files_directory = os.path.join(os.path.dirname(__file__), 'files')
        # Example schema for extraction
        self.schema = {
            "document_topic": "string",
            "document_sentiment": "float",
        }

    def test_extract_from_file_with_multiple_extractions(self):
        # Path to the file you want to test
        file_path = os.path.join(self.files_directory, 'example.pdf')

        # Call the real API for extracting data from the file
        try:
            result = extract_from_file(
                file_path=file_path,
                schema=self.schema,
                ai_model="gpt-4o-mini",
                multiple_extractions=True,
                text_only=True,
                ai_extraction=False,
                host_images=False
            )
            print("Extract from file result:", result)

            # Basic assertions to ensure extraction happened
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
            # Check if the extracted data matches the schema
            # since multiple extractions is enabled, we have the 'extractions' key for each chunk
            # containing all the extractions.
            # the result looks like: [{'chunk_index': 0, 'source': 'example.pdf', 'extraction': [{'document_topic': 'Density PDFs in Supersonic Turbulence', 'document_sentiment': None}]}]
            for item in result:
                self.assertIsInstance(item, dict)
                if 'extractions' in item:
                    for extraction in item['extractions']:
                        self.assertIsInstance(extraction, dict)
                        for key in self.schema:
                            self.assertIn(key, extraction)

        except Exception as e:
            self.fail(f"test_extract_from_file failed with error: {e}")

    def test_extract_from_url_with_one_extraction(self):
        # URL you want to extract information from
        url = 'https://thepi.pe/'  # Update this with your actual URL

        # Call the real API for extracting data from the URL
        try:
            result = extract_from_url(
                url=url,
                schema=self.schema,
                ai_model="gpt-4o-mini",
                multiple_extractions=False,
                text_only=True,
                ai_extraction=False,
                host_images=False
            )
            print("Extract from URL result:", result)

            # Basic assertions to ensure extraction happened
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)

            # Check if the extracted data matches the schema
            # since multiple extractions is disabled, we don't have the 'extractions' key for each chunk
            # [{'chunk_index': 0, 'source': 'https://thepi.pe/', 'document_topic': 'AI document extraction and data processing', 'document_sentiment': 0.8}]
            for item in result:
                self.assertIsInstance(item, dict)
                for key in self.schema:
                    self.assertIn(key, item)

        except Exception as e:
            self.fail(f"test_extract_from_url failed with error: {e}")


if __name__ == '__main__':
    unittest.main()
"""