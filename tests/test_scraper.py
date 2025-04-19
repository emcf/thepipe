import unittest
import os
import sys

sys.path.append("..")
import thepipe.core as core
import thepipe.scraper as scraper


class test_scraper(unittest.TestCase):
    def setUp(self):
        self.files_directory = os.path.join(os.path.dirname(__file__), "files")
        self.outputs_directory = "outputs"

    def tearDown(self):
        # clean up outputs
        if os.path.exists(self.outputs_directory):
            for file in os.listdir(self.outputs_directory):
                os.remove(os.path.join(self.outputs_directory, file))
            os.rmdir(self.outputs_directory)

    def test_scrape_html(self):
        filepath = self.files_directory + "/example.html"
        chunks = scraper.scrape_file(filepath, verbose=True)
        # verify it scraped the url into chunks
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        # verify it scraped markdown data
        self.assertTrue(any(chunk.text and len(chunk.text) > 0 for chunk in chunks))
        # verify it scraped to markdown correctly
        self.assertTrue(any("# Heading 1" in (chunk.text or "") for chunk in chunks))
        self.assertTrue(any("## Heading 2" in (chunk.text or "") for chunk in chunks))
        self.assertTrue(any("### Heading 3" in (chunk.text or "") for chunk in chunks))
        self.assertTrue(
            any("| Name | Age | Country |" in (chunk.text or "") for chunk in chunks)
        )
        for chunk in chunks:
            print("CHUNK TEXT: ", chunk.text)
        self.assertTrue(any("**bold text**" in (chunk.text or "") for chunk in chunks))
        self.assertTrue(any("*italic text*" in (chunk.text or "") for chunk in chunks))
        # ensure javascript was not scraped
        self.assertFalse(
            any("function highlightText()" in (chunk.text or "") for chunk in chunks)
        )

    def test_scrape_zip(self):
        chunks = scraper.scrape_file(
            self.files_directory + "/example.zip", verbose=True
        )
        # verify it scraped the zip file into chunks
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        # verify it scraped text data
        self.assertTrue(any(len(chunk.text) > 0 for chunk in chunks))
        # verify it scraped image data
        self.assertTrue(any(len(chunk.images) > 0 for chunk in chunks))

    def test_scrape_ipynb(self):
        chunks = scraper.scrape_file(
            self.files_directory + "/example.ipynb", verbose=True
        )
        # verify it scraped the ipynb file into chunks
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        # verify it scraped text data
        self.assertTrue(any(len(chunk.text) > 0 for chunk in chunks))
        # verify it scraped image data
        self.assertTrue(any(len(chunk.images) > 0 for chunk in chunks))

    # requires modal token to run
    def test_scrape_pdf_with_ai_extraction(self):
        chunks = scraper.scrape_file(
            "tests/files/example.pdf", ai_extraction=True, verbose=True
        )
        # verify it scraped the pdf file into chunks
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        # verify it scraped the data
        for chunk in chunks:
            self.assertIsNotNone(chunk.text or chunk.images)

    def test_scrape_docx(self):
        chunks = scraper.scrape_file(
            self.files_directory + "/example.docx", verbose=True
        )
        # verify it scraped the docx file into chunks
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        # verify it scraped text data
        self.assertTrue(any(len(chunk.text) > 0 for chunk in chunks))
        # verify it scraped image data
        self.assertTrue(any(len(chunk.images) > 0 for chunk in chunks))

    def test_extract_pdf_without_ai_extraction(self):
        chunks = scraper.scrape_file(
            self.files_directory + "/example.pdf",
            ai_extraction=False,
            verbose=True,
        )
        # verify it scraped the pdf file into chunks
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        # verify it scraped text data
        self.assertTrue(any(len(chunk.text) > 0 for chunk in chunks))
        # verify it scraped image data
        self.assertTrue(any(len(chunk.images) > 0 for chunk in chunks))

    def test_scrape_audio(self):
        chunks = scraper.scrape_file(
            self.files_directory + "/example.mp3", verbose=True
        )
        # verify it scraped the audio file into chunks
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        # verify it scraped audio data
        self.assertTrue(any(len(chunk.text) > 0 for chunk in chunks))
        # verify it transcribed the audio correctly, i.e., 'citizens' is in the scraped text
        self.assertTrue(
            any(
                "citizens" in chunk.text.lower()
                for chunk in chunks
                if chunk.text is not None
            )
        )

    def test_scrape_video(self):
        chunks = scraper.scrape_file(
            self.files_directory + "/example.mp4", verbose=True
        )
        # verify it scraped the video file into chunks
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        # verify it scraped visual data
        self.assertTrue(any(len(chunk.images) > 0 for chunk in chunks))
        # verify it scraped audio data
        self.assertTrue(any(len(chunk.text) > 0 for chunk in chunks))
        # verify it transcribed the audio correctly, i.e., 'citizens' is in the scraped text
        self.assertTrue(
            any(
                "citizens" in chunk.text.lower()
                for chunk in chunks
                if chunk.text is not None
            )
        )

    def test_scrape_pptx(self):
        chunks = scraper.scrape_file(
            self.files_directory + "/example.pptx", verbose=True
        )
        # verify it scraped the pptx file into chunks
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        # verify it scraped text data
        self.assertTrue(any(len(chunk.text) > 0 for chunk in chunks))
        # verify it scraped image data
        self.assertTrue(any(len(chunk.images) > 0 for chunk in chunks))

    def test_scrape_tweet(self):
        tweet_url = "https://x.com/ylecun/status/1796734866156843480"
        chunks = scraper.scrape_url(tweet_url)
        # verify it returned chunks representing the tweet
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)
        self.assertEqual(type(chunks[0]), core.Chunk)
        # verify it scraped the tweet contents
        self.assertTrue(len(chunks[0].text) > 0)
        self.assertTrue(len(chunks[0].images) > 0)

    # unable to run on github actions due to https://github.com/pytube/pytube/issues/399
    # uncomment with corresponding header fixes to run locally
    # def test_scrape_youtube(self):
    #    chunks = scraper.scrape_url("https://www.youtube.com/watch?v=So7TNRhIYJ8")
    #    # verify it scraped the youtube video into chunks
    #    self.assertEqual(type(chunks), list)
    #    self.assertNotEqual(len(chunks), 0)
    #    self.assertEqual(type(chunks[0]), core.Chunk)
    #    # verify it scraped visual data
    #    self.assertTrue(any(len(chunk.images) > 0 for chunk in chunks))
    #    # verify it scraped text data
    #    self.assertTrue(any(len(chunk.text) > 0 for chunk in chunks))
    #    # verify it transcribed the audio correctly, i.e., 'citizens' is in the scraped text
    #    self.assertTrue(any('graphics card' in chunk.text.lower() for chunk in chunks if chunk.text is not None))

    def test_scrape_url(self):
        # verify web page scrape result
        chunks = scraper.scrape_url("https://en.wikipedia.org/wiki/Piping")
        for chunk in chunks:
            self.assertEqual(type(chunk), core.Chunk)
            self.assertEqual(chunk.path, "https://en.wikipedia.org/wiki/Piping")
        # assert if any of the texts in chunk.texts contains 'pipe'
        self.assertGreater(len(chunk.text), 0)
        self.assertIn("pipe", chunk.text)
        # verify if at least one image was scraped
        self.assertTrue(any(len(chunk.images) > 0 for chunk in chunks))
        # verify file url scrape result
        chunks = scraper.scrape_url(
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        )
        self.assertEqual(len(chunks), 1)

    def test_scrape_url_with_ai_extraction(self):
        # verify web page scrape result with ai extraction
        chunks = scraper.scrape_url(
            "https://en.wikipedia.org/wiki/Piping", ai_extraction=True
        )
        for chunk in chunks:
            self.assertEqual(type(chunk), core.Chunk)
            self.assertEqual(chunk.path, "https://en.wikipedia.org/wiki/Piping")
        # assert if any of the texts in chunk.texts contains 'pipe'
        self.assertGreater(len(chunk.text), 0)
        self.assertIn("pipe", chunk.text)
        # verify if at least one image was scraped
        self.assertTrue(any(len(chunk.images) > 0 for chunk in chunks))

    @unittest.skipUnless(os.environ.get("GITHUB_TOKEN"), "requires GITHUB_TOKEN")
    def test_scrape_github(self):
        chunks = scraper.scrape_url("https://github.com/emcf/thepipe")
        self.assertEqual(type(chunks), list)
        self.assertNotEqual(len(chunks), 0)  # should have some repo contents

    def test_scrape_directory(self):
        # verify scraping entire example directory, bar the 'unknown' file
        chunks = scraper.scrape_directory(
            dir_path=self.files_directory, include_regex="^(?!.*unknown).*"
        )
        self.assertEqual(type(chunks), list)
        for chunk in chunks:
            self.assertEqual(type(chunk), core.Chunk)
            self.assertIsNotNone(chunk.path)
            self.assertIsNotNone(chunk.text or chunk.images)
