import json
import tempfile
from typing import cast
import unittest
import os
import sys
import zipfile
from PIL import Image
import pandas as pd

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

    def test_scrape_directory(self):
        # verify scraping entire example directory, bar the 'unknown' file
        chunks = scraper.scrape_directory(
            dir_path=self.files_directory, inclusion_pattern="^(?!.*unknown).*"
        )
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, core.Chunk)
            # ensure at least one of text/images is non-empty
            if not (chunk.text or chunk.images):
                self.fail("Empty chunk found: {}".format(chunk.path))
            self.assertTrue(chunk.text or chunk.images)

    def test_scrape_directory_inclusion_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            # ignored folder
            os.makedirs(os.path.join(tmp, "node_modules"))
            with open(os.path.join(tmp, "node_modules", "a.txt"), "w") as f:
                f.write("x")
            # ignored extension
            with open(os.path.join(tmp, "bad.pyc"), "w") as f:
                f.write("x")
            # valid file
            good = os.path.join(tmp, "good.txt")
            with open(good, "w") as f:
                f.write("Y")

            chunks = scraper.scrape_directory(tmp, inclusion_pattern="good")

        self.assertEqual(len(chunks), 1)

        # cast .text to str so Pylance knows it's not None
        text = cast(str, chunks[0].text)
        self.assertIn("Y", text)

    def test_scrape_html(self):
        filepath = os.path.join(self.files_directory, "example.html")
        chunks = scraper.scrape_file(filepath, verbose=True)
        # verify it scraped the url into chunks
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        # verify it scraped markdown data
        self.assertTrue(any(chunk.text and len(chunk.text) > 0 for chunk in chunks))
        # verify it scraped to markdown correctly
        self.assertTrue(any("# Heading 1" in (chunk.text or "") for chunk in chunks))
        self.assertTrue(any("## Heading 2" in (chunk.text or "") for chunk in chunks))
        self.assertTrue(any("### Heading 3" in (chunk.text or "") for chunk in chunks))
        self.assertTrue(
            any("| Name | Age | Country |" in (chunk.text or "") for chunk in chunks)
        )
        # verify bold and italic
        self.assertTrue(any("**bold text**" in (chunk.text or "") for chunk in chunks))
        self.assertTrue(any("*italic text*" in (chunk.text or "") for chunk in chunks))
        # ensure javascript was not scraped
        self.assertFalse(
            any("function highlightText()" in (chunk.text or "") for chunk in chunks)
        )

    def test_scrape_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            txt = os.path.join(tmp, "a.txt")
            with open(txt, "w") as f:
                f.write("TXT")
            imgf = os.path.join(tmp, "i.jpg")
            Image.new("RGB", (10, 10)).save(imgf)
            zf = os.path.join(tmp, "test.zip")
            with zipfile.ZipFile(zf, "w") as z:
                z.write(txt, arcname="a.txt")
                z.write(imgf, arcname="i.jpg")
            chunks = scraper.scrape_file(zf)

        self.assertTrue(any("TXT" in cast(str, c.text) for c in chunks))
        self.assertTrue(any(c.images for c in chunks))

    def test_scrape_spreadsheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame({"a": [1, 2]})
            csvp = os.path.join(tmp, "t.csv")
            df.to_csv(csvp, index=False)
            chunks_csv = scraper.scrape_spreadsheet(csvp, "application/vnd.ms-excel")
            self.assertEqual(len(chunks_csv), 2)
            for i, c in enumerate(chunks_csv):
                self.assertIsNotNone(c.text)
                rec = json.loads(cast(str, c.text))
                self.assertEqual(rec["a"], i + 1)
                self.assertEqual(rec["row index"], i)

            xlsx = os.path.join(tmp, "t.xlsx")
            df.to_excel(xlsx, index=False)
            chunks_xlsx = scraper.scrape_spreadsheet(
                xlsx,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            self.assertEqual(len(chunks_xlsx), 2)

    def test_scrape_ipynb(self):
        chunks = scraper.scrape_file(
            os.path.join(self.files_directory, "example.ipynb"), verbose=True
        )
        # verify it scraped the ipynb file into chunks
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], core.Chunk)
        # verify it scraped text data
        self.assertTrue(
            any(chunk.text and len(chunk.text or "") > 0 for chunk in chunks)
        )
        # verify it scraped image data
        self.assertTrue(
            any(chunk.images and len(chunk.images or []) > 0 for chunk in chunks)
        )

    # requires LLM server to be set up
    def test_scrape_pdf_with_ai_extraction(self):
        chunks = scraper.scrape_file(
            os.path.join(self.files_directory, "example.pdf"),
            ai_extraction=True,
            verbose=True,
        )
        # verify it scraped the pdf file into chunks
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], core.Chunk)
        # verify it scraped the data
        for chunk in chunks:
            self.assertTrue(
                (chunk.text and len(chunk.text or "") > 0)
                or (chunk.images and len(chunk.images or []) > 0)
            )

    def test_scrape_docx(self):
        chunks = scraper.scrape_file(
            os.path.join(self.files_directory, "example.docx"), verbose=True
        )
        # verify it scraped the docx file into chunks
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], core.Chunk)
        # verify it scraped data
        self.assertTrue(
            any(len(chunk.text or "") or len(chunk.images or []) for chunk in chunks)
        )

    def test_extract_pdf_without_ai_extraction(self):
        chunks = scraper.scrape_file(
            os.path.join(self.files_directory, "example.pdf"),
            ai_extraction=False,
            verbose=True,
        )
        # verify it scraped the pdf file into chunks
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], core.Chunk)
        # verify it scraped text data
        self.assertTrue(
            any(chunk.text and len(chunk.text or "") > 0 for chunk in chunks)
        )
        # verify it scraped image data
        self.assertTrue(
            any(chunk.images and len(chunk.images or []) > 0 for chunk in chunks)
        )

    def test_scrape_audio(self):
        chunks = scraper.scrape_file(
            os.path.join(self.files_directory, "example.mp3"), verbose=True
        )
        # verify it scraped the audio file into chunks
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], core.Chunk)
        # verify it scraped audio data
        self.assertTrue(
            any(chunk.text and len(chunk.text or "") > 0 for chunk in chunks)
        )
        # verify it transcribed the audio correctly
        self.assertTrue(
            any(chunk.text and "citizens" in chunk.text.lower() for chunk in chunks)
        )

    def test_scrape_video(self):
        chunks = scraper.scrape_file(
            os.path.join(self.files_directory, "example.mp4"), verbose=True
        )
        # verify it scraped the video file into chunks
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], core.Chunk)
        # verify it scraped visual data
        self.assertTrue(
            any(chunk.images and len(chunk.images or []) > 0 for chunk in chunks)
        )
        # verify it scraped audio data
        self.assertTrue(
            any(chunk.text and len(chunk.text or "") > 0 for chunk in chunks)
        )
        # verify it transcribed the audio correctly
        self.assertTrue(
            any(chunk.text and "citizens" in chunk.text.lower() for chunk in chunks)
        )

    def test_scrape_pptx(self):
        chunks = scraper.scrape_file(
            os.path.join(self.files_directory, "example.pptx"), verbose=True
        )
        # verify it scraped the pptx file into chunks
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], core.Chunk)
        # verify it scraped text data
        self.assertTrue(
            any(chunk.text and len(chunk.text or "") > 0 for chunk in chunks)
        )
        # verify it scraped image data
        self.assertTrue(
            any(chunk.images and len(chunk.images or []) > 0 for chunk in chunks)
        )

    def test_scrape_tweet(self):
        tweet_url = "https://x.com/ylecun/status/1796734866156843480"
        chunks = scraper.scrape_url(tweet_url)
        # verify it returned chunks representing the tweet
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)
        self.assertIsInstance(chunks[0], core.Chunk)
        # verify it scraped the tweet contents
        self.assertTrue(chunks[0].text and len(chunks[0].text or "") > 0)
        self.assertTrue(chunks[0].images and len(chunks[0].images or []) > 0)

    def test_scrape_url(self):
        # verify web page scrape result
        chunks = scraper.scrape_url("https://en.wikipedia.org/wiki/Piping")
        for chunk in chunks:
            self.assertIsInstance(chunk, core.Chunk)
            self.assertEqual(chunk.path, "https://en.wikipedia.org/wiki/Piping")
        # assert if any of the texts contains 'pipe'
        self.assertTrue(
            any(chunk.text and "pipe" in (chunk.text or "") for chunk in chunks)
        )
        # verify if at least one image was scraped
        self.assertTrue(
            any(chunk.images and len(chunk.images or []) > 0 for chunk in chunks)
        )
        # verify file url scrape result
        chunks_pdf = scraper.scrape_url(
            "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
        )
        self.assertEqual(len(chunks_pdf), 1)

    def test_scrape_url_with_ai_extraction(self):
        # verify web page scrape result with ai extraction
        chunks = scraper.scrape_url(
            "https://en.wikipedia.org/wiki/Piping", ai_extraction=True
        )
        for chunk in chunks:
            self.assertIsInstance(chunk, core.Chunk)
            self.assertEqual(chunk.path, "https://en.wikipedia.org/wiki/Piping")
        # assert if any of the texts contains 'pipe'
        print("AI scraped webpage chunks:", [chunk.text for chunk in chunks])
        self.assertTrue(
            any(chunk.text and "pipe" in (chunk.text or "") for chunk in chunks)
        )
        # verify if at least one image was scraped
        self.assertTrue(
            any(chunk.images and len(chunk.images or []) > 0 for chunk in chunks)
        )

    @unittest.skipUnless(os.environ.get("GITHUB_TOKEN"), "requires GITHUB_TOKEN")
    def test_scrape_github(self):
        chunks = scraper.scrape_url("https://github.com/emcf/thepipe")
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)  # should have some repo contents
