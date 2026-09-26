import json
import tempfile
from typing import cast
import unittest
import os
import sys
import zipfile
from PIL import Image, ImageStat
import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional in CI
    OpenAI = None  # type: ignore[assignment]

try:
    import whisper  # noqa: F401  # pragma: no cover - optional dependency

    HAS_WHISPER = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_WHISPER = False

sys.path.append("..")
import thepipe.core as core
import thepipe.scraper as scraper


class test_scraper(unittest.TestCase):
    def setUp(self):
        self.files_directory = os.path.join(os.path.dirname(__file__), "files")
        self.outputs_directory = "outputs"
        # create a client we can re-use for ai_extraction scenarios
        self.client = OpenAI() if OpenAI is not None else None

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
        # provenance points at the archive, not the extraction tempdir
        self.assertTrue(all(c.path == zf for c in chunks))
        self.assertEqual(
            {c.metadata["archive_member"] for c in chunks}, {"a.txt", "i.jpg"}
        )

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
    @unittest.skipIf(
        OpenAI is None or not os.getenv("OPENAI_API_KEY"), "OpenAI API key required"
    )
    def test_scrape_pdf_with_ai_extraction(self):
        chunks = scraper.scrape_file(
            os.path.join(self.files_directory, "example.pdf"),
            verbose=True,
            openai_client=self.client,
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

    # Layout of the generated test page (letter, 612x792 pt). Figures are solid
    # colour fills so the returned crops can be matched back to them by colour.
    FIGURE_BLUE = (30, 90, 200)
    FIGURE_ORANGE = (230, 120, 20)
    FIGURE_RECTS = {
        FIGURE_BLUE: (50, 190, 290, 350),  # x0, y0, x1, y1 in points
        FIGURE_ORANGE: (322, 190, 562, 350),
    }

    def _make_figure_pdf(self, path: str) -> None:
        import pymupdf
        from io import BytesIO

        def solid_png(colour):
            buf = BytesIO()
            Image.new("RGB", (480, 320), colour).save(buf, format="PNG")
            return buf.getvalue()

        doc = pymupdf.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text((50, 70), "Quarterly Widget Report", fontsize=22)
        page.insert_text(
            (50, 110),
            "Sales of the Zephyr widget grew steadily across all regions this quarter.",
            fontsize=11,
        )
        page.insert_text(
            (50, 128),
            "Figure 1 (left) shows the northern market; Figure 2 (right) the southern.",
            fontsize=11,
        )
        for colour, rect in self.FIGURE_RECTS.items():
            page.insert_image(pymupdf.Rect(*rect), stream=solid_png(colour))
        page.insert_text((50, 372), "Figure 1: Northern market", fontsize=9)
        page.insert_text((322, 372), "Figure 2: Southern market", fontsize=9)
        # table: 3 columns x 3 rows with ruled lines
        rows = [("Region", "Units", "Revenue"), ("North", "1200", "$48,000"), ("South", "950", "$38,000")]
        x_cols, y0, row_h = (50, 240, 430), 400, 24
        for i, row in enumerate(rows):
            y = y0 + i * row_h
            page.draw_line((50, y), (562, y))
            for x, cell in zip(x_cols, row):
                page.insert_text((x + 4, y + 17), cell, fontsize=11)
        page.draw_line((50, y0 + 3 * row_h), (562, y0 + 3 * row_h))
        doc.save(path)
        doc.close()

    @unittest.skipIf(
        OpenAI is None or not os.getenv("OPENAI_API_KEY"), "OpenAI API key required"
    )
    def test_scrape_pdf_crops_figures_with_vlm(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = os.path.join(tmp, "figures.pdf")
            self._make_figure_pdf(pdf_path)
            chunks = scraper.scrape_pdf(pdf_path, openai_client=self.client)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]

        # text, headings and the table are transcribed into markdown
        text = cast(str, chunk.text)
        self.assertIn("Quarterly Widget Report", text)
        self.assertIn("Zephyr", text)
        for cell in ("North", "1200", "South", "950"):
            self.assertIn(cell, text)

        # exactly the two figures are cropped out - not the table, not the page
        self.assertEqual(len(chunk.images), 2)
        self.assertEqual(chunk.metadata["page"], 1)
        self.assertEqual(chunk.metadata["model"], core.DEFAULT_AI_MODEL)
        figures = chunk.metadata["figures"]
        self.assertEqual(len(figures), 2)

        matched = set()
        for crop, figure in zip(chunk.images, figures):
            self.assertTrue(figure["description"])
            r, g, b = ImageStat.Stat(crop.convert("RGB")).mean
            colour = min(
                self.FIGURE_RECTS,
                key=lambda c: abs(c[0] - r) + abs(c[1] - g) + abs(c[2] - b),
            )
            # the crop must be dominated by that figure's fill colour (white is ~400 away)
            self.assertLess(
                abs(colour[0] - r) + abs(colour[1] - g) + abs(colour[2] - b), 150
            )
            matched.add(colour)
            # and its reported bbox must overlap the true figure rect (IoU)
            x0, y0, x1, y1 = self.FIGURE_RECTS[colour]
            truth = (x0 / 612, y0 / 792, x1 / 612, y1 / 792)
            self.assertGreater(self._iou(figure["bbox"], truth), 0.5)
        self.assertEqual(matched, set(self.FIGURE_RECTS))

    @staticmethod
    def _iou(a, b) -> float:
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
        return inter / (area(a) + area(b) - inter)

    def test_crop_figures(self):
        page = Image.new("RGB", (200, 100), "white")
        page.paste(Image.new("RGB", (50, 20), "red"), (100, 40))
        red = scraper.FigureBox(
            description="red box", x_min=0.5, y_min=0.4, x_max=0.75, y_max=0.6
        )
        # degenerate box is invalid
        dot = scraper.FigureBox(description="dot", x_min=0.1, y_min=0.1, x_max=0.11, y_max=0.9)
        # out-of-range coordinates are clamped
        edge = scraper.FigureBox(description="edge", x_min=-1, y_min=0.0, x_max=0.25, y_max=2)
        self.assertTrue(red.is_valid)
        self.assertFalse(dot.is_valid)
        self.assertEqual(edge.bbox, (0.0, 0.0, 0.25, 1.0))

        crops = scraper.crop_figures(page, [red, edge])
        self.assertEqual(crops[0].size, (50, 20))
        self.assertEqual(crops[0].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(crops[1].size, (50, 100))

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

    @unittest.skipUnless(HAS_WHISPER, "Whisper extra is not installed")
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

    @unittest.skipUnless(HAS_WHISPER, "Whisper extra is not installed")
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
        self.assertEqual(chunks[0].metadata["slide"], 1)

    def test_scraper_provenance(self):
        f = self.files_directory
        pdf = scraper.scrape_file(os.path.join(f, "example.pdf"))
        self.assertEqual([c.metadata["page"] for c in pdf], list(range(1, len(pdf) + 1)))

        ipynb = scraper.scrape_file(os.path.join(f, "example.ipynb"))
        self.assertEqual(ipynb[0].metadata["cell"], 0)
        self.assertIn(ipynb[0].metadata["cell_type"], {"markdown", "code", "raw"})

        docx = scraper.scrape_file(os.path.join(f, "example.docx"))
        self.assertIn(docx[0].metadata["block_type"], {"paragraph", "table"})
        self.assertIsInstance(docx[0].metadata["block"], int)

        rows = scraper.scrape_file(os.path.join(f, "example.csv"))
        self.assertEqual([c.metadata["row"] for c in rows], list(range(len(rows))))


if __name__ == "__main__":
    unittest.main()
