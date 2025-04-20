import argparse
import base64
import shutil
from typing import List, cast
import unittest
import os
import sys

sys.path.append("..")
import thepipe.core as core
import thepipe.scraper as scraper
from PIL import Image
from io import BytesIO


class test_core(unittest.TestCase):
    def setUp(self):
        self.files_directory = os.path.join(os.path.dirname(__file__), "files")
        self.outputs_directory = "outputs"

    def tearDown(self):
        # clean up outputs
        if os.path.exists(self.outputs_directory):
            for file in os.listdir(self.outputs_directory):
                os.remove(os.path.join(self.outputs_directory, file))
            os.rmdir(self.outputs_directory)

    def test_make_image_url_data_and_max_resolution(self):
        img = Image.new("RGB", (100, 50), color="purple")
        # max_resolution scales down before encoding
        url = core.make_image_url(img, host_images=False, max_resolution=25)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        b64 = url.split(",", 1)[1]
        img2 = Image.open(BytesIO(base64.b64decode(b64)))
        self.assertTrue(max(img2.size) <= 25)

    def test_make_image_url_host_images(self):

        # clear any old images
        if os.path.isdir("images"):
            shutil.rmtree("images")

        # override host for predictability
        import thepipe.core as core

        core.HOST_URL = "http://test-host"

        img = Image.new("RGB", (10, 10), color="orange")
        url = core.make_image_url(img, host_images=True)

        # URL should point to our HOST_URL
        self.assertTrue(url.startswith("http://test-host/images/"))

        # extract the image_id from the URL
        image_id = url.rsplit("/", 1)[-1]

        # confirm that exact file exists on disk
        self.assertTrue(os.path.exists(os.path.join("images", image_id)))

    def test_calculate_image_and_mixed_tokens(self):
        small = Image.new("RGB", (256, 256))
        self.assertEqual(core.calculate_image_tokens(small, detail="auto"), 85)
        large = Image.new("RGB", (2048, 2048))
        high = core.calculate_image_tokens(large, detail="high")
        self.assertGreater(high, 85)

        # Mixed text+image chunk
        txt = core.Chunk(text="abcd")  # 4 chars → 1 token
        img = core.Chunk(images=[small])  # 85 tokens
        total = core.calculate_tokens([txt, img])
        self.assertEqual(total, 1 + 85)

    def test_chunk_to_message_variants(self):
        img = Image.new("RGB", (5, 5))
        chunk = core.Chunk(path="f.md", text="![alt](foo.png)\nHello", images=[img])

        # text_only=True → no image_url entries
        msg1 = chunk.to_message(text_only=True)
        self.assertEqual(len(msg1["content"]), 1)
        self.assertEqual(msg1["content"][0]["type"], "text")

        # host_images & include_paths
        core.HOST_URL = "http://host"
        msg2 = chunk.to_message(host_images=True, include_paths=True)
        # First content block should include the <Document path="..."> wrapper
        self.assertIn('<Document path="f.md">', msg2["content"][0]["text"])
        # There must be at least one image_url entry
        self.assertTrue(any(item["type"] == "image_url" for item in msg2["content"]))

    def test_json_roundtrip(self):
        img = Image.new("RGB", (2, 2))
        chunk = core.Chunk(path="p", text="T", images=[img])
        data = chunk.to_json()
        chunk2 = core.Chunk.from_json(data)

        self.assertEqual(chunk2.path, "p")
        self.assertEqual(chunk2.text, "T")

        images = cast(List[Image.Image], chunk2.images)
        self.assertIsInstance(images, list)
        self.assertEqual(len(images), 1)

    def test_chunk_to_llamaindex(self):
        chunk = core.Chunk(
            path="example.md",
            text="This is a coloured image",
            images=[Image.new("RGB", (32, 32), color="red")],
        )
        llama_index_document = chunk.to_llamaindex()
        self.assertEqual(type(llama_index_document), list)
        self.assertEqual(len(llama_index_document), 1)
        self.assertEqual(type(llama_index_document[0]), core.ImageDocument)

    def test_chunks_to_messages(self):
        chunks = scraper.scrape_file(filepath=self.files_directory + "/example.md")
        messages = core.chunks_to_messages(chunks)
        self.assertEqual(type(messages), list)
        for message in messages:
            self.assertEqual(type(message), dict)
            self.assertIn("role", message)
            self.assertIn("content", message)
        # test chunks_to_messages with path included
        messages = core.chunks_to_messages(chunks, include_paths=True)
        for message in messages:
            self.assertIn("example.md", message["content"][0]["text"])

    def test_save_outputs_text_only_and_with_images(self):
        # Text-only
        c = core.Chunk(path="x.txt", text="XYZ")
        core.save_outputs([c], text_only=True)
        self.assertTrue(os.path.exists("outputs/prompt.txt"))
        files = os.listdir("outputs")
        self.assertEqual(files, ["prompt.txt"])
        shutil.rmtree("outputs")

        # With image
        img = Image.new("RGB", (10, 10))
        c2 = core.Chunk(path="y", text="TXT", images=[img])
        core.save_outputs([c2], text_only=False)
        files = os.listdir("outputs")
        self.assertIn("prompt.txt", files)
        self.assertTrue(any(f.endswith(".jpg") for f in files))

    def test_chunk_json(self):
        example_image_path = os.path.join(self.files_directory, "example.jpg")
        image = Image.open(example_image_path)
        chunk = core.Chunk(path="example.md", text="Hello, World!")
        # convert to json
        chunk_json = chunk.to_json()
        # verify it is a dictionary with the expected items
        self.assertEqual(type(chunk_json), dict)
        self.assertIn("text", chunk_json)
        self.assertIn("path", chunk_json)
        # convert back
        chunk = core.Chunk.from_json(chunk_json)
        # verify it is the correct Chunk object
        self.assertEqual(type(chunk), core.Chunk)
        self.assertEqual(chunk.path, "example.md")
        self.assertEqual(chunk.text, "Hello, World!")

    def test_calculate_tokens(self):
        text = "Hello, World!"
        tokens = core.calculate_tokens([core.Chunk(text=text)])
        self.assertAlmostEqual(tokens, 3.25, places=0)

    def test_calculate_image_tokens(self):
        image = Image.open(os.path.join(self.files_directory, "example.jpg"))
        image.load()  # needed to close the file
        tokens = core.calculate_image_tokens(image, detail="auto")
        self.assertAlmostEqual(tokens, 85, places=0)
        tokens = core.calculate_image_tokens(image, detail="low")
        self.assertAlmostEqual(tokens, 85, places=0)
        tokens = core.calculate_image_tokens(image, detail="high")
        self.assertAlmostEqual(tokens, 595, places=0)

    def test_make_image_url(self):
        image = Image.open(os.path.join(self.files_directory, "example.jpg"))
        image.load()  # needed to close the file
        url = core.make_image_url(image, host_images=False)
        # verify it is in the correct format
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        # verify it decodes correctly
        remove_prefix = url.replace("data:image/jpeg;base64,", "")
        image_data = base64.b64decode(remove_prefix)
        image = Image.open(BytesIO(image_data))
        self.assertEqual(image.format, "JPEG")
        # verify it hosts the image correctly
        url = core.make_image_url(image, host_images=True)
        self.assertTrue(url.startswith(core.HOST_URL))
