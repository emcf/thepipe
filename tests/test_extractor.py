import os
import sys
import unittest
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel

sys.path.append("..")
from thepipe.extract import extract
from thepipe.core import Chunk


class Receipt(BaseModel):
    store_name: Optional[str]
    subtotal_usd: Optional[float]
    tax_usd: Optional[float]
    total_usd: Optional[float]


@unittest.skipIf(not os.getenv("OPENAI_API_KEY"), "OpenAI API key required")
class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.client = OpenAI()
        self.chunks = [
            Chunk(
                path="receipt.md",
                text="""# Receipt
Store Name: Grocery Mart
## Total
Subtotal: $13.49 USD
Tax (8%): $1.08 USD
Total: $14.57 USD
""",
            )
        ]

    def _assert_receipt(self, result):
        self.assertEqual(result["store_name"], "Grocery Mart")
        self.assertEqual(result["subtotal_usd"], 13.49)
        self.assertEqual(result["tax_usd"], 1.08)
        self.assertEqual(result["total_usd"], 14.57)

    def test_extract(self):
        results, tokens_used = extract(self.chunks, Receipt, openai_client=self.client)
        self.assertEqual(len(results), 1)
        self._assert_receipt(results[0])
        self.assertGreater(tokens_used, 0)

    def test_extract_multiple(self):
        results, _ = extract(
            self.chunks, Receipt, multiple_extractions=True, openai_client=self.client
        )
        self.assertEqual(len(results[0]["extraction"]), 1)
        self._assert_receipt(results[0]["extraction"][0])


if __name__ == "__main__":
    unittest.main()
