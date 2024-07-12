# test_extractor.py

import unittest
import sys
import os
import json
sys.path.append('..')
from thepipe.extract import extract
from thepipe.core import Chunk

class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.example_receipt = """# Receipt
Store: GroceryMart
## Total
Subtotal: $13.49 USD
Tax (8%): $1.08 USD
Total: $14.57 USD
"""

        self.schema = {
            "store_name": "string",
            "subtotal_usd": "float",
            "tax_usd": "float",
            "total_usd": "float",
        }

        self.chunks = [Chunk(path="receipt.md", texts=[self.example_receipt])]

    def test_extract(self):
        results, total_tokens_used = extract(
            chunks=self.chunks,
            schema=self.schema,
        )

        # Check if we got a result
        self.assertEqual(len(results), 1)
        result = results[0]

        # Check if all expected fields are present
        expected_fields = ["store_name", "subtotal_usd", "tax_usd", "total_usd"]
        for field in expected_fields:
            self.assertIn(field, result)

        # Check some specific values
        self.assertEqual(result["store_name"], "GroceryMart")
        self.assertEqual(result["subtotal_usd"], 13.49)
        self.assertEqual(result["tax_usd"], 1.08)
        self.assertEqual(result["total_usd"], 14.57)

        # Check if tokens were used
        self.assertGreater(total_tokens_used, 0)

if __name__ == '__main__':
    unittest.main()
