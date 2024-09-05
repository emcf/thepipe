# test_extractor.py

import unittest
import sys
import os
import json
sys.path.append('..')
from thepipe.extract import extract, extract_json_from_response
from thepipe.core import Chunk

class TestExtractor(unittest.TestCase):
    def setUp(self):
        self.example_receipt = """# Receipt
Store Name: Grocery Mart
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

    def test_extract_json_from_response(self):
        # List of test cases with expected results
        test_cases = [
            # Case 1: JSON enclosed in triple backticks
            {
                "input": "```json\n{\"key1\": \"value1\", \"key2\": 2}\n```",
                "expected": {"key1": "value1", "key2": 2}
            },
            # Case 2: JSON directly in the response
            {
                "input": "{\"key1\": \"value1\", \"key2\": 2}",
                "expected": {"key1": "value1", "key2": 2}
            },
            # Case 3: Response contains multiple JSON objects
            {
                "input": "Random text {\"key1\": \"value1\"} and another {\"key2\": 2}",
                "expected": [{"key1": "value1"}, {"key2": 2}]
            },
            # Case 4: Response contains incomplete JSON
            {
                "input": "Random text {\"key1\": \"value1\"} and another {\"key2\": 2",
                "expected": {"key1": "value1"}
            }

        ]

        for i, case in enumerate(test_cases):
            with self.subTest(i=i):
                result = extract_json_from_response(case["input"])
                self.assertEqual(result, case["expected"])

    def test_extract(self):
        results, total_tokens_used = extract(
            chunks=self.chunks, # receipt
            schema=self.schema,
        )

        # Check if we got a result
        self.assertEqual(len(results), 1)
        result = results[0]

        print("test_extract result:", json.dumps(result, indent=2))

        # Check if all expected fields are present
        expected_fields = ["store_name", "subtotal_usd", "tax_usd", "total_usd"]
        for field in expected_fields:
            self.assertIn(field, result)

        # Check some specific values
        self.assertEqual(result["store_name"], "Grocery Mart")
        self.assertEqual(result["subtotal_usd"], 13.49)
        self.assertEqual(result["tax_usd"], 1.08)
        self.assertEqual(result["total_usd"], 14.57)

        # Check if tokens were used
        self.assertGreater(total_tokens_used, 0)

if __name__ == '__main__':
    unittest.main()
