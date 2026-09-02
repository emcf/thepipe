import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from thepipe.scraper import scrape_directory


def create_outside_link(link_path: Path, target_path: Path) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)
    else:
        os.symlink(target_path, link_path, target_is_directory=True)


class test_scrape_directory_security(unittest.TestCase):
    def test_scrape_directory_keeps_normal_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "inside.txt").write_text("INSIDE_FILE", encoding="utf-8")

            chunks = scrape_directory(str(root))

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].text, "INSIDE_FILE")

    def test_scrape_directory_stays_within_root(self):
        canary = "OUTSIDE_ROOT_CANARY"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan_root = root / "scan_root"
            outside_root = root / "outside_root"
            linked_dir = scan_root / "linked_outside"
            scan_root.mkdir()
            outside_root.mkdir()

            (scan_root / "inside.txt").write_text("INSIDE_FILE", encoding="utf-8")
            (outside_root / "secret.txt").write_text(canary, encoding="utf-8")

            try:
                create_outside_link(linked_dir, outside_root)
            except (OSError, RuntimeError) as exc:
                self.skipTest(f"unable to create outside-root link: {exc}")

            chunks = scrape_directory(str(scan_root))
            texts = [chunk.text or "" for chunk in chunks]
            paths = [Path(chunk.path).resolve() for chunk in chunks if chunk.path]

            self.assertTrue(any("INSIDE_FILE" in text for text in texts))
            self.assertFalse(any(canary in text for text in texts))
            self.assertFalse(any(path == (outside_root / "secret.txt") for path in paths))


if __name__ == "__main__":
    unittest.main()
