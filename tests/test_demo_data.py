import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_demo_data.py"
SPEC = importlib.util.spec_from_file_location("generate_demo_data", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class DemoDataTest(unittest.TestCase):
    def test_generated_files_have_expected_schema_and_size(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            blur_path, synced_path = MODULE.generate_demo_data(Path(tmp_dir), rows=72)

            with blur_path.open(encoding="utf-8") as handle:
                blur_rows = list(csv.DictReader(handle))
            with synced_path.open(encoding="utf-8") as handle:
                synced_rows = list(csv.DictReader(handle))

            self.assertEqual(len(blur_rows), 72)
            self.assertEqual(len(synced_rows), 72)
            self.assertIn("blur_index", blur_rows[0])
            self.assertIn("MOR_1A", blur_rows[0])
            self.assertIn("visibility_mor_raw", synced_rows[0])
            self.assertIn("high_freq_ratio", synced_rows[0])

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_paths = MODULE.generate_demo_data(Path(first), rows=60)
            second_paths = MODULE.generate_demo_data(Path(second), rows=60)
            self.assertEqual(first_paths[0].read_bytes(), second_paths[0].read_bytes())
            self.assertEqual(first_paths[1].read_bytes(), second_paths[1].read_bytes())


if __name__ == "__main__":
    unittest.main()
