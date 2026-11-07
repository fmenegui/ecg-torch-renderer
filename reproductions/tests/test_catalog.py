import unittest

from reproductions.fetch_catalog import describe


class CatalogTests(unittest.TestCase):
    def test_inventory_separates_records_images_and_sources(self):
        files = [
            {"name": "D0/D0_000022/D0_000022.hea", "totalBytes": 600},
            {"name": "D0/D0_000022/D0_000022_02.png", "totalBytes": 1000},
            {"name": "D0/D0_000022/D0_000022_01.pdf", "totalBytes": 2000},
            {"name": "D1/D1_000022/D1_000022_04.JPEG", "totalBytes": 3000},
            {"name": "dataset_description.xlsx", "totalBytes": 4000},
        ]
        result = describe(files)
        self.assertEqual(result["D0"], {"files": 3, "image_files": 1, "bytes": 3600, "records": 1})
        self.assertEqual(result["D1"], {"files": 1, "image_files": 1, "bytes": 3000, "records": 1})
        self.assertEqual(result["other"]["records"], 0)


if __name__ == "__main__":
    unittest.main()
