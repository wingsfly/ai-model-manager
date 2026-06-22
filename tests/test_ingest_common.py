import unittest
import aim


class StorageFieldTests(unittest.TestCase):
    def test_storage_defaults_empty_and_round_trips(self):
        e = aim.ModelEntry(id="m1")
        self.assertEqual(e.storage, {})
        e.storage = {"class": "managed-hf", "store_path": "store/x", "shims": []}
        d = e.to_dict()
        self.assertEqual(d["storage"]["class"], "managed-hf")
        e2 = aim.ModelEntry.from_dict(d)
        self.assertEqual(e2.storage["store_path"], "store/x")

    def test_from_dict_without_storage_defaults_empty(self):
        e = aim.ModelEntry.from_dict({"id": "m2"})
        self.assertEqual(e.storage, {})


if __name__ == "__main__":
    unittest.main()
