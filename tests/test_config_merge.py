import unittest
import aim


class ConfigMergeTests(unittest.TestCase):
    def test_backfills_new_engines(self):
        old = {"version": 1, "roots": [{"id": "primary", "path": "/x"}],
               "engines": {"ollama": {"enabled": True, "model_dir": "ollama/models", "native_cas": True}}}
        merged = aim._merge_config_defaults(dict(old))
        self.assertIn("pytorch-hub", merged["engines"])
        self.assertIn("whisper-cache", merged["engines"])
        self.assertIn("modelscope", merged["engines"])
        self.assertTrue(merged["engines"]["pytorch-hub"]["native_cas"])

    def test_user_engine_override_preserved(self):
        old = {"engines": {"ollama": {"enabled": False, "model_dir": "custom"}}}
        merged = aim._merge_config_defaults(dict(old))
        self.assertEqual(merged["engines"]["ollama"], {"enabled": False, "model_dir": "custom"})

    def test_backfills_missing_toplevel_keys(self):
        merged = aim._merge_config_defaults({"engines": {}})
        self.assertIn("sources", merged)
        self.assertIn("env", merged)

    def test_preserves_existing_toplevel(self):
        merged = aim._merge_config_defaults({"roots": [{"id": "x", "path": "/p"}], "engines": {}})
        self.assertEqual(merged["roots"], [{"id": "x", "path": "/p"}])


if __name__ == "__main__":
    unittest.main()
