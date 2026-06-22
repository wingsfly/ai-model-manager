import unittest
from pathlib import Path
import aim


class SourcesConfigTests(unittest.TestCase):
    def test_default_config_has_sources_and_env(self):
        cfg = aim.default_config()
        self.assertIn("sources", cfg)
        self.assertEqual(cfg["sources"], {})
        self.assertIn("env", cfg)
        self.assertFalse(cfg["env"]["managed"])

    def test_sync_writes_cache_paths(self):
        cfg = aim.default_config()
        det = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        aim._sync_sources_cache_paths(cfg, det)
        self.assertEqual(cfg["sources"]["huggingface"]["cache_path"], "/h/.cache/huggingface/hub")
        self.assertEqual(cfg["sources"]["pytorch-hub"]["cache_path"], "/h/.cache/torch/hub")

    def test_sync_preserves_existing_keys(self):
        cfg = aim.default_config()
        cfg["sources"]["huggingface"] = {"managed_env": {"HF_ENDPOINT": "https://hf-mirror.com"}}
        det = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        aim._sync_sources_cache_paths(cfg, det)
        self.assertIn("cache_path", cfg["sources"]["huggingface"])
        self.assertEqual(cfg["sources"]["huggingface"]["managed_env"]["HF_ENDPOINT"],
                         "https://hf-mirror.com")


if __name__ == "__main__":
    unittest.main()
