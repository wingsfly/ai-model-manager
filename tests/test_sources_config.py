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


class AdapterBasePathTests(unittest.TestCase):
    def _cfg(self, cache_path):
        cfg = aim.default_config()
        cfg["sources"]["huggingface"] = {"cache_path": cache_path}
        return cfg

    def test_hf_adapter_uses_sources_cache_path(self):
        cfg = self._cfg("/real/cache/huggingface/hub")
        root = aim.StorageRoot(id="primary", path="/home/u/AI")
        ad = aim.HuggingFaceAdapter(cfg, root)
        self.assertEqual(ad.base_path, Path("/real/cache/huggingface/hub"))

    def test_hf_adapter_falls_back_to_model_dir(self):
        cfg = aim.default_config()
        root = aim.StorageRoot(id="primary", path="/home/u/AI")
        ad = aim.HuggingFaceAdapter(cfg, root)
        self.assertEqual(ad.base_path, Path("/home/u/AI/huggingface/hub"))

    def test_non_cas_engine_unaffected(self):
        cfg = aim.default_config()
        cfg["sources"]["huggingface"] = {"cache_path": "/real/hub"}
        root = aim.StorageRoot(id="primary", path="/home/u/AI")
        ad = aim.OMLXAdapter(cfg, root)
        self.assertEqual(ad.base_path, Path("/home/u/AI/omlx"))


class ScanAutoAlignTests(unittest.TestCase):
    def test_sync_helper_populates_hf_cache_path(self):
        cfg = aim.default_config()
        det = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        aim._sync_sources_cache_paths(cfg, det)
        self.assertTrue(cfg["sources"]["huggingface"]["cache_path"].endswith("/.cache/huggingface/hub"))

    def test_op_scan_calls_sync(self):
        # op_scan should call _sync_sources_cache_paths at entry; verify via monkeypatch
        cfg = aim.default_config()
        called = {"n": 0}
        orig = aim._sync_sources_cache_paths
        def spy(c, d):
            called["n"] += 1
            return orig(c, d)
        aim._sync_sources_cache_paths = spy
        try:
            reg = aim.Registry()
            aim.op_scan(cfg, reg, engine_filter="huggingface")
        finally:
            aim._sync_sources_cache_paths = orig
        self.assertGreaterEqual(called["n"], 1)


if __name__ == "__main__":
    unittest.main()
