import unittest
from pathlib import Path

import aim


class DownloadCoreTests(unittest.TestCase):
    def test_parse_download_source_hf(self):
        src, model_id, err = aim._parse_download_source("hf:Qwen/Qwen2-7B")
        self.assertEqual(err, "")
        self.assertEqual(src["type"], "huggingface")
        self.assertEqual(src["repo_id"], "Qwen/Qwen2-7B")
        self.assertEqual(model_id, "qwen2-7b")

    def test_parse_download_source_invalid(self):
        src, model_id, err = aim._parse_download_source("bad-source")
        self.assertIsNone(src)
        self.assertEqual(model_id, "")
        self.assertTrue(err)

    def test_infer_category(self):
        src = {"type": "huggingface", "repo_id": "black-forest-labs/FLUX.1-dev"}
        cat = aim._infer_download_category(src, "flux-1-dev")
        self.assertEqual(cat, "image-gen/checkpoint")

    def test_resolve_download_dest_auto(self):
        root = aim.StorageRoot(id="primary", path="/tmp/ai")
        dest, mode = aim._resolve_download_dest(root, "m1", "llm/chat", "")
        self.assertEqual(mode, "auto")
        self.assertEqual(dest, Path("/tmp/ai") / "store" / "llm/chat" / "m1")

    def test_resolve_download_dest_explicit(self):
        root = aim.StorageRoot(id="primary", path="/tmp/ai")
        dest, mode = aim._resolve_download_dest(root, "m1", "llm/chat", "~/custom/model")
        self.assertEqual(mode, "explicit")
        self.assertTrue(str(dest).endswith("custom/model"))

    def test_store_path_normalization_base_root(self):
        root = aim.StorageRoot(id="r1", path="/tmp/ai")
        self.assertEqual(root.store_path, Path("/tmp/ai/store"))

    def test_store_path_normalization_store_root(self):
        root = aim.StorageRoot(id="r2", path="/tmp/ai/store")
        self.assertEqual(root.store_path, Path("/tmp/ai/store"))

    def test_resolve_download_dest_when_root_already_store(self):
        root = aim.StorageRoot(id="primary", path="/tmp/ai/store")
        dest, mode = aim._resolve_download_dest(root, "m2", "llm/chat", "")
        self.assertEqual(mode, "auto")
        self.assertEqual(dest, Path("/tmp/ai/store/llm/chat/m2"))


if __name__ == "__main__":
    unittest.main()
