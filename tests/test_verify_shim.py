import unittest
import io
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import aim
from tests.test_ingest_hf import make_hf_cache
from tests.test_ingest_ollama import make_ollama_cache


class VerifyShimTests(unittest.TestCase):
    def _cfg(self, home):
        cfg = aim.default_config()
        cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        return cfg

    def test_verify_fix_rebuilds_hf_shim(self):
        home = Path(tempfile.mkdtemp())
        cfg = self._cfg(home)
        repo_dir = make_hf_cache(home, files=(("config.json", b"{}"), ("model.safetensors", b"W" * 40)))
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="hf-org-model", native_cas=True,
                      source={"type": "huggingface", "repo_id": "Org/Model"},
                      category="asr/model", canonical={"root": "primary", "path": str(repo_dir)})]
        aim.op_ingest(cfg, reg, "hf-org-model", registry_save=False)
        shutil.rmtree(repo_dir)                       # destroy shim
        with redirect_stdout(io.StringIO()):
            aim.op_verify(cfg, reg, fix=True)
        link = repo_dir / "snapshots" / "abc123" / "model.safetensors"
        self.assertTrue(link.exists())
        self.assertEqual(link.resolve().read_bytes(), b"W" * 40)

    def test_verify_fix_rebuilds_ollama_shim(self):
        home = Path(tempfile.mkdtemp())
        cfg = self._cfg(home)
        man_path, gguf_digest, gguf = make_ollama_cache(home)
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="ollama-qwen3", native_cas=True,
                      source={"type": "ollama", "repo_id": "qwen3:latest"},
                      category="llm/chat", canonical={"root": "primary", "path": str(man_path)})]
        aim.op_ingest(cfg, reg, "ollama-qwen3", registry_save=False)
        shutil.rmtree(home / "blobs")                 # destroy cache (store keeps the bytes)
        shutil.rmtree(home / "manifests")
        with redirect_stdout(io.StringIO()):
            aim.op_verify(cfg, reg, fix=True)
        self.assertTrue((home / "blobs" / gguf_digest).exists())
        self.assertEqual((home / "blobs" / gguf_digest).read_bytes(), gguf)
        self.assertTrue(man_path.exists())


if __name__ == "__main__":
    unittest.main()
