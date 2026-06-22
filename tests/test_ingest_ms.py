import unittest
import os
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def make_ms_cache(cache: Path, layout="models", org="Qwen", repo="Qwen3-ASR-0___6B"):
    base = cache / ("hub/models" if layout == "hub-models" else "models")
    d = base / org / repo
    _write(d / "config.json", b'{"a":1}')
    _write(d / "model.safetensors", b"W" * 20)
    _write(d / ".msc", b"meta")
    _write(d / ".mdl", b"meta")
    return d


class ModelScopeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.cache = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["sources"]["modelscope"] = {"cache_path": str(self.cache)}
        self.root = aim.StorageRoot(id="primary", path=str(self.cache / "AI"))

    def test_registered(self):
        self.assertIn("modelscope", aim.ADAPTERS)
        self.assertIn("modelscope", aim.ENGINE_NAMES)

    def test_scan_finds_models_both_layouts(self):
        make_ms_cache(self.cache, "models", "Qwen", "Qwen3-ASR-0___6B")
        make_ms_cache(self.cache, "hub-models", "funasr", "paraformer-zh")
        (self.cache / "models" / "._____temp").mkdir(parents=True, exist_ok=True)
        ad = aim.ModelScopeAdapter(self.config, self.root)
        scanned = ad.scan()
        repo_ids = sorted(s.source["repo_id"] for s in scanned)
        self.assertIn("Qwen/Qwen3-ASR-0.6B", repo_ids)
        self.assertIn("funasr/paraformer-zh", repo_ids)
        self.assertTrue(all(s.native_cas and s.is_directory for s in scanned))


class MSIngestTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]

    def test_ms_read_native(self):
        d = make_ms_cache(self.home / "cache", "models", "Qwen", "Qwen3-ASR-0___6B")
        info = aim._ms_read_native(d)
        names = sorted(f["name"] for f in info["files"])
        self.assertIn("model.safetensors", names)
        self.assertIn(".msc", names)

    def test_ms_ingest_dir_symlink_shim(self):
        d = make_ms_cache(self.home / "cache", "models", "Qwen", "Qwen3-ASR-0___6B")
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="ms-qwen", native_cas=True,
                      source={"type": "modelscope", "repo_id": "Qwen/Qwen3-ASR-0.6B"},
                      category="asr/model", canonical={"root": "primary", "path": str(d)})]
        ok = aim.op_ingest(self.config, reg, "ms-qwen", registry_save=False)
        self.assertTrue(ok)
        e = reg.find("ms-qwen")
        self.assertEqual(e.storage["class"], "managed-ms")
        store = Path(self.config["roots"][0]["path"]) / e.storage["store_path"]
        self.assertEqual((store / "model.safetensors").read_bytes(), b"W" * 20)
        self.assertTrue((store / ".msc").exists())
        self.assertTrue(d.is_symlink())
        self.assertEqual(d.resolve(), store.resolve())
        self.assertEqual((d / "model.safetensors").read_bytes(), b"W" * 20)

    def test_ms_no_dataloss_on_symlink_failure(self):
        d = make_ms_cache(self.home / "cache", "models", "Qwen", "Qwen3-ASR-0___6B")
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="ms-qwen", native_cas=True,
                      source={"type": "modelscope", "repo_id": "Qwen/Qwen3-ASR-0.6B"},
                      category="asr/model", canonical={"root": "primary", "path": str(d)})]
        orig = aim.os.symlink
        aim.os.symlink = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        try:
            ok = aim.op_ingest(self.config, reg, "ms-qwen", registry_save=False)
        finally:
            aim.os.symlink = orig
        self.assertFalse(ok)
        self.assertTrue(d.is_dir() and not d.is_symlink())          # original dir intact
        self.assertEqual((d / "model.safetensors").read_bytes(), b"W" * 20)
        self.assertTrue(reg.find("ms-qwen").native_cas)
        self.assertEqual(reg.find("ms-qwen").storage, {})


if __name__ == "__main__":
    unittest.main()
