import unittest
import os
import json
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def make_ollama_cache(home: Path, model="qwen3", tag="latest"):
    import hashlib
    gguf = b"GGUF" + b"W" * 200
    params = b'{"stop":["</s>"]}'
    cfg = b'{"model_format":"gguf"}'
    def dg(b): return "sha256-" + hashlib.sha256(b).hexdigest()
    blobs = home / "blobs"
    for b in (gguf, params, cfg):
        _write(blobs / dg(b), b)
    manifest = {
        "schemaVersion": 2,
        "config": {"digest": dg(cfg), "size": len(cfg), "mediaType": "application/vnd.ollama.image.config"},
        "layers": [
            {"digest": dg(gguf), "size": len(gguf), "mediaType": "application/vnd.ollama.image.model"},
            {"digest": dg(params), "size": len(params), "mediaType": "application/vnd.ollama.image.params"},
        ],
    }
    man_path = home / "manifests" / "registry.ollama.ai" / "library" / model / tag
    _write(man_path, json.dumps(manifest).encode())
    return man_path, dg(gguf), gguf


class OllamaIngestTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]

    def test_read_native_finds_gguf_layer(self):
        man_path, gguf_digest, gguf = make_ollama_cache(self.home)
        info = aim._ollama_read_native(man_path, self.home)
        self.assertEqual(info["gguf"]["digest"], gguf_digest)
        self.assertEqual(Path(info["gguf"]["real_path"]).read_bytes(), gguf)
        self.assertEqual(len(info["small_blobs"]), 2)  # params + config
        self.assertEqual(info["model"], "qwen3")
        self.assertEqual(info["tag"], "latest")

    def test_ingest_ollama_shares_inode_cache_intact(self):
        man_path, gguf_digest, gguf = make_ollama_cache(self.home)
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="ollama-qwen3", native_cas=True,
                      source={"type": "ollama", "repo_id": "qwen3:latest"},
                      category="llm/chat", canonical={"root": "primary", "path": str(man_path)})]
        ok = aim.op_ingest(self.config, reg, "ollama-qwen3", registry_save=False)
        self.assertTrue(ok)
        e = reg.find("ollama-qwen3")
        self.assertEqual(e.storage["class"], "managed-ollama")
        store = Path(self.config["roots"][0]["path"]) / e.storage["store_path"]
        gguf_store = store / "ollama-qwen3.gguf"
        self.assertEqual(gguf_store.read_bytes(), gguf)
        blob = self.home / "blobs" / gguf_digest
        self.assertEqual(os.stat(blob).st_ino, os.stat(gguf_store).st_ino)  # shared inode
        self.assertTrue(man_path.exists())                                  # cache intact
        self.assertTrue(blob.exists())
        self.assertTrue((store / "manifest.json").exists())

    def test_ollama_no_dataloss_on_link_failure(self):
        man_path, gguf_digest, gguf = make_ollama_cache(self.home)
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="ollama-qwen3", native_cas=True,
                      source={"type": "ollama", "repo_id": "qwen3:latest"},
                      category="llm/chat", canonical={"root": "primary", "path": str(man_path)})]
        orig = aim.os.link
        aim.os.link = lambda *a, **k: (_ for _ in ()).throw(OSError("xdev"))
        try:
            ok = aim.op_ingest(self.config, reg, "ollama-qwen3", registry_save=False)
        finally:
            aim.os.link = orig
        self.assertFalse(ok)
        self.assertTrue((self.home / "blobs" / gguf_digest).exists())   # cache intact
        self.assertTrue(man_path.exists())
        store = Path(self.config["roots"][0]["path"]) / "store" / "llm/chat" / "ollama-qwen3"
        self.assertFalse(store.exists())                                # rolled back
        self.assertTrue(reg.find("ollama-qwen3").native_cas)

    def test_ollama_build_shim_rebuilds_missing_cache(self):
        # store has the gguf + small blobs + manifest; cache is empty -> rebuild
        man_path, gguf_digest, gguf = make_ollama_cache(self.home)
        info = aim._ollama_read_native(man_path, self.home)
        store = self.home / "AI" / "store" / "llm/chat" / "ollama-qwen3"
        store.mkdir(parents=True)
        _write(store / "ollama-qwen3.gguf", gguf)
        for b in info["small_blobs"]:
            _write(store / f"blob-{b['digest']}", Path(b["real_path"]).read_bytes())
        (store / "manifest.json").write_text(json.dumps(info["manifest"]))
        # wipe the cache
        import shutil as _sh
        _sh.rmtree(self.home / "blobs")
        man_path.unlink()
        # rebuild from store
        aim._ollama_build_shim(info, store, self.home)
        self.assertTrue((self.home / "blobs" / gguf_digest).exists())
        self.assertEqual((self.home / "blobs" / gguf_digest).read_bytes(), gguf)
        self.assertTrue(man_path.exists())


if __name__ == "__main__":
    unittest.main()
