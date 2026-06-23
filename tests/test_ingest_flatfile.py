import unittest
import os
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


class FlatFileHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_read_native_single_file(self):
        f = _write(self.tmp / "cache" / "checkpoints" / "wav2vec2.pth", b"W" * 50)
        info = aim._flatfile_read_native(f)
        self.assertEqual(len(info["files"]), 1)
        self.assertEqual(info["files"][0]["name"], "wav2vec2.pth")
        self.assertEqual(info["files"][0]["size"], 50)
        self.assertEqual(Path(info["files"][0]["real_path"]).read_bytes(), b"W" * 50)

    def test_build_shim_replaces_file_with_symlink(self):
        store = _write(self.tmp / "store" / "m" / "x.pt", b"DATA")
        orig = _write(self.tmp / "cache" / "x.pt", b"DATA")
        aim._flatfile_build_shim(orig, store)
        self.assertTrue(orig.is_symlink())
        self.assertEqual(orig.resolve(), store.resolve())
        self.assertEqual(orig.read_bytes(), b"DATA")
        self.assertFalse((self.tmp / "cache" / "x.pt.aim-old").exists())

    def test_build_shim_restores_original_on_symlink_failure(self):
        store = _write(self.tmp / "store" / "m" / "x.pt", b"NEW")
        orig = _write(self.tmp / "cache" / "x.pt", b"ORIGINAL")
        real_symlink = aim.os.symlink
        aim.os.symlink = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        try:
            with self.assertRaises(OSError):
                aim._flatfile_build_shim(orig, store)
        finally:
            aim.os.symlink = real_symlink
        self.assertTrue(orig.is_file() and not orig.is_symlink())
        self.assertEqual(orig.read_bytes(), b"ORIGINAL")
        self.assertFalse((self.tmp / "cache" / "x.pt.aim-old").exists())


class FlatFileIngestTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]
        self.hub = Path(self.config["roots"][0]["path"]) / "torch" / "hub"
        self.config["sources"]["pytorch-hub"] = {"cache_path": str(self.hub)}

    def _torch_entry(self, fpath):
        return aim.ModelEntry(id="torch-wav2vec2", native_cas=True,
                              source={"type": "pytorch-hub", "repo_id": "wav2vec2"},
                              category="asr/model", canonical={"root": "primary", "path": str(fpath)})

    def test_ingest_torch_checkpoint(self):
        ckpt = _write(self.hub / "checkpoints" / "wav2vec2.pth", b"W" * 64)
        reg = aim.Registry()
        reg.models = [self._torch_entry(ckpt)]
        ok = aim.op_ingest(self.config, reg, "torch-wav2vec2", registry_save=False)
        self.assertTrue(ok)
        e = reg.find("torch-wav2vec2")
        self.assertFalse(e.native_cas)
        self.assertEqual(e.storage["class"], "managed-torch")
        store = Path(self.config["roots"][0]["path"]) / e.storage["store_path"]
        self.assertEqual((store / "wav2vec2.pth").read_bytes(), b"W" * 64)
        self.assertTrue(ckpt.is_symlink())
        self.assertEqual(ckpt.resolve(), (store / "wav2vec2.pth").resolve())
        shim = e.storage["shims"][0]
        self.assertEqual(shim["kind"], "flat-file")
        self.assertEqual(shim["tool"], "pytorch-hub")
        self.assertEqual(shim["reconstruct"], {"filename": "wav2vec2.pth", "rel": "checkpoints/wav2vec2.pth"})

    def test_ingest_dry_run_changes_nothing(self):
        ckpt = _write(self.hub / "checkpoints" / "wav2vec2.pth", b"W" * 10)
        reg = aim.Registry()
        reg.models = [self._torch_entry(ckpt)]
        aim.op_ingest(self.config, reg, "torch-wav2vec2", dry_run=True, registry_save=False)
        self.assertTrue(ckpt.is_file() and not ckpt.is_symlink())
        self.assertEqual(reg.find("torch-wav2vec2").storage, {})


if __name__ == "__main__":
    unittest.main()
