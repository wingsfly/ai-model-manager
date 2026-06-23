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

    def test_ingest_rel_is_structural_regardless_of_cache_path(self):
        ckpt = _write(self.hub / "checkpoints" / "wav2vec2.pth", b"W" * 8)
        self.config["sources"]["pytorch-hub"]["cache_path"] = "/totally/unrelated"  # must NOT affect rel
        reg = aim.Registry()
        reg.models = [self._torch_entry(ckpt)]
        ok = aim.op_ingest(self.config, reg, "torch-wav2vec2", registry_save=False)
        self.assertTrue(ok)
        self.assertEqual(reg.find("torch-wav2vec2").storage["shims"][0]["reconstruct"]["rel"],
                         "checkpoints/wav2vec2.pth")


class WhisperIngestTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]
        self.wc = Path(self.config["roots"][0]["path"]) / "wcache" / "whisper"
        self.config["sources"]["whisper-cache"] = {"cache_path": str(self.wc)}

    def test_ingest_whisper_pt(self):
        pt = _write(self.wc / "base.pt", b"W" * 40)
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="whisper-base", native_cas=True,
                      source={"type": "whisper-cache", "repo_id": "base"}, category="asr/model",
                      canonical={"root": "primary", "path": str(pt)})]
        ok = aim.op_ingest(self.config, reg, "whisper-base", registry_save=False)
        self.assertTrue(ok)
        e = reg.find("whisper-base")
        self.assertEqual(e.storage["class"], "managed-whisper")
        store = Path(self.config["roots"][0]["path"]) / e.storage["store_path"]
        self.assertEqual((store / "base.pt").read_bytes(), b"W" * 40)
        self.assertTrue(pt.is_symlink())
        self.assertEqual(e.storage["shims"][0]["reconstruct"], {"filename": "base.pt", "rel": "base.pt"})


import io
from contextlib import redirect_stdout


class FlatFileRetargetVerifyTests(unittest.TestCase):
    def test_retarget_torch_and_whisper(self):
        det = aim.EnvDetector(home=Path("/h"), rc_files=[],
                              shell_value=lambda v: {"TORCH_HOME": "/tgt/torch"}.get(v))
        e = aim.ModelEntry(id="t", storage={"store_path": "store/x", "shims": [
            {"tool": "pytorch-hub", "kind": "flat-file", "location": "/OLD/x.pth",
             "reconstruct": {"filename": "x.pth", "rel": "checkpoints/x.pth"}}]})
        aim._retarget_shim_locations(e, det)
        self.assertEqual(e.storage["shims"][0]["location"], "/tgt/torch/hub/checkpoints/x.pth")

        det2 = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        e2 = aim.ModelEntry(id="w", storage={"store_path": "store/x", "shims": [
            {"tool": "whisper-cache", "kind": "flat-file", "location": "/OLD/base.pt",
             "reconstruct": {"filename": "base.pt", "rel": "base.pt"}}]})
        aim._retarget_shim_locations(e2, det2)
        self.assertEqual(e2.storage["shims"][0]["location"], "/h/.cache/whisper/base.pt")

    def test_verify_fix_rebuilds_flatfile_shim(self):
        home = Path(tempfile.mkdtemp())
        cfg = aim.default_config(); cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        hub = Path(cfg["roots"][0]["path"]) / "torch" / "hub"
        cfg["sources"]["pytorch-hub"] = {"cache_path": str(hub)}
        ckpt = _write(hub / "checkpoints" / "wav2vec2.pth", b"W" * 32)
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="torch-wav2vec2", native_cas=True,
                      source={"type": "pytorch-hub", "repo_id": "wav2vec2"}, category="asr/model",
                      canonical={"root": "primary", "path": str(ckpt)})]
        aim.op_ingest(cfg, reg, "torch-wav2vec2", registry_save=False)
        ckpt.unlink()  # destroy the shim symlink
        with redirect_stdout(io.StringIO()):
            aim.op_verify(cfg, reg, fix=True)
        self.assertTrue(ckpt.is_symlink())
        store = Path(cfg["roots"][0]["path"]) / reg.find("torch-wav2vec2").storage["store_path"]
        self.assertEqual(ckpt.resolve(), (store / "wav2vec2.pth").resolve())

    def test_restore_roundtrip_crossmachine_flatfile(self):
        src = Path(tempfile.mkdtemp())
        scfg = aim.default_config(); scfg["roots"] = [{"id": "primary", "path": str(src / "AI")}]
        shub = Path(scfg["roots"][0]["path"]) / "torch" / "hub"
        scfg["sources"]["pytorch-hub"] = {"cache_path": str(shub)}
        ckpt = _write(shub / "checkpoints" / "wav2vec2.pth", b"W" * 48)
        sreg = aim.Registry()
        sreg.models = [aim.ModelEntry(id="torch-wav2vec2", native_cas=True,
                       source={"type": "pytorch-hub", "repo_id": "wav2vec2"}, category="asr/model",
                       canonical={"root": "primary", "path": str(ckpt)})]
        aim.op_ingest(scfg, sreg, "torch-wav2vec2", registry_save=False)
        backup = src / "bk"
        with redirect_stdout(io.StringIO()):
            aim.op_backup(scfg, sreg, str(backup))
        tgt = Path(tempfile.mkdtemp())
        tgt_torch = tgt / "torchcache"
        tcfg = aim.default_config(); tcfg["roots"] = [{"id": "primary", "path": str(tgt / "AI")}]
        det = aim.EnvDetector(home=tgt, rc_files=[],
                              shell_value=lambda v: str(tgt_torch) if v == "TORCH_HOME" else None)
        with redirect_stdout(io.StringIO()):
            rc = aim.op_restore(tcfg, aim.Registry(), str(backup), detector=det, registry_save=False)
        self.assertEqual(rc, aim.EXIT_OK)
        link = tgt_torch / "hub" / "checkpoints" / "wav2vec2.pth"
        self.assertTrue(link.is_symlink())
        tgt_store = Path(tcfg["roots"][0]["path"]) / "store" / "asr" / "model" / "torch-wav2vec2"
        self.assertEqual(link.resolve(), (tgt_store / "wav2vec2.pth").resolve())
        self.assertEqual(link.resolve().read_bytes(), b"W" * 48)


if __name__ == "__main__":
    unittest.main()
