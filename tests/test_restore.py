import unittest
import json
import tempfile
import io
from contextlib import redirect_stdout
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def _det(vals):
    return aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: vals.get(v))


class ReadManifestTests(unittest.TestCase):
    def test_reads_valid(self):
        d = Path(tempfile.mkdtemp())
        (d / "aim-backup.json").write_text(json.dumps({"aim_backup_version": 1, "models": []}))
        self.assertEqual(aim._read_backup_manifest(d)["aim_backup_version"], 1)

    def test_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            aim._read_backup_manifest(Path(tempfile.mkdtemp()))

    def test_bad_version_raises(self):
        d = Path(tempfile.mkdtemp())
        (d / "aim-backup.json").write_text(json.dumps({"aim_backup_version": 99}))
        with self.assertRaises(ValueError):
            aim._read_backup_manifest(d)


class RetargetTests(unittest.TestCase):
    def test_retarget_hf(self):
        e = aim.ModelEntry(id="x", storage={"shims": [{"kind": "hf-cas", "location": "/OLD/models--Org--M",
              "reconstruct": {"repo_id": "Org/M"}}]})
        aim._retarget_shim_locations(e, _det({"HF_HOME": "/tgt/hf"}))
        self.assertEqual(e.storage["shims"][0]["location"], "/tgt/hf/hub/models--Org--M")

    def test_retarget_ollama(self):
        e = aim.ModelEntry(id="x", storage={"shims": [{"kind": "ollama-cas", "location": "/OLD",
              "reconstruct": {"manifest_rel": "registry.ollama.ai/library/q/latest"}}]})
        aim._retarget_shim_locations(e, _det({"OLLAMA_MODELS": "/tgt/ollama"}))
        self.assertEqual(e.storage["shims"][0]["location"],
                         "/tgt/ollama/manifests/registry.ollama.ai/library/q/latest")

    def test_retarget_ms(self):
        e = aim.ModelEntry(id="x", storage={"shims": [{"kind": "ms-dir", "location": "/OLD",
              "reconstruct": {"repo_id": "Qwen/Q", "dir_name": "Q___6B"}}]})
        aim._retarget_shim_locations(e, _det({"MODELSCOPE_CACHE": "/tgt/ms"}))
        self.assertEqual(e.storage["shims"][0]["location"], "/tgt/ms/models/Qwen/Q___6B")


from tests.test_ingest_hf import make_hf_cache


class RestoreRoundTripTests(unittest.TestCase):
    def _ingest_source(self):
        src_home = Path(tempfile.mkdtemp())
        cfg = aim.default_config(); cfg["roots"] = [{"id": "primary", "path": str(src_home / "AI")}]
        repo = make_hf_cache(src_home, org="Org", repo="M",
                             files=(("config.json", b"{}"), ("model.safetensors", b"W" * 40)))
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="hf-org-m", native_cas=True,
                      source={"type": "huggingface", "repo_id": "Org/M"}, category="asr/model",
                      canonical={"root": "primary", "path": str(repo)})]
        aim.op_ingest(cfg, reg, "hf-org-m", registry_save=False)
        return src_home, cfg, reg

    def test_restore_roundtrip_crossmachine(self):
        src_home, src_cfg, src_reg = self._ingest_source()
        backup = src_home / "backup"
        with redirect_stdout(io.StringIO()):
            aim.op_backup(src_cfg, src_reg, str(backup))
        tgt_home = Path(tempfile.mkdtemp())
        tgt_hf = tgt_home / "hfcache"
        tgt_cfg = aim.default_config(); tgt_cfg["roots"] = [{"id": "primary", "path": str(tgt_home / "AI")}]
        tgt_reg = aim.Registry()
        det = aim.EnvDetector(home=tgt_home, rc_files=[],
                              shell_value=lambda v: str(tgt_hf) if v == "HF_HOME" else None)
        with redirect_stdout(io.StringIO()):
            rc = aim.op_restore(tgt_cfg, tgt_reg, str(backup), detector=det, registry_save=False)
        self.assertEqual(rc, aim.EXIT_OK)
        tgt_store = Path(tgt_cfg["roots"][0]["path"]) / "store" / "asr" / "model" / "hf-org-m"
        self.assertEqual((tgt_store / "model.safetensors").read_bytes(), b"W" * 40)
        self.assertIsNotNone(tgt_reg.find("hf-org-m"))
        snaps = tgt_hf / "hub" / "models--Org--M" / "snapshots"
        snap = next(d for d in snaps.iterdir() if d.is_dir())
        self.assertEqual((snap / "model.safetensors").resolve().read_bytes(), b"W" * 40)
        self.assertTrue(str(snap.resolve()).startswith(str(tgt_home.resolve())))   # shim -> target store, not source
        self.assertFalse(str(snap.resolve()).startswith(str(src_home.resolve())))

    def test_restore_idempotent(self):
        src_home, src_cfg, src_reg = self._ingest_source()
        backup = src_home / "backup"
        with redirect_stdout(io.StringIO()):
            aim.op_backup(src_cfg, src_reg, str(backup))
        tgt_home = Path(tempfile.mkdtemp())
        tgt_cfg = aim.default_config(); tgt_cfg["roots"] = [{"id": "primary", "path": str(tgt_home / "AI")}]
        det = aim.EnvDetector(home=tgt_home, rc_files=[],
                              shell_value=lambda v: str(tgt_home / "hf") if v == "HF_HOME" else None)
        with redirect_stdout(io.StringIO()):
            aim.op_restore(tgt_cfg, aim.Registry(), str(backup), detector=det, registry_save=False)
            rc = aim.op_restore(tgt_cfg, aim.Registry(), str(backup), detector=det, registry_save=False)
        self.assertEqual(rc, aim.EXIT_OK)

    def test_restore_continues_on_shim_failure(self):
        src_home, src_cfg, src_reg = self._ingest_source()
        backup = src_home / "backup"
        with redirect_stdout(io.StringIO()):
            aim.op_backup(src_cfg, src_reg, str(backup))
        tgt_home = Path(tempfile.mkdtemp())
        tgt_cfg = aim.default_config(); tgt_cfg["roots"] = [{"id": "primary", "path": str(tgt_home / "AI")}]
        det = aim.EnvDetector(home=tgt_home, rc_files=[],
                              shell_value=lambda v: str(tgt_home / "hf") if v == "HF_HOME" else None)
        orig = aim._rebuild_shim_from_storage
        aim._rebuild_shim_from_storage = lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
        try:
            with redirect_stdout(io.StringIO()):
                rc = aim.op_restore(tgt_cfg, aim.Registry(), str(backup), detector=det, registry_save=False)
        finally:
            aim._rebuild_shim_from_storage = orig
        self.assertEqual(rc, aim.EXIT_FAILED)
        tgt_store = Path(tgt_cfg["roots"][0]["path"]) / "store" / "asr" / "model" / "hf-org-m"
        self.assertTrue((tgt_store / "model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
