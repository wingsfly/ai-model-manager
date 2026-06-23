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


if __name__ == "__main__":
    unittest.main()
