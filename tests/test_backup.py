import unittest
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


class SyncStoreDirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_copies_then_idempotent(self):
        src = self.tmp / "s"
        _write(src / "a" / "x.bin", b"X" * 10)
        _write(src / "y.bin", b"Y" * 5)
        dst = self.tmp / "d"
        copied, skipped = aim._sync_store_dir(src, dst)
        self.assertEqual((copied, skipped), (2, 0))
        self.assertEqual((dst / "a" / "x.bin").read_bytes(), b"X" * 10)
        copied2, skipped2 = aim._sync_store_dir(src, dst)
        self.assertEqual((copied2, skipped2), (0, 2))

    def test_verify_detects_content_change(self):
        src = self.tmp / "s"; _write(src / "f.bin", b"AAAA")
        dst = self.tmp / "d"; _write(dst / "f.bin", b"BBBB")
        copied, skipped = aim._sync_store_dir(src, dst, verify=False)
        self.assertEqual((copied, skipped), (0, 1))
        copied, skipped = aim._sync_store_dir(src, dst, verify=True)
        self.assertEqual((copied, skipped), (1, 0))
        self.assertEqual((dst / "f.bin").read_bytes(), b"AAAA")

    def test_missing_src_is_noop(self):
        self.assertEqual(aim._sync_store_dir(self.tmp / "nope", self.tmp / "d"), (0, 0))


import io
import json
from contextlib import redirect_stdout


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]
        store = Path(self.config["roots"][0]["path"]) / "store" / "asr" / "model" / "m1"
        _write(store / "model.safetensors", b"W" * 100)
        _write(store / "config.json", b"{}")
        self.reg = aim.Registry()
        self.reg.models = [aim.ModelEntry(
            id="m1", native_cas=False, category="asr/model",
            canonical={"root": "primary", "path": "store/asr/model/m1"},
            storage={"class": "managed-hf", "store_path": "store/asr/model/m1", "shims": []})]

    def test_backup_mirrors_store_and_writes_manifest(self):
        dest = self.home / "backup"
        aim.op_backup(self.config, self.reg, str(dest))
        self.assertEqual((dest / "store" / "asr" / "model" / "m1" / "model.safetensors").read_bytes(), b"W" * 100)
        man = json.loads((dest / "aim-backup.json").read_text())
        self.assertEqual(man["aim_backup_version"], 1)
        self.assertEqual(len(man["models"]), 1)
        self.assertEqual(man["models"][0]["id"], "m1")
        self.assertTrue(any(sf["path"].endswith("model.safetensors") for sf in man["store_files"]))

    def test_backup_idempotent(self):
        dest = self.home / "backup"
        aim.op_backup(self.config, self.reg, str(dest))
        copied, _ = aim._sync_store_dir(Path(self.config["roots"][0]["path"]) / "store", dest / "store")
        self.assertEqual(copied, 0)

    def test_backup_warns_uningested_native(self):
        self.reg.models.append(aim.ModelEntry(id="nat", native_cas=True))
        buf = io.StringIO()
        with redirect_stdout(buf):
            aim.op_backup(self.config, self.reg, str(self.home / "b2"))
        self.assertIn("not ingested", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
