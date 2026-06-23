import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import aim


class BackupCliParseTests(unittest.TestCase):
    def test_backup_parse(self):
        a = aim.build_parser().parse_args(["backup", "/tmp/bk", "--verify"])
        self.assertEqual(a.command, "backup")
        self.assertEqual(a.dest, "/tmp/bk")
        self.assertTrue(a.verify)

    def test_restore_parse(self):
        a = aim.build_parser().parse_args(["restore", "/tmp/bk", "--root", "ext", "--apply-env"])
        self.assertEqual(a.command, "restore")
        self.assertEqual(a.src, "/tmp/bk")
        self.assertEqual(a.root_id, "ext")
        self.assertTrue(a.apply_env)


class JsonOutputCleanTests(unittest.TestCase):
    def test_backup_json_is_pure_json(self):
        home = Path(tempfile.mkdtemp())
        cfg = aim.default_config(); cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        store = Path(cfg["roots"][0]["path"]) / "store" / "asr" / "model" / "m1"
        store.mkdir(parents=True); (store / "w.bin").write_bytes(b"W" * 10)
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="m1", native_cas=False,
                      canonical={"root": "primary", "path": "store/asr/model/m1"},
                      storage={"class": "managed-flat", "store_path": "store/asr/model/m1", "shims": []}),
                      aim.ModelEntry(id="nat", native_cas=True)]  # would normally trigger a warning
        buf = io.StringIO()
        with redirect_stdout(buf):
            aim.op_backup(cfg, reg, str(home / "bk"), json_output=True)
        json.loads(buf.getvalue())  # must parse cleanly (no warning lines on stdout)

    def test_restore_json_is_pure_json(self):
        home = Path(tempfile.mkdtemp())
        cfg = aim.default_config(); cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        bk = home / "bk"; (bk / "store").mkdir(parents=True)
        (bk / "aim-backup.json").write_text(json.dumps({"aim_backup_version": 1, "models": [], "sources": {}, "env": {}}))
        buf = io.StringIO()
        with redirect_stdout(buf):
            aim.op_restore(cfg, aim.Registry(), str(bk), registry_save=False, json_output=True)
        json.loads(buf.getvalue())  # must parse cleanly (no hint line on stdout)


if __name__ == "__main__":
    unittest.main()
