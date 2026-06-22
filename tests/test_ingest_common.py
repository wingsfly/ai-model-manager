import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import aim


class StorageFieldTests(unittest.TestCase):
    def test_storage_defaults_empty_and_round_trips(self):
        e = aim.ModelEntry(id="m1")
        self.assertEqual(e.storage, {})
        e.storage = {"class": "managed-hf", "store_path": "store/x", "shims": []}
        d = e.to_dict()
        self.assertEqual(d["storage"]["class"], "managed-hf")
        e2 = aim.ModelEntry.from_dict(d)
        self.assertEqual(e2.storage["store_path"], "store/x")

    def test_from_dict_without_storage_defaults_empty(self):
        e = aim.ModelEntry.from_dict({"id": "m2"})
        self.assertEqual(e.storage, {})


class IngestCliTests(unittest.TestCase):
    def test_parser_accepts_ingest(self):
        args = aim.build_parser().parse_args(["ingest", "m1", "--dry-run"])
        self.assertEqual(args.command, "ingest")
        self.assertEqual(args.model_id, "m1")
        self.assertTrue(args.dry_run)

    def test_parser_accepts_all_native(self):
        args = aim.build_parser().parse_args(["ingest", "--all-native"])
        self.assertTrue(args.all_native)

    def test_ingest_all_native_skips_non_native(self):
        cfg = aim.default_config()
        home = Path(tempfile.mkdtemp())
        cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="managed", native_cas=False)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            n = aim.op_ingest_all(cfg, reg, dry_run=True, registry_save=False)
        self.assertEqual(n, 0)


class ConvertDelegationTests(unittest.TestCase):
    def test_convert_delegates_to_ingest_and_builds_shim(self):
        from tests.test_ingest_hf import make_hf_cache
        home = Path(tempfile.mkdtemp())
        cfg = aim.default_config()
        cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        repo_dir = make_hf_cache(home, files=(("config.json", b"{}"), ("model.safetensors", b"W" * 30)))
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="hf-org-model", native_cas=True,
                      source={"type": "huggingface", "repo_id": "Org/Model"},
                      category="asr/model", canonical={"root": "primary", "path": str(repo_dir)})]
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = aim.op_convert_native_to_store(cfg, reg, "hf-org-model", keep_native=False, json_output=False)
        self.assertTrue(ok)
        e = reg.find("hf-org-model")
        self.assertEqual(e.storage["class"], "managed-hf")
        self.assertFalse((repo_dir / "blobs").exists())
        self.assertIn("deprecated", buf.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
