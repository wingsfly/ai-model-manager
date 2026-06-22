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


class CrossSourceDedupTests(unittest.TestCase):
    """Prove that ingesting the same model file from two sources (HF + MS)
    produces store copies with identical _quick_hash values, so op_dedup
    would hardlink them if they were above its 100 MB threshold.

    NOTE: op_dedup only considers files > 100 MB (min_size = 100_000_000).
    Creating a real 100 MB file in tests is impractical, so this test
    verifies the hash equivalence that is the precondition for dedup.
    The full hardlink code path is exercised by op_dedup's own tests at
    appropriate scale.
    """

    def _write(self, p, data=b"x"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())
        return p

    def test_cross_source_ingest_produces_identical_hashes(self):
        from tests.test_ingest_hf import make_hf_cache

        home = Path(tempfile.mkdtemp())
        cfg = aim.default_config()
        cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]

        weight_data = b"SHAREDWEIGHTS" * 100   # same payload in both sources

        # --- HF source ---
        hf_repo_dir = make_hf_cache(
            home, org="Org", repo="Model",
            files=(("config.json", b"{}"), ("model.safetensors", weight_data)),
        )
        reg1 = aim.Registry()
        reg1.models = [aim.ModelEntry(
            id="hf-org-model", native_cas=True,
            source={"type": "huggingface", "repo_id": "Org/Model"},
            category="asr/model",
            canonical={"root": "primary", "path": str(hf_repo_dir)},
        )]
        buf = io.StringIO()
        with redirect_stdout(buf):
            aim.op_ingest(cfg, reg1, "hf-org-model", registry_save=False)
        hf_entry = reg1.find("hf-org-model")
        hf_store = Path(cfg["roots"][0]["path"]) / hf_entry.storage["store_path"]
        hf_weight = hf_store / "model.safetensors"
        self.assertTrue(hf_weight.exists(), "HF store weight missing")

        # --- MS source: build an inline minimal ModelScope dir ---
        ms_dir = home / "ms_cache" / "models" / "Org" / "AltModel"
        self._write(ms_dir / "config.json", b"{}")
        self._write(ms_dir / "model.safetensors", weight_data)   # identical bytes
        self._write(ms_dir / ".msc", b"meta")

        reg2 = aim.Registry()
        reg2.models = [aim.ModelEntry(
            id="ms-org-model", native_cas=True,
            source={"type": "modelscope", "repo_id": "Org/AltModel"},
            category="asr/model",
            canonical={"root": "primary", "path": str(ms_dir)},
        )]
        with redirect_stdout(io.StringIO()):
            aim.op_ingest(cfg, reg2, "ms-org-model", registry_save=False)
        ms_entry = reg2.find("ms-org-model")
        ms_store = Path(cfg["roots"][0]["path"]) / ms_entry.storage["store_path"]
        ms_weight = ms_store / "model.safetensors"
        self.assertTrue(ms_weight.exists(), "MS store weight missing")

        # Both store copies hold identical content → same _quick_hash
        # → op_dedup would hardlink them once files exceed the 100 MB threshold.
        self.assertEqual(
            aim._quick_hash(hf_weight),
            aim._quick_hash(ms_weight),
            "Store copies from different sources have different hashes; op_dedup cannot merge them",
        )


if __name__ == "__main__":
    unittest.main()
