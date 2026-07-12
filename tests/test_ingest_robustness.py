import unittest
import json
import tempfile
from pathlib import Path
import aim


class OllamaNullLayersTests(unittest.TestCase):
    def test_read_native_handles_null_layers(self):
        # ollama CLOUD stub: manifest has "layers": null (no local blobs)
        root = Path(tempfile.mkdtemp())
        mpath = root / "manifests" / "registry.ollama.ai" / "library" / "gemma3n-cloud" / "latest"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps({"layers": None, "config": None}))
        info = aim._ollama_read_native(mpath, root)
        self.assertIsNone(info["gguf"])
        self.assertEqual(info["small_blobs"], [])

    def test_scan_excludes_cloud_stub_with_null_layers(self):
        root = Path(tempfile.mkdtemp())
        mpath = root / "manifests" / "registry.ollama.ai" / "library" / "cloud-model" / "latest"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps({"layers": None, "config": None}))
        cfg = {"roots": [{"id": "primary", "path": str(root)}],
               "engines": {"ollama": {"enabled": True, "native_cas": True}},
               "sources": {"ollama": {"cache_path": str(root)}}}
        adapter = aim.OllamaAdapter(cfg, aim.StorageRoot(id="primary", path=str(root)))
        self.assertEqual(adapter.scan(), [])


class IngestAllRobustnessTests(unittest.TestCase):
    def test_ingest_all_continues_past_exception(self):
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="good1", native_cas=True),
                      aim.ModelEntry(id="bad", native_cas=True),
                      aim.ModelEntry(id="good2", native_cas=True)]
        calls = []
        orig = aim.op_ingest

        def fake(config, registry, mid, **kw):
            calls.append(mid)
            if mid == "bad":
                raise TypeError("boom")
            return True

        aim.op_ingest = fake
        try:
            done = aim.op_ingest_all({}, reg, dry_run=True, registry_save=False)
        finally:
            aim.op_ingest = orig
        self.assertEqual(calls, ["good1", "bad", "good2"])  # all attempted; one bad model did NOT abort
        self.assertEqual(done, 2)  # two succeeded


class HfReadNativeStaleRefTests(unittest.TestCase):
    def test_read_native_uses_existing_snapshot_when_ref_dangling(self):
        # refs/main points to a commit whose snapshot dir is GONE (e.g. an interrupted download);
        # only a DIFFERENT snapshot actually exists. _hf_read_native must report the REAL snapshot's
        # commit — not the stale ref — else the ingest rebuilds/refs a dangling snapshot and a later
        # HF resolve of the true commit re-fetches / fails "cannot find files".
        root = Path(tempfile.mkdtemp())
        repo = root / "models--Org--Model"
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text("staledeadbeef")   # snapshots/staledeadbeef absent
        snap = repo / "snapshots" / "realcommit00"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text("{}")
        info = aim._hf_read_native(repo)
        self.assertEqual(info["commit"], "realcommit00")       # real snapshot, NOT the stale ref
        self.assertTrue(any(f["name"] == "config.json" for f in info["files"]))


if __name__ == "__main__":
    unittest.main()
