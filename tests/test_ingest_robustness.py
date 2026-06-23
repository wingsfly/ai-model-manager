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


if __name__ == "__main__":
    unittest.main()
