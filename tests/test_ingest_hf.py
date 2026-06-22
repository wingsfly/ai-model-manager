import unittest
import os
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def make_hf_cache(home: Path, org="Org", repo="Model", commit="abc123",
                  files=(("config.json", b'{"a":1}'), ("model.safetensors", b"WEIGHTS"))):
    repo_dir = home / "hub" / f"models--{org}--{repo}"
    blobs = repo_dir / "blobs"
    snap = repo_dir / "snapshots" / commit
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(commit)
    for i, (name, data) in enumerate(files):
        blob = blobs / f"sha{i}"
        blob.write_bytes(data)
        os.symlink(os.path.relpath(blob, snap), snap / name)
    return repo_dir


class HFReadNativeTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_read_native_lists_real_files_and_commit(self):
        repo_dir = make_hf_cache(self.home)
        info = aim._hf_read_native(repo_dir)
        self.assertEqual(info["repo_id"], "Org/Model")
        self.assertEqual(info["commit"], "abc123")
        names = sorted(f["name"] for f in info["files"])
        self.assertEqual(names, ["config.json", "model.safetensors"])
        weights = next(f for f in info["files"] if f["name"] == "model.safetensors")
        self.assertEqual(Path(weights["real_path"]).read_bytes(), b"WEIGHTS")

    def test_read_native_fallback_when_no_refs(self):
        repo_dir = make_hf_cache(self.home, commit="zzz999")
        (repo_dir / "refs" / "main").unlink()  # force fallback
        info = aim._hf_read_native(repo_dir)
        self.assertEqual(info["commit"], "zzz999")               # recovered from snapshot dir name
        names = sorted(f["name"] for f in info["files"])
        self.assertEqual(names, ["config.json", "model.safetensors"])  # NOT hash-prefixed


class IngestToStoreTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_copies_flat_no_doubling(self):
        src = self.home / "src"
        _write(src / "config.json", b'{"a":1}')
        _write(src / "model.safetensors", b"W" * 100)
        files = [{"name": "config.json", "real_path": str(src / "config.json"), "size": 7},
                 {"name": "model.safetensors", "real_path": str(src / "model.safetensors"), "size": 100}]
        dest = self.home / "store" / "asr" / "model" / "m1"
        total = aim._ingest_to_store(files, dest)
        self.assertEqual(total, 107)
        self.assertEqual((dest / "model.safetensors").read_bytes(), b"W" * 100)
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["config.json", "model.safetensors"])

    def test_nested_name_preserved(self):
        src = self.home / "src"
        _write(src / "sub" / "f.bin", b"z")
        files = [{"name": "sub/f.bin", "real_path": str(src / "sub" / "f.bin"), "size": 1}]
        dest = self.home / "store" / "x"
        aim._ingest_to_store(files, dest)
        self.assertTrue((dest / "sub" / "f.bin").exists())


class HFIngestTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.root = aim.StorageRoot(id="primary", path=str(self.home / "AI"))
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]

    def _entry_for(self, repo_dir):
        return aim.ModelEntry(id="hf-org-model", name="Org/Model", native_cas=True,
                              source={"type": "huggingface", "repo_id": "Org/Model"},
                              category="asr/model", canonical={"root": "primary", "path": str(repo_dir)})

    def test_ingest_hf_end_to_end(self):
        repo_dir = make_hf_cache(self.home, files=(("config.json", b'{"a":1}'),
                                                   ("model.safetensors", b"W" * 50)))
        reg = aim.Registry()
        reg.models = [self._entry_for(repo_dir)]
        ok = aim.op_ingest(self.config, reg, "hf-org-model", registry_save=False)
        self.assertTrue(ok)
        e = reg.find("hf-org-model")
        self.assertFalse(e.native_cas)
        self.assertEqual(e.storage["class"], "managed-hf")
        store = Path(self.root.path) / e.storage["store_path"]
        self.assertEqual((store / "model.safetensors").read_bytes(), b"W" * 50)
        link = repo_dir / "snapshots" / "abc123" / "model.safetensors"
        self.assertEqual(link.resolve().read_bytes(), b"W" * 50)
        total = sum(p.stat().st_size for p in store.rglob("*") if p.is_file() and not p.is_symlink())
        self.assertEqual(total, 50 + 7)

    def test_ingest_dry_run_changes_nothing(self):
        repo_dir = make_hf_cache(self.home)
        reg = aim.Registry()
        reg.models = [self._entry_for(repo_dir)]
        aim.op_ingest(self.config, reg, "hf-org-model", dry_run=True, registry_save=False)
        self.assertTrue((repo_dir / "blobs").exists())
        self.assertEqual(reg.find("hf-org-model").storage, {})


class HFBuildShimTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_shim_snapshots_symlink_into_store(self):
        store = self.home / "store" / "asr" / "model" / "m1"
        _write(store / "config.json", b'{"a":1}')
        _write(store / "model.safetensors", b"WEIGHTS")
        repo_dir = self.home / "hub" / "models--Org--Model"
        files = [{"name": "config.json"}, {"name": "model.safetensors"}]
        aim._hf_build_shim(repo_dir, store, commit="abc123", files=files)
        self.assertEqual((repo_dir / "refs" / "main").read_text(), "abc123")
        link = repo_dir / "snapshots" / "abc123" / "model.safetensors"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), (store / "model.safetensors").resolve())
        self.assertEqual(link.read_bytes(), b"WEIGHTS")
        self.assertFalse((repo_dir / "blobs").exists())


if __name__ == "__main__":
    unittest.main()
