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


if __name__ == "__main__":
    unittest.main()
