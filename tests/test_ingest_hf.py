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


if __name__ == "__main__":
    unittest.main()
