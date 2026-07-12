import os
import tempfile
import unittest
from pathlib import Path

import aim


def _mk_config(main: Path, ext: Path, ext_mounted: bool = True):
    return {
        "roots": [
            {"id": "primary", "path": str(main), "priority": 1},
            {"id": "backup", "path": str(ext), "priority": 2, "removable": True},
        ],
        "engines": {
            "huggingface": {"enabled": True, "model_dir": "huggingface/hub", "native_cas": True},
        },
    }


class OffloadHelperTests(unittest.TestCase):
    def test_root_available(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            self.assertTrue(aim.root_available(aim.StorageRoot(id="r", path=t)))
            self.assertFalse(aim.root_available(aim.StorageRoot(id="r", path=t + "/nope")))

    def test_is_offloaded(self) -> None:
        e = aim.ModelEntry(id="m")
        self.assertFalse(aim.is_offloaded(e))
        e.offload = {"status": "offline", "root": "backup"}
        self.assertTrue(aim.is_offloaded(e))

    def test_source_to_download_str(self) -> None:
        self.assertEqual(aim._source_to_download_str({"type": "huggingface", "repo_id": "a/b"}), "hf:a/b")
        self.assertEqual(aim._source_to_download_str({"type": "modelscope", "repo_id": "a/b"}), "ms:a/b")
        self.assertEqual(aim._source_to_download_str({"type": "url", "url": "http://x/y"}), "url:http://x/y")

    def test_from_dict_backward_compat(self) -> None:
        # a registry entry written before the offload field must still load
        e = aim.ModelEntry.from_dict({"id": "m", "category": "llm/chat"})
        self.assertEqual(e.offload, {})


class OffloadRestoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.main = base / "main"
        self.ext = base / "ext"
        self.main.mkdir()
        self.ext.mkdir()
        self._orig_home = aim.AIM_HOME
        aim.AIM_HOME = base / ".aim"
        aim.AIM_HOME.mkdir()
        self.config = _mk_config(self.main, self.ext)
        self.reg = aim.Registry()

    def tearDown(self) -> None:
        aim.AIM_HOME = self._orig_home
        self.tmp.cleanup()

    def _managed_model(self, mid="m1", cat="llm/chat", payload=b"x" * 1000):
        store = self.main / "store" / cat / mid
        store.mkdir(parents=True)
        (store / "w.bin").write_bytes(payload)
        e = aim.ModelEntry(id=mid, category=cat, size_bytes=len(payload),
                           source={"type": "huggingface", "repo_id": f"org/{mid}"},
                           canonical={"root": "primary", "path": f"store/{cat}/{mid}"})
        self.reg.models = [e]
        return e, store

    def test_offload_managed_moves_dir_and_marks_offline(self) -> None:
        e, store = self._managed_model()
        ok = aim.op_offload(self.config, self.reg, "m1", "backup")
        self.assertTrue(ok)
        self.assertFalse(store.exists(), "source dir should be gone from main")
        self.assertEqual((self.ext / "store/llm/chat/m1/w.bin").read_bytes(), b"x" * 1000)
        self.assertTrue(aim.is_offloaded(e))
        self.assertEqual(e.offload["root"], "backup")
        self.assertEqual(e.offload["source"], "hf:org/m1")

    def test_restore_managed_moves_back_and_clears(self) -> None:
        e, store = self._managed_model()
        aim.op_offload(self.config, self.reg, "m1", "backup")
        ok = aim.op_offload_restore(self.config, self.reg, "m1")
        self.assertTrue(ok)
        self.assertEqual((self.main / "store/llm/chat/m1/w.bin").read_bytes(), b"x" * 1000)
        self.assertFalse(aim.is_offloaded(e))
        self.assertFalse((self.ext / "store/llm/chat/m1").exists())

    def test_offload_removes_provision_links(self) -> None:
        e, store = self._managed_model()
        eng_dir = self.main / "comfyui" / "models"
        eng_dir.mkdir(parents=True)
        link = eng_dir / "w.bin"
        os.symlink(store / "w.bin", link)
        e.provisions = [{"engine": "comfyui", "target": "comfyui/models/w.bin", "link_type": "symlink"}]
        aim.op_offload(self.config, self.reg, "m1", "backup")
        self.assertFalse(link.exists() or link.is_symlink(), "dangling provision link should be removed")

    def test_offload_to_unmounted_root_fails(self) -> None:
        self._managed_model()
        self.config["roots"][1]["path"] = str(self.ext / "not-mounted")
        ok = aim.op_offload(self.config, self.reg, "m1", "backup")
        self.assertFalse(ok)

    def test_restore_when_unmounted_is_noop_with_message(self) -> None:
        e, _ = self._managed_model()
        aim.op_offload(self.config, self.reg, "m1", "backup")
        # simulate the drive being unplugged
        self.config["roots"][1]["path"] = str(self.ext / "gone")
        ok = aim.op_offload_restore(self.config, self.reg, "m1")
        self.assertFalse(ok)
        self.assertTrue(aim.is_offloaded(e), "entry must remain offloaded, not corrupted")

    def test_verify_skips_offloaded(self) -> None:
        e, store = self._managed_model()
        aim.op_offload(self.config, self.reg, "m1", "backup")
        e.storage = {"shims": [{"tool": "huggingface", "kind": "huggingface-cas",
                                 "reconstruct": {"repo_id": "org/m1"}}]}
        issues = aim.op_verify(self.config, self.reg)
        self.assertFalse([i for i in issues if i.get("model") == "m1"],
                         "offloaded model must not be reported as missing")

    def test_offload_native_cas_ingests_to_external(self) -> None:
        # build a minimal HF cache repo
        commit = "deadbeef"
        repo = self.main / "huggingface" / "hub" / "models--org--nat"
        (repo / "blobs").mkdir(parents=True)
        (repo / "snapshots" / commit).mkdir(parents=True)
        (repo / "refs").mkdir(parents=True)
        (repo / "refs" / "main").write_text(commit)
        blob = repo / "blobs" / "sha0"
        blob.write_bytes(b"N" * 2048)
        os.symlink(os.path.relpath(blob, repo / "snapshots" / commit),
                   repo / "snapshots" / commit / "model.bin")
        e = aim.ModelEntry(id="nat", category="llm/chat", native_cas=True, size_bytes=2048,
                           source={"type": "huggingface", "repo_id": "org/nat"},
                           canonical={"root": "primary", "path": "huggingface/hub/models--org--nat"})
        self.reg.models = [e]
        ok = aim.op_offload(self.config, self.reg, "nat", "backup")
        self.assertTrue(ok)
        self.assertEqual((self.ext / "store/llm/chat/nat/model.bin").read_bytes(), b"N" * 2048)
        self.assertFalse(e.native_cas, "should be managed after ingest")
        self.assertTrue(aim.is_offloaded(e))
        self.assertFalse((repo / "blobs").exists(), "cache blobs removed by default")
        self.assertEqual(e.storage["shims"][0]["kind"], "hf-cas")
        self.assertNotIn("location", e.storage["shims"][0],
                         "offline annotations should not require a mounted local shim")
        self.assertTrue(aim.op_offload_restore(self.config, self.reg, "nat"))
        shim = e.storage["shims"][0]
        self.assertEqual(shim["kind"], "hf-cas")
        self.assertEqual(shim["location"], str(repo))
        self.assertFalse([i for i in aim.op_verify(self.config, self.reg)
                          if i.get("model") == "nat"])

    def test_offload_list(self) -> None:
        e, _ = self._managed_model()
        aim.op_offload(self.config, self.reg, "m1", "backup")
        rows = aim.op_offload_list(self.config, self.reg, json_output=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "m1")
        self.assertTrue(rows[0]["mounted"])


if __name__ == "__main__":
    unittest.main()
