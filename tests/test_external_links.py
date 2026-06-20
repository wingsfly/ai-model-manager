"""Tests for external-dependency tracking: external_links field + link/unlink/scan
commands + info/verify/migrate integration."""
import shutil
import tempfile
import unittest
from pathlib import Path

import aim


class ExternalLinksTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "AI"
        self.model_dir = self.root / "store" / "asr" / "model" / "m1"
        self.model_dir.mkdir(parents=True)
        (self.model_dir / "w.bin").write_text("x")
        # isolate the registry to the temp dir
        self._orig_home = aim.AIM_HOME
        aim.AIM_HOME = self.tmp / ".aim"
        self.config = {"version": 1, "engines": {},
                       "roots": [{"id": "primary", "path": str(self.root), "priority": 1}]}
        self.reg = aim.Registry()
        self.reg.models = [aim.ModelEntry(
            id="m1", category="asr/model",
            canonical={"root": "primary", "path": "store/asr/model/m1"})]
        self.reg.save()

    def tearDown(self):
        aim.AIM_HOME = self._orig_home
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_field_default_and_backward_compat(self):
        self.assertEqual(aim.ModelEntry(id="x").external_links, [])
        # old registry entry (no external_links key) loads with the default
        self.assertEqual(aim.ModelEntry.from_dict({"id": "y"}).external_links, [])

    def test_serialization_roundtrip(self):
        e = aim.ModelEntry(id="x")
        e.external_links.append({"path": "/p", "consumer": "a", "link_type": "symlink"})
        self.assertEqual(aim.ModelEntry.from_dict(e.to_dict()).external_links, e.external_links)

    def test_link_create_records_and_makes_symlink(self):
        ext = self.tmp / "extlink"
        ok = aim.op_link(self.config, self.reg, "m1", str(ext), consumer="app", create=True)
        self.assertTrue(ok)
        self.assertTrue(ext.is_symlink())
        m = self.reg.find("m1")
        self.assertEqual(len(m.external_links), 1)
        self.assertEqual(m.external_links[0]["consumer"], "app")
        self.assertEqual(m.external_links[0]["path"], str(ext))

    def test_link_dedup_by_path(self):
        ext = self.tmp / "e"
        aim.op_link(self.config, self.reg, "m1", str(ext), consumer="a", link_type="reference")
        aim.op_link(self.config, self.reg, "m1", str(ext), consumer="b", link_type="reference")
        m = self.reg.find("m1")
        self.assertEqual(len(m.external_links), 1)
        self.assertEqual(m.external_links[0]["consumer"], "b")  # updated, not duplicated

    def test_unlink_remove_deletes_symlink(self):
        ext = self.tmp / "extlink"
        aim.op_link(self.config, self.reg, "m1", str(ext), create=True)
        aim.op_unlink(self.config, self.reg, "m1", str(ext), remove=True)
        self.assertEqual(self.reg.find("m1").external_links, [])
        self.assertFalse(ext.exists())

    def test_verify_flags_dangling_external(self):
        aim.op_link(self.config, self.reg, "m1", str(self.tmp / "nope"), link_type="reference")
        issues = aim.op_verify(self.config, self.reg)
        self.assertTrue(any(i.get("error") == "external_missing" for i in issues))

    def test_verify_ok_when_symlink_points_to_canonical(self):
        ext = self.tmp / "good"
        aim.op_link(self.config, self.reg, "m1", str(ext), create=True)
        issues = aim.op_verify(self.config, self.reg)
        self.assertFalse(any(i.get("error", "").startswith("external_") for i in issues))

    def test_scan_registers_symlink_into_root(self):
        cache = self.tmp / "cache"
        cache.mkdir()
        ext = cache / "link"
        ext.symlink_to(self.model_dir)
        aim.op_link_scan(self.config, self.reg, [str(cache)], consumer="scan", apply=True)
        m = self.reg.find("m1")
        self.assertTrue(any(e["path"] == str(ext) for e in m.external_links))

    def test_scan_dry_run_does_not_register(self):
        cache = self.tmp / "cache"
        cache.mkdir()
        (cache / "link").symlink_to(self.model_dir)
        aim.op_link_scan(self.config, self.reg, [str(cache)], apply=False)
        self.assertEqual(self.reg.find("m1").external_links, [])


if __name__ == "__main__":
    unittest.main()
