import unittest
from pathlib import Path
import tempfile
import aim


class EnvDetectorResolveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.home = Path(self.tmp)

    def _detector(self, shell_values=None, rc_files=None):
        return aim.EnvDetector(
            home=self.home,
            rc_files=rc_files if rc_files is not None else [],
            shell_value=lambda v: (shell_values or {}).get(v),
        )

    def test_resolve_from_login_shell(self):
        d = self._detector(shell_values={"HF_HOME": "/data/hf"})
        entry = {"name": "HF_HOME", "detect": ["env", "rc", "tool"], "default": "~/.cache/huggingface"}
        r = d.resolve("huggingface", entry)
        self.assertEqual(r["effective_value"], "/data/hf")
        self.assertEqual(r["source"], "env")
        self.assertEqual(r["status"], "ok")

    def test_resolve_unset_falls_back_to_default(self):
        d = self._detector(shell_values={})
        entry = {"name": "HF_HOME", "detect": ["env", "rc"], "default": "~/.cache/huggingface"}
        r = d.resolve("huggingface", entry)
        self.assertEqual(r["status"], "unset")
        self.assertEqual(r["effective_value"], str(self.home / ".cache/huggingface"))
        self.assertEqual(r["source"], "default")

    def test_scan_rc_finds_export_and_conflict(self):
        rc1 = self.home / ".zshrc"
        rc1.write_text('export HF_HOME=/a/hf\n')
        rc2 = self.home / ".bashrc"
        rc2.write_text('export HF_HOME="/b/hf"\n')
        d = aim.EnvDetector(home=self.home, rc_files=[rc1, rc2], shell_value=lambda v: None)
        hits = d.scan_rc("HF_HOME")
        self.assertEqual({v for _, v in hits}, {"/a/hf", "/b/hf"})
        entry = {"name": "HF_HOME", "detect": ["env", "rc"], "default": "~/.cache/huggingface"}
        r = d.resolve("huggingface", entry)
        self.assertEqual(r["status"], "conflict")

    def test_scan_rc_parses_fish_set(self):
        rc = self.home / "config.fish"
        rc.write_text('set -gx TORCH_HOME /data/torch\n')
        d = aim.EnvDetector(home=self.home, rc_files=[rc], shell_value=lambda v: None)
        self.assertEqual(d.scan_rc("TORCH_HOME"), [(str(rc), "/data/torch")])


class EnvDetectorParsingTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_scan_rc_strips_inline_comment(self):
        rc = self.home / ".zshrc"
        rc.write_text('export HF_HOME=/data/hf # primary cache\n')
        d = aim.EnvDetector(home=self.home, rc_files=[rc], shell_value=lambda v: None)
        self.assertEqual(d.scan_rc("HF_HOME"), [(str(rc), "/data/hf")])

    def test_scan_rc_ignores_non_exported_fish(self):
        rc = self.home / "config.fish"
        rc.write_text('set TORCH_HOME /a\nset -l TORCH_HOME /b\n')
        d = aim.EnvDetector(home=self.home, rc_files=[rc], shell_value=lambda v: None)
        self.assertEqual(d.scan_rc("TORCH_HOME"), [])

    def test_conflict_not_reported_when_live_env_present(self):
        rc1 = self.home / ".zshrc"; rc1.write_text("export HF_HOME=/a\n")
        rc2 = self.home / ".bashrc"; rc2.write_text("export HF_HOME=/b\n")
        d = aim.EnvDetector(home=self.home, rc_files=[rc1, rc2],
                            shell_value=lambda v: "/live/hf")
        entry = {"name": "HF_HOME", "detect": ["env", "rc"], "default": "~/.cache/huggingface"}
        r = d.resolve("huggingface", entry)
        self.assertEqual(r["source"], "env")
        self.assertEqual(r["effective_value"], "/live/hf")
        self.assertNotEqual(r["status"], "conflict")


class EnvDetectorCacheDirTests(unittest.TestCase):
    def test_cache_dir_uses_subpath(self):
        d = aim.EnvDetector(home=Path("/h"), rc_files=[],
                            shell_value=lambda v: "/data/hf" if v == "HF_HOME" else None)
        self.assertEqual(d.cache_dir("huggingface"), Path("/data/hf/hub"))

    def test_cache_dir_override_wins(self):
        vals = {"HF_HOME": "/data/hf", "HF_HUB_CACHE": "/fast/hub"}
        d = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: vals.get(v))
        self.assertEqual(d.cache_dir("huggingface"), Path("/fast/hub"))

    def test_cache_dir_default_when_unset(self):
        d = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        self.assertEqual(d.cache_dir("huggingface"), Path("/h/.cache/huggingface/hub"))

    def test_report_covers_all_env_entries(self):
        d = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        rows = d.report()
        names = {r["name"] for r in rows}
        self.assertIn("HF_HOME", names)
        self.assertIn("TORCH_HOME", names)
        self.assertIn("MODELSCOPE_CACHE", names)

    def test_report_row_has_source_key_and_origin(self):
        d = aim.EnvDetector(home=Path("/h"), rc_files=[],
                            shell_value=lambda v: "/data/hf" if v == "HF_HOME" else None)
        rows = d.report()
        hf_home = next(r for r in rows if r["name"] == "HF_HOME")
        self.assertEqual(hf_home["source_key"], "huggingface")  # which source
        self.assertEqual(hf_home["source"], "env")              # resolution origin


class EnvDetectorShellCacheTests(unittest.TestCase):
    def test_login_env_spawned_once_and_cached(self):
        calls = {"n": 0}

        class FakeResult:
            stdout = "HF_HOME=/x/hf\nTORCH_HOME=/x/torch\n"

        def fake_run(*a, **k):
            calls["n"] += 1
            return FakeResult()

        orig = aim.subprocess.run
        aim.subprocess.run = fake_run
        try:
            d = aim.EnvDetector(home=Path("/h"), rc_files=[])  # no shell_value → real path, faked subprocess
            self.assertEqual(d.login_shell_value("HF_HOME"), "/x/hf")
            self.assertEqual(d.login_shell_value("TORCH_HOME"), "/x/torch")
            self.assertIsNone(d.login_shell_value("NOPE"))
            self.assertEqual(calls["n"], 1)  # spawned once, then cached
        finally:
            aim.subprocess.run = orig


if __name__ == "__main__":
    unittest.main()
