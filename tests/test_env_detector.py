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


if __name__ == "__main__":
    unittest.main()
