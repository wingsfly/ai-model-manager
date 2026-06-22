import unittest
from pathlib import Path
import tempfile
import aim


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.w = aim.ShellWriter(home=Path("/h"))

    def test_render_sh(self):
        out = self.w.render_env_file([("huggingface", "HF_ENDPOINT", "https://hf-mirror.com")], fmt="sh")
        self.assertIn("# --- huggingface ---", out)
        self.assertIn('export HF_ENDPOINT="https://hf-mirror.com"', out)
        self.assertIn('[ -f "$HOME/.aim/secrets.env" ]', out)

    def test_render_fish(self):
        out = self.w.render_env_file([("pytorch-hub", "TORCH_HOME", "/data/torch")], fmt="fish")
        self.assertIn('set -gx TORCH_HOME "/data/torch"', out)
        self.assertIn('source "$HOME/.aim/secrets.env"', out)

    def test_source_block_markers(self):
        b = self.w.source_block("sh")
        self.assertIn(aim.AIM_ENV_BEGIN, b)
        self.assertIn(aim.AIM_ENV_END, b)
        self.assertIn('. "$HOME/.aim/env.sh"', b)


class WireRcTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.w = aim.ShellWriter(home=self.home)

    def test_append_then_idempotent_replace(self):
        rc = self.home / ".zshrc"
        rc.write_text("# my config\nexport PATH=$PATH:/x\n")
        r1 = self.w.wire_rc(rc, fmt="sh")
        self.assertEqual(r1["action"], "append")
        first = rc.read_text()
        self.assertIn(aim.AIM_ENV_BEGIN, first)
        self.assertIn("# my config", first)
        r2 = self.w.wire_rc(rc, fmt="sh")
        self.assertEqual(r2["action"], "replace")
        self.assertEqual(rc.read_text(), first)

    def test_backup_created_once(self):
        rc = self.home / ".zshrc"
        rc.write_text("orig\n")
        self.w.wire_rc(rc, fmt="sh")
        bak = rc.with_suffix(rc.suffix + ".aim.bak")
        self.assertTrue(bak.exists())
        self.assertEqual(bak.read_text(), "orig\n")

    def test_dry_run_writes_nothing(self):
        rc = self.home / ".zshrc"
        rc.write_text("orig\n")
        r = self.w.wire_rc(rc, fmt="sh", dry_run=True)
        self.assertFalse(r["wrote"])
        self.assertEqual(rc.read_text(), "orig\n")

    def test_target_rc_map(self):
        self.assertEqual(self.w.target_rc("zsh"), (self.home / ".zshrc", "sh"))
        self.assertEqual(self.w.target_rc("fish"), (self.home / ".config/fish/config.fish", "fish"))
        self.assertEqual(self.w.target_rc("bash"), (self.home / ".bashrc", "sh"))
        self.assertEqual(self.w.target_rc("sh"), (self.home / ".profile", "sh"))

    def test_empty_rc_no_leading_blank(self):
        rc = self.home / ".bashrc"  # does not exist yet → empty case
        self.w.wire_rc(rc, fmt="sh")
        text = rc.read_text()
        self.assertFalse(text.startswith("\n"))
        self.assertTrue(text.startswith(aim.AIM_ENV_BEGIN))
        # second run is idempotent
        before = rc.read_text()
        self.w.wire_rc(rc, fmt="sh")
        self.assertEqual(rc.read_text(), before)


if __name__ == "__main__":
    unittest.main()
