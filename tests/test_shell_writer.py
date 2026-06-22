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


if __name__ == "__main__":
    unittest.main()
