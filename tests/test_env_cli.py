import unittest
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import aim


def _detector(home):
    return aim.EnvDetector(home=home, rc_files=[], shell_value=lambda v: None)


class EnvShowTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_env_show_json(self):
        cfg = aim.default_config()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = aim.op_env_show(cfg, detector=_detector(self.home), json_output=True)
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        names = {r["name"] for r in data["env"]}
        self.assertIn("HF_HOME", names)
        self.assertIn("huggingface", data["cache_dirs"])

    def test_env_show_masks_secrets(self):
        cfg = aim.default_config()
        det = aim.EnvDetector(home=self.home, rc_files=[],
                              shell_value=lambda v: "tok" if v == "CIVITAI_API_TOKEN" else None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            aim.op_env_show(cfg, detector=det, json_output=False)
        out = buf.getvalue()
        self.assertNotIn("tok", out)
        self.assertIn("set (****)", out)
        self.assertIn("civitai", out)  # source_key column shown

    def test_env_path(self):
        cfg = aim.default_config()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = aim.op_env_path(cfg, "huggingface", detector=_detector(self.home))
        self.assertEqual(rc, 0)
        self.assertIn(".cache/huggingface/hub", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
