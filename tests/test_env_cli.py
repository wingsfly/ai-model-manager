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

    def test_env_show_json_masks_secrets(self):
        cfg = aim.default_config()
        det = aim.EnvDetector(home=self.home, rc_files=[],
                              shell_value=lambda v: "supertoken" if v == "HF_TOKEN" else None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            aim.op_env_show(cfg, detector=det, json_output=True)
        out = buf.getvalue()
        self.assertNotIn("supertoken", out)
        data = json.loads(out)
        hf_token = next(r for r in data["env"] if r["name"] == "HF_TOKEN")
        self.assertEqual(hf_token["effective_value"], "***")
        self.assertTrue(hf_token["secret"])

    def test_env_path_no_cache_dir(self):
        cfg = aim.default_config()
        rc = aim.op_env_path(cfg, "url", detector=_detector(self.home))
        self.assertEqual(rc, aim.EXIT_FAILED)


class EnvApplyTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def _run(self, cfg, **kw):
        writer = aim.ShellWriter(home=self.home)
        return aim.op_env_apply(cfg, registry=None, writer=writer, home=self.home, **kw)

    def test_apply_writes_env_file_and_wires_rc(self):
        cfg = aim.default_config()
        cfg["sources"]["huggingface"] = {"managed_env": {"HF_ENDPOINT": "https://hf-mirror.com"}}
        rc = self.home / ".zshrc"; rc.write_text("# mine\n")
        rc_code = self._run(cfg, shell="zsh", set_vars=[], service=False, dry_run=False)
        self.assertEqual(rc_code, 0)
        envf = (self.home / ".aim" / "env.sh").read_text()
        self.assertIn('export HF_ENDPOINT="https://hf-mirror.com"', envf)
        self.assertIn(aim.AIM_ENV_BEGIN, rc.read_text())
        self.assertIn("# mine", rc.read_text())

    def test_apply_set_flag_persists_to_config(self):
        cfg = aim.default_config()
        rc = self.home / ".zshrc"; rc.write_text("")
        self._run(cfg, shell="zsh", set_vars=["HF_HUB_ENABLE_HF_TRANSFER=1"],
                  service=False, dry_run=False)
        self.assertEqual(
            cfg["sources"]["huggingface"]["managed_env"]["HF_HUB_ENABLE_HF_TRANSFER"], "1")

    def test_apply_dry_run_writes_nothing(self):
        cfg = aim.default_config()
        cfg["sources"]["huggingface"] = {"managed_env": {"HF_ENDPOINT": "https://x"}}
        rc = self.home / ".zshrc"; rc.write_text("orig\n")
        self._run(cfg, shell="zsh", set_vars=[], service=False, dry_run=True)
        self.assertFalse((self.home / ".aim" / "env.sh").exists())
        self.assertEqual(rc.read_text(), "orig\n")

    def test_apply_set_secret_goes_to_secretstore_not_config(self):
        cfg = aim.default_config()
        (self.home / ".zshrc").write_text("")
        self._run(cfg, shell="zsh", set_vars=["HF_TOKEN=SECRET123"], service=False, dry_run=False)
        self.assertNotIn("HF_TOKEN",
                         cfg.get("sources", {}).get("huggingface", {}).get("managed_env", {}))
        secrets = (self.home / ".aim" / "secrets.env").read_text()
        self.assertIn('export HF_TOKEN="SECRET123"', secrets)
        self.assertNotIn("SECRET123", (self.home / ".aim" / "env.sh").read_text())

    def test_apply_bash_chains_profile_to_bashrc(self):
        cfg = aim.default_config()
        self._run(cfg, shell="bash", set_vars=[], service=False, dry_run=False)
        self.assertIn(aim.AIM_ENV_BEGIN, (self.home / ".bashrc").read_text())
        self.assertIn(".bashrc", (self.home / ".bash_profile").read_text())

    def test_apply_bash_no_duplicate_chain(self):
        cfg = aim.default_config()
        (self.home / ".bash_profile").write_text("source ~/.bashrc\n")
        self._run(cfg, shell="bash", set_vars=[], service=False, dry_run=False)
        self.assertNotIn(aim.AIM_BASH_CHAIN_BEGIN, (self.home / ".bash_profile").read_text())

    def test_apply_service_expands_ollama_tilde(self):
        cfg = aim.default_config()
        buf = io.StringIO()
        with redirect_stdout(buf):
            self._run(cfg, shell="zsh", set_vars=[], service=True, dry_run=False)
        out = buf.getvalue()
        self.assertNotIn('"~/.ollama', out)
        self.assertIn(str(self.home / ".ollama" / "models"), out)


class SourcesCliTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_sources_list_json(self):
        cfg = aim.default_config()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = aim.op_sources_list(cfg, detector=_detector(self.home), json_output=True)
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        keys = {s["key"] for s in data["sources"]}
        self.assertEqual(keys, set(aim.SOURCES))
        hf = next(s for s in data["sources"] if s["key"] == "huggingface")
        self.assertIn("tools", hf)
        self.assertIn("cache_dir", hf)


if __name__ == "__main__":
    unittest.main()
