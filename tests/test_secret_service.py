import unittest
import os
import stat
import tempfile
from pathlib import Path
import aim


class SecretStoreTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.s = aim.SecretStore(home=self.home)

    def test_set_secret_writes_600_file(self):
        self.s.set_secret("CIVITAI_API_TOKEN", "abc123")
        p = self.home / ".aim" / "secrets.env"
        self.assertTrue(p.exists())
        mode = stat.S_IMODE(os.stat(p).st_mode)
        self.assertEqual(mode, 0o600)
        self.assertIn('export CIVITAI_API_TOKEN="abc123"', p.read_text())

    def test_set_secret_replaces_not_duplicates(self):
        self.s.set_secret("CIVITAI_API_TOKEN", "old")
        self.s.set_secret("CIVITAI_API_TOKEN", "new")
        text = (self.home / ".aim" / "secrets.env").read_text()
        self.assertEqual(text.count("CIVITAI_API_TOKEN"), 1)
        self.assertIn("new", text)

    def test_mask(self):
        self.assertEqual(aim.SecretStore.mask(""), "unset")
        self.assertEqual(aim.SecretStore.mask("supersecret"), "set (****)")


class ServiceEnvTests(unittest.TestCase):
    def test_macos_uses_launchctl(self):
        cmds = aim.ServiceEnv.ollama_commands("/data/ollama/models", "Darwin")
        self.assertTrue(any("launchctl setenv OLLAMA_MODELS" in c for c in cmds))

    def test_linux_uses_systemd(self):
        cmds = aim.ServiceEnv.ollama_commands("/data/ollama/models", "Linux")
        self.assertTrue(any("systemd" in c or "systemctl" in c for c in cmds))

    def test_linux_path_with_spaces_preserved(self):
        cmds = aim.ServiceEnv.ollama_commands("/data/AI Models/ollama", "Linux")
        self.assertTrue(any("/data/AI Models/ollama" in c for c in cmds))
        self.assertTrue(any("printf '" in c for c in cmds))  # single-quoted format


if __name__ == "__main__":
    unittest.main()
