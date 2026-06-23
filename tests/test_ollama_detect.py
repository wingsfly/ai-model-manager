import unittest
from pathlib import Path
import aim


class _R:
    """Minimal fake subprocess.CompletedProcess (only .stdout is read)."""
    def __init__(self, out=""):
        self.stdout = out


# ── pure parsers (portable, no subprocess / no /proc) ─────────────────────────

class ParseEnvTests(unittest.TestCase):
    def test_token_line_finds_value(self):
        line = "hjma 123 /Applications/Ollama.app OLLAMA_MODELS=/x/y OLLAMA_HOST=0.0.0.0"
        self.assertEqual(aim._parse_env_token_line(line, "OLLAMA_MODELS"), "/x/y")

    def test_token_line_absent_or_empty(self):
        self.assertIsNone(aim._parse_env_token_line("PATH=/usr/bin OTHER=z", "OLLAMA_MODELS"))
        self.assertIsNone(aim._parse_env_token_line("OLLAMA_MODELS=", "OLLAMA_MODELS"))

    def test_environ_bytes_is_space_safe(self):
        data = b"PATH=/usr/bin\x00OLLAMA_MODELS=/data/AI Models/ollama\x00HOME=/h\x00"
        self.assertEqual(aim._parse_environ_bytes(data, "OLLAMA_MODELS"), "/data/AI Models/ollama")

    def test_environ_bytes_absent(self):
        self.assertIsNone(aim._parse_environ_bytes(b"PATH=/usr/bin\x00", "OLLAMA_MODELS"))


# ── _detect_ollama_models orchestration (mock subprocess + _pid_env_value) ─────

class DetectOllamaModelsTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "_orig_run"):
            aim.subprocess.run = self._orig_run
        if hasattr(self, "_orig_pid"):
            aim._pid_env_value = self._orig_pid

    def _patch_run(self, fn):
        self._orig_run = aim.subprocess.run
        aim.subprocess.run = fn

    def _patch_pid(self, fn):
        self._orig_pid = aim._pid_env_value
        aim._pid_env_value = fn

    def test_from_launchctl(self):
        self._patch_run(lambda cmd, **k: _R("/srv/ollama/models\n")
                        if cmd[:2] == ["launchctl", "getenv"] else _R(""))
        self.assertEqual(aim._detect_ollama_models(), "/srv/ollama/models")

    def test_from_process_when_launchctl_empty(self):
        self._patch_run(lambda cmd, **k: _R("123\n456\n") if cmd[0] == "pgrep" else _R(""))
        self._patch_pid(lambda pid, var: "/app/ollama/models" if pid == "456" else None)
        self.assertEqual(aim._detect_ollama_models(), "/app/ollama/models")

    def test_none_when_absent_everywhere(self):
        self._patch_run(lambda cmd, **k: _R("123\n") if cmd[0] == "pgrep" else _R(""))
        self._patch_pid(lambda pid, var: None)
        self.assertIsNone(aim._detect_ollama_models())

    def test_never_raises_on_subprocess_failure(self):
        def boom(cmd, **k):
            raise OSError("not found")
        self._patch_run(boom)
        self.assertIsNone(aim._detect_ollama_models())  # must swallow, return None


# ── resolve() wiring ──────────────────────────────────────────────────────────

class ResolveOllamaToolFallbackTests(unittest.TestCase):
    def test_resolve_uses_builtin_tool_fallback_when_shell_unset(self):
        orig = aim._detect_ollama_models
        aim._detect_ollama_models = lambda: "/data/ollama/models"
        try:
            d = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
            entry = next(e for e in aim.SOURCES["ollama"]["env"] if e["name"] == "OLLAMA_MODELS")
            r = d.resolve("ollama", entry)
        finally:
            aim._detect_ollama_models = orig
        self.assertEqual(r["effective_value"], "/data/ollama/models")
        self.assertEqual(r["source"], "tool")

    def test_shell_value_still_wins_over_tool(self):
        d = aim.EnvDetector(home=Path("/h"), rc_files=[],
                            shell_value=lambda v: "/shell/ollama" if v == "OLLAMA_MODELS" else None,
                            tool_probe=lambda var, e: "/service/ollama")
        entry = next(e for e in aim.SOURCES["ollama"]["env"] if e["name"] == "OLLAMA_MODELS")
        r = d.resolve("ollama", entry)
        self.assertEqual(r["effective_value"], "/shell/ollama")
        self.assertEqual(r["source"], "env")

    def test_builtin_probe_returns_none_for_non_ollama_vars(self):
        # HF_HOME / HF_TOKEN / TORCH_HOME also list "tool" in detect; builtin must not touch them.
        self.assertIsNone(aim._builtin_tool_probe("HF_HOME", {}))
        self.assertIsNone(aim._builtin_tool_probe("HF_TOKEN", {}))
        self.assertIsNone(aim._builtin_tool_probe("TORCH_HOME", {}))


if __name__ == "__main__":
    unittest.main()
