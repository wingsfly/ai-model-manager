import unittest
from pathlib import Path
import aim


class _R:
    """Minimal fake subprocess.CompletedProcess (only .stdout is read)."""
    def __init__(self, out=""):
        self.stdout = out


class DetectOllamaModelsTests(unittest.TestCase):
    def _patch(self, fn):
        self._orig = aim.subprocess.run
        aim.subprocess.run = fn

    def tearDown(self):
        if hasattr(self, "_orig"):
            aim.subprocess.run = self._orig

    def test_from_launchctl(self):
        self._patch(lambda cmd, **k: _R("/srv/ollama/models\n")
                    if cmd[:2] == ["launchctl", "getenv"] else _R(""))
        self.assertEqual(aim._detect_ollama_models(), "/srv/ollama/models")

    def test_from_process_env_when_launchctl_empty(self):
        def fr(cmd, **k):
            if cmd[:2] == ["launchctl", "getenv"]:
                return _R("")            # launchctl has nothing
            if cmd[0] == "pgrep":
                return _R("123\n")
            if cmd[0] == "ps":
                return _R("hjma 123 /Applications/Ollama.app OLLAMA_MODELS=/app/ollama/models OLLAMA_HOST=0.0.0.0")
            return _R("")
        self._patch(fr)
        self.assertEqual(aim._detect_ollama_models(), "/app/ollama/models")

    def test_none_when_absent_everywhere(self):
        self._patch(lambda cmd, **k: _R(""))
        self.assertIsNone(aim._detect_ollama_models())


class ResolveOllamaToolFallbackTests(unittest.TestCase):
    def test_resolve_uses_builtin_tool_fallback_when_shell_unset(self):
        orig = aim.subprocess.run
        aim.subprocess.run = lambda cmd, **k: _R("/data/ollama/models\n") \
            if cmd[:2] == ["launchctl", "getenv"] else _R("")
        try:
            d = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
            entry = next(e for e in aim.SOURCES["ollama"]["env"] if e["name"] == "OLLAMA_MODELS")
            r = d.resolve("ollama", entry)
        finally:
            aim.subprocess.run = orig
        self.assertEqual(r["effective_value"], "/data/ollama/models")
        self.assertEqual(r["source"], "tool")

    def test_shell_value_still_wins_over_tool(self):
        # if OLLAMA_MODELS IS set in the shell, that takes precedence over service detection
        d = aim.EnvDetector(home=Path("/h"), rc_files=[],
                            shell_value=lambda v: "/shell/ollama" if v == "OLLAMA_MODELS" else None,
                            tool_probe=lambda var, e: "/service/ollama")
        entry = next(e for e in aim.SOURCES["ollama"]["env"] if e["name"] == "OLLAMA_MODELS")
        r = d.resolve("ollama", entry)
        self.assertEqual(r["effective_value"], "/shell/ollama")
        self.assertEqual(r["source"], "env")


if __name__ == "__main__":
    unittest.main()
