import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import aim


class AssumeYesTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("AIM_ASSUME_YES", None)

    def tearDown(self):
        os.environ.pop("AIM_ASSUME_YES", None)
        if self._saved is not None:
            os.environ["AIM_ASSUME_YES"] = self._saved

    def test_flag_true(self):
        self.assertTrue(aim._assume_yes({}, True))

    def test_all_false(self):
        self.assertFalse(aim._assume_yes({}, False))

    def test_env_truthy_values(self):
        for v in ("1", "true", "YES", "on", "y", "True"):
            os.environ["AIM_ASSUME_YES"] = v
            self.assertTrue(aim._assume_yes({}, False), v)

    def test_env_falsy_values(self):
        for v in ("0", "false", "", "no", "off"):
            os.environ["AIM_ASSUME_YES"] = v
            self.assertFalse(aim._assume_yes({}, False), v)

    def test_config_opt_in(self):
        self.assertTrue(aim._assume_yes({"defaults": {"auto_install_backend": True}}, False))

    def test_config_default_false(self):
        self.assertFalse(aim._assume_yes({"defaults": {}}, False))


class EnsureBackendTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("AIM_ASSUME_YES", None)
        self._orig_check = aim._check_backend_available
        aim._check_backend_available = lambda t, c: False  # backend always missing
        self._orig_run = aim.subprocess.run
        self.cfg = {"roots": [{"path": "/x"}]}

    def tearDown(self):
        aim._check_backend_available = self._orig_check
        aim.subprocess.run = self._orig_run
        os.environ.pop("AIM_ASSUME_YES", None)
        if self._saved is not None:
            os.environ["AIM_ASSUME_YES"] = self._saved

    def _record_run(self, calls):
        def run(cmd, **kw):
            calls.append(cmd)
            return mock.Mock(returncode=0)
        return run

    @staticmethod
    def _boom_input(*a, **k):
        raise AssertionError("input() must not be called in non-interactive mode")

    def test_non_tty_no_optin_clear_error_no_prompt(self):
        calls = []
        aim.subprocess.run = self._record_run(calls)
        buf = io.StringIO()
        with mock.patch("builtins.input", self._boom_input), \
             mock.patch.object(sys.stdin, "isatty", return_value=False), \
             redirect_stdout(io.StringIO()), redirect_stderr(buf):
            ok, err = aim._ensure_backend("modelscope", self.cfg, json_output=False, auto_confirm=False)
        self.assertFalse(ok)
        self.assertEqual(calls, [])  # no install attempted without opt-in
        msg = buf.getvalue().lower()
        self.assertIn("modelscope", msg)
        self.assertTrue("aim_assume_yes" in msg or "-y" in msg)  # tells caller how to enable

    def test_env_optin_installs_without_prompt(self):
        os.environ["AIM_ASSUME_YES"] = "1"
        calls = []
        aim.subprocess.run = self._record_run(calls)
        with mock.patch("builtins.input", self._boom_input), \
             mock.patch.object(sys.stdin, "isatty", return_value=False), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            aim._ensure_backend("modelscope", self.cfg, json_output=False, auto_confirm=False)
        self.assertGreaterEqual(len(calls), 1)  # install attempted

    def test_config_optin_installs_without_prompt(self):
        calls = []
        aim.subprocess.run = self._record_run(calls)
        cfg = {"roots": [{"path": "/x"}], "defaults": {"auto_install_backend": True}}
        with mock.patch("builtins.input", self._boom_input), \
             mock.patch.object(sys.stdin, "isatty", return_value=False), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            aim._ensure_backend("modelscope", cfg, json_output=False, auto_confirm=False)
        self.assertGreaterEqual(len(calls), 1)

    def test_json_no_optin_unchanged(self):
        ok, err = aim._ensure_backend("modelscope", self.cfg, json_output=True, auto_confirm=False)
        self.assertFalse(ok)
        self.assertIn("BACKEND_NOT_FOUND", err)

    def test_interactive_tty_yes_installs(self):
        calls = []
        aim.subprocess.run = self._record_run(calls)
        with mock.patch("builtins.input", lambda *a, **k: "y"), \
             mock.patch.object(sys.stdin, "isatty", return_value=True), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            aim._ensure_backend("modelscope", self.cfg, json_output=False, auto_confirm=False)
        self.assertGreaterEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
