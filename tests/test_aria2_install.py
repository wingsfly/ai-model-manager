"""_try_install_aria2: best-effort aria2c install with graceful fallback.

hfd.sh hard-requires aria2c; when it's missing we try to install it (no-sudo managers first),
and callers fall back to the pure-Python `hf` CLI when it still can't be installed — so HF
downloads keep working on boxes without aria2c (e.g. the DGX before aria2c was added).
"""
import unittest
from unittest import mock

import aim


class TryInstallAria2Tests(unittest.TestCase):
    def test_returns_true_when_already_present(self):
        # aria2c already on PATH → short-circuit True, no install attempted.
        with mock.patch("aim.shutil.which", return_value="/usr/bin/aria2c") as which, \
             mock.patch("aim.subprocess.run") as run:
            self.assertTrue(aim._try_install_aria2())
            which.assert_called_once_with("aria2c")
            run.assert_not_called()

    def test_returns_false_when_absent_and_no_managers(self):
        # aria2c absent and no package manager present → no attempts → False (caller falls back).
        with mock.patch("aim.shutil.which", return_value=None), \
             mock.patch("aim.subprocess.run") as run:
            self.assertFalse(aim._try_install_aria2())
            run.assert_not_called()

    def test_installs_via_available_manager(self):
        # aria2c absent initially; brew present; after a successful install it appears → True.
        seen = {"aria2c": 0}

        def fake_which(name):
            if name == "aria2c":
                seen["aria2c"] += 1
                return None if seen["aria2c"] == 1 else "/usr/local/bin/aria2c"
            return "/usr/local/bin/brew" if name == "brew" else None

        with mock.patch("aim.shutil.which", side_effect=fake_which), \
             mock.patch("aim.subprocess.run") as run:
            self.assertTrue(aim._try_install_aria2())
            run.assert_called_once()
            self.assertEqual(run.call_args[0][0][:2], ["brew", "install"])


if __name__ == "__main__":
    unittest.main()
