import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AIM = ROOT / "aim.py"


def _write_exec(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class DownloadE2ETests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.bin = Path(self.tmp.name) / "bin"
        self.home.mkdir(parents=True, exist_ok=True)
        self.bin.mkdir(parents=True, exist_ok=True)
        self.env = os.environ.copy()
        self.env["HOME"] = str(self.home)
        self.env["PATH"] = f"{self.bin}:/usr/bin:/bin"
        self.py = sys.executable

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, *args: str, check: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.py, str(AIM), *args],
            cwd=str(ROOT),
            env=self.env,
            text=True,
            capture_output=True,
            check=check,
        )

    def _install_wget_success(self) -> None:
        _write_exec(
            self.bin / "wget",
            """#!/bin/sh
set -eu
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -O) out="$2"; shift 2 ;;
    -c) shift ;;
    --timeout|--connect-timeout|--tries|--limit-rate) shift 2 ;;
    -e) shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$(dirname "$out")"
dd if=/dev/zero of="$out" bs=1024 count=1 status=none
echo "Length: 1024 (1.0K) [application/octet-stream]" 1>&2
echo "100% [==================>] 1,024  1.0MB/s  eta 0s" 1>&2
exit 0
""",
        )

    def _install_wget_slow(self) -> None:
        _write_exec(
            self.bin / "wget",
            """#!/bin/sh
set -eu
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    -O) out="$2"; shift 2 ;;
    -c) shift ;;
    --timeout|--connect-timeout|--tries|--limit-rate) shift 2 ;;
    -e) shift 2 ;;
    *) shift ;;
  esac
done
mkdir -p "$(dirname "$out")"
trap 'exit 143' TERM INT
i=0
while [ "$i" -lt 50 ]; do
  i=$((i+1))
  dd if=/dev/zero of="$out" bs=128 count=1 seek=$((i-1)) conv=notrunc status=none
  pct=$((i*2))
  [ "$pct" -gt 99 ] && pct=99
  echo "$pct% [====>] $((i*128))  64KB/s  eta 9s" 1>&2
  sleep 0.2
done
exit 0
""",
        )

    def test_url_download_success_and_category_path(self) -> None:
        self._install_wget_success()
        p = self._run(
            "download",
            "url:http://example.com/model.bin",
            "--name",
            "m-ok",
            "--category",
            "llm/chat",
            "--json",
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = [json.loads(x) for x in p.stdout.strip().splitlines() if x.strip()]
        summary = lines[-1]
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["model_id"], "m-ok")
        self.assertEqual(summary["category"], "llm/chat")
        self.assertTrue(summary["path"].endswith("/AI/store/llm/chat/m-ok"))
        self.assertTrue((Path(summary["path"]) / "model.bin").exists())

    def test_explicit_path_overrides_category(self) -> None:
        self._install_wget_success()
        explicit = self.home / "custom" / "m-explicit"
        p = self._run(
            "download",
            "url:http://example.com/model.bin",
            "--name",
            "m-explicit",
            "--category",
            "llm/chat",
            "--path",
            str(explicit),
            "--json",
            "--force-redownload",
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        summary = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(summary["placement_mode"], "explicit")
        self.assertEqual(Path(summary["path"]).resolve(), explicit.resolve())
        self.assertTrue((explicit / "model.bin").exists())

    def test_idempotent_already_exists(self) -> None:
        self._install_wget_success()
        p1 = self._run(
            "download",
            "url:http://example.com/model.bin",
            "--name",
            "m-idem",
            "--category",
            "llm/chat",
            "--json",
        )
        self.assertEqual(p1.returncode, 0, p1.stderr)
        p2 = self._run(
            "download",
            "url:http://example.com/model.bin",
            "--name",
            "m-idem",
            "--category",
            "llm/chat",
            "--json",
        )
        self.assertEqual(p2.returncode, 0, p2.stderr)
        s2 = json.loads(p2.stdout.strip().splitlines()[-1])
        self.assertEqual(s2["status"], "already_exists")

    def test_no_progress_outputs_summary_only(self) -> None:
        self._install_wget_success()
        p = self._run(
            "download",
            "url:http://example.com/model.bin",
            "--name",
            "m-noprog",
            "--category",
            "llm/chat",
            "--json",
            "--no-progress",
            "--force-redownload",
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        lines = [x for x in p.stdout.strip().splitlines() if x.strip()]
        self.assertEqual(len(lines), 1)
        summary = json.loads(lines[0])
        self.assertIn(summary["status"], {"completed", "already_exists"})

    def test_no_resume_removes_resume_flag_from_backend_command(self) -> None:
        self._install_wget_success()
        p = self._run(
            "download",
            "url:http://example.com/model.bin",
            "--name",
            "m-nr",
            "--category",
            "llm/chat",
            "--json",
            "--no-resume",
            "--force-redownload",
        )
        self.assertEqual(p.returncode, 0, p.stderr)
        summary = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(summary["backend_tool"], "wget")
        self.assertNotIn("-c", summary["backend_command"])

    def test_backend_missing_exit_code_4(self) -> None:
        env = self.env.copy()
        env["PATH"] = str(self.bin)
        p = subprocess.run(
            [self.py, str(AIM), "download", "url:http://example.com/model.bin", "--name", "m-miss", "--json", "--retry", "0", "--force-redownload"],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(p.returncode, 4)
        summary = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["error"]["code"], "BACKEND_NOT_FOUND")

    def test_cancel_flow(self) -> None:
        self._install_wget_slow()
        jobs_dir = self.home / ".aim" / "download-jobs"
        proc = subprocess.Popen(
            [
                self.py,
                str(AIM),
                "download",
                "url:http://example.com/big.bin",
                "--name",
                "m-cancel",
                "--category",
                "llm/chat",
                "--json",
                "--force-redownload",
            ],
            cwd=str(ROOT),
            env=self.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        job_id = None
        deadline = time.time() + 8
        while time.time() < deadline and job_id is None:
            if jobs_dir.exists():
                files = sorted(jobs_dir.glob("dl-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if files:
                    job_id = files[0].stem
                    break
            time.sleep(0.1)
        self.assertIsNotNone(job_id)
        assert job_id is not None

        # give it a moment to start running
        time.sleep(0.5)
        c = self._run("download", "cancel", job_id, "--json")
        self.assertEqual(c.returncode, 0, c.stderr)
        c_obj = json.loads(c.stdout.strip().splitlines()[-1])
        self.assertEqual(c_obj["status"], "cancel_requested")

        out, err = proc.communicate(timeout=20)
        self.assertEqual(proc.returncode, 2, err)
        last = json.loads([x for x in out.strip().splitlines() if x.strip()][-1])
        self.assertEqual(last["status"], "canceled")

        st = self._run("download", "status", job_id, "--json")
        self.assertEqual(st.returncode, 0, st.stderr)
        state = json.loads(st.stdout.strip())
        self.assertEqual(state["status"], "canceled")


if __name__ == "__main__":
    unittest.main()
