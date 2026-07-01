import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from unittest import mock
from pathlib import Path

import aim


class DownloadDirSizeTests(unittest.TestCase):
    """_download_dir_size sums real bytes on disk, ignoring aria2 control files
    and symlinks, so it can drive aggregate progress for multi-file downloads."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sums_nested_files(self) -> None:
        (self.dir / "a.bin").write_bytes(b"x" * 100)
        sub = self.dir / "sub"
        sub.mkdir()
        (sub / "b.bin").write_bytes(b"y" * 250)
        self.assertEqual(aim._download_dir_size(self.dir), 350)

    def test_skips_aria2_control_files(self) -> None:
        (self.dir / "model.safetensors").write_bytes(b"z" * 500)
        (self.dir / "model.safetensors.aria2").write_bytes(b"c" * 40)
        self.assertEqual(aim._download_dir_size(self.dir), 500)

    def test_skips_symlinks(self) -> None:
        real = self.dir / "real.bin"
        real.write_bytes(b"w" * 128)
        os.symlink(real, self.dir / "link.bin")
        self.assertEqual(aim._download_dir_size(self.dir), 128)

    def test_missing_dir_is_zero(self) -> None:
        self.assertEqual(aim._download_dir_size(self.dir / "nope"), 0)

    def test_sparse_file_counts_real_blocks_not_logical_size(self) -> None:
        # aria2 seeks to a high offset, giving a large logical size but few real
        # blocks; progress must reflect bytes actually on disk, not logical size.
        sparse = self.dir / "big.safetensors"
        with open(sparse, "wb") as f:
            f.seek(2_000_000_000)   # 2 GB logical
            f.write(b"x")           # only a few KB actually allocated
        logical = os.path.getsize(sparse)
        self.assertGreaterEqual(logical, 2_000_000_000)
        counted = aim._download_dir_size(self.dir)
        # must be far below the logical 2 GB (real allocation is tiny)
        self.assertLess(counted, 100_000_000)


class MultiFileTickFilterTests(unittest.TestCase):
    """For multi-file backends (modelscope / huggingface), per-file progress lines
    parsed from backend stdout must NOT drive the job's aggregate progress — a single
    small file finishing at 100% would otherwise lock the whole job at 100%."""

    def test_per_file_tick_ignored_for_multifile(self) -> None:
        tick = {"percent": 100.0, "downloaded_bytes": 1200, "total_bytes": 1200,
                "speed_bps": 5_000_000, "backend_tool": "modelscope"}
        # per-file ticks must be ignored entirely so one finished file can't lock 100%
        self.assertIsNone(aim._multifile_filter(tick, is_multi_file=True))

    def test_aggregate_tick_passes_through(self) -> None:
        tick = {"downloaded_bytes": 5_000_000_000, "aggregate": True,
                "backend_tool": "modelscope"}
        out = aim._multifile_filter(tick, is_multi_file=True)
        self.assertIsNotNone(out)
        self.assertEqual(out.get("downloaded_bytes"), 5_000_000_000)

    def test_single_file_tick_untouched(self) -> None:
        tick = {"percent": 42.0, "downloaded_bytes": 4200, "total_bytes": 10000,
                "backend_tool": "wget"}
        out = aim._multifile_filter(tick, is_multi_file=False)
        self.assertIsNotNone(out)
        self.assertEqual(out.get("percent"), 42.0)
        self.assertEqual(out.get("downloaded_bytes"), 4200)


class ModelScopeCommandTests(unittest.TestCase):
    """modelscope's CLI has no --timeout flag; passing one makes every timed
    download fail with 'unrecognized arguments: --timeout'."""

    def _capture_cmd(self, **opt_kwargs):
        captured = {}

        def _fake_exec(cmd, env, options, job_state, backend_tool, on_progress=None):
            captured["cmd"] = cmd
            return aim.DownloadResult(success=True)

        opts = aim.DownloadOptions(**opt_kwargs)
        with mock.patch.object(aim, "_execute_backend_command", _fake_exec):
            aim._download_modelscope("Qwen/Qwen3-Embedding-4B", Path("/tmp/x"),
                                     opts, {"job_id": "j"})
        return captured["cmd"]

    def test_timeout_not_passed_to_modelscope(self) -> None:
        cmd = self._capture_cmd(timeout=60)
        self.assertNotIn("--timeout", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("--local_dir", cmd)

    def test_backend_args_still_forwarded(self) -> None:
        cmd = self._capture_cmd(timeout=60, backend_args=["--max-workers", "8"])
        self.assertNotIn("--timeout", cmd)
        self.assertIn("--max-workers", cmd)
        self.assertIn("8", cmd)


class ActiveFileAndFormattingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_active_file_from_aria2_control(self) -> None:
        (self.dir / "model-00001-of-00002.safetensors.aria2").write_bytes(b"c")
        self.assertEqual(aim._active_download_file(self.dir),
                         "model-00001-of-00002.safetensors")

    def test_active_file_from_temp_dir(self) -> None:
        temp = self.dir / "._____temp"
        temp.mkdir()
        (temp / "big.safetensors").write_bytes(b"x" * 500)
        (temp / "small.bin").write_bytes(b"y" * 10)
        self.assertEqual(aim._active_download_file(self.dir), "big.safetensors")

    def test_active_file_none(self) -> None:
        self.assertEqual(aim._active_download_file(self.dir), "")

    def test_format_duration(self) -> None:
        self.assertEqual(aim._format_duration(45), "45s")
        self.assertEqual(aim._format_duration(200), "3m20s")
        self.assertEqual(aim._format_duration(4320), "1h12m")

    def test_download_line_has_file_percent_speed_eta(self) -> None:
        line = aim._format_download_line({
            "status": "downloading", "current_file": "model.safetensors",
            "percent": 2.0, "downloaded_bytes": 180_000_000, "total_bytes": 8_060_000_000,
            "speed_bps": 1_800_000, "eta_seconds": 4320,
        })
        self.assertIn("model.safetensors", line)
        self.assertIn("2%", line)
        self.assertIn("/s", line)
        self.assertIn("ETA", line)


class RemoteTotalSizeTests(unittest.TestCase):
    """_fetch_remote_total_size sums repo file sizes from the backend API so
    multi-file downloads can show a real percentage."""

    def _patched_opener(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")

        @contextmanager
        def _open(url, timeout=0):
            resp = mock.Mock()
            resp.read.return_value = body
            yield resp

        opener = mock.Mock()
        opener.open = _open
        return mock.patch("urllib.request.build_opener", return_value=opener)

    def test_modelscope_sums_blob_sizes(self) -> None:
        payload = {"Data": {"Files": [
            {"Type": "blob", "Size": 1000},
            {"Type": "tree", "Size": 0},
            {"Type": "blob", "Size": 8_000_000_000},
        ]}}
        with self._patched_opener(payload):
            total = aim._fetch_remote_total_size({"type": "modelscope", "repo_id": "Qwen/X"})
        self.assertEqual(total, 8_000_001_000)

    def test_huggingface_sums_file_sizes(self) -> None:
        payload = [
            {"type": "file", "size": 500},
            {"type": "directory", "size": 0},
            {"type": "file", "size": 43_000_000_000},
        ]
        with self._patched_opener(payload):
            total = aim._fetch_remote_total_size({"type": "huggingface", "repo_id": "org/Y"})
        self.assertEqual(total, 43_000_000_500)

    def test_unknown_source_returns_zero(self) -> None:
        self.assertEqual(aim._fetch_remote_total_size({"type": "url", "repo_id": ""}), 0)

    def test_network_error_returns_zero(self) -> None:
        with mock.patch("urllib.request.build_opener", side_effect=OSError("no net")):
            total = aim._fetch_remote_total_size({"type": "modelscope", "repo_id": "Qwen/X"})
        self.assertEqual(total, 0)


if __name__ == "__main__":
    unittest.main()
