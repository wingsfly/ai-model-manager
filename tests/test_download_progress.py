import unittest

import aim


class DownloadProgressParseTests(unittest.TestCase):
    def test_parse_curl_progress(self):
        line = " 12  100M   12 12.0M    0     0  6144k      0  0:00:16  0:00:02  0:00:14 6144k"
        out = aim._parse_progress_line(line, backend_tool="curl")
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["percent"], 12.0, places=1)
        self.assertGreater(out["speed_bps"], 0)
        self.assertEqual(out["eta_seconds"], 14.0)
        self.assertGreater(out["downloaded_bytes"], 0)
        self.assertGreater(out["total_bytes"], out["downloaded_bytes"])

    def test_parse_wget_progress(self):
        line = " 42% [=======>           ] 44,040,704  11.2MB/s  eta 9s"
        out = aim._parse_progress_line(line, backend_tool="wget")
        self.assertIsNotNone(out)
        self.assertAlmostEqual(out["percent"], 42.0, places=1)
        self.assertGreater(out["downloaded_bytes"], 0)
        self.assertGreater(out["total_bytes"], out["downloaded_bytes"])
        self.assertGreater(out["speed_bps"], 0)
        self.assertEqual(out["eta_seconds"], 9.0)

    def test_parse_hfd_like_progress(self):
        line = "[#123 12MiB/1.0GiB(1%) CN:16 DL:2.3MiB ETA:7m12s]"
        out = aim._parse_progress_line(line, backend_tool="hfd")
        self.assertIsNotNone(out)
        self.assertGreater(out["downloaded_bytes"], 0)
        self.assertGreater(out["total_bytes"], out["downloaded_bytes"])
        self.assertGreater(out["percent"], 0)
        self.assertEqual(out["eta_seconds"], 432.0)

    def test_non_progress_line_returns_none(self):
        self.assertIsNone(aim._parse_progress_line("curl: (60) SSL certificate problem", backend_tool="curl"))


if __name__ == "__main__":
    unittest.main()
