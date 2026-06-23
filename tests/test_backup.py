import unittest
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


class SyncStoreDirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_copies_then_idempotent(self):
        src = self.tmp / "s"
        _write(src / "a" / "x.bin", b"X" * 10)
        _write(src / "y.bin", b"Y" * 5)
        dst = self.tmp / "d"
        copied, skipped = aim._sync_store_dir(src, dst)
        self.assertEqual((copied, skipped), (2, 0))
        self.assertEqual((dst / "a" / "x.bin").read_bytes(), b"X" * 10)
        copied2, skipped2 = aim._sync_store_dir(src, dst)
        self.assertEqual((copied2, skipped2), (0, 2))

    def test_verify_detects_content_change(self):
        src = self.tmp / "s"; _write(src / "f.bin", b"AAAA")
        dst = self.tmp / "d"; _write(dst / "f.bin", b"BBBB")
        copied, skipped = aim._sync_store_dir(src, dst, verify=False)
        self.assertEqual((copied, skipped), (0, 1))
        copied, skipped = aim._sync_store_dir(src, dst, verify=True)
        self.assertEqual((copied, skipped), (1, 0))
        self.assertEqual((dst / "f.bin").read_bytes(), b"AAAA")

    def test_missing_src_is_noop(self):
        self.assertEqual(aim._sync_store_dir(self.tmp / "nope", self.tmp / "d"), (0, 0))


if __name__ == "__main__":
    unittest.main()
