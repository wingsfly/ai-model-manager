import unittest
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


class PyTorchHubAdapterTests(unittest.TestCase):
    def setUp(self):
        self.hub = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["sources"]["pytorch-hub"] = {"cache_path": str(self.hub)}
        self.root = aim.StorageRoot(id="primary", path=str(self.hub / "AI"))

    def test_registered(self):
        self.assertIn("pytorch-hub", aim.ADAPTERS)
        self.assertIn("pytorch-hub", aim.ENGINE_NAMES)

    def test_scan_finds_checkpoints_only(self):
        _write(self.hub / "checkpoints" / "wav2vec2_base.pth", b"W" * 30)
        _write(self.hub / "checkpoints" / "resnet50.pt", b"R" * 20)
        _write(self.hub / "snakers4_silero-vad_master" / "hubconf.py", b"code")
        _write(self.hub / "trusted_list", b"x")
        ad = aim.PyTorchHubAdapter(self.config, self.root)
        scanned = ad.scan()
        names = sorted(s.name for s in scanned)
        self.assertEqual(names, ["resnet50", "wav2vec2_base"])
        self.assertTrue(all(s.native_cas and not s.is_directory for s in scanned))
        wav = next(s for s in scanned if s.name == "wav2vec2_base")
        self.assertEqual(wav.source, {"type": "pytorch-hub", "repo_id": "wav2vec2_base"})
        self.assertEqual(wav.category, "asr/model")


if __name__ == "__main__":
    unittest.main()
