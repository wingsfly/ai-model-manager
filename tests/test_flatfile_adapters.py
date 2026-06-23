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


class WhisperCacheAdapterTests(unittest.TestCase):
    def setUp(self):
        self.wc = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["sources"]["whisper-cache"] = {"cache_path": str(self.wc)}
        self.root = aim.StorageRoot(id="primary", path=str(self.wc / "AI"))

    def test_registered_and_source(self):
        self.assertIn("whisper-cache", aim.ADAPTERS)
        self.assertIn("whisper-cache", aim.ENGINE_NAMES)
        self.assertIn("whisper-cache", aim.SOURCES)

    def test_cache_dir_resolves_to_xdg_whisper(self):
        det = aim.EnvDetector(home=Path("/h"), rc_files=[],
                              shell_value=lambda v: "/x/cache" if v == "XDG_CACHE_HOME" else None)
        self.assertEqual(det.cache_dir("whisper-cache"), Path("/x/cache/whisper"))
        det2 = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        self.assertEqual(det2.cache_dir("whisper-cache"), Path("/h/.cache/whisper"))

    def test_scan_finds_pt_files(self):
        _write(self.wc / "base.pt", b"B" * 12)
        _write(self.wc / "large-v3.pt", b"L" * 15)
        ad = aim.WhisperCacheAdapter(self.config, self.root)
        scanned = ad.scan()
        self.assertEqual(sorted(s.name for s in scanned), ["whisper-base", "whisper-large-v3"])
        self.assertTrue(all(s.native_cas and s.category == "asr/model" for s in scanned))
        self.assertEqual(scanned[0].source["type"], "whisper-cache")


if __name__ == "__main__":
    unittest.main()
