# SP4 — flat-file Ingest (PyTorch Hub + Whisper) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest single-weight-file models — PyTorch Hub `checkpoints/*.pth` and openai-whisper `*.pt` — into the flat store with a file-symlink shim left in place, reusing the SP2/SP3 ingest/verify/restore machinery.

**Architecture:** Single-file `aim.py` (stdlib only). New "flat-file" storage kind: copy the one weight file into `store/{cat}/{id}/`, replace the original cache file with a symlink (rename-aside-then-restore, like `_ms_build_shim`). Two scan-only adapters (`PyTorchHubAdapter`, `WhisperCacheAdapter`) discover the files; `op_ingest` / `_retarget_shim_locations` / `_rebuild_shim_from_storage` each gain a `flat-file` branch so backup/restore/verify/dedup work unchanged.

**Tech Stack:** Python 3.10+ stdlib. Tests: stdlib `unittest`, synthetic caches in tempdirs, run from repo ROOT via `make test`.

**Reference spec:** `docs/superpowers/specs/2026-06-23-aim-sp4-flatfile-ingest-design.md`. Add code to the existing `# ── Native Ingest (SP2) ──` section; new adapters go beside the other adapters (before the `ADAPTERS` dict at [aim.py:1099]). Locate anchors by symbol (`grep -n`), never absolute line.

Shared test helper (top of each new test file):
```python
def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p
```

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `aim.py` | `_flatfile_read_native`, `_flatfile_build_shim`; `op_ingest` flat-file branch; `PyTorchHubAdapter`, `WhisperCacheAdapter` + registration; `whisper-cache` in `SOURCES`; `pytorch-hub` cache_layout→`flat-file`; flat-file cases in `_retarget_shim_locations` + `_rebuild_shim_from_storage`; `default_config` engines | Modify |
| `tests/test_ingest_flatfile.py` | `_flatfile_read_native`/`_flatfile_build_shim`; op_ingest torch+whisper end-to-end; retarget/verify/restore flat-file | Create |
| `tests/test_flatfile_adapters.py` | `PyTorchHubAdapter`/`WhisperCacheAdapter` scan; `cache_dir` resolution | Create |
| `README.md`, `AIM-MANUAL.md` | document the two new sources | Modify |

---

## Task 1: `_flatfile_read_native` + `_flatfile_build_shim`

**Files:** Modify `aim.py` (SP2 section); Test `tests/test_ingest_flatfile.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_flatfile.py
import unittest
import os
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


class FlatFileHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_read_native_single_file(self):
        f = _write(self.tmp / "cache" / "checkpoints" / "wav2vec2.pth", b"W" * 50)
        info = aim._flatfile_read_native(f)
        self.assertEqual(len(info["files"]), 1)
        self.assertEqual(info["files"][0]["name"], "wav2vec2.pth")
        self.assertEqual(info["files"][0]["size"], 50)
        self.assertEqual(Path(info["files"][0]["real_path"]).read_bytes(), b"W" * 50)

    def test_build_shim_replaces_file_with_symlink(self):
        store = _write(self.tmp / "store" / "m" / "x.pt", b"DATA")
        orig = _write(self.tmp / "cache" / "x.pt", b"DATA")  # the original cache file
        aim._flatfile_build_shim(orig, store)
        self.assertTrue(orig.is_symlink())
        self.assertEqual(orig.resolve(), store.resolve())
        self.assertEqual(orig.read_bytes(), b"DATA")  # loads through the shim
        self.assertFalse((self.tmp / "cache" / "x.pt.aim-old").exists())  # no leftover backup

    def test_build_shim_restores_original_on_symlink_failure(self):
        store = _write(self.tmp / "store" / "m" / "x.pt", b"NEW")
        orig = _write(self.tmp / "cache" / "x.pt", b"ORIGINAL")
        real_symlink = aim.os.symlink
        aim.os.symlink = lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        try:
            with self.assertRaises(OSError):
                aim._flatfile_build_shim(orig, store)
        finally:
            aim.os.symlink = real_symlink
        self.assertTrue(orig.is_file() and not orig.is_symlink())   # original restored
        self.assertEqual(orig.read_bytes(), b"ORIGINAL")
        self.assertFalse((self.tmp / "cache" / "x.pt.aim-old").exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_ingest_flatfile -v` → FAIL (no `_flatfile_read_native`).

- [ ] **Step 3: Implement** — add to the `# ── Native Ingest (SP2) ──` section (near `_ms_build_shim`):

```python
def _flatfile_read_native(file_path) -> dict:
    """A single weight file IS the model. Returns {files:[{name, real_path, size}]}."""
    f = Path(file_path)
    size = f.stat().st_size if f.exists() else 0
    return {"files": [{"name": f.name, "real_path": str(f), "size": size}]}


def _flatfile_build_shim(orig_file: Path, store_file: Path) -> None:
    """Replace a single cache file with a symlink -> store_file, safely (rename-aside + restore).
    The original is renamed aside first; if creating the symlink fails it is restored."""
    orig_file = Path(orig_file)
    orig_file.parent.mkdir(parents=True, exist_ok=True)
    link = store_file.resolve()
    backup = orig_file.parent / (orig_file.name + ".aim-old")
    if backup.is_symlink() or backup.is_file():
        backup.unlink()
    elif backup.is_dir():
        shutil.rmtree(backup)
    had_original = orig_file.exists() or orig_file.is_symlink()
    if had_original:
        os.rename(orig_file, backup)
    try:
        os.symlink(link, orig_file)
    except OSError:
        if had_original and not (orig_file.exists() or orig_file.is_symlink()):
            os.rename(backup, orig_file)
        raise
    if had_original and (backup.exists() or backup.is_symlink()):
        if backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup)
        else:
            backup.unlink()
```

- [ ] **Step 4: Run** `python3 -m unittest tests.test_ingest_flatfile -v` → PASS (3). `make test`; `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_flatfile.py
git commit -m "feat(sp4): flat-file read + symlink shim (rename-aside, no data loss)"
```

---

## Task 2: `PyTorchHubAdapter` (scan checkpoints)

**Files:** Modify `aim.py` (adapter + registration + SOURCES/config); Test `tests/test_flatfile_adapters.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_flatfile_adapters.py
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
        self.hub = Path(tempfile.mkdtemp())  # stands in for $TORCH_HOME/hub
        self.config = aim.default_config()
        self.config["sources"]["pytorch-hub"] = {"cache_path": str(self.hub)}
        self.root = aim.StorageRoot(id="primary", path=str(self.hub / "AI"))

    def test_registered(self):
        self.assertIn("pytorch-hub", aim.ADAPTERS)
        self.assertIn("pytorch-hub", aim.ENGINE_NAMES)

    def test_scan_finds_checkpoints_only(self):
        _write(self.hub / "checkpoints" / "wav2vec2_base.pth", b"W" * 30)
        _write(self.hub / "checkpoints" / "resnet50.pt", b"R" * 20)
        # a code repo dir + trusted_list must be IGNORED
        _write(self.hub / "snakers4_silero-vad_master" / "hubconf.py", b"code")
        _write(self.hub / "trusted_list", b"x")
        ad = aim.PyTorchHubAdapter(self.config, self.root)
        scanned = ad.scan()
        names = sorted(s.name for s in scanned)
        self.assertEqual(names, ["resnet50", "wav2vec2_base"])
        self.assertTrue(all(s.native_cas and not s.is_directory for s in scanned))
        wav = next(s for s in scanned if s.name == "wav2vec2_base")
        self.assertEqual(wav.source, {"type": "pytorch-hub", "repo_id": "wav2vec2_base"})
        self.assertEqual(wav.category, "asr/model")  # inferred


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_flatfile_adapters -v` → FAIL.

- [ ] **Step 3: Implement** — add the adapter (before the `ADAPTERS` dict) and register it.

```python
class PyTorchHubAdapter(EngineAdapter):
    name = "pytorch-hub"

    def supported_formats(self) -> list[str]:
        return ["pt", "pth"]

    def scan(self) -> list[ScannedModel]:
        results: list[ScannedModel] = []
        ckpt = self.base_path / "checkpoints"   # base_path = $TORCH_HOME/hub (sources.cache_path)
        if not ckpt.is_dir():
            return results
        for f in sorted(ckpt.iterdir()):
            if not f.is_file() or f.is_symlink() or f.suffix not in (".pt", ".pth"):
                continue
            stem = f.stem
            results.append(ScannedModel(
                id=self._make_id(f"torch-{stem}"), name=stem, path=str(f), engine=self.name,
                format=f.suffix[1:], size_bytes=f.stat().st_size,
                category=self._infer_cat(stem), tags=["pytorch-hub"],
                source={"type": "pytorch-hub", "repo_id": stem},
                native_cas=True, is_directory=False))
        return results

    def _infer_cat(self, name: str) -> str:
        n = name.lower()
        if any(k in n for k in ["wav2vec", "w2v", "whisper", "asr", "hubert", "wavlm", "conformer"]):
            return "asr/model"
        if "vad" in n:
            return "audio/vad"
        return "uncategorized"

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        return []

    def unprovision(self, model: ModelEntry) -> bool:
        return False
```

Register: add `"pytorch-hub": PyTorchHubAdapter,` to `ADAPTERS`; add `"pytorch-hub"` to `ENGINE_NAMES`; add to `default_config()["engines"]`:
```python
            "pytorch-hub": {"enabled": True, "model_dir": "torch/hub", "native_cas": True},
```
In `SOURCES["pytorch-hub"]`, change `"cache_layout": "torch-hub"` to `"cache_layout": "flat-file"`.

> `base_path` (SP1) returns `sources["pytorch-hub"]["cache_path"]` because `native_cas=True` + cache_path is set. `_make_id`, `ScannedModel`, `Provision` exist.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_flatfile_adapters -v` → PASS (2). `make test` (confirm SP1 `test_sources_registry` still passes after the cache_layout change — `flat-file` is in its `ALLOWED_LAYOUT`). `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_flatfile_adapters.py
git commit -m "feat(sp4): PyTorchHubAdapter (scan checkpoints/*.pth, ignore code repos)"
```

---

## Task 3: `op_ingest` flat-file branch (PyTorch Hub end-to-end)

**Files:** Modify `aim.py` (`op_ingest`); Test `tests/test_ingest_flatfile.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
class FlatFileIngestTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]
        # torch hub cache lives UNDER the root here (so canonical/rel resolve cleanly)
        self.hub = Path(self.config["roots"][0]["path"]) / "torch" / "hub"
        self.config["sources"]["pytorch-hub"] = {"cache_path": str(self.hub)}

    def _torch_entry(self, fpath):
        return aim.ModelEntry(id="torch-wav2vec2", native_cas=True,
                              source={"type": "pytorch-hub", "repo_id": "wav2vec2"},
                              category="asr/model", canonical={"root": "primary", "path": str(fpath)})

    def test_ingest_torch_checkpoint(self):
        ckpt = _write(self.hub / "checkpoints" / "wav2vec2.pth", b"W" * 64)
        reg = aim.Registry()
        reg.models = [self._torch_entry(ckpt)]
        ok = aim.op_ingest(self.config, reg, "torch-wav2vec2", registry_save=False)
        self.assertTrue(ok)
        e = reg.find("torch-wav2vec2")
        self.assertFalse(e.native_cas)
        self.assertEqual(e.storage["class"], "managed-torch")
        store = Path(self.config["roots"][0]["path"]) / e.storage["store_path"]
        self.assertEqual((store / "wav2vec2.pth").read_bytes(), b"W" * 64)
        # original replaced by a symlink resolving into store
        self.assertTrue(ckpt.is_symlink())
        self.assertEqual(ckpt.resolve(), (store / "wav2vec2.pth").resolve())
        shim = e.storage["shims"][0]
        self.assertEqual(shim["kind"], "flat-file")
        self.assertEqual(shim["tool"], "pytorch-hub")
        self.assertEqual(shim["reconstruct"], {"filename": "wav2vec2.pth", "rel": "checkpoints/wav2vec2.pth"})

    def test_ingest_dry_run_changes_nothing(self):
        ckpt = _write(self.hub / "checkpoints" / "wav2vec2.pth", b"W" * 10)
        reg = aim.Registry()
        reg.models = [self._torch_entry(ckpt)]
        aim.op_ingest(self.config, reg, "torch-wav2vec2", dry_run=True, registry_save=False)
        self.assertTrue(ckpt.is_file() and not ckpt.is_symlink())
        self.assertEqual(reg.find("torch-wav2vec2").storage, {})
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_ingest_flatfile.FlatFileIngestTests -v` → FAIL (no flat-file branch).

- [ ] **Step 3: Implement** — in `op_ingest`, add this branch immediately BEFORE the final `print(f"Error: ingest not yet implemented...` line:

```python
    if tool in ("pytorch-hub", "whisper-cache"):
        info = _flatfile_read_native(cache_repo)
        fname = info["files"][0]["name"]
        target_id = _sanitize_ingest_id(entry, new_id)
        target_cat = category or entry.category or "uncategorized"
        store_dir = (root.store_path / target_cat / target_id)
        store_rel = str(store_dir.relative_to(Path(root.path)))
        if dry_run:
            print(f"[dry-run] would ingest {model_id} -> {store_rel}/{fname}, symlink original back")
            return True
        if store_dir.exists():
            print(f"Error: store dir already exists: {store_dir}", file=sys.stderr)
            return False
        try:
            size = _ingest_to_store(info["files"], store_dir)
            _flatfile_build_shim(cache_repo, store_dir / fname)
        except OSError as ex:
            if store_dir.exists():
                shutil.rmtree(store_dir, ignore_errors=True)
            print(f"Error: ingest failed, rolled back: {ex}", file=sys.stderr)
            return False
        cache_base = Path(config.get("sources", {}).get(tool, {}).get("cache_path", ""))
        try:
            rel = str(cache_repo.relative_to(cache_base)) if str(cache_base) else fname
        except ValueError:
            rel = fname
        cls = "managed-torch" if tool == "pytorch-hub" else "managed-whisper"
        cache_root_var = "TORCH_HOME" if tool == "pytorch-hub" else "XDG_CACHE_HOME"
        entry.native_cas = False
        entry.id = target_id
        entry.category = target_cat
        entry.size_bytes = size
        entry.format = fname.rsplit(".", 1)[-1] if "." in fname else entry.format
        entry.canonical = {"root": root.id, "path": store_rel}
        entry.storage = {
            "class": cls, "store_path": store_rel, "ingested_at": _now_iso(),
            "shims": [{
                "tool": tool, "kind": "flat-file",
                "location": str(cache_repo.relative_to(Path(root.path))) if str(cache_repo).startswith(str(root.path)) else str(cache_repo),
                "cache_root_var": cache_root_var,
                "reconstruct": {"filename": fname, "rel": rel},
            }],
        }
        registry.add(entry)
        if registry_save:
            registry.save()
        print(f"Ingested {model_id} -> {store_rel}")
        return True
```

> `cache_repo` is the entry's canonical path resolved to absolute (set near the top of `op_ingest`); for these adapters it's the `.pth`/`.pt` file. `_sanitize_ingest_id`, `_ingest_to_store`, `_now_iso` exist.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_ingest_flatfile -v` → PASS (5). `make test`; `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_flatfile.py
git commit -m "feat(sp4): op_ingest flat-file branch (PyTorch Hub checkpoints end-to-end)"
```

---

## Task 4: `WhisperCacheAdapter` + `whisper-cache` source

**Files:** Modify `aim.py` (adapter + SOURCES + register); Test `tests/test_flatfile_adapters.py` (append) + `tests/test_ingest_flatfile.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_flatfile_adapters.py`:
```python
class WhisperCacheAdapterTests(unittest.TestCase):
    def setUp(self):
        self.wc = Path(tempfile.mkdtemp())  # stands in for ${XDG_CACHE_HOME}/whisper
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
```

Append to `tests/test_ingest_flatfile.py`:
```python
class WhisperIngestTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]
        self.wc = Path(self.config["roots"][0]["path"]) / "wcache" / "whisper"
        self.config["sources"]["whisper-cache"] = {"cache_path": str(self.wc)}

    def test_ingest_whisper_pt(self):
        pt = _write(self.wc / "base.pt", b"W" * 40)
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="whisper-base", native_cas=True,
                      source={"type": "whisper-cache", "repo_id": "base"}, category="asr/model",
                      canonical={"root": "primary", "path": str(pt)})]
        ok = aim.op_ingest(self.config, reg, "whisper-base", registry_save=False)
        self.assertTrue(ok)
        e = reg.find("whisper-base")
        self.assertEqual(e.storage["class"], "managed-whisper")
        store = Path(self.config["roots"][0]["path"]) / e.storage["store_path"]
        self.assertEqual((store / "base.pt").read_bytes(), b"W" * 40)
        self.assertTrue(pt.is_symlink())
        self.assertEqual(e.storage["shims"][0]["reconstruct"], {"filename": "base.pt", "rel": "base.pt"})
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_flatfile_adapters tests.test_ingest_flatfile -v` → FAIL.

- [ ] **Step 3: Implement** — add `WhisperCacheAdapter` (beside PyTorchHubAdapter), register it, and add the `whisper-cache` SOURCES entry.

```python
class WhisperCacheAdapter(EngineAdapter):
    name = "whisper-cache"

    def supported_formats(self) -> list[str]:
        return ["pt"]

    def scan(self) -> list[ScannedModel]:
        results: list[ScannedModel] = []
        base = self.base_path   # ${XDG_CACHE_HOME:-~/.cache}/whisper
        if not base.is_dir():
            return results
        for f in sorted(base.iterdir()):
            if not f.is_file() or f.is_symlink() or f.suffix != ".pt":
                continue
            results.append(ScannedModel(
                id=self._make_id(f"whisper-{f.stem}"), name=f"whisper-{f.stem}", path=str(f),
                engine=self.name, format="pt", size_bytes=f.stat().st_size,
                category="asr/model", tags=["whisper", "openai-whisper"],
                source={"type": "whisper-cache", "repo_id": f.stem},
                native_cas=True, is_directory=False))
        return results

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        return []

    def unprovision(self, model: ModelEntry) -> bool:
        return False
```

Add to `SOURCES` (after the `pytorch-hub` entry):
```python
    "whisper-cache": {
        "aliases": ["whisper-dl"],
        "cache_layout": "flat-file",
        "tools": [{"name": "whisper", "check": "which",
                   "install_cmd": "pip3 install --break-system-packages -U openai-whisper",
                   "description": "openai-whisper"}],
        "env": [{"name": "XDG_CACHE_HOME", "role": "cache_dir", "default": "~/.cache",
                 "subpath": "whisper", "detect": ["env", "rc"], "manage": "none", "secret": False}],
    },
```
Register: add `"whisper-cache": WhisperCacheAdapter,` to `ADAPTERS`; add `"whisper-cache"` to `ENGINE_NAMES`; add to `default_config()["engines"]`:
```python
            "whisper-cache": {"enabled": True, "model_dir": ".cache/whisper", "native_cas": True},
```

> `cache_dir("whisper-cache")` resolves `XDG_CACHE_HOME` (default `~/.cache`) + subpath `whisper`. The `op_ingest` flat-file branch (Task 3) already handles `tool == "whisper-cache"`.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_flatfile_adapters tests.test_ingest_flatfile -v` → PASS. `make test`; `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_flatfile_adapters.py tests/test_ingest_flatfile.py
git commit -m "feat(sp4): WhisperCacheAdapter + whisper-cache source (openai-whisper cache)"
```

---

## Task 5: flat-file in retarget + verify + restore round-trip

**Files:** Modify `aim.py` (`_retarget_shim_locations`, `_rebuild_shim_from_storage`); Test `tests/test_ingest_flatfile.py` (append)

- [ ] **Step 1: Write the failing tests (append)**

```python
import io
from contextlib import redirect_stdout


class FlatFileRetargetVerifyTests(unittest.TestCase):
    def test_retarget_torch_and_whisper(self):
        det = aim.EnvDetector(home=Path("/h"), rc_files=[],
                              shell_value=lambda v: {"TORCH_HOME": "/tgt/torch"}.get(v))
        e = aim.ModelEntry(id="t", storage={"store_path": "store/x", "shims": [
            {"tool": "pytorch-hub", "kind": "flat-file", "location": "/OLD/x.pth",
             "reconstruct": {"filename": "x.pth", "rel": "checkpoints/x.pth"}}]})
        aim._retarget_shim_locations(e, det)
        self.assertEqual(e.storage["shims"][0]["location"], "/tgt/torch/hub/checkpoints/x.pth")

        det2 = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        e2 = aim.ModelEntry(id="w", storage={"store_path": "store/x", "shims": [
            {"tool": "whisper-cache", "kind": "flat-file", "location": "/OLD/base.pt",
             "reconstruct": {"filename": "base.pt", "rel": "base.pt"}}]})
        aim._retarget_shim_locations(e2, det2)
        self.assertEqual(e2.storage["shims"][0]["location"], "/h/.cache/whisper/base.pt")

    def test_verify_fix_rebuilds_flatfile_shim(self):
        home = Path(tempfile.mkdtemp())
        cfg = aim.default_config(); cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        hub = Path(cfg["roots"][0]["path"]) / "torch" / "hub"
        cfg["sources"]["pytorch-hub"] = {"cache_path": str(hub)}
        ckpt = _write(hub / "checkpoints" / "wav2vec2.pth", b"W" * 32)
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="torch-wav2vec2", native_cas=True,
                      source={"type": "pytorch-hub", "repo_id": "wav2vec2"}, category="asr/model",
                      canonical={"root": "primary", "path": str(ckpt)})]
        aim.op_ingest(cfg, reg, "torch-wav2vec2", registry_save=False)
        ckpt.unlink()  # destroy the shim symlink
        with redirect_stdout(io.StringIO()):
            aim.op_verify(cfg, reg, fix=True)
        self.assertTrue(ckpt.is_symlink())
        store = Path(cfg["roots"][0]["path"]) / reg.find("torch-wav2vec2").storage["store_path"]
        self.assertEqual(ckpt.resolve(), (store / "wav2vec2.pth").resolve())

    def test_restore_roundtrip_crossmachine_flatfile(self):
        # SOURCE
        src = Path(tempfile.mkdtemp())
        scfg = aim.default_config(); scfg["roots"] = [{"id": "primary", "path": str(src / "AI")}]
        shub = Path(scfg["roots"][0]["path"]) / "torch" / "hub"
        scfg["sources"]["pytorch-hub"] = {"cache_path": str(shub)}
        ckpt = _write(shub / "checkpoints" / "wav2vec2.pth", b"W" * 48)
        sreg = aim.Registry()
        sreg.models = [aim.ModelEntry(id="torch-wav2vec2", native_cas=True,
                       source={"type": "pytorch-hub", "repo_id": "wav2vec2"}, category="asr/model",
                       canonical={"root": "primary", "path": str(ckpt)})]
        aim.op_ingest(scfg, sreg, "torch-wav2vec2", registry_save=False)
        backup = src / "bk"
        with redirect_stdout(io.StringIO()):
            aim.op_backup(scfg, sreg, str(backup))
        # TARGET: different root + different TORCH_HOME
        tgt = Path(tempfile.mkdtemp())
        tgt_torch = tgt / "torchcache"
        tcfg = aim.default_config(); tcfg["roots"] = [{"id": "primary", "path": str(tgt / "AI")}]
        det = aim.EnvDetector(home=tgt, rc_files=[],
                              shell_value=lambda v: str(tgt_torch) if v == "TORCH_HOME" else None)
        with redirect_stdout(io.StringIO()):
            rc = aim.op_restore(tcfg, aim.Registry(), str(backup), detector=det, registry_save=False)
        self.assertEqual(rc, aim.EXIT_OK)
        link = tgt_torch / "hub" / "checkpoints" / "wav2vec2.pth"
        self.assertTrue(link.is_symlink())
        tgt_store = Path(tcfg["roots"][0]["path"]) / "store" / "asr" / "model" / "torch-wav2vec2"
        self.assertEqual(link.resolve(), (tgt_store / "wav2vec2.pth").resolve())
        self.assertEqual(link.resolve().read_bytes(), b"W" * 48)
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_ingest_flatfile.FlatFileRetargetVerifyTests -v` → FAIL.

- [ ] **Step 3: Implement** — add a `flat-file` case to each function.

In `_retarget_shim_locations`, after the `ms-dir` branch:
```python
        elif kind == "flat-file":
            base = detector.cache_dir(shim.get("tool", ""))
            rel = rc.get("rel", "")
            if base and rel:
                shim["location"] = str(base / rel)
```

In `_rebuild_shim_from_storage`, after the `ollama-cas` branch:
```python
        elif shim["kind"] == "flat-file":
            store_file = store_dir / rc.get("filename", "")
            _flatfile_build_shim(cache_path, store_file)
            rebuilt = True
```

- [ ] **Step 4: Run** `python3 -m unittest tests.test_ingest_flatfile -v` → PASS. Run `make test` (confirm SP2 `test_verify_shim` + SP3 `test_restore` still pass). `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_flatfile.py
git commit -m "feat(sp4): flat-file retarget + verify --fix + cross-machine restore"
```

---

## Task 6: docs + final suite

**Files:** Modify `README.md`, `AIM-MANUAL.md`

- [ ] **Step 1: README.md** — under the `aim ingest` / `aim sources` area, add a note:
```markdown
# Ingestable sources now include single-file weights:
#   PyTorch Hub  ($TORCH_HOME/hub/checkpoints/*.pth)  -> aim ingest <torch-id>
#   openai-whisper (${XDG_CACHE_HOME:-~/.cache}/whisper/*.pt) -> aim ingest <whisper-id>
# (aim scan discovers them; ingest leaves a file symlink so the tool still loads.)
```

- [ ] **Step 2: AIM-MANUAL.md** — in the `aim ingest` section, add a Chinese paragraph: aim 现在还能摄取**单文件权重**模型——PyTorch Hub 的 `checkpoints/*.pth`(源 `pytorch-hub`,`TORCH_HOME`)和 openai-whisper 的 `*.pt`(源 `whisper-cache`,`${XDG_CACHE_HOME:-~/.cache}/whisper`)。`aim scan` 发现它们,`aim ingest` 复制进 store 并在原处留**文件软链**(工具照常按原路径加载);PyTorch Hub 仅摄取 checkpoints,代码仓不纳管。

- [ ] **Step 3:** Run `make lint && make test` → all PASS.

- [ ] **Step 4: Commit**
```bash
git add README.md AIM-MANUAL.md
git commit -m "docs(sp4): document flat-file ingest (PyTorch Hub + whisper)"
```

---

## Self-Review

**1. Spec coverage:**
- §1 flat-file kind (copy + file symlink + annotation) → T1 (`_flatfile_build_shim`), T3 (op_ingest branch builds the annotation) ✓
- §2 components (`_flatfile_read_native`, `_flatfile_build_shim`, op_ingest branch, two adapters, retarget/rebuild) → T1/T2/T3/T4/T5 ✓
- §3 SOURCES/registration (pytorch-hub cache_layout→flat-file; whisper-cache entry; register both adapters + default_config engines) → T2 (torch) + T4 (whisper) ✓
- §4 ingest flow (copy-first, rename-aside shim, rollback, dry-run, keep_native moot) → T1 (shim safety) + T3 (flow/rollback/dry-run) ✓
- §5 tests (read/build/dataloss, scans, end-to-end torch+whisper, retarget, verify --fix, cross-machine restore, cache_dir resolution) → T1/T2/T3/T4/T5 ✓
- §6 acceptance → all covered across T2 (scan ignores repos), T3 (ingest+symlink, dry-run), T1 (rollback safety), T5 (verify --fix + restore + cache_dir) ✓

**2. Placeholder scan:** No TBD/"handle errors". Every code step complete. `_infer_cat` falls back to `"uncategorized"` (a valid existing category) rather than the spec's tentative `llm/chat` — a deliberate refinement to avoid mis-bucketing non-ASR checkpoints (noted here).

**3. Type consistency:** `_flatfile_read_native(file_path)->{files:[{name,real_path,size}]}` consumed by `_ingest_to_store` (uses name/real_path) and the op_ingest branch (uses `files[0]["name"]`). `_flatfile_build_shim(orig_file, store_file)` signature matches calls in op_ingest (Task 3) and `_rebuild_shim_from_storage` (Task 5). storage shim shape `{tool, kind:"flat-file", location, cache_root_var, reconstruct:{filename, rel}}` written in T3, consumed by retarget (T5: `cache_dir(tool)/rel`) and rebuild (T5: `store_dir/filename`). Adapters set `source={"type":"pytorch-hub"|"whisper-cache"}` matching the op_ingest `if tool in (...)` branch. `cache_dir("whisper-cache")` relies on the SOURCES entry added in T4 — note: the T4 `test_cache_dir_resolves_to_xdg_whisper` and the retarget test in T5 both depend on T4's SOURCES entry, so if executing strictly in order, T5's whisper retarget assertion needs T4 done first (it is — T4 precedes T5).

**Executor notes:** run from repo root (cross-test imports + relative behavior). Locate anchors by symbol. After Task 2's `cache_layout` change, re-run `tests.test_sources_registry` (it allows `flat-file`). After Task 5, re-run `tests.test_verify_shim` and `tests.test_restore` for no regressions.
