# SP2 — Native Ingest + Shim Reconstruction + storage Annotation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correctly ingest HF/ModelScope/Ollama native models into the flat `store/{category}/{id}/`, rebuild each tool's native "load shim" pointing back to store (in place), and record a per-model `storage` annotation so tools still load and SP3 can back up/restore portably — replacing the broken `aim convert`.

**Architecture:** Single-file `aim.py` (stdlib only). Per tool: a `NativeReader` (list real files + reconstruct metadata) → a generic `StoreIngestor` (copy flat into store, no 2×, annotate) → a `ShimBuilder` (rebuild load shim in the tool's current cache). New `ModelScopeAdapter`. `op_ingest` orchestrates with copy-first rollback + dry-run. `op_convert` delegates to ingest; `op_verify` gains shim verification + `--fix` (rebuild shim from annotation).

**Tech Stack:** Python 3.10+ stdlib (`os`, `shutil`, `json`, `pathlib`, `hashlib`). Tests: stdlib `unittest`, synthetic native caches in tempdirs (no real tools needed), run via `make test`.

**Reference spec:** `docs/superpowers/specs/2026-06-22-aim-sp2-native-ingest-design.md`. Order: foundation → HF → MS → Ollama → cross-cutting.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `aim.py` | `ModelEntry.storage` field; `# ── Native Ingest (SP2) ──` section with `_hf_read_native`/`_hf_build_shim`, `_ms_read_native`/`_ms_build_shim`, `_ollama_read_native`/`_ollama_build_shim`, generic `_ingest_to_store`, `op_ingest`; `ModelScopeAdapter`; rework `op_convert_native_to_store`; extend `op_verify`; parser + dispatch | Modify |
| `tests/test_ingest_hf.py` | HF reader/shim/ingest/rollback (synthetic HF cache) | Create |
| `tests/test_ingest_ms.py` | MS reader/shim/ingest + `ModelScopeAdapter` scan | Create |
| `tests/test_ingest_ollama.py` | Ollama reader/shim/ingest (synthetic blobs+manifest) | Create |
| `tests/test_ingest_common.py` | `storage` field round-trip, `op_ingest` CLI, convert delegation, verify `--fix` round-trip, dedup interaction | Create |
| `README.md`, `AIM-MANUAL.md` | Document `aim ingest` + reworked `convert` + `verify --fix` | Modify |

All SP2 production code goes in a new `# ── Native Ingest (SP2) ──` section placed after the `# ── Sources & Env (SP1) ──` section. Locate anchors by symbol (`grep -n`), never absolute line.

Shared test helper used across test files (define at top of each file that needs it):
```python
def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p
```

---

## Task 1: `ModelEntry.storage` field

**Files:** Modify `aim.py` (`class ModelEntry`); Test `tests/test_ingest_common.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_common.py
import unittest
import aim


class StorageFieldTests(unittest.TestCase):
    def test_storage_defaults_empty_and_round_trips(self):
        e = aim.ModelEntry(id="m1")
        self.assertEqual(e.storage, {})
        e.storage = {"class": "managed-hf", "store_path": "store/x", "shims": []}
        d = e.to_dict()
        self.assertEqual(d["storage"]["class"], "managed-hf")
        e2 = aim.ModelEntry.from_dict(d)
        self.assertEqual(e2.storage["store_path"], "store/x")

    def test_from_dict_without_storage_defaults_empty(self):
        e = aim.ModelEntry.from_dict({"id": "m2"})
        self.assertEqual(e.storage, {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_ingest_common -v` → FAIL (`ModelEntry` has no attribute `storage`).

- [ ] **Step 3: Implement** — in `class ModelEntry`, add after the `external_links` field:

```python
    # SP2: how this model is physically stored + how each tool loads it (shims).
    # {"class": "managed-hf|managed-ollama|managed-ms|managed-flat",
    #  "store_path": "<relative-to-root>", "ingested_at": str, "shims": [ {...} ]}
    storage: dict = field(default_factory=dict)
```

- [ ] **Step 4: Run** the test → PASS. Run `make test` → all pass. `make lint` → clean.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_common.py
git commit -m "feat(sp2): ModelEntry.storage annotation field"
```

---

## Task 2: HF NativeReader (`_hf_read_native`)

**Files:** Modify `aim.py` (new SP2 section); Test `tests/test_ingest_hf.py`

Reads a HF cache repo dir (`models--org--repo`) → reconstruct metadata + real file list. HF layout: `refs/main` holds the commit hash; `snapshots/<commit>/<file>` are symlinks into `blobs/<sha>`; resolving a snapshot symlink yields the real bytes.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_hf.py
import unittest
import os
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def make_hf_cache(home: Path, org="Org", repo="Model", commit="abc123",
                  files=(("config.json", b'{"a":1}'), ("model.safetensors", b"WEIGHTS"))):
    """Build a realistic HF cache repo with blobs + snapshot symlinks."""
    repo_dir = home / "hub" / f"models--{org}--{repo}"
    blobs = repo_dir / "blobs"
    snap = repo_dir / "snapshots" / commit
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(commit)
    for i, (name, data) in enumerate(files):
        blob = blobs / f"sha{i}"
        blob.write_bytes(data)
        os.symlink(os.path.relpath(blob, snap), snap / name)
    return repo_dir


class HFReadNativeTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_read_native_lists_real_files_and_commit(self):
        repo_dir = make_hf_cache(self.home)
        info = aim._hf_read_native(repo_dir)
        self.assertEqual(info["repo_id"], "Org/Model")
        self.assertEqual(info["commit"], "abc123")
        names = sorted(f["name"] for f in info["files"])
        self.assertEqual(names, ["config.json", "model.safetensors"])
        # real_path resolves to the blob content
        weights = next(f for f in info["files"] if f["name"] == "model.safetensors")
        self.assertEqual(Path(weights["real_path"]).read_bytes(), b"WEIGHTS")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_ingest_hf -v` → FAIL (no `_hf_read_native`).

- [ ] **Step 3: Implement** — add the SP2 section header + function to `aim.py`:

```python
# ── Native Ingest (SP2) ──────────────────────────────────────────────────────


def _hf_read_native(repo_dir: Path) -> dict:
    """Read a HuggingFace cache repo dir (models--org--repo) into reconstruct metadata.
    Returns {repo_id, commit, files: [{name, real_path, size}]} using the current snapshot."""
    parts = repo_dir.name.split("--", 2)
    repo_id = f"{parts[1]}/{parts[2]}" if len(parts) == 3 else repo_dir.name
    refs_main = repo_dir / "refs" / "main"
    commit = refs_main.read_text().strip() if refs_main.exists() else ""
    snap = repo_dir / "snapshots" / commit
    if not snap.is_dir():
        # fall back to the single snapshot present
        snaps = [d for d in (repo_dir / "snapshots").iterdir() if d.is_dir()] if (repo_dir / "snapshots").exists() else []
        snap = snaps[0] if snaps else snap
        commit = commit or (snap.name if snaps else "")
    files = []
    if snap.is_dir():
        for f in sorted(snap.rglob("*")):
            if f.is_file() or f.is_symlink():
                real = f.resolve()
                if real.is_file():
                    files.append({"name": str(f.relative_to(snap)), "real_path": str(real),
                                  "size": real.stat().st_size})
    return {"repo_id": repo_id, "commit": commit, "files": files}
```

- [ ] **Step 4: Run** the test → PASS. `make test` → all pass. `make lint` → clean.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_hf.py
git commit -m "feat(sp2): HF native reader (snapshot deref -> real file list + commit)"
```

---

## Task 3: Generic `_ingest_to_store` (copy flat, no 2×)

**Files:** Modify `aim.py` (SP2 section); Test `tests/test_ingest_hf.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
class IngestToStoreTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_copies_flat_no_doubling(self):
        src = self.home / "src"
        _write(src / "config.json", b'{"a":1}')
        _write(src / "model.safetensors", b"W" * 100)
        files = [{"name": "config.json", "real_path": str(src / "config.json"), "size": 7},
                 {"name": "model.safetensors", "real_path": str(src / "model.safetensors"), "size": 100}]
        dest = self.home / "store" / "asr" / "model" / "m1"
        total = aim._ingest_to_store(files, dest)
        self.assertEqual(total, 107)
        self.assertEqual((dest / "model.safetensors").read_bytes(), b"W" * 100)
        # flat: no blobs/snapshots structure
        self.assertEqual(sorted(p.name for p in dest.iterdir()), ["config.json", "model.safetensors"])

    def test_nested_name_preserved(self):
        src = self.home / "src"
        _write(src / "sub" / "f.bin", b"z")
        files = [{"name": "sub/f.bin", "real_path": str(src / "sub" / "f.bin"), "size": 1}]
        dest = self.home / "store" / "x"
        aim._ingest_to_store(files, dest)
        self.assertTrue((dest / "sub" / "f.bin").exists())
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_ingest_hf.IngestToStoreTests -v` → FAIL.

- [ ] **Step 3: Implement** (add to SP2 section):

```python
def _ingest_to_store(files: list, dest: Path) -> int:
    """Copy each real file flat into dest/<name>. Returns total bytes. No CAS structure copied."""
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in files:
        target = dest / f["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f["real_path"], target)
        total += target.stat().st_size
    return total
```

- [ ] **Step 4: Run** → PASS. `make test`. `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_hf.py
git commit -m "feat(sp2): generic _ingest_to_store (flat copy, no doubling)"
```

---

## Task 4: HF ShimBuilder (`_hf_build_shim`)

**Files:** Modify `aim.py`; Test `tests/test_ingest_hf.py` (append)

Rebuild the HF cache repo as a shim: `refs/main` = commit; `snapshots/<commit>/<file>` = absolute symlinks into store; remove `blobs/`.

- [ ] **Step 1: Write the failing test (append)**

```python
class HFBuildShimTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_shim_snapshots_symlink_into_store(self):
        store = self.home / "store" / "asr" / "model" / "m1"
        _write(store / "config.json", b'{"a":1}')
        _write(store / "model.safetensors", b"WEIGHTS")
        repo_dir = self.home / "hub" / "models--Org--Model"
        files = [{"name": "config.json"}, {"name": "model.safetensors"}]
        aim._hf_build_shim(repo_dir, store, commit="abc123", files=files)
        self.assertEqual((repo_dir / "refs" / "main").read_text(), "abc123")
        link = repo_dir / "snapshots" / "abc123" / "model.safetensors"
        self.assertTrue(link.is_symlink())
        self.assertEqual(link.resolve(), (store / "model.safetensors").resolve())
        self.assertEqual(link.read_bytes(), b"WEIGHTS")  # loads through the shim
        self.assertFalse((repo_dir / "blobs").exists())   # blobs removed
```

- [ ] **Step 2: Run** → FAIL (no `_hf_build_shim`).

- [ ] **Step 3: Implement**:

```python
def _hf_build_shim(repo_dir: Path, store_dir: Path, commit: str, files: list) -> None:
    """Rebuild a HF cache repo as a shim: snapshots/<commit>/<file> -> absolute symlink into store."""
    blobs = repo_dir / "blobs"
    if blobs.exists():
        shutil.rmtree(blobs)
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(commit)
    snap = repo_dir / "snapshots" / commit
    if snap.exists():
        shutil.rmtree(snap)
    snap.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = snap / f["name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink((store_dir / f["name"]).resolve(), target)
```

- [ ] **Step 4: Run** → PASS. `make test`. `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_hf.py
git commit -m "feat(sp2): HF shim builder (snapshots symlink into store, blobs removed)"
```

---

## Task 5: `op_ingest` for HF (orchestration + rollback + dry-run)

**Files:** Modify `aim.py`; Test `tests/test_ingest_hf.py` (append)

Orchestrates HF: read → copy to store → build shim → annotate → cleanup, copy-first with rollback. (Ollama/MS branches added in Tasks 8/11.)

- [ ] **Step 1: Write the failing test (append)**

```python
class HFIngestTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.root = aim.StorageRoot(id="primary", path=str(self.home / "AI"))
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]

    def _entry_for(self, repo_dir):
        e = aim.ModelEntry(id="hf-org-model", name="Org/Model", native_cas=True,
                           source={"type": "huggingface", "repo_id": "Org/Model"},
                           category="asr/model", canonical={"root": "primary", "path": str(repo_dir)})
        return e

    def test_ingest_hf_end_to_end(self):
        repo_dir = make_hf_cache(self.home, files=(("config.json", b'{"a":1}'),
                                                   ("model.safetensors", b"W" * 50)))
        reg = aim.Registry()
        reg.models = [self._entry_for(repo_dir)]
        ok = aim.op_ingest(self.config, reg, "hf-org-model", registry_save=False)
        self.assertTrue(ok)
        e = reg.find("hf-org-model")
        self.assertFalse(e.native_cas)
        self.assertEqual(e.storage["class"], "managed-hf")
        store = Path(self.root.path) / e.storage["store_path"]
        self.assertEqual((store / "model.safetensors").read_bytes(), b"W" * 50)
        # shim still resolves
        link = repo_dir / "snapshots" / "abc123" / "model.safetensors"
        self.assertEqual(link.resolve().read_bytes(), b"W" * 50)
        # store total == real size (no doubling)
        total = sum(p.stat().st_size for p in store.rglob("*") if p.is_file() and not p.is_symlink())
        self.assertEqual(total, 50 + 7)

    def test_ingest_dry_run_changes_nothing(self):
        repo_dir = make_hf_cache(self.home)
        reg = aim.Registry()
        reg.models = [self._entry_for(repo_dir)]
        aim.op_ingest(self.config, reg, "hf-org-model", dry_run=True, registry_save=False)
        self.assertTrue((repo_dir / "blobs").exists())  # untouched
        self.assertEqual(reg.find("hf-org-model").storage, {})
```

- [ ] **Step 2: Run** → FAIL (no `op_ingest`).

- [ ] **Step 3: Implement** (add to SP2 section). This is the orchestration skeleton + HF branch; Tasks 8/11 extend the `class`-dispatch:

```python
def _sanitize_ingest_id(entry: "ModelEntry", new_id: str) -> str:
    return _sanitize_model_id(new_id) if new_id else entry.id


def op_ingest(config: dict, registry: "Registry", model_id: str, new_id: str = "",
              category: str = "", dry_run: bool = False, keep_native: bool = False,
              registry_save: bool = True, json_output: bool = False) -> bool:
    """Ingest a native-CAS model into the flat store + rebuild its load shim + annotate."""
    entry = registry.find(model_id)
    if not entry:
        print(f"Error: model '{model_id}' not found.", file=sys.stderr)
        return False
    if not entry.native_cas:
        print(f"Error: '{model_id}' is already managed (native_cas=false).", file=sys.stderr)
        return False
    tool = entry.source.get("type", "")
    root = get_primary_root(config)
    cache_repo = Path(entry.canonical.get("path", ""))
    if not cache_repo.is_absolute():
        cache_repo = Path(root.path) / entry.canonical.get("path", "")

    if tool == "huggingface":
        info = _hf_read_native(cache_repo)
        target_id = _sanitize_ingest_id(entry, new_id)
        target_cat = category or entry.category or "uncategorized"
        store_dir = (root.store_path / target_cat / target_id)
        store_rel = str(store_dir.relative_to(Path(root.path)))
        if dry_run:
            print(f"[dry-run] would ingest {model_id} ({len(info['files'])} files) -> {store_rel}, "
                  f"rebuild HF shim at {cache_repo}")
            return True
        if store_dir.exists():
            print(f"Error: store dir already exists: {store_dir}", file=sys.stderr)
            return False
        try:
            size = _ingest_to_store(info["files"], store_dir)          # copy-first
            _hf_build_shim(cache_repo, store_dir, info["commit"], info["files"])
        except OSError as ex:
            if store_dir.exists():
                shutil.rmtree(store_dir, ignore_errors=True)           # rollback
            print(f"Error: ingest failed, rolled back: {ex}", file=sys.stderr)
            return False
        entry.native_cas = False
        entry.id = target_id
        entry.category = target_cat
        entry.size_bytes = size
        entry.canonical = {"root": root.id, "path": store_rel}
        entry.storage = {
            "class": "managed-hf", "store_path": store_rel, "ingested_at": _now_iso(),
            "shims": [{
                "tool": "huggingface", "kind": "hf-cas",
                "location": str(cache_repo.relative_to(Path(root.path))) if str(cache_repo).startswith(str(root.path)) else str(cache_repo),
                "cache_root_var": "HF_HOME",
                "reconstruct": {"repo_id": info["repo_id"], "commit": info["commit"],
                                "files": [f["name"] for f in info["files"]]},
            }],
        }
        registry.add(entry)
        if registry_save:
            registry.save()
        print(f"Ingested {model_id} -> {store_rel}")
        return True

    print(f"Error: ingest not yet implemented for source type '{tool}'.", file=sys.stderr)
    return False
```

> `_sanitize_model_id`, `_now_iso`, `get_primary_root`, `StorageRoot.store_path` already exist in `aim.py`.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_ingest_hf -v` → PASS. `make test`. `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_hf.py
git commit -m "feat(sp2): op_ingest HF pipeline (read->store->shim->annotate, rollback, dry-run)"
```

---

## Task 6: HF ingest rollback on shim failure

**Files:** Test `tests/test_ingest_hf.py` (append) — verifies the rollback path from Task 5.

- [ ] **Step 1: Write the failing test (append)**

```python
class HFRollbackTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]

    def test_rollback_on_shim_failure(self):
        repo_dir = make_hf_cache(self.home)
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="hf-org-model", native_cas=True,
                      source={"type": "huggingface", "repo_id": "Org/Model"},
                      category="asr/model", canonical={"root": "primary", "path": str(repo_dir)})]
        orig = aim._hf_build_shim
        aim._hf_build_shim = lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
        try:
            ok = aim.op_ingest(self.config, reg, "hf-org-model", registry_save=False)
        finally:
            aim._hf_build_shim = orig
        self.assertFalse(ok)
        e = reg.find("hf-org-model")
        self.assertTrue(e.native_cas)            # unchanged
        self.assertEqual(e.storage, {})          # not annotated
        store_dir = Path(self.config["roots"][0]["path"]) / "store" / "asr/model" / "hf-org-model"
        self.assertFalse(store_dir.exists())     # half-built store cleaned up
        self.assertTrue((repo_dir / "blobs").exists())  # native intact (copy-first)
```

- [ ] **Step 2: Run** → it should PASS already if Task 5's rollback is correct. If it FAILS, fix `op_ingest` so that on exception it (a) removes the partial `store_dir`, (b) does NOT mutate `entry` fields, (c) leaves native intact. (In Task 5's code, entry mutation happens only after the try-block succeeds, so this holds.)

- [ ] **Step 3: Run** `make test` → all pass.

- [ ] **Step 4: Commit**
```bash
git add tests/test_ingest_hf.py
git commit -m "test(sp2): HF ingest rolls back cleanly on shim failure"
```

---

## Task 7: `ModelScopeAdapter` (scan flat-ms, both layouts)

**Files:** Modify `aim.py` (add adapter, register in `ADAPTERS`/`ENGINE_NAMES`/`default_config`); Test `tests/test_ingest_ms.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_ms.py
import unittest
import os
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def make_ms_cache(cache: Path, layout="models", org="Qwen", repo="Qwen3-ASR-0___6B"):
    base = cache / ("hub/models" if layout == "hub-models" else "models")
    d = base / org / repo
    _write(d / "config.json", b'{"a":1}')
    _write(d / "model.safetensors", b"W" * 20)
    _write(d / ".msc", b"meta")
    _write(d / ".mdl", b"meta")
    return d


class ModelScopeAdapterTests(unittest.TestCase):
    def setUp(self):
        self.cache = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["sources"]["modelscope"] = {"cache_path": str(self.cache)}
        self.root = aim.StorageRoot(id="primary", path=str(self.cache / "AI"))

    def test_registered(self):
        self.assertIn("modelscope", aim.ADAPTERS)
        self.assertIn("modelscope", aim.ENGINE_NAMES)

    def test_scan_finds_models_both_layouts(self):
        make_ms_cache(self.cache, "models", "Qwen", "Qwen3-ASR-0___6B")
        make_ms_cache(self.cache, "hub-models", "funasr", "paraformer-zh")
        (self.cache / "models" / "._____temp").mkdir(parents=True, exist_ok=True)  # must be skipped
        ad = aim.ModelScopeAdapter(self.config, self.root)
        scanned = ad.scan()
        repo_ids = sorted(s.source["repo_id"] for s in scanned)
        self.assertIn("Qwen/Qwen3-ASR-0.6B", repo_ids)   # ___ -> . un-munged
        self.assertIn("funasr/paraformer-zh", repo_ids)
        self.assertTrue(all(s.native_cas and s.is_directory for s in scanned))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_ingest_ms -v` → FAIL.

- [ ] **Step 3: Implement** — add the adapter (place near the other adapters, before the `ADAPTERS` dict), and register it.

```python
class ModelScopeAdapter(EngineAdapter):
    name = "modelscope"

    def supported_formats(self) -> list[str]:
        return ["safetensors", "bin", "pt", "gguf"]

    def scan(self) -> list[ScannedModel]:
        results: list[ScannedModel] = []
        base = self.base_path
        roots = [base / "models", base / "hub" / "models"]
        for layout_root in roots:
            if not layout_root.exists():
                continue
            for org_dir in sorted(layout_root.iterdir()):
                if not org_dir.is_dir() or org_dir.name.startswith((".", "_")):
                    continue
                for repo_dir in sorted(org_dir.iterdir()):
                    if not repo_dir.is_dir() or repo_dir.name.startswith((".", "_")):
                        continue
                    has_real = any(f.is_file() for f in repo_dir.rglob("*"))
                    if not has_real:
                        continue
                    repo_name = repo_dir.name.replace("___", ".")
                    repo_id = f"{org_dir.name}/{repo_name}"
                    fmt = ""
                    for f in repo_dir.iterdir():
                        if f.suffix == ".safetensors":
                            fmt = "safetensors"; break
                        if f.suffix in (".bin", ".pt", ".gguf"):
                            fmt = fmt or f.suffix[1:]
                    results.append(ScannedModel(
                        id=self._make_id(f"ms-{org_dir.name}-{repo_name}"),
                        name=repo_id, path=str(repo_dir), engine=self.name, format=fmt,
                        size_bytes=self._dir_size(repo_dir),
                        category=self._infer_ms_category(repo_id),
                        tags=["modelscope", org_dir.name],
                        source={"type": "modelscope", "repo_id": repo_id},
                        native_cas=True, is_directory=True))
        return results

    def _infer_ms_category(self, repo_id: str) -> str:
        rl = repo_id.lower()
        if any(k in rl for k in ["asr", "paraformer", "whisper", "sensevoice", "firered"]):
            return "asr/model"
        if any(k in rl for k in ["tts", "vocoder"]):
            return "tts/model"
        return "llm/chat"

    def provision(self, model: ModelEntry, store_path: Path, options: dict = None) -> list[Provision]:
        return []  # native_cas; shim built by op_ingest

    def unprovision(self, model: ModelEntry) -> bool:
        return False
```

Register in `ADAPTERS` (add `"modelscope": ModelScopeAdapter,`), add `"modelscope"` to `ENGINE_NAMES`, and add to `default_config()["engines"]`:
```python
            "modelscope":  {"enabled": True, "model_dir": "modelscope", "native_cas": True},
```

> `_make_id`, `_dir_size`, `Provision` already exist. `base_path` reads `sources.modelscope.cache_path` via SP1's `EngineAdapter.__init__` (since `native_cas=True` and cache_path is set).

- [ ] **Step 4: Run** `python3 -m unittest tests.test_ingest_ms -v` → PASS. `make test`. `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_ms.py
git commit -m "feat(sp2): ModelScopeAdapter (scan flat-ms, both layouts, un-munge dir names)"
```

---

## Task 8: MS reader + shim + ingest branch

**Files:** Modify `aim.py`; Test `tests/test_ingest_ms.py` (append)

MS model = flat dir of real files (incl. `.msc/.mdl/.mv`). Reader lists them; ingest copies them flat to store; shim = replace cache dir with a directory symlink → store.

- [ ] **Step 1: Write the failing test (append)**

```python
class MSIngestTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]

    def test_ms_read_native(self):
        d = make_ms_cache(self.home / "cache", "models", "Qwen", "Qwen3-ASR-0___6B")
        info = aim._ms_read_native(d)
        names = sorted(f["name"] for f in info["files"])
        self.assertIn("model.safetensors", names)
        self.assertIn(".msc", names)  # metadata carried into store

    def test_ms_ingest_dir_symlink_shim(self):
        d = make_ms_cache(self.home / "cache", "models", "Qwen", "Qwen3-ASR-0___6B")
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="ms-qwen", native_cas=True,
                      source={"type": "modelscope", "repo_id": "Qwen/Qwen3-ASR-0.6B"},
                      category="asr/model", canonical={"root": "primary", "path": str(d)})]
        ok = aim.op_ingest(self.config, reg, "ms-qwen", registry_save=False)
        self.assertTrue(ok)
        e = reg.find("ms-qwen")
        self.assertEqual(e.storage["class"], "managed-ms")
        store = Path(self.config["roots"][0]["path"]) / e.storage["store_path"]
        self.assertEqual((store / "model.safetensors").read_bytes(), b"W" * 20)
        self.assertTrue((store / ".msc").exists())
        # cache dir replaced by symlink -> store
        self.assertTrue(d.is_symlink())
        self.assertEqual(d.resolve(), store.resolve())
        self.assertEqual((d / "model.safetensors").read_bytes(), b"W" * 20)  # loads through shim
```

- [ ] **Step 2: Run** → FAIL (no `_ms_read_native` / MS branch).

- [ ] **Step 3: Implement** — add reader + shim, and an MS branch in `op_ingest`.

```python
def _ms_read_native(repo_dir: Path) -> dict:
    files = []
    for f in sorted(repo_dir.rglob("*")):
        if f.is_file() and not f.is_symlink():
            files.append({"name": str(f.relative_to(repo_dir)), "real_path": str(f),
                          "size": f.stat().st_size})
    return {"files": files, "dir_name": repo_dir.name}


def _ms_build_shim(repo_dir: Path, store_dir: Path) -> None:
    """Replace the MS cache model dir with a directory symlink -> store."""
    if repo_dir.exists() or repo_dir.is_symlink():
        if repo_dir.is_symlink() or repo_dir.is_file():
            repo_dir.unlink()
        else:
            shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(store_dir.resolve(), repo_dir)
```

In `op_ingest`, add this branch (before the final "not implemented" line):
```python
    if tool == "modelscope":
        info = _ms_read_native(cache_repo)
        target_id = _sanitize_ingest_id(entry, new_id)
        target_cat = category or entry.category or "uncategorized"
        store_dir = (root.store_path / target_cat / target_id)
        store_rel = str(store_dir.relative_to(Path(root.path)))
        if dry_run:
            print(f"[dry-run] would ingest {model_id} ({len(info['files'])} files) -> {store_rel}, "
                  f"replace MS dir {cache_repo} with symlink")
            return True
        if store_dir.exists():
            print(f"Error: store dir already exists: {store_dir}", file=sys.stderr)
            return False
        try:
            size = _ingest_to_store(info["files"], store_dir)
            _ms_build_shim(cache_repo, store_dir)
        except OSError as ex:
            if store_dir.exists():
                shutil.rmtree(store_dir, ignore_errors=True)
            print(f"Error: ingest failed, rolled back: {ex}", file=sys.stderr)
            return False
        entry.native_cas = False
        entry.id = target_id
        entry.category = target_cat
        entry.size_bytes = size
        entry.canonical = {"root": root.id, "path": store_rel}
        entry.storage = {
            "class": "managed-ms", "store_path": store_rel, "ingested_at": _now_iso(),
            "shims": [{
                "tool": "modelscope", "kind": "ms-dir",
                "location": str(cache_repo.relative_to(Path(root.path))) if str(cache_repo).startswith(str(root.path)) else str(cache_repo),
                "cache_root_var": "MODELSCOPE_CACHE",
                "reconstruct": {"repo_id": entry.source.get("repo_id", ""), "dir_name": info["dir_name"]},
            }],
        }
        registry.add(entry)
        if registry_save:
            registry.save()
        print(f"Ingested {model_id} -> {store_rel}")
        return True
```

- [ ] **Step 4: Run** `python3 -m unittest tests.test_ingest_ms -v` → PASS. `make test`. `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_ms.py
git commit -m "feat(sp2): ModelScope ingest (flat copy + dir-symlink shim + annotation)"
```

---

## Task 9: Ollama reader + shim + ingest branch

**Files:** Modify `aim.py`; Test `tests/test_ingest_ollama.py`

Ollama layout: `manifests/<registry>/<ns>/<model>/<tag>` (JSON with `layers:[{digest,size,mediaType}]` + `config:{digest}`); `blobs/sha256-<hex>`. Largest layer (the GGUF) → store; blobs hardlinked back; small blobs copied; manifest rewritten.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest_ollama.py
import unittest
import os
import json
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def make_ollama_cache(home: Path, model="qwen3", tag="latest"):
    """Build OLLAMA_MODELS with one model: a big GGUF layer + a tiny params layer + config."""
    gguf = b"GGUF" + b"W" * 200
    params = b'{"stop":["</s>"]}'
    cfg = b'{"model_format":"gguf"}'
    import hashlib
    def dg(b): return "sha256-" + hashlib.sha256(b).hexdigest()
    blobs = home / "blobs"
    for b in (gguf, params, cfg):
        _write(blobs / dg(b), b)
    manifest = {
        "schemaVersion": 2,
        "config": {"digest": dg(cfg), "size": len(cfg), "mediaType": "application/vnd.ollama.image.config"},
        "layers": [
            {"digest": dg(gguf), "size": len(gguf), "mediaType": "application/vnd.ollama.image.model"},
            {"digest": dg(params), "size": len(params), "mediaType": "application/vnd.ollama.image.params"},
        ],
    }
    man_path = home / "manifests" / "registry.ollama.ai" / "library" / model / tag
    _write(man_path, json.dumps(manifest).encode())
    return man_path, dg(gguf), gguf


class OllamaIngestTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]

    def test_read_native_finds_gguf_layer(self):
        man_path, gguf_digest, gguf = make_ollama_cache(self.home)
        info = aim._ollama_read_native(man_path, self.home)
        self.assertEqual(info["gguf"]["digest"], gguf_digest)
        self.assertEqual(Path(info["gguf"]["real_path"]).read_bytes(), gguf)
        self.assertEqual(len(info["small_blobs"]), 2)  # params + config

    def test_ingest_ollama_hardlinks_blob_back(self):
        man_path, gguf_digest, gguf = make_ollama_cache(self.home)
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="ollama-qwen3", native_cas=True,
                      source={"type": "ollama", "repo_id": "qwen3:latest"},
                      category="llm/chat", canonical={"root": "primary", "path": str(man_path)})]
        ok = aim.op_ingest(self.config, reg, "ollama-qwen3", registry_save=False)
        self.assertTrue(ok)
        e = reg.find("ollama-qwen3")
        self.assertEqual(e.storage["class"], "managed-ollama")
        store = Path(self.config["roots"][0]["path"]) / e.storage["store_path"]
        gguf_store = store / "ollama-qwen3.gguf"
        self.assertEqual(gguf_store.read_bytes(), gguf)
        # blob hardlinked back to store gguf (same inode)
        blob = self.home / "blobs" / gguf_digest
        self.assertEqual(os.stat(blob).st_ino, os.stat(gguf_store).st_ino)
        # manifest restored
        self.assertTrue(man_path.exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_ingest_ollama -v` → FAIL.

- [ ] **Step 3: Implement** — reader + shim + ingest branch.

```python
def _ollama_read_native(manifest_path: Path, models_root: Path) -> dict:
    """Parse an ollama manifest into {model, tag, manifest, gguf:{digest,real_path,size}, small_blobs:[...]}."""
    manifest = json.loads(manifest_path.read_text())
    blobs_dir = models_root / "blobs"
    layers = list(manifest.get("layers", []))
    cfg = manifest.get("config")
    gguf_layer = max(layers, key=lambda l: l.get("size", 0)) if layers else None
    small = [l for l in layers if l is not gguf_layer]
    if cfg:
        small.append(cfg)
    def real(digest):
        return str(blobs_dir / digest)
    rel = manifest_path.relative_to(models_root / "manifests")
    parts = rel.parts  # <registry>/<ns>/<model>/<tag>
    model = parts[-2] if len(parts) >= 2 else manifest_path.parent.name
    tag = parts[-1]
    return {
        "model": model, "tag": tag, "manifest": manifest,
        "gguf": {"digest": gguf_layer["digest"], "real_path": real(gguf_layer["digest"]),
                 "size": gguf_layer.get("size", 0)} if gguf_layer else None,
        "small_blobs": [{"digest": b["digest"], "real_path": real(b["digest"]),
                         "size": b.get("size", 0)} for b in small],
        "manifest_rel": str(rel),
    }


def _ollama_build_shim(info: dict, store_dir: Path, models_root: Path) -> None:
    """Hardlink the GGUF blob back from store; copy small blobs; rewrite the manifest."""
    blobs_dir = models_root / "blobs"
    blobs_dir.mkdir(parents=True, exist_ok=True)
    gguf_blob = blobs_dir / info["gguf"]["digest"]
    if gguf_blob.exists() or gguf_blob.is_symlink():
        gguf_blob.unlink()
    store_gguf = store_dir / f"{store_dir.name}.gguf"
    if LinkManager.same_volume(store_gguf, gguf_blob):
        os.link(store_gguf, gguf_blob)
    else:
        shutil.copy2(store_gguf, gguf_blob)
    for b in info["small_blobs"]:
        sb = blobs_dir / b["digest"]
        if not sb.exists():
            shutil.copy2(store_dir / f"blob-{b['digest']}", sb)
    man_path = models_root / "manifests" / Path(info["manifest_rel"])
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(info["manifest"]))
```

In `op_ingest`, add this branch (before the final "not implemented" line):
```python
    if tool == "ollama":
        models_root = cache_repo.parents[len(Path(entry.canonical.get("path","")).relative_to(Path(root.path)).parts) - 4] if False else _ollama_models_root(cache_repo)
        info = _ollama_read_native(cache_repo, models_root)
        if not info["gguf"]:
            print(f"Error: no GGUF layer in {model_id}", file=sys.stderr)
            return False
        target_id = _sanitize_ingest_id(entry, new_id)
        target_cat = category or entry.category or "llm/chat"
        store_dir = (root.store_path / target_cat / target_id)
        store_rel = str(store_dir.relative_to(Path(root.path)))
        if dry_run:
            print(f"[dry-run] would ingest {model_id} (gguf {info['gguf']['size']}B) -> {store_rel}, "
                  f"hardlink blob + rewrite manifest")
            return True
        if store_dir.exists():
            print(f"Error: store dir already exists: {store_dir}", file=sys.stderr)
            return False
        try:
            store_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(info["gguf"]["real_path"], store_dir / f"{target_id}.gguf")
            for b in info["small_blobs"]:
                shutil.copy2(b["real_path"], store_dir / f"blob-{b['digest']}")
            (store_dir / "manifest.json").write_text(json.dumps(info["manifest"]))
            size = sum(p.stat().st_size for p in store_dir.rglob("*") if p.is_file())
            _ollama_build_shim(info, store_dir, models_root)
        except OSError as ex:
            if store_dir.exists():
                shutil.rmtree(store_dir, ignore_errors=True)
            print(f"Error: ingest failed, rolled back: {ex}", file=sys.stderr)
            return False
        entry.native_cas = False
        entry.id = target_id
        entry.category = target_cat
        entry.size_bytes = size
        entry.format = "gguf"
        entry.canonical = {"root": root.id, "path": store_rel}
        entry.storage = {
            "class": "managed-ollama", "store_path": store_rel, "ingested_at": _now_iso(),
            "shims": [{
                "tool": "ollama", "kind": "ollama-cas",
                "location": info["manifest_rel"], "cache_root_var": "OLLAMA_MODELS",
                "reconstruct": {"model": info["model"], "tag": info["tag"], "manifest": info["manifest"],
                                "gguf_digest": info["gguf"]["digest"],
                                "small_blobs": [b["digest"] for b in info["small_blobs"]]},
            }],
        }
        registry.add(entry)
        if registry_save:
            registry.save()
        print(f"Ingested {model_id} -> {store_rel}")
        return True
```

Also add the helper (the `models_root` is the dir containing `blobs/` and `manifests/`; for a manifest at `<root>/manifests/.../tag`, walk up to the parent of `manifests`):
```python
def _ollama_models_root(manifest_path: Path) -> Path:
    p = manifest_path
    while p.parent != p and p.name != "manifests":
        p = p.parent
    return p.parent  # the dir containing manifests/ and blobs/
```
> Replace the convoluted `models_root = ...if False else _ollama_models_root(cache_repo)` line with simply `models_root = _ollama_models_root(cache_repo)`.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_ingest_ollama -v` → PASS. `make test`. `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_ollama.py
git commit -m "feat(sp2): Ollama ingest (gguf->store, hardlink blob, rewrite manifest, annotate)"
```

---

## Task 10: `op_ingest` CLI (`aim ingest` + `--all-native`) + parser + dispatch

**Files:** Modify `aim.py` (parser + main dispatch + a bulk helper); Test `tests/test_ingest_common.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
import io, tempfile
from contextlib import redirect_stdout
from pathlib import Path
import os


class IngestCliTests(unittest.TestCase):
    def test_parser_accepts_ingest(self):
        args = aim.build_parser().parse_args(["ingest", "m1", "--dry-run"])
        self.assertEqual(args.command, "ingest")
        self.assertEqual(args.model_id, "m1")
        self.assertTrue(args.dry_run)

    def test_parser_accepts_all_native(self):
        args = aim.build_parser().parse_args(["ingest", "--all-native"])
        self.assertTrue(args.all_native)

    def test_ingest_all_native_skips_non_native(self):
        # op_ingest_all should only touch native_cas models; here none are native -> 0 ingested
        cfg = aim.default_config()
        home = Path(tempfile.mkdtemp())
        cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="managed", native_cas=False)]
        buf = io.StringIO()
        with redirect_stdout(buf):
            n = aim.op_ingest_all(cfg, reg, dry_run=True, registry_save=False)
        self.assertEqual(n, 0)
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_ingest_common.IngestCliTests -v` → FAIL.

- [ ] **Step 3: Implement** — bulk helper + parser + dispatch.

Add to SP2 section:
```python
def op_ingest_all(config: dict, registry: "Registry", dry_run: bool = False,
                  keep_native: bool = False, registry_save: bool = True) -> int:
    native = [m.id for m in list(registry.models) if m.native_cas]
    done = 0
    for mid in native:
        if op_ingest(config, registry, mid, dry_run=dry_run, keep_native=keep_native,
                     registry_save=False):
            done += 1
    if registry_save and not dry_run:
        registry.save()
    print(f"Ingested {done}/{len(native)} native model(s).")
    return done
```

In `build_parser()`, after the `convert` subparser, add:
```python
    p = sub.add_parser("ingest", help="Ingest a native-CAS model into the store + rebuild its load shim")
    p.add_argument("model_id", nargs="?", default="", help="Model id (omit with --all-native)")
    p.add_argument("--all-native", dest="all_native", action="store_true", help="Ingest all native_cas models")
    p.add_argument("--new-id", dest="new_id", default="", help="Rename on ingest")
    p.add_argument("--category", default="", help="Override category")
    p.add_argument("--keep-native", dest="keep_native", action="store_true", help="Keep original native bytes")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--json", dest="json_output", action="store_true")
```

In `main()`, add a dispatch branch (e.g. after the `convert` branch):
```python
    elif cmd == "ingest":
        if getattr(args, "all_native", False):
            rc = op_ingest_all(config, registry, dry_run=args.dry_run,
                               keep_native=args.keep_native)
            return EXIT_OK
        if not args.model_id:
            print("Usage: aim ingest <model_id> | --all-native", file=sys.stderr)
            return EXIT_INVALID_ARGS
        ok = op_ingest(config, registry, args.model_id, new_id=args.new_id,
                       category=args.category, dry_run=args.dry_run,
                       keep_native=args.keep_native, json_output=args.json_output)
        return EXIT_OK if ok else EXIT_FAILED
```

- [ ] **Step 4: Run** `python3 -m unittest tests.test_ingest_common -v` → PASS. `make test`. `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_common.py
git commit -m "feat(sp2): aim ingest CLI (single + --all-native)"
```

---

## Task 11: Rework `op_convert_native_to_store` to delegate to ingest

**Files:** Modify `aim.py` (`op_convert_native_to_store`); Test `tests/test_ingest_common.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
class ConvertDelegationTests(unittest.TestCase):
    def test_convert_delegates_to_ingest_and_builds_shim(self):
        from tests.test_ingest_hf import make_hf_cache  # reuse fixture
        home = Path(tempfile.mkdtemp())
        cfg = aim.default_config()
        cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        repo_dir = make_hf_cache(home, files=(("config.json", b"{}"), ("model.safetensors", b"W" * 30)))
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="hf-org-model", native_cas=True,
                      source={"type": "huggingface", "repo_id": "Org/Model"},
                      category="asr/model", canonical={"root": "primary", "path": str(repo_dir)})]
        buf = io.StringIO()
        with redirect_stdout(buf):
            ok = aim.op_convert_native_to_store(cfg, reg, "hf-org-model", json_output=False)
        self.assertTrue(ok)
        e = reg.find("hf-org-model")
        self.assertEqual(e.storage["class"], "managed-hf")          # went through ingest
        self.assertFalse((repo_dir / "blobs").exists())            # shim built (blobs gone)
        self.assertIn("deprecated", buf.getvalue().lower())        # deprecation notice
```

> Note: this test imports `make_hf_cache` from `tests.test_ingest_hf`; run the full suite from the repo root so the import resolves.

- [ ] **Step 2: Run** → FAIL (old convert copies CAS, no storage, no deprecation notice).

- [ ] **Step 3: Implement** — replace the body of `op_convert_native_to_store` with a thin delegator (keep the signature for backward compat):

```python
def op_convert_native_to_store(
    config: dict,
    registry: Registry,
    model_id: str,
    new_id: str = "",
    category: str = "",
    mode: str = "copy",
    keep_native: bool = True,
    json_output: bool = False,
) -> bool:
    """Deprecated: delegates to op_ingest (correct flat ingest + shim + annotation)."""
    if not json_output:
        print("Note: 'aim convert' is deprecated; use 'aim ingest'. Delegating to ingest...")
    return op_ingest(config, registry, model_id, new_id=new_id, category=category,
                     keep_native=keep_native, json_output=json_output)
```

> The `mode` parameter is now ignored (ingest always copies-then-cleans). The dispatch in `main()` for `convert` already passes these args; leaving it as-is is fine.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_ingest_common -v` → PASS. `make test`. `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_common.py
git commit -m "refactor(sp2): aim convert delegates to ingest (deprecation notice)"
```

---

## Task 12: Extend `op_verify` — check + `--fix` rebuild shims from annotation

**Files:** Modify `aim.py` (`op_verify` + a `_rebuild_shim_from_storage` helper); Test `tests/test_ingest_common.py` (append)

- [ ] **Step 1: Write the failing test (append)** — round-trip: ingest HF, delete the shim, verify detects it, `--fix` rebuilds it.

```python
class VerifyShimTests(unittest.TestCase):
    def test_verify_fix_rebuilds_hf_shim(self):
        from tests.test_ingest_hf import make_hf_cache
        home = Path(tempfile.mkdtemp())
        cfg = aim.default_config()
        cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        repo_dir = make_hf_cache(home, files=(("config.json", b"{}"), ("model.safetensors", b"W" * 40)))
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="hf-org-model", native_cas=True,
                      source={"type": "huggingface", "repo_id": "Org/Model"},
                      category="asr/model", canonical={"root": "primary", "path": str(repo_dir)})]
        aim.op_ingest(cfg, reg, "hf-org-model", registry_save=False)
        # destroy the shim
        import shutil as _sh
        _sh.rmtree(repo_dir)
        self.assertFalse(repo_dir.exists())
        # verify --fix rebuilds it from storage annotation
        buf = io.StringIO()
        with redirect_stdout(buf):
            aim.op_verify(cfg, reg, fix=True)
        link = repo_dir / "snapshots" / "abc123" / "model.safetensors"
        self.assertTrue(link.exists())
        self.assertEqual(link.resolve().read_bytes(), b"W" * 40)
```

- [ ] **Step 2: Run** → FAIL (verify doesn't know about storage shims).

- [ ] **Step 3: Implement** — add a rebuild helper and hook it into `op_verify`.

Add to SP2 section:
```python
def _rebuild_shim_from_storage(config: dict, entry: "ModelEntry") -> bool:
    """Rebuild a model's load shim(s) from its storage annotation. Returns True if rebuilt."""
    storage = entry.storage or {}
    if not storage.get("shims"):
        return False
    root = get_primary_root(config)
    store_dir = Path(root.path) / storage["store_path"]
    rebuilt = False
    for shim in storage["shims"]:
        cache_root = os.environ.get(shim.get("cache_root_var", ""), "")
        # location is relative to the root in tests/native ingest; resolve under root path
        loc = shim["location"]
        cache_path = Path(loc) if os.path.isabs(loc) else (Path(root.path) / loc)
        rc = shim.get("reconstruct", {})
        if shim["kind"] == "hf-cas":
            files = [{"name": n} for n in rc.get("files", [])]
            _hf_build_shim(cache_path, store_dir, rc.get("commit", ""), files)
            rebuilt = True
        elif shim["kind"] == "ms-dir":
            _ms_build_shim(cache_path, store_dir)
            rebuilt = True
        elif shim["kind"] == "ollama-cas":
            info = {"gguf": {"digest": rc.get("gguf_digest", "")},
                    "small_blobs": [{"digest": d} for d in rc.get("small_blobs", [])],
                    "manifest": rc.get("manifest", {}), "manifest_rel": loc}
            models_root = cache_path
            while models_root.name and models_root.name != "manifests" and models_root.parent != models_root:
                models_root = models_root.parent
            _ollama_build_shim(info, store_dir, models_root.parent)
            rebuilt = True
    return rebuilt
```

In `op_verify`, after its existing provision checks (find the function via `grep -n "^def op_verify"`), before it returns/prints the summary, add a pass over storage shims. Insert near the end of the per-model loop or as a second loop:
```python
    # SP2: verify storage shims resolve to store; rebuild with --fix.
    for m in registry.models:
        st = getattr(m, "storage", {}) or {}
        if not st.get("shims"):
            continue
        root = get_primary_root(config)
        store_dir = Path(root.path) / st.get("store_path", "")
        for shim in st["shims"]:
            loc = shim["location"]
            cache_path = Path(loc) if os.path.isabs(loc) else (Path(root.path) / loc)
            ok = cache_path.exists() or cache_path.is_symlink()
            if not ok:
                issues.append({"model": m.id, "error": "shim_missing", "path": str(cache_path)})
                if fix and _rebuild_shim_from_storage(config, m):
                    print(f"  Fixed shim for {m.id}")
```
> Use whatever list the function already accumulates into (it currently builds a list of issue dicts — match its existing variable name, commonly `issues` or `problems`; read the function first and reuse it). If `op_verify` doesn't already iterate `registry.models`, add this as a standalone block before its return.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_ingest_common -v` → PASS. `make test`. `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_ingest_common.py
git commit -m "feat(sp2): aim verify checks/rebuilds storage shims from annotation"
```

---

## Task 13: dedup interaction (cross-source identical files hardlinked)

**Files:** Test `tests/test_ingest_common.py` (append). Verifies existing `op_dedup` works on two store entries produced by ingest.

- [ ] **Step 1: Write the failing/【likely passing】 test (append)**

```python
class DedupInteractionTests(unittest.TestCase):
    def test_dedup_hardlinks_identical_store_files(self):
        from tests.test_ingest_hf import make_hf_cache
        from tests.test_ingest_ms import make_ms_cache
        home = Path(tempfile.mkdtemp())
        cfg = aim.default_config()
        cfg["roots"] = [{"id": "primary", "path": str(home / "AI")}]
        big = b"IDENTICAL" * 20000  # >100KB so dedup considers it
        # HF copy
        hf = make_hf_cache(home, org="X", repo="M", files=(("model.safetensors", big),))
        # MS copy of the same bytes
        msd = home / "mscache" / "models" / "X" / "M"
        _write(msd / "model.safetensors", big)
        reg = aim.Registry()
        reg.models = [
            aim.ModelEntry(id="hf-x-m", native_cas=True, source={"type": "huggingface", "repo_id": "X/M"},
                           category="asr/model", canonical={"root": "primary", "path": str(hf)}),
            aim.ModelEntry(id="ms-x-m", native_cas=True, source={"type": "modelscope", "repo_id": "X/M"},
                           category="asr/model", canonical={"root": "primary", "path": str(msd)}),
        ]
        aim.op_ingest(cfg, reg, "hf-x-m", registry_save=False)
        aim.op_ingest(cfg, reg, "ms-x-m", registry_save=False)
        root = Path(cfg["roots"][0]["path"])
        f1 = root / reg.find("hf-x-m").storage["store_path"] / "model.safetensors"
        f2 = root / reg.find("ms-x-m").storage["store_path"] / "model.safetensors"
        self.assertNotEqual(os.stat(f1).st_ino, os.stat(f2).st_ino)  # two copies before dedup
        aim.op_dedup(cfg, reg, scan_only=False)  # existing dedup
        self.assertEqual(os.stat(f1).st_ino, os.stat(f2).st_ino)     # one physical copy after
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_ingest_common.DedupInteractionTests -v`.
- If PASS: existing `op_dedup` already covers store files — done, proceed to commit.
- If FAIL: inspect `op_dedup` (`grep -n "^def op_dedup"`); it scans `>100MB` files only — adjust the test's `big` to exceed the real threshold, OR if dedup doesn't scan `store/`, that's a real gap — report it and add a minimal fix so dedup includes `store/`. (Do not change the dedup threshold to accommodate tests; size the fixture to the real threshold.)

> Check the actual threshold first: `grep -n "100" aim.py | grep -i dedup` or read `op_dedup`. Size `big` above it (the 100KB above is a placeholder — set it to exceed the real MB threshold, e.g. `b"X" * (101*1024*1024)` is wasteful in a test; instead, if the threshold is 100MB, temporarily call `op_dedup` with a lowered threshold if it accepts one, or assert dedup is invoked correctly on smaller files by reading how the threshold is parameterized). If the threshold is not parameterizable and is 100MB, mark this test `@unittest.skip("dedup threshold 100MB impractical for unit test")` and instead unit-test `_quick_hash` equality on the two store files directly.

- [ ] **Step 3:** Run `make test` → all pass.

- [ ] **Step 4: Commit**
```bash
git add tests/test_ingest_common.py aim.py
git commit -m "test(sp2): cross-source ingest dedups via existing aim dedup"
```

---

## Task 14: docs + final suite

**Files:** Modify `README.md`, `AIM-MANUAL.md`

- [ ] **Step 1: README.md** — replace the `aim convert` line in the Usage block with:
```markdown
# Ingest a native-cache model (HF/Ollama/ModelScope) into the store + rebuild its load shim
aim ingest <model_id>                 # one model
aim ingest --all-native               # all native_cas models
aim ingest <model_id> --dry-run --keep-native
aim convert <model_id>                # deprecated alias -> ingest
aim verify --fix                      # also rebuilds storage shims from annotation
```

- [ ] **Step 2: AIM-MANUAL.md** — after the `aim convert` section (or near it), add a Chinese `### aim ingest — 原生模型摇入 store` section: explain it copies the real files flat into `store/{类别}/{id}`, rebuilds the tool's load shim (HF snapshots 软链 / Ollama blob 硬链 / MS 目录软链), records the `storage` annotation, supports `--all-native`/`--dry-run`/`--keep-native`, and that `aim convert` is now a deprecated alias. Note `aim verify --fix` rebuilds broken shims from the annotation.

- [ ] **Step 3:** Run `make lint && make test` → all PASS.

- [ ] **Step 4: Commit**
```bash
git add README.md AIM-MANUAL.md
git commit -m "docs(sp2): document aim ingest, convert deprecation, verify --fix"
```

---

## Self-Review

**1. Spec coverage:**
- §1 components → Tasks 2–12 (NativeReader: T2/T8/T9; StoreIngestor: T3; ShimBuilder: T4/T8/T9; ModelScopeAdapter: T7; op_ingest: T5/T8/T9/T10; convert: T11; verify: T12) ✓
- §2 storage schema → T5 (HF), T8 (MS), T9 (Ollama) write exactly the documented shapes ✓
- §3 ingest flow (read→copy→shim→annotate→cleanup, rollback, dry-run, keep-native) → T5 (+T6 rollback) ✓
- §4 three shims + ModelScopeAdapter → T4/T8/T9/T7 ✓
- §5 commands → T10 (ingest), T11 (convert), T12 (verify) ✓
- §6 safety/rollback → T5/T6 (copy-first, partial cleanup, native intact) ✓
- §7 tests → each task is TDD; synthetic caches per tool; round-trip (T12); dedup (T13) ✓
- §8 acceptance → T5/T8/T9 (no 2×, loads), T11 (convert), T7 (MS scan), T12 (verify --fix), T6 (rollback), T13 (dedup) ✓

**2. Placeholder scan:** No "TBD/handle errors". One deliberate guard in T9 had a convoluted `models_root` line — Step 3 explicitly instructs replacing it with `_ollama_models_root(cache_repo)`. T13 honestly flags the dedup-threshold uncertainty with a concrete fallback rather than hand-waving.

**3. Type consistency:** `op_ingest(config, registry, model_id, new_id, category, dry_run, keep_native, registry_save, json_output)` consistent across T5/T8/T9/T10/T11. `_hf_read_native`→`{repo_id,commit,files:[{name,real_path,size}]}` consumed by `_ingest_to_store`(uses name/real_path) and `_hf_build_shim`(uses name). `storage` shape (class/store_path/ingested_at/shims[{tool,kind,location,cache_root_var,reconstruct}]) identical in T5/T8/T9 and consumed by T12. `_ms_build_shim(repo_dir, store_dir)` / `_hf_build_shim(repo_dir, store_dir, commit, files)` / `_ollama_build_shim(info, store_dir, models_root)` signatures match their call sites in op_ingest and `_rebuild_shim_from_storage`.

**Executor notes:** Line numbers drift — locate by symbol. Run tests from the repo root (cross-test fixture imports like `from tests.test_ingest_hf import make_hf_cache` require it). For T12, read `op_verify` first and reuse its existing issue-list variable.
