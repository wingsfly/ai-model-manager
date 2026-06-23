# SP3 — Portable Backup / Restore — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `aim backup <dir>` mirrors `store/` + a self-contained `aim-backup.json` (idempotent, excludes regenerable shims/caches); `aim restore <dir>` recreates `store/`, imports the registry, recomputes each tool shim's location for the *target* machine, rebuilds the shims from store, and detects+prints (optionally applies) env.

**Architecture:** Single-file `aim.py` (stdlib only). New `# ── Backup / Restore (SP3) ──` section: `_sync_store_dir` (idempotent copy), `op_backup`+`_build_backup_manifest`, `_read_backup_manifest`, `_retarget_shim_locations` (recompute per target cache roots), `op_restore`. Reuses SP1 `EnvDetector`/`op_env_apply` and SP2 `_rebuild_shim_from_storage`/`_*_build_shim`/`_quick_hash`. A small root-aware tweak to `_rebuild_shim_from_storage` lets restore target any configured root.

**Tech Stack:** Python 3.10+ stdlib (`os`, `shutil`, `json`, `pathlib`). Tests: stdlib `unittest`, synthetic data in tempdirs, run from repo ROOT via `make test`.

**Reference spec:** `docs/superpowers/specs/2026-06-23-aim-sp3-backup-restore-design.md`. Add all code to a new `# ── Backup / Restore (SP3) ──` section (place after the `# ── Native Ingest (SP2) ──` section; locate by symbol). Locate anchors with `grep -n`, never absolute line numbers.

Shared test helper (define at top of each new test file):
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
| `aim.py` | SP3 section (`_sync_store_dir`, `_build_backup_manifest`, `op_backup`, `_read_backup_manifest`, `_retarget_shim_locations`, `op_restore`); root-aware tweak to `_rebuild_shim_from_storage`; parser + dispatch | Modify |
| `tests/test_backup.py` | `_sync_store_dir`, `op_backup` (mirror/manifest/idempotent/warning) | Create |
| `tests/test_restore.py` | `_read_backup_manifest`, `_retarget_shim_locations`, `op_restore` round-trip/idempotent/failure | Create |
| `tests/test_backup_cli.py` | `backup`/`restore` parser + dispatch | Create |
| `README.md`, `AIM-MANUAL.md` | Document `aim backup` / `aim restore` | Modify |

---

## Task 1: `_sync_store_dir` (idempotent copy)

**Files:** Modify `aim.py` (new SP3 section); Test `tests/test_backup.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backup.py
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
        self.assertEqual((copied2, skipped2), (0, 2))  # idempotent: same size -> skip

    def test_verify_detects_content_change(self):
        src = self.tmp / "s"; _write(src / "f.bin", b"AAAA")
        dst = self.tmp / "d"; _write(dst / "f.bin", b"BBBB")  # same size, different content
        copied, skipped = aim._sync_store_dir(src, dst, verify=False)
        self.assertEqual((copied, skipped), (0, 1))          # size-only -> skipped
        copied, skipped = aim._sync_store_dir(src, dst, verify=True)
        self.assertEqual((copied, skipped), (1, 0))          # verify -> recopied
        self.assertEqual((dst / "f.bin").read_bytes(), b"AAAA")

    def test_missing_src_is_noop(self):
        self.assertEqual(aim._sync_store_dir(self.tmp / "nope", self.tmp / "d"), (0, 0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_backup -v` → FAIL (no `_sync_store_dir`).

- [ ] **Step 3: Implement** — add the SP3 section header + function:

```python
# ── Backup / Restore (SP3) ───────────────────────────────────────────────────


def _sync_store_dir(src: Path, dst: Path, verify: bool = False) -> tuple[int, int]:
    """Idempotently copy every real file under src/ into dst/, preserving layout.
    Skip when dst already has the file at the same size (or same quick_hash when verify).
    Returns (copied, skipped)."""
    copied = skipped = 0
    if not src.exists():
        return (0, 0)
    for f in sorted(src.rglob("*")):
        if not f.is_file() or f.is_symlink():
            continue
        target = dst / f.relative_to(src)
        if target.exists():
            same = target.stat().st_size == f.stat().st_size
            if same and verify:
                same = _quick_hash(target) == _quick_hash(f)
            if same:
                skipped += 1
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1
    return (copied, skipped)
```

> `_quick_hash(path)` already exists in `aim.py` (size + first-64KB hash).

- [ ] **Step 4: Run** `python3 -m unittest tests.test_backup -v` → PASS (3). `make test`; `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_backup.py
git commit -m "feat(sp3): _sync_store_dir idempotent store copy (size/quick_hash skip)"
```

---

## Task 2: `op_backup` + `_build_backup_manifest`

**Files:** Modify `aim.py`; Test `tests/test_backup.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
import io
import json
from contextlib import redirect_stdout


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.config = aim.default_config()
        self.config["roots"] = [{"id": "primary", "path": str(self.home / "AI")}]
        store = Path(self.config["roots"][0]["path"]) / "store" / "asr" / "model" / "m1"
        _write(store / "model.safetensors", b"W" * 100)
        _write(store / "config.json", b"{}")
        self.reg = aim.Registry()
        self.reg.models = [aim.ModelEntry(
            id="m1", native_cas=False, category="asr/model",
            canonical={"root": "primary", "path": "store/asr/model/m1"},
            storage={"class": "managed-hf", "store_path": "store/asr/model/m1", "shims": []})]

    def test_backup_mirrors_store_and_writes_manifest(self):
        dest = self.home / "backup"
        aim.op_backup(self.config, self.reg, str(dest))
        self.assertEqual((dest / "store" / "asr" / "model" / "m1" / "model.safetensors").read_bytes(), b"W" * 100)
        man = json.loads((dest / "aim-backup.json").read_text())
        self.assertEqual(man["aim_backup_version"], 1)
        self.assertEqual(len(man["models"]), 1)
        self.assertEqual(man["models"][0]["id"], "m1")
        self.assertTrue(any(sf["path"].endswith("model.safetensors") for sf in man["store_files"]))

    def test_backup_idempotent(self):
        dest = self.home / "backup"
        aim.op_backup(self.config, self.reg, str(dest))
        copied, _ = aim._sync_store_dir(Path(self.config["roots"][0]["path"]) / "store", dest / "store")
        self.assertEqual(copied, 0)  # second pass skips everything

    def test_backup_warns_uningested_native(self):
        self.reg.models.append(aim.ModelEntry(id="nat", native_cas=True))
        buf = io.StringIO()
        with redirect_stdout(buf):
            aim.op_backup(self.config, self.reg, str(self.home / "b2"))
        self.assertIn("not ingested", buf.getvalue())
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_backup.BackupTests -v` → FAIL (no `op_backup`).

- [ ] **Step 3: Implement** — add to SP3 section:

```python
def _build_backup_manifest(config: dict, registry: "Registry", store_root: Path) -> dict:
    store_files = []
    if store_root.exists():
        for f in sorted(store_root.rglob("*")):
            if f.is_file() and not f.is_symlink():
                store_files.append({"path": str(Path("store") / f.relative_to(store_root)),
                                    "size": f.stat().st_size, "quick_hash": _quick_hash(f)})
    return {
        "aim_backup_version": 1,
        "created_at": _now_iso(),
        "source_root": str(Path(get_primary_root(config).path)),
        "models": [m.to_dict() for m in registry.models],
        "sources": config.get("sources", {}),
        "env": config.get("env", {}),
        "store_files": store_files,
    }


def op_backup(config: dict, registry: "Registry", dest: str, verify: bool = False,
              json_output: bool = False) -> int:
    root = get_primary_root(config)
    store_root = root.store_path
    dest_dir = Path(dest).expanduser()
    dest_dir.mkdir(parents=True, exist_ok=True)
    native = [m.id for m in registry.models if m.native_cas]
    if native:
        shown = ", ".join(native[:5]) + (" ..." if len(native) > 5 else "")
        print(f"Warning: {len(native)} native model(s) not ingested (not in store, won't be backed up): {shown}")
        print("  Run 'aim ingest --all-native' first to include them.")
    copied, skipped = _sync_store_dir(store_root, dest_dir / "store", verify=verify)
    manifest = _build_backup_manifest(config, registry, store_root)
    (dest_dir / "aim-backup.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    out = {"status": "backed_up", "dest": str(dest_dir), "copied": copied, "skipped": skipped,
           "models": len(manifest["models"]), "native_uningested": len(native)}
    if json_output:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"Backed up {len(manifest['models'])} model(s) to {dest_dir} (copied {copied}, skipped {skipped}).")
    return EXIT_OK
```

> `get_primary_root`, `StorageRoot.store_path`, `_now_iso`, `EXIT_OK` already exist.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_backup -v` → PASS (6). `make test`; `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_backup.py
git commit -m "feat(sp3): op_backup — mirror store + self-contained manifest + uningested warning"
```

---

## Task 3: `_read_backup_manifest` + `_retarget_shim_locations`

**Files:** Modify `aim.py`; Test `tests/test_restore.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_restore.py
import unittest
import json
import tempfile
from pathlib import Path
import aim


def _write(p, data=b"x"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return p


def _det(vals):
    return aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: vals.get(v))


class ReadManifestTests(unittest.TestCase):
    def test_reads_valid(self):
        d = Path(tempfile.mkdtemp())
        (d / "aim-backup.json").write_text(json.dumps({"aim_backup_version": 1, "models": []}))
        man = aim._read_backup_manifest(d)
        self.assertEqual(man["aim_backup_version"], 1)

    def test_missing_raises(self):
        with self.assertRaises(FileNotFoundError):
            aim._read_backup_manifest(Path(tempfile.mkdtemp()))

    def test_bad_version_raises(self):
        d = Path(tempfile.mkdtemp())
        (d / "aim-backup.json").write_text(json.dumps({"aim_backup_version": 99}))
        with self.assertRaises(ValueError):
            aim._read_backup_manifest(d)


class RetargetTests(unittest.TestCase):
    def test_retarget_hf(self):
        e = aim.ModelEntry(id="x", storage={"shims": [{"kind": "hf-cas", "location": "/OLD/models--Org--M",
              "reconstruct": {"repo_id": "Org/M"}}]})
        aim._retarget_shim_locations(e, _det({"HF_HOME": "/tgt/hf"}))
        self.assertEqual(e.storage["shims"][0]["location"], "/tgt/hf/hub/models--Org--M")

    def test_retarget_ollama(self):
        e = aim.ModelEntry(id="x", storage={"shims": [{"kind": "ollama-cas", "location": "/OLD",
              "reconstruct": {"manifest_rel": "registry.ollama.ai/library/q/latest"}}]})
        aim._retarget_shim_locations(e, _det({"OLLAMA_MODELS": "/tgt/ollama"}))
        self.assertEqual(e.storage["shims"][0]["location"],
                         "/tgt/ollama/manifests/registry.ollama.ai/library/q/latest")

    def test_retarget_ms(self):
        e = aim.ModelEntry(id="x", storage={"shims": [{"kind": "ms-dir", "location": "/OLD",
              "reconstruct": {"repo_id": "Qwen/Q", "dir_name": "Q___6B"}}]})
        aim._retarget_shim_locations(e, _det({"MODELSCOPE_CACHE": "/tgt/ms"}))
        self.assertEqual(e.storage["shims"][0]["location"], "/tgt/ms/models/Qwen/Q___6B")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_restore -v` → FAIL.

- [ ] **Step 3: Implement** — add to SP3 section:

```python
def _read_backup_manifest(backup_dir: Path) -> dict:
    man_path = backup_dir / "aim-backup.json"
    if not man_path.exists():
        raise FileNotFoundError(f"no aim-backup.json in {backup_dir}")
    man = json.loads(man_path.read_text())
    if man.get("aim_backup_version") != 1:
        raise ValueError(f"unsupported backup version: {man.get('aim_backup_version')}")
    return man


def _retarget_shim_locations(entry: "ModelEntry", detector: "EnvDetector") -> None:
    """Recompute each shim's `location` for THIS machine's tool caches (cross-machine restore)."""
    for shim in entry.storage.get("shims", []):
        rc = shim.get("reconstruct", {})
        kind = shim.get("kind")
        if kind == "hf-cas":
            hub = detector.cache_dir("huggingface")
            org, _, repo = rc.get("repo_id", "").partition("/")
            if hub and org and repo:
                shim["location"] = str(hub / f"models--{org}--{repo}")
        elif kind == "ollama-cas":
            om = detector.cache_dir("ollama")
            if om and rc.get("manifest_rel"):
                shim["location"] = str(om / "manifests" / rc["manifest_rel"])
        elif kind == "ms-dir":
            ms = detector.cache_dir("modelscope")
            org, _, _ = rc.get("repo_id", "").partition("/")
            if ms and org and rc.get("dir_name"):
                shim["location"] = str(ms / "models" / org / rc["dir_name"])
```

> `EnvDetector.cache_dir("huggingface")` returns `$HF_HOME/hub`; `cache_dir("ollama")` returns `$OLLAMA_MODELS`; `cache_dir("modelscope")` returns `$MODELSCOPE_CACHE` (per SP1 `SOURCES` subpaths).

- [ ] **Step 4: Run** `python3 -m unittest tests.test_restore -v` → PASS (6). `make test`; `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_restore.py
git commit -m "feat(sp3): _read_backup_manifest + _retarget_shim_locations (per-target cache roots)"
```

---

## Task 4: `op_restore` (+ root-aware `_rebuild_shim_from_storage`)

**Files:** Modify `aim.py` (`op_restore`; small tweak to `_rebuild_shim_from_storage`); Test `tests/test_restore.py` (append)

- [ ] **Step 1: Write the failing test (append)** — cross-machine round-trip + idempotency + single-shim failure.

```python
import io
from contextlib import redirect_stdout
from tests.test_ingest_hf import make_hf_cache


class RestoreRoundTripTests(unittest.TestCase):
    def _ingest_source(self):
        src_home = Path(tempfile.mkdtemp())
        cfg = aim.default_config(); cfg["roots"] = [{"id": "primary", "path": str(src_home / "AI")}]
        repo = make_hf_cache(src_home, org="Org", repo="M",
                             files=(("config.json", b"{}"), ("model.safetensors", b"W" * 40)))
        reg = aim.Registry()
        reg.models = [aim.ModelEntry(id="hf-org-m", native_cas=True,
                      source={"type": "huggingface", "repo_id": "Org/M"}, category="asr/model",
                      canonical={"root": "primary", "path": str(repo)})]
        aim.op_ingest(cfg, reg, "hf-org-m", registry_save=False)
        return src_home, cfg, reg

    def test_restore_roundtrip_crossmachine(self):
        src_home, src_cfg, src_reg = self._ingest_source()
        backup = src_home / "backup"
        with redirect_stdout(io.StringIO()):
            aim.op_backup(src_cfg, src_reg, str(backup))
        # TARGET machine: different root + different HF_HOME
        tgt_home = Path(tempfile.mkdtemp())
        tgt_hf = tgt_home / "hfcache"
        tgt_cfg = aim.default_config(); tgt_cfg["roots"] = [{"id": "primary", "path": str(tgt_home / "AI")}]
        tgt_reg = aim.Registry()
        det = aim.EnvDetector(home=tgt_home, rc_files=[],
                              shell_value=lambda v: str(tgt_hf) if v == "HF_HOME" else None)
        with redirect_stdout(io.StringIO()):
            rc = aim.op_restore(tgt_cfg, tgt_reg, str(backup), detector=det)
        self.assertEqual(rc, aim.EXIT_OK)
        # store recreated at target root
        tgt_store = Path(tgt_cfg["roots"][0]["path"]) / "store" / "asr" / "model" / "hf-org-m"
        self.assertEqual((tgt_store / "model.safetensors").read_bytes(), b"W" * 40)
        # registry imported
        self.assertIsNotNone(tgt_reg.find("hf-org-m"))
        # shim rebuilt at the TARGET HF_HOME (not the source path), resolves to target store
        snaps = tgt_hf / "hub" / "models--Org--M" / "snapshots"
        snap = next(d for d in snaps.iterdir() if d.is_dir())
        self.assertEqual((snap / "model.safetensors").resolve().read_bytes(), b"W" * 40)

    def test_restore_idempotent(self):
        src_home, src_cfg, src_reg = self._ingest_source()
        backup = src_home / "backup"
        with redirect_stdout(io.StringIO()):
            aim.op_backup(src_cfg, src_reg, str(backup))
        tgt_home = Path(tempfile.mkdtemp())
        tgt_cfg = aim.default_config(); tgt_cfg["roots"] = [{"id": "primary", "path": str(tgt_home / "AI")}]
        det = aim.EnvDetector(home=tgt_home, rc_files=[],
                              shell_value=lambda v: str(tgt_home / "hf") if v == "HF_HOME" else None)
        with redirect_stdout(io.StringIO()):
            aim.op_restore(tgt_cfg, aim.Registry(), str(backup), detector=det)
            rc = aim.op_restore(tgt_cfg, aim.Registry(), str(backup), detector=det)
        self.assertEqual(rc, aim.EXIT_OK)  # second run clean

    def test_restore_continues_on_shim_failure(self):
        src_home, src_cfg, src_reg = self._ingest_source()
        backup = src_home / "backup"
        with redirect_stdout(io.StringIO()):
            aim.op_backup(src_cfg, src_reg, str(backup))
        tgt_home = Path(tempfile.mkdtemp())
        tgt_cfg = aim.default_config(); tgt_cfg["roots"] = [{"id": "primary", "path": str(tgt_home / "AI")}]
        det = aim.EnvDetector(home=tgt_home, rc_files=[],
                              shell_value=lambda v: str(tgt_home / "hf") if v == "HF_HOME" else None)
        orig = aim._rebuild_shim_from_storage
        aim._rebuild_shim_from_storage = lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
        try:
            with redirect_stdout(io.StringIO()):
                rc = aim.op_restore(tgt_cfg, aim.Registry(), str(backup), detector=det)
        finally:
            aim._rebuild_shim_from_storage = orig
        self.assertEqual(rc, aim.EXIT_FAILED)            # reports failure
        tgt_store = Path(tgt_cfg["roots"][0]["path"]) / "store" / "asr" / "model" / "hf-org-m"
        self.assertTrue((tgt_store / "model.safetensors").exists())  # store still restored
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_restore.RestoreRoundTripTests -v` → FAIL (no `op_restore`).

- [ ] **Step 3a: Make `_rebuild_shim_from_storage` root-aware.** Find it (`grep -n "def _rebuild_shim_from_storage"`). It currently computes `root = get_primary_root(config)`. Replace that single line with (resolve the entry's own canonical root, falling back to primary — keeps `verify --fix` correct since ingested entries carry `canonical.root`):
```python
    roots = {r.id: r for r in get_roots(config)}
    root = roots.get(entry.canonical.get("root", "")) or get_primary_root(config)
```
(`get_roots` already exists.)

- [ ] **Step 3b: Implement `op_restore`** — add to SP3 section:

```python
def op_restore(config: dict, registry: "Registry", src: str, root_id: str = "",
               apply_env: bool = False, verify: bool = False,
               detector: Optional["EnvDetector"] = None, json_output: bool = False) -> int:
    backup_dir = Path(src).expanduser()
    try:
        man = _read_backup_manifest(backup_dir)
    except (FileNotFoundError, ValueError) as ex:
        print(f"Error: {ex}", file=sys.stderr)
        return EXIT_FAILED
    roots = {r.id: r for r in get_roots(config)}
    root = roots.get(root_id) if root_id else get_primary_root(config)
    if not root:
        print(f"Error: unknown root '{root_id}'", file=sys.stderr)
        return EXIT_INVALID_ARGS
    copied, skipped = _sync_store_dir(backup_dir / "store", root.store_path, verify=verify)
    det = detector or EnvDetector()
    rebuilt = 0
    errors: list = []
    for md in man.get("models", []):
        entry = ModelEntry.from_dict(md)
        entry.canonical = dict(entry.canonical or {})
        entry.canonical["root"] = root.id
        if entry.storage.get("shims"):
            _retarget_shim_locations(entry, det)
            try:
                if _rebuild_shim_from_storage(config, entry):
                    rebuilt += 1
            except Exception as ex:
                errors.append((entry.id, str(ex)))
        registry.add(entry)
    registry.save()
    # restore source managed_env into config
    csources = config.setdefault("sources", {})
    for k, v in man.get("sources", {}).items():
        if isinstance(v, dict) and v.get("managed_env"):
            csources.setdefault(k, {}).setdefault("managed_env", {}).update(v["managed_env"])
    # env: detect + print; write only with --apply-env
    if apply_env:
        op_env_apply(config, registry)
        print("Applied env to shell config.")
    else:
        print("Recommended: run 'aim env apply' to set tool env vars on this machine.")
    for mid, err in errors:
        print(f"  shim rebuild failed for {mid}: {err}", file=sys.stderr)
    out = {"status": "restored", "root": root.id, "models": len(man.get("models", [])),
           "store_copied": copied, "store_skipped": skipped, "shims_rebuilt": rebuilt, "errors": len(errors)}
    if json_output:
        print(json.dumps(out, ensure_ascii=False))
    else:
        print(f"Restored {out['models']} model(s) to root '{root.id}' "
              f"(store copied {copied}, skipped {skipped}; shims rebuilt {rebuilt}, errors {len(errors)}).")
    return EXIT_OK if not errors else EXIT_FAILED
```

> `get_roots`, `ModelEntry.from_dict`, `_rebuild_shim_from_storage`, `EnvDetector`, `op_env_apply`, `EXIT_OK/EXIT_FAILED/EXIT_INVALID_ARGS` already exist. `Optional` is imported.

- [ ] **Step 4: Run** `python3 -m unittest tests.test_restore -v` → PASS (9). Run `make test` (no regression — confirm `verify --fix` tests still pass after the root-aware tweak). `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_restore.py
git commit -m "feat(sp3): op_restore — recreate store, retarget+rebuild shims, env detect/print; root-aware rebuild"
```

---

## Task 5: CLI (`aim backup` / `aim restore`)

**Files:** Modify `aim.py` (parser + dispatch); Test `tests/test_backup_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backup_cli.py
import unittest
import aim


class BackupCliParseTests(unittest.TestCase):
    def test_backup_parse(self):
        a = aim.build_parser().parse_args(["backup", "/tmp/bk", "--verify"])
        self.assertEqual(a.command, "backup")
        self.assertEqual(a.dest, "/tmp/bk")
        self.assertTrue(a.verify)

    def test_restore_parse(self):
        a = aim.build_parser().parse_args(["restore", "/tmp/bk", "--root", "ext", "--apply-env"])
        self.assertEqual(a.command, "restore")
        self.assertEqual(a.src, "/tmp/bk")
        self.assertEqual(a.root_id, "ext")
        self.assertTrue(a.apply_env)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run** `python3 -m unittest tests.test_backup_cli -v` → FAIL.

- [ ] **Step 3a: Add subparsers** in `build_parser()` (after the `ingest` subparser block):
```python
    p = sub.add_parser("backup", help="Back up store/ + a portable manifest to a directory")
    p.add_argument("dest", help="Destination directory (external drive / another path)")
    p.add_argument("--verify", action="store_true", help="Compare by quick-hash, not just size")
    p.add_argument("--json", dest="json_output", action="store_true")

    p = sub.add_parser("restore", help="Restore store/ + rebuild tool shims from a backup directory")
    p.add_argument("src", help="Backup directory (containing aim-backup.json)")
    p.add_argument("--root", dest="root_id", default="", help="Target storage root id (default: primary)")
    p.add_argument("--apply-env", dest="apply_env", action="store_true", help="Also write env to shell config")
    p.add_argument("--verify", action="store_true", help="Compare by quick-hash, not just size")
    p.add_argument("--json", dest="json_output", action="store_true")
```

- [ ] **Step 3b: Add dispatch** in `main()` (after the `sources` branch, before `config`):
```python
    elif cmd == "backup":
        return op_backup(config, registry, args.dest, verify=args.verify,
                         json_output=args.json_output)

    elif cmd == "restore":
        return op_restore(config, registry, args.src, root_id=args.root_id,
                          apply_env=args.apply_env, verify=args.verify,
                          json_output=args.json_output)
```

- [ ] **Step 4: Run** `python3 -m unittest tests.test_backup_cli -v` → PASS. Verify wiring: `python3 -c "import aim; aim.build_parser().parse_args(['backup','/tmp/x'])"` and `aim.build_parser().parse_args(['restore','/tmp/x'])` both exit 0. `make test`; `make lint`.

- [ ] **Step 5: Commit**
```bash
git add aim.py tests/test_backup_cli.py
git commit -m "feat(sp3): aim backup / aim restore CLI"
```

---

## Task 6: docs + final suite

**Files:** Modify `README.md`, `AIM-MANUAL.md`

- [ ] **Step 1: README.md** — in the Usage section, after the `aim verify --fix` line (added in SP2), insert:
```markdown
# Portable backup / restore (store/ + manifest; shims are regenerated on restore)
aim backup /Volumes/Backup/aim       # mirror store/ + write aim-backup.json (idempotent; re-runnable)
aim backup /Volumes/Backup/aim --verify
aim restore /Volumes/Backup/aim      # recreate store, rebuild tool shims for THIS machine, print env to set
aim restore /Volumes/Backup/aim --apply-env   # also write env to shell config
```

- [ ] **Step 2: AIM-MANUAL.md** — after the `aim verify --fix` section (added in SP2), add a Chinese section `### aim backup / restore — 可移植备份与还原`: explain backup mirrors `store/` + writes `aim-backup.json` (排除可再生壳/缓存, 幂等, 未摇入原生模型告警); restore recreates store, imports registry, **按目标机重算并重建 HF/Ollama/MS 加载壳**, 检测并打印建议 env (`--apply-env` 才写 shell), supports `--root`/`--verify`, 幂等且单壳失败不中断. Note backup只含 store 真实字节 + 小 JSON, 适合换机/换盘.

- [ ] **Step 3:** Run `make lint && make test` → all PASS.

- [ ] **Step 4: Commit**
```bash
git add README.md AIM-MANUAL.md
git commit -m "docs(sp3): document aim backup / restore"
```

---

## Self-Review

**1. Spec coverage:**
- §1 components → T1 (`_sync_store_dir`), T2 (`op_backup`+`_build_backup_manifest`), T3 (`_read_backup_manifest`+`_retarget_shim_locations`), T4 (`op_restore`), T5 (CLI) ✓
- §2 backup (content, manifest schema, idempotent sync, exclude regenerable, uningested warning) → T1/T2 ✓
- §3 restore (read manifest, recreate store, import registry w/ target root, retarget+rebuild shims, restore managed_env, env detect/print + --apply-env, continue-on-failure) → T3/T4 ✓
- §4 tests (sync, backup, retarget, cross-machine round-trip, idempotent, --apply-env path via op_env_apply, single-shim failure) → covered; **note:** `--apply-env` is exercised indirectly (op_env_apply is SP1-tested); the round-trip asserts the non-apply path. If desired, an extra `--apply-env` restore test can assert `~/.aim/env.sh` exists — added as optional.
- §5 acceptance → T2 (mirror+manifest+idempotent+warning+exclude), T4 (cross-machine rebuild to target caches, env not written by default, idempotent, continue-on-failure) ✓

**2. Placeholder scan:** No TBD/"handle errors". Every code step is complete. `op_restore` catches `Exception` per-model deliberately (continue-on-failure is the spec'd behavior, not a vague catch-all).

**3. Type consistency:** `_sync_store_dir(src,dst,verify)->(int,int)` used by op_backup/op_restore. `_build_backup_manifest(config,registry,store_root)->dict` and `_read_backup_manifest(dir)->dict` share the `aim-backup.json` shape (aim_backup_version/models/sources/env/store_files). `_retarget_shim_locations(entry,detector)` mutates `entry.storage["shims"][].location` consumed by `_rebuild_shim_from_storage` (root-aware after T4 tweak). `op_backup(config,registry,dest,verify,json_output)` / `op_restore(config,registry,src,root_id,apply_env,verify,detector,json_output)` match the CLI dispatch in T5. Reuses real SP1/SP2 symbols (`EnvDetector.cache_dir`, `op_env_apply`, `_rebuild_shim_from_storage`, `_quick_hash`, `get_roots`, `ModelEntry.from_dict`).

**Executor notes:** run from repo root (the round-trip test imports `make_hf_cache` from `tests.test_ingest_hf`). Locate anchors by symbol. After the T4 root-aware tweak, re-run `tests.test_verify_shim` to confirm no regression.
