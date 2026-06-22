# SP1 — Source/Tool Model + Env-Var Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make aim aware of every download source (HF/Ollama/ModelScope/URL/PyTorch-Hub/Civitai/Git), detect each one's real env-var config cross-shell/OS, align aim's config to where models actually live (fixing the "scans the wrong location" bug), and safely manage env config via an aim-owned env file + a single guarded `source` line.

**Architecture:** All production code stays in the single `aim.py` (preserves zero-dep, `ln -s aim.py` install). New code is a clearly-marked "Sources & Env" section with focused classes: `SOURCES` (data) → `EnvDetector` (read) → `ShellWriter` / `SecretStore` / `ServiceEnv` (write) → `op_env_*` / `op_sources_*` (CLI). SP1 *adopts current cache locations*; relocation/ingest/backup are deferred to SP2/SP3.

**Tech Stack:** Python 3.10+ stdlib only (`os`, `subprocess`, `platform`, `pathlib`, `re`, `json`, `argparse`). Tests: stdlib `unittest`, one file per component under `tests/`, run via `make test`.

**Reference spec:** `docs/superpowers/specs/2026-06-22-aim-sp1-source-env-management-design.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `aim.py` | `SOURCES` registry; `EnvDetector`, `ShellWriter`, `SecretStore`, `ServiceEnv` classes; `_sync_sources_cache_paths`; `op_env_*` / `op_sources_*`; parser + dispatch; `default_config` additions; HF/Ollama adapter `base_path` from `sources.cache_path` | Modify |
| `tests/test_sources_registry.py` | `SOURCES` schema + backward-compat of derived `_BACKEND_REGISTRY` | Create |
| `tests/test_env_detector.py` | `EnvDetector` rc-scan / resolve / cache_dir / report (stubbed shell) | Create |
| `tests/test_sources_config.py` | `default_config` additions + `_sync_sources_cache_paths` + adapter base_path | Create |
| `tests/test_shell_writer.py` | env-file render, idempotent rc wiring, backup, dry-run, shell→rc map | Create |
| `tests/test_secret_service.py` | `SecretStore` perms/masking; `ServiceEnv` command generation | Create |
| `tests/test_env_cli.py` | `op_env_*` / `op_sources_*` orchestration via temp HOME + parser | Create |
| `README.md`, `AIM-MANUAL.md` | Document `aim env` / `aim sources` | Modify |

All new `aim.py` code goes in one section placed **after** `_ensure_backend` (currently ending near [aim.py:1360]) and **before** `_build_download_options`. Ops + parser/dispatch edits go in their existing locations.

---

## Task 1: `SOURCES` registry (data model)

**Files:**
- Modify: `aim.py` — replace `_BACKEND_REGISTRY` definition at [aim.py:1242] with `SOURCES` + a derived `_BACKEND_REGISTRY` view
- Test: `tests/test_sources_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources_registry.py
import unittest
import aim


class SourcesRegistryTests(unittest.TestCase):
    ALLOWED_ROLES = {"cache_dir", "cache_dir_override", "endpoint", "token",
                     "accel", "proxy", "offline", "regen_cache", "misc"}
    ALLOWED_MANAGE = {"env_file", "native", "service", "none"}
    ALLOWED_LAYOUT = {"cas-hf", "cas-ollama", "flat-ms", "torch-hub", "flat"}

    def test_all_expected_sources_present(self):
        for key in ["huggingface", "ollama", "modelscope", "url",
                    "pytorch-hub", "civitai", "git"]:
            self.assertIn(key, aim.SOURCES)

    def test_schema_valid(self):
        for key, spec in aim.SOURCES.items():
            self.assertIn(spec["cache_layout"], self.ALLOWED_LAYOUT, key)
            self.assertIsInstance(spec.get("tools", []), list)
            for e in spec.get("env", []):
                self.assertIn("name", e)
                self.assertIn(e["role"], self.ALLOWED_ROLES, e["name"])
                self.assertIn(e.get("manage", "none"), self.ALLOWED_MANAGE, e["name"])

    def test_backend_registry_backward_compatible(self):
        # existing download code relies on these source types + tool shape
        for key in ["huggingface", "ollama", "modelscope", "url"]:
            self.assertIn(key, aim._BACKEND_REGISTRY)
            for t in aim._BACKEND_REGISTRY[key]:
                self.assertIn("name", t)
                self.assertIn("check", t)
                self.assertIn("install_cmd", t)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_sources_registry -v`
Expected: FAIL — `AttributeError: module 'aim' has no attribute 'SOURCES'`

- [ ] **Step 3: Write minimal implementation**

Replace the `_BACKEND_REGISTRY = {...}` block at [aim.py:1242] with:

```python
SOURCES: dict[str, dict] = {
    "huggingface": {
        "aliases": ["hf"],
        "cache_layout": "cas-hf",
        "tools": [
            {"name": "hfd", "check": "path",
             "install_cmd": "curl -fSL -o {root}/hfd.sh https://hf-mirror.com/hfd/hfd.sh && chmod +x {root}/hfd.sh && brew install wget aria2",
             "description": "HuggingFace Download script (hfd.sh + wget + aria2)"},
            {"name": "hf", "check": "which",
             "install_cmd": "pip3 install --break-system-packages -U huggingface_hub",
             "description": "HuggingFace CLI (hf)"},
        ],
        "env": [
            {"name": "HF_HOME", "role": "cache_dir", "default": "~/.cache/huggingface",
             "subpath": "hub", "detect": ["env", "rc", "tool"], "manage": "env_file", "secret": False},
            {"name": "HF_HUB_CACHE", "role": "cache_dir_override", "default": "",
             "subpath": "", "detect": ["env", "rc"], "manage": "env_file", "secret": False},
            {"name": "HF_ENDPOINT", "role": "endpoint", "default": "https://huggingface.co",
             "detect": ["env", "rc"], "manage": "env_file", "secret": False},
            {"name": "HF_TOKEN", "role": "token", "default": "",
             "detect": ["env", "tool"], "manage": "native", "secret": True},
            {"name": "HF_HUB_ENABLE_HF_TRANSFER", "role": "accel", "default": "0",
             "detect": ["env", "rc"], "manage": "env_file", "secret": False},
            {"name": "HF_XET_CACHE", "role": "regen_cache", "default": "",
             "detect": ["env", "rc"], "manage": "none", "secret": False},
        ],
    },
    "ollama": {
        "aliases": [],
        "cache_layout": "cas-ollama",
        "tools": [
            {"name": "ollama", "check": "which", "install_cmd": "brew install ollama",
             "description": "Ollama CLI"},
        ],
        "env": [
            {"name": "OLLAMA_MODELS", "role": "cache_dir", "default": "~/.ollama/models",
             "subpath": "", "detect": ["env", "rc"], "manage": "service", "secret": False},
            {"name": "OLLAMA_HOST", "role": "endpoint", "default": "127.0.0.1:11434",
             "detect": ["env", "rc"], "manage": "service", "secret": False},
        ],
    },
    "modelscope": {
        "aliases": ["ms"],
        "cache_layout": "flat-ms",
        "tools": [
            {"name": "modelscope", "check": "which",
             "install_cmd": "pip3 install --break-system-packages modelscope",
             "description": "ModelScope CLI"},
        ],
        "env": [
            {"name": "MODELSCOPE_CACHE", "role": "cache_dir", "default": "~/.cache/modelscope",
             "subpath": "", "detect": ["env", "rc"], "manage": "env_file", "secret": False},
        ],
    },
    "url": {
        "aliases": [],
        "cache_layout": "flat",
        "tools": [
            {"name": "wget", "check": "which", "install_cmd": "brew install wget", "description": "GNU Wget"},
            {"name": "curl", "check": "which", "install_cmd": "brew install curl", "description": "cURL"},
        ],
        "env": [],
    },
    "pytorch-hub": {
        "aliases": ["torch", "pytorch"],
        "cache_layout": "torch-hub",
        "tools": [
            {"name": "python3", "check": "which", "install_cmd": "brew install python",
             "description": "Python (torch.hub)"},
        ],
        "env": [
            {"name": "TORCH_HOME", "role": "cache_dir", "default": "~/.cache/torch",
             "subpath": "hub", "detect": ["env", "rc", "tool"], "manage": "env_file", "secret": False},
        ],
    },
    "civitai": {
        "aliases": [],
        "cache_layout": "flat",
        "tools": [
            {"name": "civitdl", "check": "which",
             "install_cmd": "pip3 install --break-system-packages civitdl",
             "description": "Civitai downloader (civitdl)"},
        ],
        "env": [
            {"name": "CIVITAI_API_TOKEN", "role": "token", "default": "",
             "detect": ["env"], "manage": "native", "secret": True},
        ],
    },
    "git": {
        "aliases": ["git-lfs"],
        "cache_layout": "flat",
        "tools": [
            {"name": "git", "check": "which", "install_cmd": "brew install git", "description": "Git"},
            {"name": "git-lfs", "check": "which", "install_cmd": "brew install git-lfs", "description": "Git LFS"},
        ],
        "env": [],
    },
}

# Derived backward-compat view: existing download code reads _BACKEND_REGISTRY[type] -> [tool,...]
_BACKEND_REGISTRY: dict[str, list[dict]] = {
    key: spec["tools"] for key, spec in SOURCES.items() if spec.get("tools")
}
```

> Note: `_check_backend_available` / `_ensure_backend` (below the block) keep working unchanged — they read `_BACKEND_REGISTRY`, now a derived view of `SOURCES`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_sources_registry -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run existing tests to confirm no regression**

Run: `make test`
Expected: all existing download tests still PASS (derived `_BACKEND_REGISTRY` preserves behavior)

- [ ] **Step 6: Commit**

```bash
git add aim.py tests/test_sources_registry.py
git commit -m "feat(sp1): unified SOURCES registry; derive _BACKEND_REGISTRY from it"
```

---

## Task 2: `EnvDetector` — rc scan + login-shell + resolve

**Files:**
- Modify: `aim.py` — add `EnvDetector` class in the new "Sources & Env" section (after `_ensure_backend`)
- Test: `tests/test_env_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_detector.py
import unittest
from pathlib import Path
import tempfile
import aim


class EnvDetectorResolveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.home = Path(self.tmp)

    def _detector(self, shell_values=None, rc_files=None):
        return aim.EnvDetector(
            home=self.home,
            rc_files=rc_files if rc_files is not None else [],
            shell_value=lambda v: (shell_values or {}).get(v),
        )

    def test_resolve_from_login_shell(self):
        d = self._detector(shell_values={"HF_HOME": "/data/hf"})
        entry = {"name": "HF_HOME", "detect": ["env", "rc", "tool"], "default": "~/.cache/huggingface"}
        r = d.resolve("huggingface", entry)
        self.assertEqual(r["effective_value"], "/data/hf")
        self.assertEqual(r["source"], "env")
        self.assertEqual(r["status"], "ok")

    def test_resolve_unset_falls_back_to_default(self):
        d = self._detector(shell_values={})
        entry = {"name": "HF_HOME", "detect": ["env", "rc"], "default": "~/.cache/huggingface"}
        r = d.resolve("huggingface", entry)
        self.assertEqual(r["status"], "unset")
        self.assertEqual(r["effective_value"], str(self.home / ".cache/huggingface"))
        self.assertEqual(r["source"], "default")

    def test_scan_rc_finds_export_and_conflict(self):
        rc1 = self.home / ".zshrc"
        rc1.write_text('export HF_HOME=/a/hf\n')
        rc2 = self.home / ".bashrc"
        rc2.write_text('export HF_HOME="/b/hf"\n')
        d = aim.EnvDetector(home=self.home, rc_files=[rc1, rc2], shell_value=lambda v: None)
        hits = d.scan_rc("HF_HOME")
        self.assertEqual({v for _, v in hits}, {"/a/hf", "/b/hf"})
        entry = {"name": "HF_HOME", "detect": ["env", "rc"], "default": "~/.cache/huggingface"}
        r = d.resolve("huggingface", entry)
        self.assertEqual(r["status"], "conflict")

    def test_scan_rc_parses_fish_set(self):
        rc = self.home / "config.fish"
        rc.write_text('set -gx TORCH_HOME /data/torch\n')
        d = aim.EnvDetector(home=self.home, rc_files=[rc], shell_value=lambda v: None)
        self.assertEqual(d.scan_rc("TORCH_HOME"), [(str(rc), "/data/torch")])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_env_detector -v`
Expected: FAIL — `module 'aim' has no attribute 'EnvDetector'`

- [ ] **Step 3: Write minimal implementation**

Add to `aim.py`:

```python
# ── Sources & Env (SP1) ──────────────────────────────────────────────────────


class EnvDetector:
    """Read-only resolver for source env vars across shells/tools."""

    def __init__(self, sources: Optional[dict] = None, home: Optional[Path] = None,
                 rc_files: Optional[list] = None, shell_value=None, tool_probe=None,
                 recommended_map: Optional[dict] = None):
        self.sources = sources if sources is not None else SOURCES
        self.home = Path(home) if home else Path.home()
        self._rc_files = rc_files            # None -> auto-discover; [] -> none
        self._shell_value = shell_value      # callable(var)->Optional[str]
        self._tool_probe = tool_probe        # callable(var, entry)->Optional[str]
        self._recommended = recommended_map or {}

    def rc_files(self) -> list[Path]:
        if self._rc_files is not None:
            return [Path(p) for p in self._rc_files]
        names = [".zshenv", ".zprofile", ".zshrc", ".bash_profile", ".bashrc", ".profile"]
        out = [self.home / n for n in names]
        out.append(self.home / ".config" / "fish" / "config.fish")
        return [p for p in out if p.exists()]

    def expand(self, val: str) -> str:
        return os.path.expanduser(val) if val else val

    def scan_rc(self, var: str) -> list[tuple[str, str]]:
        pat_sh = re.compile(r'^\s*(?:export\s+)?' + re.escape(var) + r'=(.+)$')
        pat_fish = re.compile(r'^\s*set\s+(?:-\S+\s+)*' + re.escape(var) + r'\s+(.+)$')
        found: list[tuple[str, str]] = []
        for f in self.rc_files():
            try:
                for line in f.read_text().splitlines():
                    if line.strip().startswith("#"):
                        continue
                    m = pat_sh.match(line) or pat_fish.match(line)
                    if m:
                        found.append((str(f), m.group(1).strip().strip('"').strip("'")))
            except OSError:
                continue
        return found

    def login_shell_value(self, var: str) -> Optional[str]:
        if self._shell_value is not None:
            return self._shell_value(var)
        shell = os.environ.get("SHELL", "/bin/sh")
        base = os.path.basename(shell)
        try:
            if base == "fish":
                cmd = [shell, "-c", f"echo ${var}"]
            else:
                cmd = [shell, "-ic", f'printf "%s" "${var}"']
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return r.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def resolve(self, key: str, entry: dict) -> dict:
        var = entry["name"]
        detect = entry.get("detect", ["env"])
        rc_hits = self.scan_rc(var) if "rc" in detect else []
        rc_values = {v for _, v in rc_hits}
        effective: Optional[str] = None
        source = "default"
        if "env" in detect:
            live = self.login_shell_value(var)
            if live:
                effective, source = live, "env"
        if effective is None and rc_hits:
            effective, source = rc_hits[0][1], f"rc:{rc_hits[0][0]}"
        if effective is None and "tool" in detect and self._tool_probe:
            tv = self._tool_probe(var, entry)
            if tv:
                effective, source = tv, "tool"
        recommended = self._recommended.get(var, "")
        if effective is None:
            status = "unset"
            effective = self.expand(entry.get("default", "")) or ""
            source = "default"
        elif len(rc_values) > 1:
            status = "conflict"
        elif recommended and effective != recommended:
            status = "drift"
        else:
            status = "ok"
        return {"name": var, "role": entry.get("role", "misc"), "effective_value": effective,
                "source": source, "aim_recommended": recommended, "status": status,
                "secret": entry.get("secret", False)}
```

> The `re` module is already imported at the top of `aim.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_env_detector -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_env_detector.py
git commit -m "feat(sp1): EnvDetector rc-scan + login-shell resolve with conflict/unset status"
```

---

## Task 3: `EnvDetector.cache_dir` + `report`

**Files:**
- Modify: `aim.py` — add `cache_dir` and `report` methods to `EnvDetector`
- Test: `tests/test_env_detector.py` (append)

- [ ] **Step 1: Write the failing test (append to test file)**

```python
class EnvDetectorCacheDirTests(unittest.TestCase):
    def test_cache_dir_uses_subpath(self):
        d = aim.EnvDetector(home=Path("/h"), rc_files=[],
                            shell_value=lambda v: "/data/hf" if v == "HF_HOME" else None)
        self.assertEqual(d.cache_dir("huggingface"), Path("/data/hf/hub"))

    def test_cache_dir_override_wins(self):
        vals = {"HF_HOME": "/data/hf", "HF_HUB_CACHE": "/fast/hub"}
        d = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: vals.get(v))
        self.assertEqual(d.cache_dir("huggingface"), Path("/fast/hub"))

    def test_cache_dir_default_when_unset(self):
        d = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        self.assertEqual(d.cache_dir("huggingface"), Path("/h/.cache/huggingface/hub"))

    def test_report_covers_all_env_entries(self):
        d = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        rows = d.report()
        names = {r["name"] for r in rows}
        self.assertIn("HF_HOME", names)
        self.assertIn("TORCH_HOME", names)
        self.assertIn("MODELSCOPE_CACHE", names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_env_detector.EnvDetectorCacheDirTests -v`
Expected: FAIL — `'EnvDetector' object has no attribute 'cache_dir'`

- [ ] **Step 3: Write minimal implementation (add methods to `EnvDetector`)**

```python
    def cache_dir(self, key: str) -> Optional[Path]:
        spec = self.sources.get(key, {})
        env = spec.get("env", [])
        override = next((e for e in env if e.get("role") == "cache_dir_override"), None)
        if override:
            r = self.resolve(key, override)
            if r["status"] != "unset" and r["effective_value"]:
                return Path(self.expand(r["effective_value"]))
        base = next((e for e in env if e.get("role") == "cache_dir"), None)
        if not base:
            return None
        r = self.resolve(key, base)
        root = Path(self.expand(r["effective_value"]))
        sub = base.get("subpath", "")
        return root / sub if sub else root

    def report(self) -> list[dict]:
        rows: list[dict] = []
        for key, spec in self.sources.items():
            for entry in spec.get("env", []):
                rows.append({"source": key, **self.resolve(key, entry)})
        return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_env_detector -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_env_detector.py
git commit -m "feat(sp1): EnvDetector.cache_dir (override-aware) + report"
```

---

## Task 4: config schema additions + `_sync_sources_cache_paths`

**Files:**
- Modify: `aim.py` — `default_config()` at [aim.py:180]; add `_sync_sources_cache_paths` in the Sources & Env section
- Test: `tests/test_sources_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources_config.py
import unittest
from pathlib import Path
import aim


class SourcesConfigTests(unittest.TestCase):
    def test_default_config_has_sources_and_env(self):
        cfg = aim.default_config()
        self.assertIn("sources", cfg)
        self.assertEqual(cfg["sources"], {})
        self.assertIn("env", cfg)
        self.assertFalse(cfg["env"]["managed"])

    def test_sync_writes_cache_paths(self):
        cfg = aim.default_config()
        det = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        aim._sync_sources_cache_paths(cfg, det)
        self.assertEqual(cfg["sources"]["huggingface"]["cache_path"], "/h/.cache/huggingface/hub")
        self.assertEqual(cfg["sources"]["pytorch-hub"]["cache_path"], "/h/.cache/torch/hub")

    def test_sync_preserves_existing_keys(self):
        cfg = aim.default_config()
        cfg["sources"]["huggingface"] = {"managed_env": {"HF_ENDPOINT": "https://hf-mirror.com"}}
        det = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        aim._sync_sources_cache_paths(cfg, det)
        self.assertIn("cache_path", cfg["sources"]["huggingface"])
        self.assertEqual(cfg["sources"]["huggingface"]["managed_env"]["HF_ENDPOINT"],
                         "https://hf-mirror.com")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_sources_config -v`
Expected: FAIL — `KeyError: 'sources'`

- [ ] **Step 3: Write minimal implementation**

In `default_config()` ([aim.py:180]), add two keys to the returned dict (after the `"engines": {...}` block, inside the same dict literal):

```python
        "sources": {},
        "env": {"managed": False, "shells": [], "files": {}},
```

Add to the Sources & Env section:

```python
def _sync_sources_cache_paths(config: dict, detector: "EnvDetector") -> dict:
    """Write each source's detected cache location into config['sources'][k]['cache_path']."""
    sources = config.setdefault("sources", {})
    for key in SOURCES:
        cd = detector.cache_dir(key)
        if cd is not None:
            sources.setdefault(key, {})["cache_path"] = str(cd)
    return config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_sources_config -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_sources_config.py
git commit -m "feat(sp1): config sources/env sections + _sync_sources_cache_paths"
```

---

## Task 5: scan integration — HF/Ollama adapters read `sources.cache_path`

**Files:**
- Modify: `aim.py` — `EngineAdapter.__init__` ([aim.py:399]) + `base_path` ([aim.py:407])
- Test: `tests/test_sources_config.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
class AdapterBasePathTests(unittest.TestCase):
    def _cfg(self, cache_path):
        cfg = aim.default_config()
        cfg["sources"]["huggingface"] = {"cache_path": cache_path}
        return cfg

    def test_hf_adapter_uses_sources_cache_path(self):
        cfg = self._cfg("/real/cache/huggingface/hub")
        root = aim.StorageRoot(id="primary", path="/home/u/AI")
        ad = aim.HuggingFaceAdapter(cfg, root)
        self.assertEqual(ad.base_path, Path("/real/cache/huggingface/hub"))

    def test_hf_adapter_falls_back_to_model_dir(self):
        cfg = aim.default_config()  # no sources.huggingface.cache_path
        root = aim.StorageRoot(id="primary", path="/home/u/AI")
        ad = aim.HuggingFaceAdapter(cfg, root)
        self.assertEqual(ad.base_path, Path("/home/u/AI/huggingface/hub"))

    def test_non_cas_engine_unaffected(self):
        cfg = aim.default_config()
        cfg["sources"]["huggingface"] = {"cache_path": "/real/hub"}
        root = aim.StorageRoot(id="primary", path="/home/u/AI")
        ad = aim.OMLXAdapter(cfg, root)
        self.assertEqual(ad.base_path, Path("/home/u/AI/omlx"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_sources_config.AdapterBasePathTests -v`
Expected: FAIL — `base_path` returns `/home/u/AI/huggingface/hub`, not `/real/cache/huggingface/hub`

- [ ] **Step 3: Write minimal implementation**

In `EngineAdapter.__init__` ([aim.py:399]) add after `self.native_cas = engine_cfg.get("native_cas", False)`:

```python
        self._cache_path = config.get("sources", {}).get(self.name, {}).get("cache_path", "")
```

Replace `base_path` property ([aim.py:407]) with:

```python
    @property
    def base_path(self) -> Path:
        # Native-CAS engines (HF/Ollama): prefer the detected real cache location.
        if self.native_cas and self._cache_path:
            return Path(self._cache_path)
        return Path(self.root.path) / self.model_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_sources_config -v`
Expected: PASS (6 tests). Also run `make test` — no regression.

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_sources_config.py
git commit -m "feat(sp1): HF/Ollama scan reads sources.cache_path (fixes wrong-location scan)"
```

---

## Task 6: `ShellWriter` — env-file rendering + source block

**Files:**
- Modify: `aim.py` — add `AIM_ENV_BEGIN`/`AIM_ENV_END` constants + `ShellWriter` class
- Test: `tests/test_shell_writer.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_shell_writer.py
import unittest
from pathlib import Path
import tempfile
import aim


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.w = aim.ShellWriter(home=Path("/h"))

    def test_render_sh(self):
        out = self.w.render_env_file([("huggingface", "HF_ENDPOINT", "https://hf-mirror.com")], fmt="sh")
        self.assertIn("# --- huggingface ---", out)
        self.assertIn('export HF_ENDPOINT="https://hf-mirror.com"', out)
        self.assertIn('[ -f "$HOME/.aim/secrets.env" ]', out)

    def test_render_fish(self):
        out = self.w.render_env_file([("pytorch-hub", "TORCH_HOME", "/data/torch")], fmt="fish")
        self.assertIn('set -gx TORCH_HOME "/data/torch"', out)
        self.assertIn('source "$HOME/.aim/secrets.env"', out)

    def test_source_block_markers(self):
        b = self.w.source_block("sh")
        self.assertIn(aim.AIM_ENV_BEGIN, b)
        self.assertIn(aim.AIM_ENV_END, b)
        self.assertIn('. "$HOME/.aim/env.sh"', b)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_shell_writer -v`
Expected: FAIL — `module 'aim' has no attribute 'ShellWriter'`

- [ ] **Step 3: Write minimal implementation**

```python
AIM_ENV_BEGIN = "# >>> aim env >>>"
AIM_ENV_END = "# <<< aim env <<<"


class ShellWriter:
    """Generate aim-owned env files and wire a single guarded source line into rc files."""

    def __init__(self, home: Optional[Path] = None):
        self.home = Path(home) if home else Path.home()

    def render_env_file(self, managed: list, fmt: str = "sh") -> str:
        lines = ["# Generated by aim — do not edit. Run `aim env apply` to regenerate."]
        cur = None
        for source, name, value in managed:
            if source != cur:
                lines.append(f"# --- {source} ---")
                cur = source
            if fmt == "fish":
                lines.append(f'set -gx {name} "{value}"')
            else:
                lines.append(f'export {name}="{value}"')
        if fmt == "fish":
            lines.append('test -f "$HOME/.aim/secrets.env"; and source "$HOME/.aim/secrets.env"')
        else:
            lines.append('[ -f "$HOME/.aim/secrets.env" ] && . "$HOME/.aim/secrets.env"')
        return "\n".join(lines) + "\n"

    def source_block(self, fmt: str = "sh") -> str:
        if fmt == "fish":
            body = 'test -f "$HOME/.aim/env.fish"; and source "$HOME/.aim/env.fish"'
        else:
            body = '[ -f "$HOME/.aim/env.sh" ] && . "$HOME/.aim/env.sh"'
        return f"{AIM_ENV_BEGIN}\n{body}\n{AIM_ENV_END}\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_shell_writer -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_shell_writer.py
git commit -m "feat(sp1): ShellWriter env-file rendering (sh/fish) + guarded source block"
```

---

## Task 7: `ShellWriter` — idempotent rc wiring + backup + dry-run + shell map

**Files:**
- Modify: `aim.py` — add `wire_rc`, `target_rc`, `detect_shell` to `ShellWriter`
- Test: `tests/test_shell_writer.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
class WireRcTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.w = aim.ShellWriter(home=self.home)

    def test_append_then_idempotent_replace(self):
        rc = self.home / ".zshrc"
        rc.write_text("# my config\nexport PATH=$PATH:/x\n")
        r1 = self.w.wire_rc(rc, fmt="sh")
        self.assertEqual(r1["action"], "append")
        first = rc.read_text()
        self.assertIn(aim.AIM_ENV_BEGIN, first)
        self.assertIn("# my config", first)  # user content preserved
        r2 = self.w.wire_rc(rc, fmt="sh")
        self.assertEqual(r2["action"], "replace")
        self.assertEqual(rc.read_text(), first)  # idempotent

    def test_backup_created_once(self):
        rc = self.home / ".zshrc"
        rc.write_text("orig\n")
        self.w.wire_rc(rc, fmt="sh")
        bak = rc.with_suffix(rc.suffix + ".aim.bak")
        self.assertTrue(bak.exists())
        self.assertEqual(bak.read_text(), "orig\n")

    def test_dry_run_writes_nothing(self):
        rc = self.home / ".zshrc"
        rc.write_text("orig\n")
        r = self.w.wire_rc(rc, fmt="sh", dry_run=True)
        self.assertFalse(r["wrote"])
        self.assertEqual(rc.read_text(), "orig\n")

    def test_target_rc_map(self):
        self.assertEqual(self.w.target_rc("zsh"), (self.home / ".zshrc", "sh"))
        self.assertEqual(self.w.target_rc("fish"), (self.home / ".config/fish/config.fish", "fish"))
        self.assertEqual(self.w.target_rc("bash"), (self.home / ".bashrc", "sh"))
        self.assertEqual(self.w.target_rc("sh"), (self.home / ".profile", "sh"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_shell_writer.WireRcTests -v`
Expected: FAIL — `'ShellWriter' object has no attribute 'wire_rc'`

- [ ] **Step 3: Write minimal implementation (add methods to `ShellWriter`)**

```python
    def wire_rc(self, rc_path: Path, fmt: str = "sh", dry_run: bool = False) -> dict:
        block = self.source_block(fmt)
        existing = rc_path.read_text() if rc_path.exists() else ""
        if AIM_ENV_BEGIN in existing and AIM_ENV_END in existing:
            pre = existing.split(AIM_ENV_BEGIN)[0]
            post = existing.split(AIM_ENV_END, 1)[1]
            new = pre + block.rstrip("\n") + post
            action = "replace"
        else:
            sep = "" if (existing == "" or existing.endswith("\n")) else "\n"
            new = existing + sep + "\n" + block
            action = "append"
        if dry_run:
            return {"action": action, "path": str(rc_path), "wrote": False}
        bak = rc_path.with_suffix(rc_path.suffix + ".aim.bak")
        if rc_path.exists() and not bak.exists():
            bak.write_text(existing)
        rc_path.parent.mkdir(parents=True, exist_ok=True)
        rc_path.write_text(new)
        return {"action": action, "path": str(rc_path), "wrote": True}

    def target_rc(self, shell: str) -> tuple:
        if shell == "zsh":
            return self.home / ".zshrc", "sh"
        if shell == "bash":
            return self.home / ".bashrc", "sh"
        if shell == "fish":
            return self.home / ".config" / "fish" / "config.fish", "fish"
        return self.home / ".profile", "sh"

    def detect_shell(self) -> str:
        base = os.path.basename(os.environ.get("SHELL", ""))
        return base if base in ("zsh", "bash", "fish") else "sh"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_shell_writer -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_shell_writer.py
git commit -m "feat(sp1): ShellWriter idempotent rc wiring with backup, dry-run, shell map"
```

---

## Task 8: `SecretStore` + `ServiceEnv`

**Files:**
- Modify: `aim.py` — add `SecretStore` and `ServiceEnv` classes
- Test: `tests/test_secret_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_secret_service.py
import unittest
import os
import stat
import tempfile
from pathlib import Path
import aim


class SecretStoreTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.s = aim.SecretStore(home=self.home)

    def test_set_secret_writes_600_file(self):
        self.s.set_secret("CIVITAI_API_TOKEN", "abc123")
        p = self.home / ".aim" / "secrets.env"
        self.assertTrue(p.exists())
        mode = stat.S_IMODE(os.stat(p).st_mode)
        self.assertEqual(mode, 0o600)
        self.assertIn('export CIVITAI_API_TOKEN="abc123"', p.read_text())

    def test_set_secret_replaces_not_duplicates(self):
        self.s.set_secret("CIVITAI_API_TOKEN", "old")
        self.s.set_secret("CIVITAI_API_TOKEN", "new")
        text = (self.home / ".aim" / "secrets.env").read_text()
        self.assertEqual(text.count("CIVITAI_API_TOKEN"), 1)
        self.assertIn("new", text)

    def test_mask(self):
        self.assertEqual(aim.SecretStore.mask(""), "unset")
        self.assertEqual(aim.SecretStore.mask("supersecret"), "set (****)")


class ServiceEnvTests(unittest.TestCase):
    def test_macos_uses_launchctl(self):
        cmds = aim.ServiceEnv.ollama_commands("/data/ollama/models", "Darwin")
        self.assertTrue(any("launchctl setenv OLLAMA_MODELS" in c for c in cmds))

    def test_linux_uses_systemd(self):
        cmds = aim.ServiceEnv.ollama_commands("/data/ollama/models", "Linux")
        self.assertTrue(any("systemd" in c or "systemctl" in c for c in cmds))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_secret_service -v`
Expected: FAIL — `module 'aim' has no attribute 'SecretStore'`

- [ ] **Step 3: Write minimal implementation**

```python
class SecretStore:
    """Store secrets (tokens) outside shell rc / env files, in a 0600 file."""

    def __init__(self, home: Optional[Path] = None):
        self.home = Path(home) if home else Path.home()
        self.path = self.home / ".aim" / "secrets.env"

    def set_secret(self, name: str, value: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        if self.path.exists():
            lines = [l for l in self.path.read_text().splitlines()
                     if not l.strip().startswith(f"export {name}=")]
        lines.append(f'export {name}="{value}"')
        self.path.write_text("\n".join(lines) + "\n")
        os.chmod(self.path, 0o600)

    @staticmethod
    def mask(value: str) -> str:
        return "set (****)" if value else "unset"


class ServiceEnv:
    """Render commands to set daemon-level env (e.g. ollama server) that ignores shell rc."""

    @staticmethod
    def ollama_commands(models_path: str, system: str) -> list[str]:
        if system == "Darwin":
            return [f'launchctl setenv OLLAMA_MODELS "{models_path}"',
                    "# restart the Ollama app for the change to take effect"]
        if system == "Linux":
            return ["mkdir -p ~/.config/systemd/user/ollama.service.d",
                    f'printf "[Service]\\nEnvironment=OLLAMA_MODELS={models_path}\\n" '
                    "> ~/.config/systemd/user/ollama.service.d/aim.conf",
                    "systemctl --user daemon-reload && systemctl --user restart ollama"]
        return [f"# set OLLAMA_MODELS={models_path} in your service manager"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_secret_service -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_secret_service.py
git commit -m "feat(sp1): SecretStore (0600 + masking) and ServiceEnv command generation"
```

---

## Task 9: `op_env_show` / `op_env_path` + parser + dispatch

**Files:**
- Modify: `aim.py` — add ops; add `env` subparser at [aim.py:3722] area; add dispatch before `config` at [aim.py:3951]
- Test: `tests/test_env_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env_cli.py
import unittest
import io
import json
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import aim


def _detector(home):
    return aim.EnvDetector(home=home, rc_files=[], shell_value=lambda v: None)


class EnvShowTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_env_show_json(self):
        cfg = aim.default_config()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = aim.op_env_show(cfg, detector=_detector(self.home), json_output=True)
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        names = {r["name"] for r in data["env"]}
        self.assertIn("HF_HOME", names)
        self.assertIn("huggingface", data["cache_dirs"])

    def test_env_show_masks_secrets(self):
        cfg = aim.default_config()
        det = aim.EnvDetector(home=self.home, rc_files=[],
                              shell_value=lambda v: "tok" if v == "CIVITAI_API_TOKEN" else None)
        buf = io.StringIO()
        with redirect_stdout(buf):
            aim.op_env_show(cfg, detector=det, json_output=False)
        out = buf.getvalue()
        self.assertNotIn("tok", out)
        self.assertIn("set (****)", out)

    def test_env_path(self):
        cfg = aim.default_config()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = aim.op_env_path(cfg, "huggingface", detector=_detector(self.home))
        self.assertEqual(rc, 0)
        self.assertIn(".cache/huggingface/hub", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_env_cli -v`
Expected: FAIL — `module 'aim' has no attribute 'op_env_show'`

- [ ] **Step 3: Write minimal implementation**

Add ops in the Sources & Env section:

```python
def op_env_show(config: dict, detector: Optional["EnvDetector"] = None,
                json_output: bool = False) -> int:
    det = detector or EnvDetector()
    rows = det.report()
    cache_dirs = {k: str(det.cache_dir(k) or "") for k in SOURCES}
    if json_output:
        print(json.dumps({"env": rows, "cache_dirs": cache_dirs}, ensure_ascii=False))
        return 0
    print(f"{'SOURCE':<13}{'VARIABLE':<28}{'STATUS':<10}VALUE  [SOURCE]")
    for r in rows:
        val = SecretStore.mask(r["effective_value"]) if r["secret"] else r["effective_value"]
        print(f"{r['source']:<13}{r['name']:<28}{r['status']:<10}{val}  [{r['source_for_display'] if False else r['source']}]")
    print()
    print("Resolved cache directories:")
    for k, v in cache_dirs.items():
        if v:
            print(f"  {k:<13} {v}")
    return 0


def op_env_path(config: dict, source: str, detector: Optional["EnvDetector"] = None) -> int:
    det = detector or EnvDetector()
    if source not in SOURCES:
        print(f"Unknown source: {source}. Known: {', '.join(SOURCES)}", file=sys.stderr)
        return EXIT_INVALID_ARGS
    cd = det.cache_dir(source)
    if cd is None:
        print(f"Source '{source}' has no cache directory.", file=sys.stderr)
        return EXIT_FAILED
    print(str(cd))
    return 0
```

> The `r['source']` field already disambiguates (env / rc:<file> / tool / default). The ternary is a no-op kept only to satisfy the column layout; an executor may simplify to `[{r['source']}]`.

Add the `env` subparser. After the `config` subparser block at [aim.py:3722-3724], add:

```python
    p_env = sub.add_parser("env", help="Detect/manage download-source environment variables")
    env_sub = p_env.add_subparsers(dest="env_command")
    pe = env_sub.add_parser("show", help="Show detected env vars and cache dirs")
    pe.add_argument("--json", dest="json_output", action="store_true")
    pe = env_sub.add_parser("apply", help="Write aim env files and wire shell rc")
    pe.add_argument("--shell", default="", help="zsh|bash|fish|all (default: detected)")
    pe.add_argument("--set", dest="set_vars", action="append", default=[], metavar="VAR=VALUE")
    pe.add_argument("--service", action="store_true", help="Also emit daemon-level env commands")
    pe.add_argument("--dry-run", dest="dry_run", action="store_true")
    pe = env_sub.add_parser("path", help="Print resolved cache dir for a source")
    pe.add_argument("source", help="Source key, e.g. huggingface")
```

Add dispatch in `main()` just before `elif cmd == "config":` ([aim.py:3951]):

```python
    elif cmd == "env":
        sub_cmd = getattr(args, "env_command", None)
        if sub_cmd == "show" or sub_cmd is None:
            return op_env_show(config, json_output=getattr(args, "json_output", False))
        elif sub_cmd == "path":
            return op_env_path(config, args.source)
        elif sub_cmd == "apply":
            return op_env_apply(config, registry, shell=args.shell, set_vars=args.set_vars,
                                service=args.service, dry_run=args.dry_run)
```

> `op_env_apply` is implemented in Task 10; this dispatch line references it ahead of definition, which is fine because both live in the same module loaded before `main()` runs. If implementing strictly task-by-task, temporarily stub `op_env_apply` returning `0` until Task 10. (The Task 9 tests do not exercise `apply`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_env_cli -v`
Expected: PASS (3 tests). Also `python3 -c "import aim; aim.build_parser().parse_args(['env','show','--json'])"` exits 0.

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_env_cli.py
git commit -m "feat(sp1): aim env show/path commands"
```

---

## Task 10: `op_env_apply` (regenerate env files + wire rc + --set + dry-run)

**Files:**
- Modify: `aim.py` — add `op_env_apply` (replace the Task-9 stub if used)
- Test: `tests/test_env_cli.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
class EnvApplyTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def _run(self, cfg, **kw):
        writer = aim.ShellWriter(home=self.home)
        return aim.op_env_apply(cfg, registry=None, writer=writer,
                                home=self.home, **kw)

    def test_apply_writes_env_file_and_wires_rc(self):
        cfg = aim.default_config()
        cfg["sources"]["huggingface"] = {"managed_env": {"HF_ENDPOINT": "https://hf-mirror.com"}}
        rc = self.home / ".zshrc"; rc.write_text("# mine\n")
        rc_code = self._run(cfg, shell="zsh", set_vars=[], service=False, dry_run=False)
        self.assertEqual(rc_code, 0)
        envf = (self.home / ".aim" / "env.sh").read_text()
        self.assertIn('export HF_ENDPOINT="https://hf-mirror.com"', envf)
        self.assertIn(aim.AIM_ENV_BEGIN, rc.read_text())
        self.assertIn("# mine", rc.read_text())

    def test_apply_set_flag_persists_to_config(self):
        cfg = aim.default_config()
        rc = self.home / ".zshrc"; rc.write_text("")
        self._run(cfg, shell="zsh", set_vars=["HF_HUB_ENABLE_HF_TRANSFER=1"],
                  service=False, dry_run=False)
        self.assertEqual(
            cfg["sources"]["huggingface"]["managed_env"]["HF_HUB_ENABLE_HF_TRANSFER"], "1")

    def test_apply_dry_run_writes_nothing(self):
        cfg = aim.default_config()
        cfg["sources"]["huggingface"] = {"managed_env": {"HF_ENDPOINT": "https://x"}}
        rc = self.home / ".zshrc"; rc.write_text("orig\n")
        self._run(cfg, shell="zsh", set_vars=[], service=False, dry_run=True)
        self.assertFalse((self.home / ".aim" / "env.sh").exists())
        self.assertEqual(rc.read_text(), "orig\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_env_cli.EnvApplyTests -v`
Expected: FAIL — `op_env_apply` missing or stub ignores args

- [ ] **Step 3: Write minimal implementation**

```python
def _managed_env_pairs(config: dict) -> list:
    """Collect (source, name, value) for manage:env_file vars from config['sources'][*]['managed_env']."""
    pairs = []
    for key, spec in SOURCES.items():
        managed_env = config.get("sources", {}).get(key, {}).get("managed_env", {})
        env_file_names = [e["name"] for e in spec.get("env", [])
                          if e.get("manage") == "env_file" and not e.get("secret")]
        for name in env_file_names:
            if name in managed_env:
                pairs.append((key, name, managed_env[name]))
    return pairs


def _source_key_for_var(var: str) -> Optional[str]:
    for key, spec in SOURCES.items():
        if any(e["name"] == var for e in spec.get("env", [])):
            return key
    return None


def op_env_apply(config: dict, registry, writer: Optional["ShellWriter"] = None,
                 home: Optional[Path] = None, shell: str = "", set_vars: Optional[list] = None,
                 service: bool = False, dry_run: bool = False) -> int:
    home = Path(home) if home else Path.home()
    writer = writer or ShellWriter(home=home)
    # 1) fold --set VAR=VALUE into config.sources[key].managed_env
    for item in (set_vars or []):
        if "=" not in item:
            print(f"Invalid --set '{item}', expected VAR=VALUE", file=sys.stderr)
            return EXIT_INVALID_ARGS
        var, value = item.split("=", 1)
        key = _source_key_for_var(var)
        if not key:
            print(f"Unknown variable: {var}", file=sys.stderr)
            return EXIT_INVALID_ARGS
        config.setdefault("sources", {}).setdefault(key, {}).setdefault("managed_env", {})[var] = value
    # 2) render env files
    pairs = _managed_env_pairs(config)
    shells = ["zsh", "bash", "fish"] if shell == "all" else [shell or writer.detect_shell()]
    if dry_run:
        print("[dry-run] would write ~/.aim/env.sh, ~/.aim/env.fish and wire:", ", ".join(shells))
        for s in shells:
            rc_path, fmt = writer.target_rc(s)
            r = writer.wire_rc(rc_path, fmt=fmt, dry_run=True)
            print(f"  {s}: {r['action']} {r['path']}")
        return 0
    aim_dir = home / ".aim"
    aim_dir.mkdir(parents=True, exist_ok=True)
    (aim_dir / "env.sh").write_text(writer.render_env_file(pairs, fmt="sh"))
    (aim_dir / "env.fish").write_text(writer.render_env_file(pairs, fmt="fish"))
    # 3) wire rc for target shells
    wired = []
    for s in shells:
        rc_path, fmt = writer.target_rc(s)
        r = writer.wire_rc(rc_path, fmt=fmt, dry_run=False)
        wired.append(r["path"])
    # 4) record env management state
    config.setdefault("env", {})
    config["env"]["managed"] = True
    config["env"]["shells"] = shells
    config["env"]["files"] = {"posix": str(aim_dir / "env.sh"), "fish": str(aim_dir / "env.fish")}
    if registry is not None:
        save_config(config)
    # 5) optional service env hints
    if service:
        models = config.get("sources", {}).get("ollama", {}).get("cache_path", "~/.ollama/models")
        print("Service-level env (run manually):")
        for c in ServiceEnv.ollama_commands(models, platform.system()):
            print(f"  {c}")
    print(f"Wrote {aim_dir/'env.sh'}, {aim_dir/'env.fish'}; wired: {', '.join(wired)}")
    return 0
```

> `platform` is already imported at the top of `aim.py`. `save_config` is only called when a live `registry` is passed (production path); tests pass `registry=None` to avoid touching the real `~/.aim/config.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_env_cli -v`
Expected: PASS (6 tests total). Run `make test` — no regression.

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_env_cli.py
git commit -m "feat(sp1): aim env apply — render env files, wire rc, --set, --service, --dry-run"
```

---

## Task 11: `op_sources_list` / `op_sources_install` + parser + dispatch

**Files:**
- Modify: `aim.py` — add ops; add `sources` subparser; add dispatch
- Test: `tests/test_env_cli.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
class SourcesCliTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_sources_list_json(self):
        cfg = aim.default_config()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = aim.op_sources_list(cfg, detector=_detector(self.home), json_output=True)
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        keys = {s["key"] for s in data["sources"]}
        self.assertEqual(keys, set(aim.SOURCES))
        hf = next(s for s in data["sources"] if s["key"] == "huggingface")
        self.assertIn("tools", hf)
        self.assertIn("cache_dir", hf)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_env_cli.SourcesCliTests -v`
Expected: FAIL — `module 'aim' has no attribute 'op_sources_list'`

- [ ] **Step 3: Write minimal implementation**

```python
def op_sources_list(config: dict, detector: Optional["EnvDetector"] = None,
                    json_output: bool = False) -> int:
    det = detector or EnvDetector()
    out = []
    for key, spec in SOURCES.items():
        tools = []
        for t in spec.get("tools", []):
            tools.append({"name": t["name"], "installed": _check_backend_available(t, config)})
        out.append({
            "key": key,
            "cache_layout": spec.get("cache_layout"),
            "cache_dir": str(det.cache_dir(key) or ""),
            "tools": tools,
            "env_vars": [e["name"] for e in spec.get("env", [])],
        })
    if json_output:
        print(json.dumps({"sources": out}, ensure_ascii=False))
        return 0
    for s in out:
        tool_str = ", ".join(f"{t['name']}{'✓' if t['installed'] else '✗'}" for t in s["tools"])
        print(f"{s['key']:<13} layout={s['cache_layout']:<10} tools=[{tool_str}]")
        if s["cache_dir"]:
            print(f"              cache: {s['cache_dir']}")
    return 0


def op_sources_install(config: dict, source: str, json_output: bool = False,
                       auto_confirm: bool = False) -> int:
    if source not in SOURCES:
        print(f"Unknown source: {source}. Known: {', '.join(SOURCES)}", file=sys.stderr)
        return EXIT_INVALID_ARGS
    ok, err = _ensure_backend(source, config, json_output, auto_confirm)
    if not ok:
        if err:
            print(err)
        return EXIT_BACKEND_MISSING
    print(f"Source '{source}' has at least one usable tool.")
    return 0
```

Add parser after the `env` subparser block:

```python
    p_src = sub.add_parser("sources", help="List/manage download sources and their tools")
    src_sub = p_src.add_subparsers(dest="sources_command")
    ps = src_sub.add_parser("list", help="List sources, tool install state, env summary")
    ps.add_argument("--json", dest="json_output", action="store_true")
    ps = src_sub.add_parser("install", help="Install a download tool for a source")
    ps.add_argument("source", help="Source key, e.g. huggingface")
    ps.add_argument("-y", "--yes", dest="auto_confirm", action="store_true")
    ps.add_argument("--json", dest="json_output", action="store_true")
```

Add dispatch in `main()` after the `env` block:

```python
    elif cmd == "sources":
        sub_cmd = getattr(args, "sources_command", None)
        if sub_cmd == "install":
            return op_sources_install(config, args.source,
                                      json_output=getattr(args, "json_output", False),
                                      auto_confirm=getattr(args, "auto_confirm", False))
        return op_sources_list(config, json_output=getattr(args, "json_output", False))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests.test_env_cli -v`
Expected: PASS (7 tests total)

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_env_cli.py
git commit -m "feat(sp1): aim sources list/install commands"
```

---

## Task 12: wire detection into `aim scan` (auto-align cache_path)

**Files:**
- Modify: `aim.py` — `op_scan` at [aim.py:1058] (now shifted by Task 1's larger block; locate via `def op_scan`)
- Test: `tests/test_sources_config.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
class ScanAutoAlignTests(unittest.TestCase):
    def test_scan_populates_sources_cache_path(self):
        cfg = aim.default_config()
        det = aim.EnvDetector(home=Path("/h"), rc_files=[], shell_value=lambda v: None)
        # simulate what op_scan does at entry
        aim._sync_sources_cache_paths(cfg, det)
        self.assertTrue(cfg["sources"]["huggingface"]["cache_path"].endswith("/.cache/huggingface/hub")
                        or cfg["sources"]["huggingface"]["cache_path"].endswith("\\.cache\\huggingface\\hub"))
```

- [ ] **Step 2: Run test to verify it fails / passes-trivially, then wire production path**

Run: `python3 -m unittest tests.test_sources_config.ScanAutoAlignTests -v`
Expected: PASS (this asserts the helper; the production wiring below makes `op_scan` call it).

- [ ] **Step 3: Wire into `op_scan`**

At the top of `op_scan` ([aim.py:1058], `def op_scan(config, registry, engine_filter="")`), after `root = get_primary_root(config)`, add:

```python
    # Align config to where tools actually cache models, so we scan the right dirs.
    _sync_sources_cache_paths(config, EnvDetector())
```

- [ ] **Step 4: Verify import + smoke**

Run: `python3 -c "import aim; print('ok')"`
Expected: `ok`
Run: `make test`
Expected: all PASS (no regression; scan now self-aligns).

- [ ] **Step 5: Commit**

```bash
git add aim.py tests/test_sources_config.py
git commit -m "feat(sp1): aim scan auto-aligns sources.cache_path before scanning"
```

---

## Task 13: docs + full suite + lint

**Files:**
- Modify: `README.md`, `AIM-MANUAL.md`

- [ ] **Step 1: Add to `README.md` Usage section** (after the existing `aim status` block):

```markdown
# Detect / manage download-source env vars (HF, Ollama, ModelScope, PyTorch Hub, Civitai, Git)
aim env show                          # detected vars + resolved cache dirs (read-only)
aim env show --json
aim env path huggingface              # resolved cache dir for a source
aim env apply --shell zsh             # write ~/.aim/env.{sh,fish} + wire rc (one guarded line)
aim env apply --set HF_ENDPOINT=https://hf-mirror.com --set HF_HUB_ENABLE_HF_TRANSFER=1
aim env apply --dry-run               # preview, write nothing
aim sources list                      # sources, tool install state, env summary
aim sources install huggingface -y
```

- [ ] **Step 2: Add a section to `AIM-MANUAL.md`** (after the `aim config show` section), documenting `aim env` and `aim sources` in Chinese, mirroring the spec §8.1 command table.

- [ ] **Step 3: Run the full suite + lint**

Run: `make lint && make test`
Expected: `py_compile` clean; ALL tests PASS (existing + 6 new test files).

- [ ] **Step 4: Commit**

```bash
git add README.md AIM-MANUAL.md
git commit -m "docs(sp1): document aim env and aim sources commands"
```

---

## Self-Review

**1. Spec coverage** (each SP1 spec section → task):
- §3 SOURCES descriptor → Task 1 ✓
- §3.1 env inventory (all 7 sources incl. PyTorch Hub/Civitai/Git) → Task 1 ✓
- §4 EnvDetector (rc/login-shell/tool/default; conflict/unset; cache_dir; reality-sync) → Tasks 2,3,4 ✓
- §4 config-reality-sync (fix scan bug) → Tasks 4,5,12 ✓
- §5 ShellWriter (env files sh/fish; guarded rc block; idempotent; backup; dry-run; shell/OS) → Tasks 6,7 ✓
- §6 SecretStore (600; masking; not in env file/rc) → Task 8 ✓
- §7 ServiceEnv (detect/report; launchctl/systemd) → Task 8 ✓
- §8.1 commands (`env show/apply/path`, `sources list/install`) → Tasks 9,10,11 ✓
- §8.2 config schema → Task 4 ✓
- §8.3 engines→sources cache authority cleanup → Task 5 ✓
- §10 testing strategy (pure unit, stubbed shell, cross-shell render) → all tasks ✓
- §11 acceptance → covered by Tasks 5 (scan finds models), 7 (idempotent/backup/dry-run), 8 (secret masking), 11 (sources list)
- Non-goals (relocation/ingest/backup, Windows mutate) → correctly absent ✓

**2. Placeholder scan:** No "TBD/TODO/handle edge cases" — every code step has complete code. The one `if False` ternary in `op_env_show` is annotated as a deliberate no-op an executor may simplify; not a placeholder for missing logic.

**3. Type consistency:** `EnvDetector(home, rc_files, shell_value, tool_probe, recommended_map)` used consistently across Tasks 2/3/4/5/9/10/11/12. `ShellWriter` methods (`render_env_file`, `source_block`, `wire_rc`, `target_rc`, `detect_shell`) consistent across Tasks 6/7/10. `SecretStore.mask` / `ServiceEnv.ollama_commands` signatures match call sites. Ops (`op_env_show/op_env_path/op_env_apply/op_sources_list/op_sources_install`) signatures match dispatch + tests. Config keys (`sources.<k>.cache_path`, `sources.<k>.managed_env`, `env.managed/shells/files`) consistent across Tasks 4/5/10/11/12.

**Note for executor:** Task 9 dispatch references `op_env_apply` (Task 10). If executing strictly in order, add a one-line `def op_env_apply(*a, **k): return 0` stub during Task 9, replaced in Task 10. Line numbers drift as code is added — locate anchors by `def`/symbol name, not absolute line.
