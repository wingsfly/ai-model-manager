# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`aim.py` is a single-file (~140KB), zero-dependency Python 3.10+ CLI for managing AI models across 9 inference engines (Ollama, HuggingFace, oMLX, ComfyUI, Whisper, Coqui, SparkTTS, Piper, Fish-Speech). It maintains a canonical model store at `~/AI/store/` with engine-provisioning via symlinks/hardlinks.

## Build & Test Commands

```bash
make lint        # Syntax check (py_compile)
make test        # All tests (unit + e2e)
make test-unit   # Unit tests only
make test-core   # Core tests (test_download_core, test_download_progress)
make test-e2e    # E2E tests with mocked backends and temp HOME
```

Single test file: `python3 -m unittest tests.test_download_core -v`

## Architecture

```
aim.py
├── Data model (dataclasses)        # StorageRoot, ModelEntry, Provision, ScannedModel, DownloadOptions, DownloadResult
├── Config system                   # default_config(), load_config(), save_config(), get_roots(), get_primary_root()
├── Registry class                 # In-memory model index backed by ~/.aim/registry.json
├── LinkManager (static methods)    # create_link(), remove_link(), verify_link(), same_volume()
├── EngineAdapter (base class)      # scan(), provision(), unprovision() — polymorphic per engine
├── Engine adapters (9 subclasses) # OllamaAdapter, HuggingFaceAdapter, OMLXAdapter, ComfyUIAdapter, WhisperAdapter, CoquiAdapter, SparkTTSAdapter, PiperAdapter, FishSpeechAdapter
├── op_* operations                # op_scan, op_download, op_provision, op_import, op_convert_native_to_store, op_dedup, op_verify, op_orphans, op_organize, op_delete, op_migrate
├── download helpers               # _parse_download_source(), _infer_download_category(), _resolve_download_dest(), _parse_progress_line(), _build_download_options(), _run_command()
├── display helpers                # display_model_list(), display_model_info(), display_status(), format_size()
└── main() + build_parser()       # argparse dispatch — command routing and CLI arg definitions
```

### Key architectural patterns

- **Engine adapter pattern**: `get_adapter(name, config, root)` factory returns an `EngineAdapter` subclass. All 9 engines share the same interface (`scan()`, `provision()`, `unprovision()`), with engine-specific `native_cas` semantics (Ollama/HuggingFace use content-addressable storage and skip the canonical store).
- **Canonical store model**: `aim organize` moves models into `store/<category>/<id>/` with symlinks/hardlinks back to engine dirs. `aim verify` and `aim dedup` operate on these links.
- **Download orchestration**: AIM wraps external backend tools (`hfd`, `huggingface-cli`, `wget`, `curl`, `ollama pull`, `modelscope`). Progress is parsed from backend output via `_parse_progress_line()` which supports curl/wget/hfd formats. Job state is persisted in `~/.aim/download-jobs/<job_id>.json`.
- **Multi-root storage**: Multiple `StorageRoot` entries support multi-disk setups. `LinkManager.same_volume()` uses `st_dev` comparison to decide hardlink vs symlink.

## Exit Codes

| Code | Constant | Meaning |
|------|----------|---------|
| 0 | `EXIT_OK` | Success / already_exists |
| 1 | `EXIT_FAILED` | Generic failure |
| 2 | `EXIT_CANCELED` | Download canceled |
| 3 | `EXIT_INVALID_ARGS` | Invalid arguments |
| 4 | `EXIT_BACKEND_MISSING` | Backend tool not found |
| 5 | `EXIT_AUTH_FAILED` | Authentication failure |

## File Layout

```
aim.py              # Single-file monolith: all logic, all engines
AIM-MANUAL.md       # Full user manual (Chinese)
docs/
  download-prd.md           # Download feature PRD
  download-implementation-plan.md  # Implementation plan for download refactor
  download-json-contract.md # JSON output contract (progress events + summary)
tests/
  test_download_core.py     # Core parsing logic tests
  test_download_progress.py # Progress line parser tests
  test_download_e2e.py      # Full end-to-end tests with mocked backends
```

## Conventions

- All `op_*` functions are command implementations; all `_parse_*` and `_infer_*` are pure parsing helpers.
- Engine names are kebab-free strings: `fish-speech`, `sparktts`, `huggingface`.
- Model IDs are lowercase with hyphens.
- Registry auto-backs up to `registry.json.bak` before saving.
- The `get_adapter()` factory at line ~1043 instantiates the correct engine adapter.
