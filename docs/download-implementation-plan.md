# AIM Download Implementation Plan (From Current `aim.py`)

- Date: 2026-04-11
- Scope: Minimal invasive refactor aligned with `docs/download-prd.md`

## 1. Current State Snapshot

`aim.py` currently has:

1. `op_download(...)` handling source parsing + download execution + registry add.
2. Backend functions:
   - `_download_hf`
   - `_download_ollama`
   - `_download_url`
   - `_download_modelscope`
3. No `download --json`, no job id, no status/cancel subcommands.
4. Destination path currently `store/<model_id>` (missing category segment).

## 2. Target Minimal Refactor

Keep single-file architecture first; add an internal manager layer without splitting packages yet.

## 3. Function-level Changes

## 3.1 New Data Structures

Add dataclasses near existing model dataclasses:

1. `DownloadOptions`
2. `DownloadEvent`
3. `DownloadSummary`
4. `DownloadError`

Purpose:

1. Keep event schema stable.
2. Centralize JSON emission logic.

## 3.2 Source Parsing Extraction

Extract from `op_download`:

1. `_parse_download_source(source_str, name) -> tuple[source_dict, model_id]`

This supports reuse in future `status/create` flow.

## 3.3 Placement Resolver

Add:

1. `_infer_category(source, model_id, explicit_category) -> str`
2. `_resolve_download_dest(root, model_id, category, explicit_path) -> tuple[Path, placement_mode, final_category]`

Behavior:

1. `--path` wins.
2. else category path under `store/<category>/<model_id>`.
3. fallback `uncategorized`.

## 3.4 Backend Adapter-like Wrapper

Keep old backend functions but wrap with:

1. `_run_download_backend(source, dest, config, options) -> BackendRunResult`

`BackendRunResult` includes:

1. `success`
2. `backend_tool`
3. `backend_command`
4. `error`

## 3.5 JSON Event Emission

Add helpers:

1. `_emit_event(event, json_output)`
2. `_emit_summary(summary, json_output)`

Rules:

1. `--json`: print JSONL events + final JSON summary.
2. non-json: preserve human-readable output.

## 3.6 op_download Signature Upgrade

Current:

`op_download(config, registry, source_str, name="", category="")`

Proposed:

`op_download(config, registry, source_str, name="", category="", path="", json_output=False, options: DownloadOptions | None = None) -> int`

Return code instead of entry:

1. `0` success/already_exists
2. `1` failure
3. `2` canceled
4. `3` arg error
5. `4` backend missing
6. `5` auth failure

## 3.7 CLI Parser Changes

In `build_parser()`:

1. `download` add args:
   - `--json`
   - `--path`
   - `--proxy`
   - `--timeout`
   - `--connect-timeout`
   - `--retry`
   - `--retry-backoff`
   - `--max-speed`
   - `--concurrency`
   - `--no-verify-ssl`
   - `--backend-arg` (append)
   - `--resume` / `--force-redownload`
2. Add subcommands:
   - `download status <job_id> [--json]`
   - `download cancel <job_id>`

Implementation note:

Use nested subparsers under `download` to avoid breaking existing syntax:

1. Default action path for source download.
2. Named actions for status/cancel.

## 3.8 Main Dispatch Changes

In `main()`:

1. route `download` default/status/cancel.
2. call `sys.exit(code)` using new return codes.

## 4. Job State (Minimal P0 Design)

Store job state in:

`~/.aim/download-jobs/<job_id>.json`

Each file keeps:

1. job metadata (source, model_id, created_at)
2. runtime status (queued/downloading/completed/failed/canceled)
3. summary/error payload
4. pid (when running)

Cancel logic:

1. read pid from job file.
2. send terminate signal.
3. update status to canceled.

## 5. Config Changes

Extend `default_config()`:

```json
"download": {
  "proxy": "",
  "timeout": 0,
  "connect_timeout": 0,
  "retry": 2,
  "retry_backoff": 1.5,
  "max_speed": "",
  "concurrency": 0,
  "verify_ssl": true,
  "backend_priority": {
    "huggingface": ["hfd", "huggingface-cli"],
    "url": ["wget", "curl"],
    "modelscope": ["modelscope"],
    "ollama": ["ollama"]
  }
}
```

## 6. Backward Compatibility

1. Existing `aim download <source> [--name] [--category]` remains valid.
2. Human-readable output unchanged unless `--json`.
3. Registry shape unchanged (new fields additive only when needed).

## 7. Testing Plan (Immediate)

## 7.1 Unit

1. source parser.
2. category inference + placement resolver.
3. error code mapping.
4. config precedence resolver.

## 7.2 Integration

1. mock backend command success/failure.
2. JSON event and summary contract validation.
3. status/cancel lifecycle with fake job pid.

## 8. Suggested Delivery Sequence

1. Step 1: destination/category/path + `--json` summary only.
2. Step 2: JSON progress events + backend metadata.
3. Step 3: status/cancel with job files.
4. Step 4: proxy/retry/timeout/concurrency/passthrough.

## 9. Definition of Done

1. All acceptance criteria in `docs/download-prd.md` P0 pass.
2. Existing non-json workflows keep behavior.
3. CI includes at least 10 E2E cases from PRD Appendix D.

