# AIM Download PRD

- Version: v1.0
- Date: 2026-04-11
- Status: Draft
- Owner: AIM

## 1. Overview

AIM download should become an integration-friendly, observable, and controllable model acquisition pipeline:

1. Stable machine-readable contract (`--json`) for progress/result/error.
2. Category-based canonical placement by default.
3. Reuse official/mature backend tools instead of reimplementing download engines.
4. Cancellation, status querying, resume/idempotency.
5. Advanced network capabilities (proxy/retry/timeout/concurrency).

## 2. Background

Current `aim download` works for manual usage, but deep integrations are blocked by:

1. No stable structured progress protocol.
2. No official cancel/status command.
3. Placement behavior not strongly category-driven.
4. Insufficient proxy and advanced network controls.
5. Backend tool reuse policy not explicit.

## 3. Goals / Non-goals

### 3.1 Goals

1. Provide stable JSON contract for event stream and final summary.
2. Place models into `<root>/store/<category>/<model_id>/` by default.
3. Keep AIM as orchestrator; leverage backend CLIs for real transfer.
4. Add cancel/status/resume/idempotent behavior.
5. Support enterprise-grade network config.

### 3.2 Non-goals

1. Do not build AIM-native generic downloader as primary path.
2. Do not write non-native-CAS downloads directly into engine dirs.

## 4. Users

1. Platform integrators (automation, orchestration, observability).
2. Researchers/engineers (manual yet robust CLI usage).
3. Operations teams (proxy and policy governance).

## 5. Principles

1. Orchestration first: AIM standardizes I/O, job control, registry updates.
2. Tool reuse first: prefer official/mature tools.
3. Stable contract: backend can change, output contract cannot.
4. Store-first: canonical store is default destination (except native CAS semantics).

## 6. Scope

### 6.1 P0

1. `aim download --json` structured output.
2. Standard progress fields (`percent/speed/eta`).
3. `cancel` and `status` commands.
4. Standard completion fields (`path/model_id/source`).
5. Category auto-placement.
6. Proxy + baseline advanced network params.

### 6.2 P1/P2

1. Job center (`create/list/logs`).
2. Webhook callbacks.
3. Queueing and richer parallel scheduling.

## 7. Functional Requirements

### FR-DL-001 Command Modes

1. Synchronous: `aim download <source> ...`
2. JSON mode: `aim download <source> --json`
3. Status: `aim download status <job_id> --json`
4. Cancel: `aim download cancel <job_id>`

### FR-DL-002 Supported Sources

1. `hf:org/repo`
2. `ms:org/repo`
3. `ollama:model:tag`
4. `url:https://...`

### FR-DL-003 Backend Reuse & Fallback

1. HuggingFace: `hfd` (if configured/available) -> `huggingface-cli`.
2. ModelScope: `modelscope` CLI.
3. Ollama: `ollama pull`.
4. URL: `wget` -> `curl`.
5. Emit `backend_tool` and sanitized `backend_command`.

### FR-DL-004 Placement Rules

Default destination:

`<root>/store/<category>/<model_id>/`

Priority:

1. Explicit `--path`
2. Explicit `--category`
3. Inferred category
4. `uncategorized` fallback

### FR-DL-005 Category Inference

1. Use `--category` when provided.
2. Otherwise infer from source metadata, extension, naming rules.
3. Failover to `uncategorized` and mark `placement_mode=fallback`.

### FR-DL-006 Progress Event Contract (JSONL)

Each line is one event with stable keys:

`job_id,status,timestamp,percent,speed_bps,eta_seconds,downloaded_bytes,total_bytes,model_id,source,backend_tool`

### FR-DL-007 Completion Summary Contract

Stable fields:

`job_id,status,model_id,source,path,final_path,category,placement_mode,size_bytes,duration_ms,checksum,registered,storage_mode,backend_tool,backend_command`

### FR-DL-008 Error Contract

On failure:

`error.code,error.message,error.retryable,error.details`

Minimum codes:

`AUTH_FAILED,FORBIDDEN,RATE_LIMITED,NETWORK_TIMEOUT,DISK_FULL,VERIFY_FAILED,BACKEND_NOT_FOUND,CANCELED,UNKNOWN`

### FR-DL-009 Cancel/Interrupt

1. `cancel` transitions job to `canceled`.
2. Sync mode handles `SIGINT` gracefully and emits canceled summary.

### FR-DL-010 Resume/Idempotency

1. `--resume` enabled by default.
2. Existing verified artifact returns `already_exists` summary.
3. `--force-redownload` bypasses idempotency.

### FR-DL-011 Advanced Network Controls

`--proxy --timeout --connect-timeout --retry --retry-backoff --max-speed --concurrency --no-verify-ssl --backend-arg`

### FR-DL-012 Config Precedence

CLI args > project config > global config > env defaults.

## 8. CLI Design (Proposed)

1. `aim download <source> [--name] [--category] [--path] [--json] [--no-progress] [--resume]`
2. `aim download status <job_id> [--json]`
3. `aim download cancel <job_id>`
4. `aim download logs <job_id> [--json]` (P1)

## 9. Config Additions (Proposed)

Add under `download`:

1. `proxy`
2. `timeout`
3. `connect_timeout`
4. `retry`
5. `retry_backoff`
6. `max_speed`
7. `concurrency`
8. `verify_ssl`
9. `backend_priority.huggingface`
10. `backend_priority.modelscope`

## 10. Exit Codes

1. `0` success (including `already_exists`)
2. `1` generic failure
3. `2` canceled
4. `3` invalid argument
5. `4` backend missing (`BACKEND_NOT_FOUND`)
6. `5` auth failure

## 11. Integration with Existing AIM Modules

1. Download completion must update registry.
2. Non-native-CAS should resolve through canonical store path.
3. Engine usability remains `provision` responsibility.

## 12. Non-functional Requirements

1. Observability: traceable by `job_id`.
2. Compatibility: additive JSON evolution only.
3. Performance: orchestration overhead minimal.
4. Security: redact secrets in logs/JSON.

## 13. UAT Acceptance

1. JSON mode produces valid progress events and completion summary.
2. Cancel effective within 3 seconds and yields `canceled`.
3. Default placement follows category path.
4. Proxy config effective for HF/MS/URL routes.
5. Duplicate request is idempotent.
6. Common failures map to standard error codes.

## 14. Milestones

1. P0 (2 weeks): contract, cancel/status, placement, proxy/retry.
2. P1 (2 weeks): logs, stronger resume/verify, backend passthrough.
3. P2 (2-4 weeks): job center, webhook, richer scheduling.

## 15. Risks & Mitigations

1. Backend output variability -> adapter parser tests + graceful degraded progress.
2. Proxy/auth inconsistencies -> mapping table + passthrough support.
3. Native CAS semantics -> explicit `storage_mode` in summary.

---

## Appendix A: JSON Schemas

### A.1 Download Event Schema

```json
{
  "$id": "https://aim.dev/schema/download-event.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AIM Download Event",
  "type": "object",
  "required": ["job_id", "status", "timestamp"],
  "properties": {
    "job_id": { "type": "string", "minLength": 1 },
    "status": {
      "type": "string",
      "enum": ["queued", "downloading", "verifying", "registering", "completed", "failed", "canceled"]
    },
    "timestamp": { "type": "string", "format": "date-time" },
    "model_id": { "type": "string" },
    "source": {
      "type": "object",
      "properties": {
        "type": { "type": "string", "enum": ["huggingface", "modelscope", "ollama", "url"] },
        "repo_id": { "type": "string" },
        "url": { "type": "string" },
        "tag": { "type": "string" }
      },
      "additionalProperties": true
    },
    "percent": { "type": "number", "minimum": 0, "maximum": 100 },
    "speed_bps": { "type": "number", "minimum": 0 },
    "eta_seconds": { "type": "number", "minimum": 0 },
    "downloaded_bytes": { "type": "integer", "minimum": 0 },
    "total_bytes": { "type": "integer", "minimum": 0 },
    "backend_tool": { "type": "string" }
  },
  "additionalProperties": true
}
```

### A.2 Download Summary Schema

```json
{
  "$id": "https://aim.dev/schema/download-summary.json",
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "AIM Download Summary",
  "type": "object",
  "required": ["job_id", "status", "model_id", "source", "path"],
  "properties": {
    "job_id": { "type": "string" },
    "status": { "type": "string", "enum": ["completed", "failed", "canceled", "already_exists"] },
    "model_id": { "type": "string" },
    "source": { "type": "object" },
    "path": { "type": "string" },
    "final_path": { "type": "string" },
    "category": { "type": "string" },
    "placement_mode": { "type": "string", "enum": ["auto", "explicit", "fallback"] },
    "size_bytes": { "type": "integer", "minimum": 0 },
    "duration_ms": { "type": "integer", "minimum": 0 },
    "checksum": { "type": "string" },
    "registered": { "type": "boolean" },
    "storage_mode": { "type": "string", "enum": ["canonical_store", "native_cas"] },
    "backend_tool": { "type": "string" },
    "backend_command": { "type": "array", "items": { "type": "string" } },
    "error": {
      "type": "object",
      "properties": {
        "code": { "type": "string" },
        "message": { "type": "string" },
        "retryable": { "type": "boolean" },
        "details": { "type": "object" }
      },
      "required": ["code", "message", "retryable"]
    }
  },
  "additionalProperties": true
}
```

## Appendix B: Adapter Interface (Python)

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Iterable

@dataclass
class DownloadRequest:
    job_id: str
    source: dict
    model_id: str
    category: str
    dest: Path
    proxy: str = ""
    timeout: int = 0
    connect_timeout: int = 0
    retry: int = 0
    retry_backoff: float = 0.0
    max_speed: str = ""
    concurrency: int = 0
    verify_ssl: bool = True
    backend_args: list[str] | None = None

@dataclass
class ProgressEvent:
    job_id: str
    status: str
    timestamp: str
    percent: float | None = None
    speed_bps: float | None = None
    eta_seconds: float | None = None
    downloaded_bytes: int | None = None
    total_bytes: int | None = None
    model_id: str | None = None
    source: dict | None = None
    backend_tool: str | None = None

@dataclass
class DownloadResult:
    success: bool
    status: str
    path: str
    size_bytes: int = 0
    checksum: str = ""
    error: dict | None = None
    backend_tool: str = ""
    backend_command: list[str] | None = None

class DownloadAdapter(Protocol):
    name: str
    def supports(self, source: dict) -> bool: ...
    def build_command(self, req: DownloadRequest) -> list[str]: ...
    def run(self, req: DownloadRequest) -> Iterable[ProgressEvent]: ...
    def cancel(self, job_id: str) -> bool: ...
    def finalize(self, req: DownloadRequest) -> DownloadResult: ...
```

## Appendix C: Placement Pseudocode

```python
def resolve_download_path(root: Path, model_id: str, category: str, explicit_path: str):
    if explicit_path:
        return Path(explicit_path), "explicit", category or "uncategorized"
    final_category = category or infer_category(model_id=model_id) or "uncategorized"
    mode = "auto" if category else "fallback"
    return root / "store" / final_category / model_id, mode, final_category
```

## Appendix D: CI E2E Cases

1. HF + `--json` emits progress and completed summary.
2. ModelScope lands in `store/<category>/<model_id>`.
3. URL without category falls back to `uncategorized`.
4. Explicit `--path` overrides category placement.
5. Cancel moves job to `canceled` and exit code `2`.
6. Duplicate request returns `already_exists`.
7. Proxy config applied in backend invocation.
8. Auth failure maps to `AUTH_FAILED`.
9. Missing backend maps to `BACKEND_NOT_FOUND` and exit code `4`.
10. Retry behavior deterministic for flaky network.

