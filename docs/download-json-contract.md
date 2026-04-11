# AIM Download JSON Contract

- Version: v1
- Date: 2026-04-11
- Scope: `aim download`, `aim download status`, `aim download cancel`

## 1. JSON Output Modes

`aim download` has two output patterns:

1. Default mode: human-readable text.
2. `--json`: machine-readable output.

In `--json` mode:

1. Progress is emitted as JSON Lines (one JSON object per line), unless `--no-progress`.
2. Final result is always emitted as one JSON summary object.

## 2. Progress Event (JSON Line)

Each progress line includes:

1. `job_id` (string)
2. `status` (`queued|downloading|registering|completed|failed|canceled`)
3. `timestamp` (ISO datetime)
4. `model_id` (string)
5. `source` (object)
6. `backend_tool` (string|null)
7. `percent` (number|null, 0..100)
8. `speed_bps` (number|null)
9. `eta_seconds` (number|null)
10. `downloaded_bytes` (integer|null)
11. `total_bytes` (integer|null)

Example:

```json
{"job_id":"dl-1775878239291-1521","status":"downloading","timestamp":"2026-04-11T03:30:47.008807+00:00","model_id":"progress-http-test","source":{"type":"url","url":"http://speedtest.tele2.net/1MB.zip"},"backend_tool":"wget","percent":78.0,"speed_bps":52345.2,"eta_seconds":6.9,"downloaded_bytes":826028,"total_bytes":1048576}
```

## 3. Final Summary Object

Final line fields:

1. `job_id`
2. `status` (`completed|failed|canceled|already_exists`)
3. `model_id`
4. `source`
5. `path`
6. `final_path`
7. `category`
8. `placement_mode` (`auto|explicit|fallback|existing`)
9. `size_bytes`
10. `duration_ms`
11. `checksum`
12. `registered` (boolean)
13. `storage_mode` (`canonical_store|native_cas`)
14. `backend_tool`
15. `backend_command` (array)
16. `timestamp`
17. `error` (object, only for failed/canceled flows)

Success example:

```json
{"job_id":"dl-1775879802826-16434","status":"completed","model_id":"final-http-ok","source":{"type":"url","url":"http://speedtest.tele2.net/1MB.zip"},"path":"/Users/you/AI/store/llm/chat/final-http-ok","final_path":"/Users/you/AI/store/llm/chat/final-http-ok","category":"llm/chat","placement_mode":"auto","size_bytes":1048576,"duration_ms":18755,"checksum":"","registered":true,"storage_mode":"canonical_store","backend_tool":"wget","backend_command":["wget","-c","-O","/Users/you/AI/store/llm/chat/final-http-ok/1MB.zip","http://speedtest.tele2.net/1MB.zip","--timeout","20"],"timestamp":"2026-04-11T03:57:01.586589+00:00"}
```

Failure example:

```json
{"job_id":"dl-1775879846393-16754","status":"failed","model_id":"nonexistent","source":{"type":"ollama","repo_id":"library/nonexistent","tag":"latest"},"path":"/Users/you/AI/store/llm/chat/nonexistent","final_path":"/Users/you/AI/store/llm/chat/nonexistent","category":"llm/chat","placement_mode":"fallback","size_bytes":0,"duration_ms":2,"checksum":"","registered":false,"storage_mode":"native_cas","backend_tool":"ollama","backend_command":["ollama","pull","nonexistent:latest"],"timestamp":"2026-04-11T03:57:26.395765+00:00","error":{"code":"BACKEND_NOT_FOUND","message":"[Errno 2] No such file or directory: 'ollama'","retryable":false,"details":{}}}
```

## 4. `--no-progress` Behavior

With `--json --no-progress`:

1. Progress JSON lines are suppressed.
2. Final summary JSON object is still printed.

## 5. Status/Cancel Commands

## 5.1 Status

`aim download status <job_id> --json`

Returns current job state object from `~/.aim/download-jobs/<job_id>.json`:

1. `job_id`, `status`, `model_id`, `source`
2. `path`, `category`, `placement_mode`
3. `progress` (if available)
4. `summary` (when terminal state reached)

Not found:

```json
{"error":{"code":"JOB_NOT_FOUND","message":"Job 'xxx' not found","retryable":false}}
```

## 5.2 Cancel

`aim download cancel <job_id> --json`

Successful cancel request:

```json
{"job_id":"dl-...","status":"cancel_requested"}
```

Not found:

```json
{"error":{"code":"JOB_NOT_FOUND","message":"Job 'xxx' not found","retryable":false}}
```

## 6. Error Code Contract

Common `error.code` values:

1. `INVALID_SOURCE`
2. `JOB_NOT_FOUND`
3. `AUTH_FAILED`
4. `FORBIDDEN`
5. `RATE_LIMITED`
6. `NETWORK_TIMEOUT`
7. `DISK_FULL`
8. `VERIFY_FAILED`
9. `BACKEND_NOT_FOUND`
10. `CANCELED`
11. `UNKNOWN`

## 7. Exit Codes

1. `0` success (`completed` / `already_exists`)
2. `1` generic failure
3. `2` canceled
4. `3` invalid args
5. `4` backend missing
6. `5` auth failure

