# Offload / Restore — Offline-Aware Model Archival

**Status:** Design approved 2026-07-02 · Target release `0.3.0`

## Motivation

The primary disk is nearly full. Some models (image-generation checkpoints, a 48 GB
`voxtral-small-24b`) are not used day-to-day and should live on an external drive
(`/Volumes/Backup`), freeing main-disk space, while remaining **reversible** and
**index-safe**: bringing them back must land cleanly under aim management, and
having the drive unmounted must never corrupt the registry or trigger destructive
"missing/orphan" handling.

`aim migrate` does not fit this: it moves a model's *engine provisioning* to the
destination root (an engine reading from the main disk would lose the model), and
it refuses `native_cas` models outright. So we add a dedicated **offload / restore**
capability with first-class **offline awareness**.

## Concepts

- **Offloaded model** — a managed (canonical-store) model whose bytes have been
  moved to a *removable* root. The registry keeps a lightweight record; the model
  is marked `offline`.
- **Removable root** — a storage root flagged as external/removable (e.g. the
  external drive). Offline detection keys off whether its path is currently
  accessible (mounted).
- **Offline** — the model's data is not currently reachable (offloaded to a root
  whose path is missing, i.e. drive unmounted, or offloaded at all).

## Data model

Add one optional field to `ModelEntry` (backward compatible — absent means online):

```jsonc
entry.offload = {
  "status": "offline",                       // "online" (restored) | "offline"
  "root": "backup",                          // removable root id holding the data
  "path": "store/image-gen/checkpoint/flux1-dev",  // rel path within that root
  "source": "hf:mistralai/Voxtral-Small-24B-2507", // original download source, for re-download fallback
  "offloaded_at": "2026-07-02T...Z"
}
```

Root config gains an optional `"removable": true` marker (defaults false), set when
a root is added for an external drive.

Availability of a root is derived, not stored: `root_available(root)` ⇔ the root
path exists (mounted). Never persisted, so unplug/replug needs no bookkeeping.

## Commands

### `aim offload <id> --to <root_id> [--keep-cache]`

1. Resolve the target removable root; error if not found or not currently mounted.
2. **native_cas model** (e.g. HF-cache `voxtral`): ingest **directly to the target
   root** — copy the real blobs from the HF cache into `<root>/store/<cat>/<id>/`,
   rebuild nothing on main. This converts it to a managed model *and* places it on
   the external drive in one pass, so the main disk never holds a transient second
   copy (safe on a nearly-full disk). Unless `--keep-cache`, remove the original HF
   cache blobs afterward.
3. **already-managed (canonical-store) model** (e.g. ComfyUI `flux`): `move` the
   store directory from its current root to `<target_root>/store/<cat>/<id>/`
   (cross-volume move = copy-then-delete; main disk peak is zero-additional).
4. Remove the model's engine provision links on the main disk (they would dangle).
5. Record `entry.offload = {status: offline, root, path, source, offloaded_at}` and
   `registry.save()`.

Result: main disk freed; data on the external drive; aim still tracks the model as
an offline entry.

### `aim restore <id>`

1. If `entry.offload.root` is mounted:
   - `move` the store directory back to the primary root's canonical location.
   - Re-provision engine links (rebuild whatever the model had).
   - Clear `entry.offload` (back to `online`), `registry.save()`.
   - The model is seamlessly managed again.
2. If the root is **not mounted**: do not touch anything; print a clear instruction —
   mount the drive, or `aim download <source>` to re-fetch.

### `aim offload --list` (or surfaced via `aim list`)

List offloaded models with their root and mount state.

## Offline awareness (the core value)

- **`aim list` / `status` / `info`**: an offloaded model shows `⚠ offline (on
  <root>)`; if its root is unmounted, `⚠ offline (drive not mounted)`.
- **`verify` / `scan` / `orphans`**: **skip** models with `offload.status == offline`.
  They neither report them as `missing_canonical`/orphan nor mutate them. This is the
  index-safety guarantee — an unmounted drive produces no false alarms and no
  destructive action.
- **resolve / provision / any use**: if the target is an offline model, print
  "model is offloaded to '<root>' — mount <path> or `aim download <source>`" and
  exit non-zero rather than emit a broken path.

## Safety & reversibility

- Registry auto-backs up to `registry.json.bak` before save (existing behavior).
- Offload and restore are **copy-then-delete**: the source survives until the copy
  completes, so an interruption never loses data (at worst a partial copy on the
  destination to clean up).
- `source` is recorded, so even a physically failed external drive leaves a known
  re-download path.
- No existing command deletes registry entries for missing files (`scan` is
  additive; `orphans`/`verify` are read-only by default), so pre-feature behavior is
  already non-corrupting; this feature additionally removes the false-alarm noise.

## Out of scope (YAGNI)

- Automatic re-download on restore (user runs `aim download` explicitly).
- Partial/streaming offload; per-file offload.
- Multiple external roots juggling / auto-tiering.

## Testing plan (TDD)

Pure/unit-testable helpers first, then command-level with a temp HOME + fake
external root (a second temp dir standing in for the mount):

- `root_available()` — true when path exists, false when missing.
- offline-state predicate on `ModelEntry`.
- `verify` / `orphans` / `scan` **skip** offline entries (regression-guard against
  false `missing_canonical`).
- `list`/`info` render the offline marker.
- `offload` (managed model): store dir moved to external root, provisions removed,
  entry marked offline; source path freed.
- `offload` (native_cas): ingested directly to external root, entry becomes managed
  + offline, HF blobs removed unless `--keep-cache`.
- `restore` (root mounted): dir moved back, provisions rebuilt, entry online.
- `restore` (root unmounted): no-op with clear message, non-zero exit.
- resolve of an offline model: instructive error, non-zero exit.

Coverage target ≥ 80% on new code; full existing suite stays green.
