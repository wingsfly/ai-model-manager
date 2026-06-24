# aim — non-interactive backend auto-install (design)

- Date: 2026-06-24
- Status: approved
- Author: wingsfly + Claude

## Problem

`aim download <ref>` auto-installs a missing backend (e.g. ModelScope CLI) but asks interactively:
`Install ModelScope CLI? [y/N]:`. When another app (openspeechapi) spawns `aim download` headless
(no TTY/stdin), the prompt gets EOF → prints `Aborted` → fails silently → caller falls back. The
existing `-y/--yes` flag already does non-interactive auto-install, but (a) a caller that can't inject
`-y` into argv has no switch, and (b) the headless failure is vague/silent and hard to diagnose.

## Behavior (current, for reference)

`_ensure_backend(source_type, config, json_output, auto_confirm)` ([aim.py](../../aim.py)):
- `-y` → installs, no prompt (already works headless).
- `--json` (no `-y`) → returns structured `BACKEND_NOT_FOUND` JSON with `install_hint` (no hang).
- plain (no `-y`, no `--json`) → interactive `[y/N]` → headless EOF → `Aborted`. ← the trap.

## Design

### 1. Effective "assume-yes" — three opt-ins, OR'd, resolved once inside `_ensure_backend`
```
effective_auto = auto_confirm (-y/--yes)
              OR env AIM_ASSUME_YES truthy   # "1","true","yes","on","y" (case-insensitive)
              OR config defaults.auto_install_backend == true
```
- New helper `_assume_yes(config, auto_confirm) -> bool`. Called at the top of `_ensure_backend`;
  the rest of the function uses the resolved value. All callers (`download`/`resolve`/`provision`)
  benefit with no call-site change.
- Default stays **opt-in**: nothing auto-installs unless one of the three is set.

### 2. Non-TTY safety (the silent-abort fix)
In the non-`--json` branch, before prompting: if `not sys.stdin.isatty()` (headless), **do not call
`input()`**. Print a clear, actionable error to **stderr** — names the missing backend + install
command + the three ways to enable auto-install (`-y`, `AIM_ASSUME_YES=1`,
`defaults.auto_install_backend`) — and return `(False, "")`. Interactive TTY path unchanged.

### 3. `--json` unchanged
Still returns `BACKEND_NOT_FOUND` / `INSTALL_FAILED`. Extend `install_hint.note` to mention env/config
switches in addition to `-y`.

### 4. Config / docs
- Add `defaults.auto_install_backend: false` to `default_config()` (read defensively with `.get`, so
  pre-existing configs without the key behave as `false` — no migration needed).
- Update `-y` help to mention env/config equivalents; document headless integration in README +
  AIM-MANUAL.
- Bump `VERSION` to `0.2.0`; re-publish to PyPI.

### Out of scope
Install-target cleanliness (keeps `pip3 install --break-system-packages`). The venv / `uv tool --with`
approach is a deployment recommendation / future item. The openspeechapi silent-fallback logging is
that project's concern, not aim's.

## Tests (stdlib unittest, all synthetic — no real backends/network)
- `_assume_yes`: env truthy matrix (`1/true/YES/on`→True; `0/false/""/unset`→False); config opt-in;
  `-y` flag; precedence (any one true → True).
- `_ensure_backend` non-TTY + no opt-in + non-json → returns `(False, …)`, `input()` is **never
  called** (monkeypatch `sys.stdin.isatty`→False and `aim.input` to raise if called), message
  mentions the tool + how to enable.
- `_ensure_backend` with `config defaults.auto_install_backend=True` → skips prompt, runs install
  (mock `_check_backend_available` + `subprocess.run`).
- `_ensure_backend` with `AIM_ASSUME_YES=1` → skips prompt, runs install.
- `--json` + no opt-in → `BACKEND_NOT_FOUND` unchanged.
- Interactive TTY (isatty→True) + answer "y" → installs (existing behavior preserved).

## Acceptance
1. Headless `AIM_ASSUME_YES=1 aim download ms:<id>` (no TTY) auto-installs the backend and proceeds —
   no prompt, no abort.
2. Headless `aim download ms:<id>` with no opt-in prints a clear stderr error (named tool + enable
   hint) and fails — never a silent `Aborted`, never hangs.
3. `defaults.auto_install_backend=true` in config has the same effect as `-y`.
4. `-y` and `--json` behave as before; full suite green, stdlib-only.
