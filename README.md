# aim — AI Model Manager

Unified CLI for managing AI models across multiple inference engines.

## Features

- **Multi-engine support**: Ollama, Hugging Face, MLX (omlx), ComfyUI, Whisper, Coqui TTS, SparkTTS, Piper, Fish Speech
- **Canonical Store**: Models organized by category in `~/AI/store/{category}/{model_id}/`, with symlinks/hardlinks back to engine directories
- **Deduplication**: Detect and deduplicate identical model files across engines via hardlinks
- **Registry**: Track all models in a single JSON registry with metadata, provenance, and provision info

## Docs

- Download PRD: `docs/download-prd.md`
- Download implementation plan: `docs/download-implementation-plan.md`
- Download JSON contract: `docs/download-json-contract.md`

## Install

```bash
# Symlink to PATH
ln -sf "$(pwd)/aim.py" ~/.local/bin/aim
chmod +x aim.py
```

## Usage

```bash
# Scan all engines and register models
aim scan

# List registered models
aim list
aim list --engine comfyui --sort size
aim list --json                      # JSON array output

# Show model details
aim info <model_id>
aim info <model_id> --json           # full metadata as JSON

# Resolve model to absolute path (for inference)
aim resolve <model_id>               # prints absolute path
aim resolve <model_id> --json        # full metadata + path + resolved_file
aim resolve <model_id> --engine comfyui  # prefer engine provision path

# Organize models into canonical store (preview)
aim organize

# Organize all non-CAS models
aim organize --all

# Organize a single model
aim organize <model_id>

# Download a model
aim download hf:org/repo
aim download ollama:model:tag
aim download hf:org/repo --json
aim download status <job_id> --json
aim download cancel <job_id>
aim download hf:org/repo --category llm/chat
aim download hf:org/repo --path /custom/path
aim download url:https://example.com/model.bin --no-progress
aim download url:https://example.com/model.bin --no-resume
aim download hf:org/repo --proxy http://127.0.0.1:7890 --retry 3 --retry-backoff 1.5

# Import/register existing local model path
aim import /path/to/local/model-dir --id my-model --category llm/chat

# Convert native CAS model to managed store model
aim convert hf-org-repo --new-id hf-org-repo-managed --category llm/chat

# Provision a model for an engine
aim provision <model_id> --engine comfyui

# Verify link integrity
aim verify
aim verify --fix

# Storage overview
aim status
aim status --by category

# Find duplicates
aim dedup
aim dedup --apply

# Find unregistered model files
aim orphans
```

## JSON Output

`aim list`, `aim info`, and `aim resolve` support `--json` for programmatic integration.
`aim download --json` emits JSONL progress events and a final JSON summary.

`aim resolve <id> --json` is the recommended single-call API — it returns the full model
metadata (superset of `aim info --json`) plus two extra fields:

```jsonc
{
  "id": "whisper-large-v3-turbo",
  "name": "whisper-large-v3-turbo",
  "source": { "type": "local", "repo_id": "openai/whisper-large-v3-turbo" },
  "format": "pt",
  "size_bytes": 1617941637,
  "category": "asr/model",
  "engines": ["whisper"],
  // ... all other model fields ...
  "path": "/Users/you/AI/store/asr/model/whisper-large-v3-turbo",
  "resolved_file": "/Users/you/AI/store/asr/model/whisper-large-v3-turbo/large-v3-turbo.pt"
}
```

| Field | Description |
|-------|-------------|
| `path` | Absolute directory (or file) path, resolved via provision or canonical store |
| `resolved_file` | Primary weight file inside the directory, or `null` for sharded / complex models |

Download placement policy:

1. `--path` has highest priority
2. Else auto place under `store/<category>/<model_id>/`
3. Missing category falls back to inferred category, then `uncategorized`

Download control flags:

1. `--no-progress`: only final summary (especially useful with `--json`)
2. `--resume` / `--no-resume`: toggle resume behavior
3. `--proxy --timeout --connect-timeout --retry --retry-backoff --max-speed --concurrency`

`resolved_file` detection: scans top-level weight files (`.safetensors`, `.pt`, `.pth`, `.gguf`,
`.bin`, `.onnx`). Single file → returned directly. Multiple → picks largest matching format.
Sharded models (`model-00001-of-00006`) → `null` (load from directory).

## Store Layout

```
~/AI/store/
├── asr/model/           — Whisper models
├── image-gen/
│   ├── checkpoint/      — FLUX, SDXL checkpoints
│   ├── lora/            — LoRA weights
│   ├── text-encoder/    — CLIP, T5 encoders
│   └── vae/             — VAE models
├── llm/chat/            — LLM chat models (MLX)
└── tts/
    ├── model/           — TTS models
    └── vocoder/         — Vocoder models
```

## How It Works

1. **`aim scan`** discovers models in engine directories, registers them
2. **`aim organize`** moves models into `store/{category}/{id}/`, replaces originals with:
   - **Directories** → symlink back to engine location
   - **Files in shared dirs** → hardlink back
3. **`aim verify`** checks all links are intact
4. **`aim provision`** creates links for a model in any supported engine

## Configuration

Config is stored at `~/.aim/config.json`. Registry at `~/.aim/registry.json`.

## Requirements

- Python 3.10+
- macOS / Linux
- Optional: `gh` CLI, `hfd.sh` for downloads

## Testing

```bash
# Run syntax check
make lint

# Run all tests (unit + e2e)
make test

# Run only unit tests
make test-unit

# Run only end-to-end tests
make test-e2e
```

## License

MIT
