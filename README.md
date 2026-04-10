# aim — AI Model Manager

Unified CLI for managing AI models across multiple inference engines.

## Features

- **Multi-engine support**: Ollama, Hugging Face, MLX (omlx), ComfyUI, Whisper, Coqui TTS, SparkTTS, Piper, Fish Speech
- **Canonical Store**: Models organized by category in `~/AI/store/{category}/{model_id}/`, with symlinks/hardlinks back to engine directories
- **Deduplication**: Detect and deduplicate identical model files across engines via hardlinks
- **Registry**: Track all models in a single JSON registry with metadata, provenance, and provision info

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

## License

MIT
