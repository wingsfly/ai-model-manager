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

# Show model details
aim info <model_id>

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
